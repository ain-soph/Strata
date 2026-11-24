// metal/kernels/moe_dcomb.metal - the routed down with the combine folded in (after moe.metal).

// ---- moe_down + moe_combine_sh2 in one dispatch ("the last threadgroup finishes") -----------------------------------
// moe_down_comb_<type>_<L>_<R>_<S>: moe_down's grid (x = a block of RPT output rows, y = the expert groups); every
// threadgroup writes its ye rows (moe_down's values), fences them at device scope and bumps the row block's counter; the
// threadgroup that brings it to the group count combines the block: bo = sigmoid(sg) * ysh + the ten experts' ye rows,
// summed as moe_combine_sh2 sums them, for every token of the window, then resets the counter.  No threadgroup waits for
// another (router_route's pattern, kernels/router.metal).  ye is read back through a device-coherent pointer (the M5 Ultra's
// two dies).  Buffers 0..5 as moe_down's (no arena), 6 ysh, 7 sg (one logit a token), 8 bo, 9 the counters (one uint a
// row block at a 128-byte stride, DCOMB_STRIDE uints: no two row blocks share a cache line; zero between dispatches;
// the host keeps the grid's x <= 256).  Threadgroups past the group count (grid y > ng) leave at once, before counting.
template <int F, int LPR, int NR, int SG>
kernel void moe_down_comb(device const uchar* W [[buffer(0)]], device const float* h [[buffer(1)]],
                          device coherent(device) float* ye [[buffer(2)]], device const int* plan [[buffer(3)]],
                          constant MoeArgs& a [[buffer(4)]], device const float* hs16 [[buffer(5)]],
                          device const float* ysh [[buffer(6)]], device const float* sgl [[buffer(7)]],
                          device float* bo [[buffer(8)]], device atomic_uint* cnt [[buffer(9)]],
                          uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                          ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int RG = 32 / LPR, RPT = SG * RG * NR, TK = ROUTE_K;
    constexpr uint DCOMB_STRIDE = 32u;   // 128 B
    const int g = int(tg.y), ng = plan[P_NG];
    if (g >= ng) return;   // uniform over the threadgroup; such a threadgroup must not count
    threadgroup uint tab[fmt<F>::TAB];
    threadgroup uint last;
    device atomic_uint* ctr = cnt + tg.x * DCOMB_STRIDE;
    load_tab<F>(tab, tid, SG * 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        const int li = lane % LPR;
        const int row0 = ((int(tg.x) * SG + sg) * RG + lane / LPR) * NR;
        const ulong ex = (ulong)plan[P_GEXP + g];
        const int c = plan[P_GSTART + g], ne = plan[P_GSTART + g + 1] - c;
        device const uchar* d0 = W + ex * a.d_stride + a.down_off + (ulong)row0 * a.d_row;
        typedef device coherent(device) float* YP;
        switch (ne) {
            case 1: dn_body<F, LPR, NR, 1, YP>(d0, h, hs16, ye, plan, c, row0, li, a, tab); break;
            case 2: dn_body<F, LPR, NR, 2, YP>(d0, h, hs16, ye, plan, c, row0, li, a, tab); break;
            case 3: dn_body<F, LPR, NR, 3, YP>(d0, h, hs16, ye, plan, c, row0, li, a, tab); break;
            case 4: dn_body<F, LPR, NR, 4, YP>(d0, h, hs16, ye, plan, c, row0, li, a, tab); break;
            default:
                for (int k = 0; k < ne; k += 4) {
                    const int m = min(4, ne - k);
                    if (m == 4) dn_body<F, LPR, NR, 4, YP>(d0, h, hs16, ye, plan, c + k, row0, li, a, tab);
                    else if (m == 3) dn_body<F, LPR, NR, 3, YP>(d0, h, hs16, ye, plan, c + k, row0, li, a, tab);
                    else if (m == 2) dn_body<F, LPR, NR, 2, YP>(d0, h, hs16, ye, plan, c + k, row0, li, a, tab);
                    else dn_body<F, LPR, NR, 1, YP>(d0, h, hs16, ye, plan, c + k, row0, li, a, tab);
                }
        }
    }
    // every thread's ye writes are committed before thread 0 signals (router_route's sequence)
    threadgroup_barrier(mem_flags::mem_device);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    if (tid == 0) last = atomic_fetch_add_explicit(ctr, 1u, memory_order_relaxed) == uint(ng) - 1u ? 1u : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!last) return;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_device);
    const int n = int(a.n_embd), rb = int(tg.x) * RPT;
    for (int i = tid; i < RPT * int(a.n_tok); i += SG * 32) {
        const int t = i / RPT, k = rb + i % RPT;
        if (k >= n) continue;
        int e[TK];
        for (int j = 0; j < TK; ++j) e[j] = plan[P_TOKENT + t * TK + j];
        const float gs = 1.0f / (1.0f + exp(-sgl[t]));
        const float shv = ysh[t * n + k];
        float y[TK];
        for (int j = 0; j < TK; ++j) y[j] = ye[e[j] * n + k];
        float s = 0.0f;
        for (int j = 0; j < TK; ++j) s += y[j];
        bo[t * n + k] = fma(gs, shv, s);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) atomic_store_explicit(ctr, 0u, memory_order_relaxed);
}
#define DNC_(F, L, R, S) template [[host_name("moe_down_comb_" #F "_" #L "_" #R "_" #S)]] kernel decltype(moe_down_comb<F, L, R, S>) moe_down_comb<F, L, R, S>;
#define DNC(F, L, R, S) DNC_(F, L, R, S)
DNC(T_IQ4_NL, 4, 2, 4) DNC(T_IQ4_NL, 4, 1, 4) DNC(T_IQ4_NL, 8, 2, 4) DNC(T_IQ4_NL, 8, 1, 4)
