// metal/kernels/dense_mm.metal - dense projections for a PROMPT chunk (after quant/units/mma.metal):
//   act_f16    X[t][k] -> half (the matrix units' operand)
//   dq_f16<F>  a quantized matrix -> half rows (ref_quant's values rounded to fp16), once per chunk
//   gemm_f16   Y[t][n] = Xh[t] . Wh[n] (fp32) by MetalPerformancePrimitives matmul2d, TM x TN tiles, K dynamic;
//              rows of Y past n_out / t past n_tok are not written (the tensors carry the extents)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

kernel void act_f16(device const float* x [[buffer(0)]], device half* xh [[buffer(1)]], constant uint& n [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid < n) xh[gid] = half(x[gid]);
}

struct DqArgs {
    uint n_in;       // values per row
    uint n_out;      // rows
    uint row_bytes;
    uint perm;       // 0: row r -> r; p > 0: row r = c * (n_out / p) + d -> d * p + c (the HC up matrix with the p
                     // streams of an output interleaved, for gemm_hc_up's epilogue)
};

// one thread per unit: grid (units per row / 64, n_out), 64 threads
template <int F>
kernel void dq_f16(device const uchar* W [[buffer(0)]], device half* Wh [[buffer(1)]], constant DqArgs& a [[buffer(2)]],
                   uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int UNIT = fmt<F>::UNIT, H = UNIT / 2;
    threadgroup uint tab[fmt<F>::TAB];
    load_tab<F>(tab, tid, 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int u = int(tg.x) * 64 + tid, r = int(tg.y);
    if (u * UNIT >= int(a.n_in)) return;
    // the unit's chunk origin: a 128-wide chunk holds whole units of every format
    const int k0 = (u * UNIT) / 128 * 128;
    float w[UNIT];
    int p0, p1;
    DW<F>::run(W + ulong(r) * a.row_bytes, u, k0, w, p0, p1, tab);
    const int q = a.perm ? int(a.n_out / a.perm) : 1;
    const int ro = a.perm ? (r % q) * int(a.perm) + r / q : r;
    device half* o = Wh + ulong(ro) * a.n_in + k0;
    for (int j = 0; j < H; ++j) {
        o[p0 + j] = half(w[j]);
        o[p1 + j] = half(w[H + j]);
    }
}

struct GemmArgs { int M, N, K; int pad; };   // Y is M rows of N floats
struct DotArgs { uint M, K; };               // Y[M] = Xh[M x K] . half(W[K])

template <> [[host_name("dq_f16_8")]]
kernel void dq_f16<T_Q8_0>(device const uchar* W [[buffer(0)]], device half* Wh [[buffer(1)]], constant DqArgs& a [[buffer(2)]],
                           uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    const int u = int(tg.x) * 64 + tid, r = int(tg.y);
    if (u * 32 >= int(a.n_in)) return;
    device const uchar* b = W + ulong(r) * a.row_bytes + ulong(u) * 34;
    const float d = rdh(b);
    device const packed_ushort4* p = (device const packed_ushort4*)(b + 2);
    const int q = a.perm ? int(a.n_out / a.perm) : 1;
    const int ro = a.perm ? (r % q) * int(a.perm) + r / q : r;
    device half4* o4 = (device half4*)(Wh + ulong(ro) * a.n_in + ulong(u) * 32);
    for (int j = 0; j < 4; ++j) {
        const ushort4 v = ushort4(p[j]);
        const half2 a0 = half2(float2(float(as_type<char>(uchar(v.x & 255))), float(as_type<char>(uchar(v.x >> 8)))) * d);
        const half2 a1 = half2(float2(float(as_type<char>(uchar(v.y & 255))), float(as_type<char>(uchar(v.y >> 8)))) * d);
        const half2 a2 = half2(float2(float(as_type<char>(uchar(v.z & 255))), float(as_type<char>(uchar(v.z >> 8)))) * d);
        const half2 a3 = half2(float2(float(as_type<char>(uchar(v.w & 255))), float(as_type<char>(uchar(v.w >> 8)))) * d);
        o4[2 * j]     = half4(a0, a1);
        o4[2 * j + 1] = half4(a2, a3);
    }
}

kernel void dot_f16_f32(device const half* A [[buffer(0)]], device const float* W [[buffer(1)]],
                        device float* Y [[buffer(2)]], constant DotArgs& a [[buffer(3)]],
                        uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                        ushort lane [[thread_index_in_simdgroup]]) {
    const int t = int(tg) * 8 + int(sg);
    if (t >= int(a.M)) return;
    device const half* ar = A + ulong(t) * a.K;
    float acc = 0.0f;
    for (int k = int(lane) * 4; k + 3 < int(a.K); k += 128) {
        const half4 av = *(device const half4*)(ar + k);
        const float4 wv = *(device const float4*)(W + k);
        acc += dot(float4(av), float4(half4(wv)));
    }
    const int tail = int(a.K) & ~3;
    if (tail + int(lane) < int(a.K))
        acc += float(ar[tail + int(lane)]) * float(half(W[tail + int(lane)]));
    acc = simd_sum(acc);
    if (lane == 0) Y[t] = acc;
}

template <int TM, int TN, int SG>
kernel void gemm_f16(device half* A [[buffer(0)]], device half* B [[buffer(1)]], device float* Y [[buffer(2)]],
                     constant GemmArgs& a [[buffer(3)]], uint3 tg [[threadgroup_position_in_grid]]) {
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tA(A, dextents<int32_t, 2>(a.K, a.M));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tB(B, dextents<int32_t, 2>(a.K, a.N));
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tY(Y, dextents<int32_t, 2>(a.N, a.M));
    constexpr auto desc = matmul2d_descriptor(TM, TN, dynamic_length_v<int>, false, true, false);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto mA = tA.slice(0, int(tg.y) * TM);
    auto mB = tB.slice(0, int(tg.x) * TN);
    auto mY = tY.slice(int(tg.x) * TN, int(tg.y) * TM);
    op.run(mA, mB, mY);
}

#define DQ(F) template [[host_name("dq_f16_" #F)]] kernel decltype(dq_f16<F>) dq_f16<F>;
#define DQ_(F) DQ(F)
DQ_(T_F32) DQ_(T_BF16) DQ_(T_Q4_0) DQ_(T_Q5_0) DQ_(T_IQ4_NL) DQ_(T_IQ4_XS) DQ_(T_Q2_0) DQ_(T_IQ3_XXS) DQ_(T_IQ3_S)
DQ_(T_Q2_K) DQ_(T_Q3_K) DQ_(T_Q4_K) DQ_(T_Q5_K) DQ_(T_Q6_K) DQ_(T_NVFP4)
template [[host_name("gemm_f16_64_64_4")]] kernel decltype(gemm_f16<64, 64, 4>) gemm_f16<64, 64, 4>;
template [[host_name("gemm_f16_128_64_4")]] kernel decltype(gemm_f16<128, 64, 4>) gemm_f16<128, 64, 4>;
template [[host_name("gemm_f16_64_128_4")]] kernel decltype(gemm_f16<64, 128, 4>) gemm_f16<64, 128, 4>;
template [[host_name("gemm_f16_64_32_4")]] kernel decltype(gemm_f16<64, 32, 4>) gemm_f16<64, 32, 4>;
template [[host_name("gemm_f16_128_128_4")]] kernel decltype(gemm_f16<128, 128, 4>) gemm_f16<128, 128, 4>;
template [[host_name("gemm_f16_128_128_8")]] kernel decltype(gemm_f16<128, 128, 8>) gemm_f16<128, 128, 8>;
template [[host_name("gemm_f16_128_64_8")]] kernel decltype(gemm_f16<128, 64, 8>) gemm_f16<128, 64, 8>;
template [[host_name("gemm_f16_64_64_8")]] kernel decltype(gemm_f16<64, 64, 8>) gemm_f16<64, 64, 8>;
template [[host_name("gemm_f16_256_64_8")]] kernel decltype(gemm_f16<256, 64, 8>) gemm_f16<256, 64, 8>;
template [[host_name("gemm_f16_64_256_8")]] kernel decltype(gemm_f16<64, 256, 8>) gemm_f16<64, 256, 8>;
template [[host_name("gemm_f16_128_32_4")]] kernel decltype(gemm_f16<128, 32, 4>) gemm_f16<128, 32, 4>;
template [[host_name("gemm_f16_32_128_4")]] kernel decltype(gemm_f16<32, 128, 4>) gemm_f16<32, 128, 4>;
template [[host_name("gemm_f16_16_64_2")]] kernel decltype(gemm_f16<16, 64, 2>) gemm_f16<16, 64, 2>;
