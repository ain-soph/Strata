// metal/tests/gdn_segs_parity.cpp - concurrent MTP (ain-soph serving): gdn_step with GDN_ROWS + GDN_SEGS (function constants 31, 33)
// against the solo kernels it stands for.  A batch window's rows are slots; the rows of one slot are adjacent (its token and its
// drafts).  For every segment of every layout:
//   - y of its rows bitwise the solo gdn_step#n window's (n = the segment's rows) from the slot's state and conv history;
//   - a one-row segment commits bitwise as gdn_step#1 with commit (state, and the history into the other parity);
//   - a longer segment leaves the slot's state and both history parities untouched (Window::commit_rows folds later).
// And with only one-row segments the GDN_SEGS kernel equals the GDN_ROWS kernel bitwise (y, states, histories).
#include "mtl_rt.h"
#include "ref_quant.h"
#include "engine/geo.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
constexpr int S = strata_metal::geo::SS, HK = strata_metal::geo::HK, HV = strata_metal::geo::HV, C = (2 * HK + HV) * S, MAXT = 8;
constexpr int NSET = 3;
struct GdnArgs { uint32_t h_k, h_v, channels, commit; float eps; uint32_t plane; };
}  // namespace

int main() {
    mtl::Context cx;
    const int lib = cx.compile({"gdn.metal"}, "#define NO_GGML_COMMON 1\n");
    std::printf("device %s\n", cx.device_name().c_str());
    const size_t SE = (size_t) S * HV * S, HE = (size_t) C * 3;
    mtl::Buffer big = cx.alloc(4 * SE * NSET), hb = cx.alloc(4 * HE * 2 * NSET), big0 = cx.alloc(4 * SE * NSET), hb0 = cx.alloc(4 * HE * 2 * NSET),
                qkv = cx.alloc(4 * C * MAXT), al = cx.alloc(4 * HV * MAXT), be = cx.alloc(4 * HV * MAXT), z = cx.alloc(4 * HV * S * MAXT),
                cw = cx.alloc(4 * C * 4), dt = cx.alloc(4 * HV), sa = cx.alloc(4 * HV), gm = cx.alloc(4 * S), y = cx.alloc(4 * HV * S * MAXT),
                y2 = cx.alloc(4 * HV * S * MAXT), st = cx.alloc(4 * SE), hi = cx.alloc(4 * HE), ho = cx.alloc(4 * HE),
                rinfo = cx.alloc(4 * (8 + 4 * MAXT));
    ref::Rng r(23);
    for (size_t i = 0; i < SE * NSET; ++i) big0.as<float>()[i] = r.sym() * 0.1f;
    for (size_t i = 0; i < HE * 2 * NSET; ++i) hb0.as<float>()[i] = r.sym();
    for (int i = 0; i < C * 4; ++i) cw.as<float>()[i] = r.sym() * 0.5f;
    for (int i = 0; i < HV; ++i) { dt.as<float>()[i] = r.sym(); sa.as<float>()[i] = -std::exp(r.sym()); }
    for (int i = 0; i < S; ++i) gm.as<float>()[i] = 1.0f + 0.2f * r.sym();
    mtl::Pipeline prow = cx.pipeline(lib, "gdn_step", mtl::Constants().i(2, 1).b(31, true));
    mtl::Pipeline pseg = cx.pipeline(lib, "gdn_step", mtl::Constants().i(2, 1).b(31, true).b(33, true));
    std::printf("gdn_step#segs: max %d threads per threadgroup (the kernel needs 512)\n", pseg.max_threads);
    if (pseg.max_threads < 512) { std::printf("FAIL\n"); return 1; }
    auto rows_disp = [&](const mtl::Pipeline& p, int nrows) {
        GdnArgs a{HK, HV, C, 1u, 1e-6f, 0};
        mtl::Dispatch d;
        d.p = &p;
        d.args = {mtl::Arg::B(big), mtl::Arg::B(hb), mtl::Arg::B(qkv), mtl::Arg::B(al), mtl::Arg::B(be), mtl::Arg::B(z), mtl::Arg::B(cw),
                  mtl::Arg::B(dt), mtl::Arg::B(sa), mtl::Arg::B(gm), mtl::Arg::B(y), mtl::Arg::V(a), mtl::Arg::B(hb), mtl::Arg::B(rinfo)};
        d.grid[0] = HV;
        d.grid[1] = (uint32_t) nrows;
        d.tg[0] = 512;
        return d;
    };
    // (layout: the physical set of each row; parities of the sets)
    const std::vector<std::vector<int>> layouts = {{0, 0, 0, 1, 1}, {1, 0, 0}, {0, 1}, {2, 2, 1, 1, 1, 0}, {1, 1, 2, 2}, {0, 1, 2}, {2, 2, 2}};
    const int par[NSET] = {0, 1, 0};
    int fails = 0;
    for (size_t li = 0; li < layouts.size(); ++li) {
        const std::vector<int>& L = layouts[li];
        const int n = (int) L.size();
        for (int i = 0; i < C * n; ++i) qkv.as<float>()[i] = r.sym() * 2.0f;
        for (int i = 0; i < HV * n; ++i) { al.as<float>()[i] = r.sym() * 2.0f; be.as<float>()[i] = r.sym() * 2.0f; }
        for (int i = 0; i < HV * S * n; ++i) z.as<float>()[i] = r.sym() * 2.0f;
        uint32_t* ri = rinfo.as<uint32_t>();
        std::memset(ri, 0, 4 * (8 + 4 * MAXT));
        ri[5] = (uint32_t) SE;
        ri[6] = (uint32_t) HE;
        for (int t = 0; t < n; ++t) { ri[8 + 4 * t] = (uint32_t) L[t]; ri[8 + 4 * t + 1] = 100 + t; ri[8 + 4 * t + 2] = (uint32_t) par[L[t]]; }
        std::memcpy(big.ptr, big0.ptr, 4 * SE * NSET);
        std::memcpy(hb.ptr, hb0.ptr, 4 * HE * 2 * NSET);
        std::memset(y.ptr, 0, 4 * HV * S * MAXT);
        cx.run({rows_disp(pseg, n)});
        std::vector<float> yseg(y.as<float>(), y.as<float>() + HV * S * n), bseg(big.as<float>(), big.as<float>() + SE * NSET),
            hseg(hb.as<float>(), hb.as<float>() + HE * 2 * NSET);
        bool single = true;
        for (int t = 1; t < n; ++t) single = single && L[t] != L[t - 1];
        bool ok = true;
        std::string why;
        if (single) {   // the plain GDN_ROWS kernel: bitwise the same
            std::memcpy(big.ptr, big0.ptr, 4 * SE * NSET);
            std::memcpy(hb.ptr, hb0.ptr, 4 * HE * 2 * NSET);
            cx.run({rows_disp(prow, n)});
            if (std::memcmp(y.ptr, yseg.data(), 4 * HV * S * n) || std::memcmp(big.ptr, bseg.data(), 4 * SE * NSET) ||
                std::memcmp(hb.ptr, hseg.data(), 4 * HE * 2 * NSET)) { ok = false; why += " rows!=segs"; }
        }
        for (int t = 0; t < n;) {
            int m = 1;
            while (t + m < n && L[t + m] == L[t]) ++m;
            const int s = L[t];
            std::memcpy(st.ptr, big0.as<float>() + SE * s, 4 * SE);
            std::memcpy(hi.ptr, hb0.as<float>() + HE * (2 * s + par[s]), 4 * HE);
            std::memset(ho.ptr, 0, 4 * HE);
            mtl::Pipeline pn = cx.pipeline(lib, "gdn_step", mtl::Constants().i(2, m));
            GdnArgs a{HK, HV, C, m == 1 ? 1u : 0u, 1e-6f, 0};
            mtl::Dispatch d;
            d.p = &pn;
            d.args = {mtl::Arg::B(st), mtl::Arg::B(hi), mtl::Arg::B(qkv, 4 * (size_t) C * t), mtl::Arg::B(al, 4 * (size_t) HV * t),
                      mtl::Arg::B(be, 4 * (size_t) HV * t), mtl::Arg::B(z, 4 * (size_t) HV * S * t), mtl::Arg::B(cw), mtl::Arg::B(dt),
                      mtl::Arg::B(sa), mtl::Arg::B(gm), mtl::Arg::B(y2, 4 * (size_t) HV * S * t), mtl::Arg::V(a), mtl::Arg::B(ho)};
            d.grid[0] = HV;
            d.tg[0] = 512;
            cx.run({d});
            if (std::memcmp(y2.as<float>() + (size_t) HV * S * t, yseg.data() + (size_t) HV * S * t, 4 * (size_t) HV * S * m)) {
                ok = false; why += " y(seg@" + std::to_string(t) + ")";
            }
            const float* sb = bseg.data() + SE * s;
            if (m == 1) {
                if (std::memcmp(sb, st.ptr, 4 * SE)) { ok = false; why += " commit-state@" + std::to_string(t); }
                if (std::memcmp(hseg.data() + HE * (2 * s + 1 - par[s]), ho.ptr, 4 * HE)) { ok = false; why += " commit-hist@" + std::to_string(t); }
            } else {
                if (std::memcmp(sb, big0.as<float>() + SE * s, 4 * SE) ||
                    std::memcmp(hseg.data() + HE * 2 * s, hb0.as<float>() + HE * 2 * s, 4 * HE * 2)) { ok = false; why += " touched@" + std::to_string(t); }
            }
            t += m;
        }
        // sets with no rows: untouched
        for (int s = 0; s < NSET; ++s) {
            bool used = false;
            for (int v : L) used = used || v == s;
            if (!used && (std::memcmp(bseg.data() + SE * s, big0.as<float>() + SE * s, 4 * SE) ||
                          std::memcmp(hseg.data() + HE * 2 * s, hb0.as<float>() + HE * 2 * s, 4 * HE * 2))) { ok = false; why += " other-set"; }
        }
        std::printf("layout %zu {", li);
        for (int t = 0; t < n; ++t) std::printf("%s%d", t ? "," : "", L[t]);
        std::printf("}: %s%s\n", ok ? "PASS" : "FAIL", why.c_str());
        fails += !ok;
    }
    std::printf("%s\n", fails ? "SOME FAILED" : "ALL PASS");
    return fails ? 1 : 0;
}
