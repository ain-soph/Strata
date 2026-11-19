// metal/tests/dense_mm_parity.cpp - dense prompt projections (dense_mm.metal): dequantize to fp16 once, then the
// MPP GEMM. Parity vs a reference with the same fp16 operands (and the exact float one), every dense format and
// awkward row counts; then the speed on the packs' shapes for a 2048-token chunk.
//   dense_mm_parity            parity
//   dense_mm_parity --bench    speed
#include "mtl_rt.h"
#include "ref_quant.h"
#include "engine/geo.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {
struct DqArgs { uint32_t n_in, n_out, row_bytes, perm; };
struct GemmArgs { int M, N, K, pad; };
struct DotArgs { uint32_t M, K; };
struct EpiArgs { uint32_t n_tok, n_embd, hc, hc_lr; };
double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
float h16(float v) { return ref::f16_to_f32(ref::f32_to_f16(v)); }
int unit_of(int t) {
    switch (t) {
        case ref::Q2_0: case ref::NVFP4: case ref::IQ3_XXS: case ref::IQ3_S: case ref::BF16: case ref::F32: return 16;
        default: return 32;
    }
}
}  // namespace

int main(int argc, char** argv) {
    const std::string arg1 = argc > 1 ? argv[1] : "";
    const bool bench = arg1 == "--bench";
    const bool m5u = arg1 == "--m5u";
    mtl::Context cx;
    const int lib = cx.compile({"quant.metal", "units.metal", "mma.metal", "dense_mm.metal"});
    std::printf("device %s\n", cx.device_name().c_str());
    mtl::Pipeline pact = cx.pipeline(lib, "act_f16");
    auto run_all = [&](int ty, const mtl::Buffer& W, size_t woff, int n_in, int n_out, int T, const mtl::Buffer& x,
                       const mtl::Buffer& xh, const mtl::Buffer& wh, const mtl::Buffer& y, const std::string& gk,
                       std::vector<mtl::Dispatch>& ds, int which) {
        static std::vector<mtl::Pipeline*> keep;
        mtl::Pipeline* pdq = new mtl::Pipeline(cx.pipeline(lib, "dq_f16_" + std::to_string(ty)));
        mtl::Pipeline* pg = new mtl::Pipeline(cx.pipeline(lib, gk));
        keep.push_back(pdq);
        keep.push_back(pg);
        int tm = 64, tn = 64, sg = 4;
        std::sscanf(gk.c_str(), "gemm_f16_%d_%d_%d", &tm, &tn, &sg);
        if (which & 1) {
            const uint32_t n = (uint32_t)T * n_in;
            mtl::Dispatch d; d.p = &pact; d.args = {mtl::Arg::B(x), mtl::Arg::B(xh), mtl::Arg::V(n)};
            d.grid[0] = (n + 255) / 256; d.tg[0] = 256; ds.push_back(d);
        }
        if (which & 2) {
            DqArgs a{(uint32_t)n_in, (uint32_t)n_out, (uint32_t)ref::row_bytes(ty, n_in), 0u};
            mtl::Dispatch d; d.p = pdq; d.args = {mtl::Arg::B(W, woff), mtl::Arg::B(wh), mtl::Arg::V(a)};
            d.grid[0] = (n_in / unit_of(ty) + 63) / 64; d.grid[1] = n_out; d.tg[0] = 64; ds.push_back(d);
        }
        if (which & 4) {
            GemmArgs g{T, n_out, n_in, 0};
            mtl::Dispatch d; d.p = pg; d.args = {mtl::Arg::B(xh), mtl::Arg::B(wh), mtl::Arg::B(y), mtl::Arg::V(g)};
            d.grid[0] = (n_out + tn - 1) / tn; d.grid[1] = (T + tm - 1) / tm; d.tg[0] = 32 * sg; ds.push_back(d);
        }
    };
    if (m5u) {
        std::setvbuf(stdout, nullptr, _IOLBF, 0);
        int fails = 0;
        // 1. dq_f16_8 (Q8_0) parity with perm=0 and perm=4
        for (uint32_t perm : {0u, 4u}) {
            const int n_in = 2560, n_out = 324;
            const size_t rb = ref::row_bytes(ref::Q8_0, n_in);
            mtl::Buffer W = cx.alloc(rb * n_out), wh = cx.alloc(2 * (size_t)n_out * n_in);
            ref::random_blocks(ref::Q8_0, W.as<uint8_t>(), n_out, n_in, 17 + (int)perm, 0.02f);
            std::vector<float> wf((size_t)n_in * n_out);
            ref::dequant(ref::Q8_0, W.ptr, wf.data(), wf.size());
            mtl::Pipeline pdq = cx.pipeline(lib, "dq_f16_8");
            DqArgs a{(uint32_t)n_in, (uint32_t)n_out, (uint32_t)rb, perm};
            mtl::Dispatch d; d.p = &pdq; d.args = {mtl::Arg::B(W), mtl::Arg::B(wh), mtl::Arg::V(a)};
            d.grid[0] = (n_in / 32 + 63) / 64; d.grid[1] = n_out; d.tg[0] = 64;
            cx.run({d});
            const int q = perm ? n_out / (int)perm : 1;
            int mismatches = 0;
            for (int r = 0; r < n_out; ++r) {
                const int ro = perm ? (r % q) * (int)perm + r / q : r;
                for (int k = 0; k < n_in; ++k) {
                    const uint16_t got = wh.as<uint16_t>()[(size_t)ro * n_in + k];
                    const uint16_t exp = ref::f32_to_f16(wf[(size_t)r * n_in + k]);
                    if (got != exp) ++mismatches;
                }
            }
            fails += mismatches > 0;
            std::printf("m5u parity dq_f16_8 perm=%u: mismatches=%d  %s\n", perm, mismatches, mismatches == 0 ? "PASS" : "FAIL");
        }
        // 2. dot_f16_f32 parity (T=1024 and T=73, K=2560)
        for (int T : {73, 1024}) {
            const int K = 2560;
            mtl::Buffer x = cx.alloc(4 * (size_t)T * K), xh = cx.alloc(2 * (size_t)T * K),
                        w = cx.alloc(4 * (size_t)K), y = cx.alloc(4 * (size_t)T);
            ref::Rng r(41 + T);
            for (size_t i = 0; i < (size_t)T * K; ++i) x.as<float>()[i] = r.sym();
            for (int k = 0; k < K; ++k) w.as<float>()[k] = r.sym();
            const uint32_t n = (uint32_t)T * K;
            mtl::Dispatch da; da.p = &pact; da.args = {mtl::Arg::B(x), mtl::Arg::B(xh), mtl::Arg::V(n)};
            da.grid[0] = (n + 255) / 256; da.tg[0] = 256;
            mtl::Pipeline pdot = cx.pipeline(lib, "dot_f16_f32");
            DotArgs g{(uint32_t)T, (uint32_t)K};
            mtl::Dispatch dd; dd.p = &pdot; dd.args = {mtl::Arg::B(xh), mtl::Arg::B(w), mtl::Arg::B(y), mtl::Arg::V(g)};
            dd.grid[0] = (T + 7) / 8; dd.tg[0] = 256;
            cx.run({da, dd});
            double se16 = 0, sr = 0;
            for (int t = 0; t < T; ++t) {
                double a16 = 0;
                for (int k = 0; k < K; ++k) a16 += (double)h16(x.as<float>()[(size_t)t * K + k]) * h16(w.as<float>()[k]);
                const double gval = y.as<float>()[t];
                se16 += (gval - a16) * (gval - a16);
                sr += a16 * a16;
            }
            const double r16 = std::sqrt(se16 / sr);
            const bool ok = r16 < 1e-5;
            fails += !ok;
            std::printf("m5u parity dot_f16_f32 T=%d K=%d: rel %.2e  %s\n", T, K, r16, ok ? "PASS" : "FAIL");
        }
        // 3. New gemm_f16 tiles parity (T=73, N=324, K=640)
        const std::vector<const char*> gemm_tiles = {
            "gemm_f16_64_64_4", "gemm_f16_128_64_4", "gemm_f16_64_128_4", "gemm_f16_64_32_4",
            "gemm_f16_128_128_4", "gemm_f16_128_128_8", "gemm_f16_128_64_8", "gemm_f16_64_64_8",
            "gemm_f16_256_64_8", "gemm_f16_64_256_8", "gemm_f16_128_32_4", "gemm_f16_32_128_4",
            "gemm_f16_16_64_2"
        };
        {
            const int T = 73, Nn = 324, K = 640;
            mtl::Buffer W = cx.alloc(ref::row_bytes(ref::BF16, K) * Nn), x = cx.alloc(4 * (size_t)T * K),
                        xh = cx.alloc(2 * (size_t)T * K), wh = cx.alloc(2 * (size_t)Nn * K), y = cx.alloc(4 * (size_t)T * Nn);
            ref::random_blocks(ref::BF16, W.as<uint8_t>(), Nn, K, 29, 0.02f);
            ref::Rng r(13);
            for (size_t i = 0; i < (size_t)T * K; ++i) x.as<float>()[i] = r.sym();
            std::vector<float> wf((size_t)Nn * K);
            ref::dequant(ref::BF16, W.ptr, wf.data(), wf.size());
            std::vector<double> ref_y((size_t)T * Nn, 0.0);
            for (int t = 0; t < T; ++t)
                for (int n = 0; n < Nn; ++n) {
                    double acc = 0;
                    for (int k = 0; k < K; ++k) acc += (double)h16(x.as<float>()[(size_t)t * K + k]) * h16(wf[(size_t)n * K + k]);
                    ref_y[(size_t)t * Nn + n] = acc;
                }
            for (const char* gk : gemm_tiles) {
                std::memset(y.ptr, 0, 4 * (size_t)T * Nn);
                std::vector<mtl::Dispatch> ds;
                run_all(ref::BF16, W, 0, K, Nn, T, x, xh, wh, y, gk, ds, 7);
                cx.run(ds);
                double se16 = 0, sr = 0;
                for (size_t i = 0; i < (size_t)T * Nn; ++i) {
                    const double e = y.as<float>()[i] - ref_y[i];
                    se16 += e * e; sr += ref_y[i] * ref_y[i];
                }
                const double r16 = std::sqrt(se16 / sr);
                const bool ok = r16 < 1e-5;
                fails += !ok;
                std::printf("m5u parity %-20s T=%d N=%d K=%d: rel %.2e  %s\n", gk, T, Nn, K, r16, ok ? "PASS" : "FAIL");
            }
        }
        // 4. gemm_hc_up parity & T=1024 benchmark
        const int lprompt = cx.compile({"quant.metal", "units.metal", "mma.metal", "dense_mm.metal", "gr.metal", "prompt.metal"}, strata_metal::geo::prelude());
        const std::vector<const char*> hc_tiles = {
            "gemm_hc_up_64_64_4", "gemm_hc_up_64_32_4", "gemm_hc_up_32_64_4"
        };
        {
            constexpr int N = strata_metal::geo::N, HC = strata_metal::geo::HC, LR = strata_metal::geo::LR;
            const int T = 1024;
            mtl::Buffer lo = cx.alloc(2 * (size_t)T * LR), wu = cx.alloc(2 * (size_t)(HC * N) * LR),
                        xnh = cx.alloc(2 * (size_t)T * HC * N), mref = cx.alloc(4 * (size_t)T * N),
                        mtest = cx.alloc(4 * (size_t)T * N), mixh = cx.alloc(2 * (size_t)T * N);
            ref::Rng r(77);
            for (size_t i = 0; i < (size_t)T * LR; ++i) lo.as<uint16_t>()[i] = ref::f32_to_f16(r.sym() * 0.5f);
            for (size_t i = 0; i < (size_t)(HC * N) * LR; ++i) wu.as<uint16_t>()[i] = ref::f32_to_f16(r.sym() * 0.1f);
            for (size_t i = 0; i < (size_t)T * HC * N; ++i) xnh.as<uint16_t>()[i] = ref::f32_to_f16(r.sym());
            auto make_hc = [&](const char* name, const mtl::Pipeline* p, const mtl::Buffer& out, int ntok) {
                int tm = 64, tn = 64, sg = 4;
                std::sscanf(name, "gemm_hc_up_%d_%d_%d", &tm, &tn, &sg);
                GemmArgs g{ntok, HC * N, LR, 0};
                EpiArgs ea{(uint32_t)ntok, (uint32_t)N, (uint32_t)HC, (uint32_t)LR};
                mtl::Dispatch d; d.p = p;
                d.args = {mtl::Arg::B(lo), mtl::Arg::B(wu), mtl::Arg::B(xnh), mtl::Arg::B(out), mtl::Arg::B(mixh), mtl::Arg::V(g), mtl::Arg::V(ea)};
                d.grid[0] = (HC * N) / tn; d.grid[1] = (ntok + tm - 1) / tm; d.tg[0] = 32 * sg;
                return d;
            };
            mtl::Pipeline pref = cx.pipeline(lprompt, "gemm_hc_up_64_64_4");
            cx.run({make_hc("gemm_hc_up_64_64_4", &pref, mref, 129)});
            for (const char* hk : hc_tiles) {
                mtl::Pipeline p = cx.pipeline(lprompt, hk);
                std::memset(mtest.ptr, 0, 4 * 129ULL * N);
                cx.run({make_hc(hk, &p, mtest, 129)});
                double se = 0, sr = 0;
                for (size_t i = 0; i < 129ULL * N; ++i) {
                    const double e = mtest.as<float>()[i] - mref.as<float>()[i], rv = mref.as<float>()[i];
                    se += e * e; sr += rv * rv;
                }
                const double rel = std::sqrt(se / sr);
                const bool ok = rel < 1e-5;
                fails += !ok;
                std::vector<mtl::Dispatch> ds;
                for (int i = 0; i < 4; ++i) ds.push_back(make_hc(hk, &p, mtest, T));
                cx.warm(ds, 0.2);
                std::vector<double> tv;
                for (int i = 0; i < 5; ++i) tv.push_back(cx.run(ds) / 4);
                const double dt = median(tv);
                const double fl = 2.0 * T * (HC * N) * LR;
                std::printf("m5u hc_up %-20s parity %.2e %s | T=%d: %6.1f us (%5.1f TFLOP/s)\n",
                            hk, rel, ok ? "PASS" : "FAIL", T, dt * 1e6, fl / dt / 1e12);
            }
        }
        // 5. T=1024 AD-4.27 shape sweep across gemm_f16 tiles + dq_f16_8 & dot_f16_f32
        struct M5Shape { const char* name; int type, n_in, n_out; };
        const std::vector<M5Shape> m5shapes = {
            {"hc down",   ref::BF16, 10240, 324},
            {"gdn fused", ref::Q4_K, 2560, 16480},
            {"gdn out",   ref::Q8_0, 6144, 2560},
            {"qsa q",     ref::Q8_0, 2560, 12288},
            {"shared gu", ref::BF16, 2560, 640},
            {"shared dn", ref::Q8_0, 640,  2560},
            {"router",    ref::F32,  2560, 512},
            {"Wsg",       ref::F32,  2560, 1},
            {"dq8 gdn",   ref::Q8_0, 6144, 2560},
            {"dq8 hc",    ref::Q8_0, 2560, 10240},
        };
        const int T = 1024;
        for (const M5Shape& s : m5shapes) {
            const size_t rb = ref::row_bytes(s.type, s.n_in);
            mtl::Buffer W = cx.alloc(std::max<size_t>(rb * s.n_out, 4096)), x = cx.alloc(4 * (size_t)T * s.n_in),
                        xh = cx.alloc(2 * (size_t)T * s.n_in), wh = cx.alloc(2 * (size_t)s.n_out * s.n_in),
                        y = cx.alloc(4 * (size_t)T * s.n_out);
            ref::random_blocks(s.type, W.as<uint8_t>(), s.n_out, s.n_in, 5, 0.02f);
            auto timeit = [&](int which, const std::string& gk) {
                std::vector<mtl::Dispatch> ds;
                for (int i = 0; i < 4; ++i) run_all(s.type, W, 0, s.n_in, s.n_out, T, x, xh, wh, y, gk, ds, which);
                cx.warm(ds, 0.2);
                std::vector<double> tv;
                for (int i = 0; i < 5; ++i) tv.push_back(cx.run(ds) / 4);
                return median(tv);
            };
            if (std::string(s.name).rfind("dq8", 0) == 0) {
                const double tdq = timeit(2, "gemm_f16_64_64_4");
                std::printf("m5u %-10s in=%5d out=%5d: dq_f16_8 %6.1f us\n", s.name, s.n_in, s.n_out, tdq * 1e6);
                continue;
            }
            if (s.n_out == 1) {
                const double tbase = timeit(6, "gemm_f16_64_32_4");
                mtl::Pipeline pdot = cx.pipeline(lib, "dot_f16_f32");
                DotArgs g{(uint32_t)T, (uint32_t)s.n_in};
                std::vector<mtl::Dispatch> ds;
                for (int i = 0; i < 4; ++i) {
                    mtl::Dispatch d; d.p = &pdot;
                    d.args = {mtl::Arg::B(xh), mtl::Arg::B(W), mtl::Arg::B(y), mtl::Arg::V(g)};
                    d.grid[0] = (T + 7) / 8; d.tg[0] = 256; ds.push_back(d);
                }
                cx.warm(ds, 0.2);
                std::vector<double> tv;
                for (int i = 0; i < 5; ++i) tv.push_back(cx.run(ds) / 4);
                const double tdot = median(tv);
                std::printf("m5u %-10s N=    1 K=%5d: dq+gemm %6.1f us vs dot_f16_f32 %6.1f us (%.2fx)\n",
                            s.name, s.n_in, tbase * 1e6, tdot * 1e6, tbase / tdot);
                continue;
            }
            double best = 1e9, t64 = 0;
            std::string bk;
            for (const char* gk : gemm_tiles) {
                const double tg = timeit(4, gk);
                if (std::string(gk) == "gemm_f16_64_64_4") t64 = tg;
                if (tg < best) { best = tg; bk = gk; }
            }
            const double fl = 2.0 * T * s.n_in * s.n_out;
            std::printf("m5u %-10s M=%d N=%5d K=%5d: 64_64_4 %6.1f us (%5.1f TFLOP/s) | best %-18s %6.1f us (%5.1f TFLOP/s, %.2fx)\n",
                        s.name, T, s.n_out, s.n_in, t64 * 1e6, fl / t64 / 1e12, bk.c_str(), best * 1e6, fl / best / 1e12, t64 / best);
        }
        std::printf("m5u parity %s\n", fails ? "SOME FAILED" : "ALL PASS");
        return fails ? 1 : 0;
    }
    const std::vector<int> types = {ref::F32, ref::BF16, ref::Q8_0, ref::Q4_0, ref::Q5_0, ref::IQ4_NL, ref::IQ4_XS, ref::Q2_0,
                                    ref::IQ3_XXS, ref::IQ3_S, ref::Q2_K, ref::Q3_K, ref::Q4_K, ref::Q5_K, ref::Q6_K, ref::NVFP4};
    if (!bench) {
        int fails = 0;
        for (int ty : types)
            for (int n_in : {2560, 640, 6144}) {
                if (n_in % 256 && ref::block_elems(ty) == 256) continue;
                for (int n_out : {100, 48}) {
                    const int T = 72;
                    const size_t rb = ref::row_bytes(ty, n_in);
                    mtl::Buffer W = cx.alloc(rb * n_out), x = cx.alloc(4 * (size_t)T * n_in), xh = cx.alloc(2 * (size_t)T * n_in),
                                wh = cx.alloc(2 * (size_t)n_out * n_in), y = cx.alloc(4 * (size_t)T * n_out);
                    ref::random_blocks(ty, W.as<uint8_t>(), n_out, n_in, 5 + ty + n_in, 0.02f);
                    ref::Rng r(3);
                    for (size_t i = 0; i < (size_t)T * n_in; ++i) x.as<float>()[i] = r.sym();
                    std::vector<mtl::Dispatch> ds;
                    run_all(ty, W, 0, n_in, n_out, T, x, xh, wh, y, "gemm_f16_64_64_4", ds, 7);
                    cx.run(ds);
                    std::vector<float> wf((size_t)n_in * n_out);
                    ref::dequant(ty, W.ptr, wf.data(), wf.size());
                    double se16 = 0, se = 0, sr = 0;
                    for (int t = 0; t < T; ++t)
                        for (int o = 0; o < n_out; ++o) {
                            double a16 = 0, ae = 0;
                            for (int k = 0; k < n_in; ++k) {
                                const float xv = x.as<float>()[(size_t)t * n_in + k], wv = wf[(size_t)o * n_in + k];
                                a16 += (double)h16(xv) * h16(wv);
                                ae += (double)xv * wv;
                            }
                            const double g = y.as<float>()[(size_t)t * n_out + o];
                            se16 += (g - a16) * (g - a16);
                            se += (g - ae) * (g - ae);
                            sr += ae * ae;
                        }
                    const double r16 = std::sqrt(se16 / sr), re = std::sqrt(se / sr);
                    const bool ok = r16 < 1e-5;
                    fails += !ok;
                    std::printf("dense mm %-7s %5d -> %4d, T %d: vs fp16 operands %.2e, vs exact %.2e  %s\n",
                                ref::type_name(ty), n_in, n_out, T, r16, re, ok ? "PASS" : "FAIL");
                }
            }
        std::printf("%s\n", fails ? "SOME FAILED" : "ALL PASS");
        return fails ? 1 : 0;
    }
    struct Shape { const char* name; int type, n_in, n_out; };
    const std::vector<Shape> shapes = {{"gdn qkv", ref::IQ4_XS, 2560, 10240}, {"gdn qkv", ref::Q3_K, 2560, 10240},
                                       {"gdn gate", ref::Q4_K, 2560, 6144},   {"gdn out", ref::Q6_K, 6144, 2560},
                                       {"qsa q", ref::Q2_0, 2560, 12288},     {"hc down", ref::BF16, 10240, 324},
                                       {"hc up", ref::BF16, 320, 10240},      {"router", ref::BF16, 2560, 512},
                                       {"alpha", ref::BF16, 2560, 48}};
    const int T = 2048;
    for (const Shape& s : shapes) {
        if (s.n_in % 128) { std::printf("%-9s %-6s: n_in %d not a multiple of 128 (needs the 32-chunk path)\n", s.name, ref::type_name(s.type), s.n_in); continue; }
        const size_t rb = ref::row_bytes(s.type, s.n_in);
        mtl::Buffer W = cx.alloc(rb * s.n_out), x = cx.alloc(4 * (size_t)T * s.n_in), xh = cx.alloc(2 * (size_t)T * s.n_in),
                    wh = cx.alloc(2 * (size_t)s.n_out * s.n_in), y = cx.alloc(4 * (size_t)T * s.n_out);
        ref::random_blocks(s.type, W.as<uint8_t>(), s.n_out, s.n_in, 3, 0.02f);
        auto timeit = [&](int which, const std::string& gk) {
            std::vector<mtl::Dispatch> ds;
            for (int i = 0; i < 4; ++i) run_all(s.type, W, 0, s.n_in, s.n_out, T, x, xh, wh, y, gk, ds, which);
            cx.warm(ds, 0.2);
            std::vector<double> t;
            for (int i = 0; i < 5; ++i) t.push_back(cx.run(ds) / 4);
            return median(t);
        };
        const double tdq = timeit(2, "gemm_f16_64_64_4"), tact = timeit(1, "gemm_f16_64_64_4");
        double best = 1e9;
        std::string bk;
        for (const char* gk : {"gemm_f16_64_64_4", "gemm_f16_128_64_4", "gemm_f16_64_128_4", "gemm_f16_64_32_4"}) {
            const double tg = timeit(4, gk);
            if (tg < best) { best = tg; bk = gk; }
        }
        const double fl = 2.0 * T * s.n_in * s.n_out;
        std::printf("%-9s %-6s %5d -> %5d, T %d: dequant %7.1f us, act %6.1f us, GEMM %7.1f us (%5.1f TFLOP/s, %s) -> %.2f us a token\n",
                    s.name, ref::type_name(s.type), s.n_in, s.n_out, T, tdq * 1e6, tact * 1e6, best * 1e6, fl / best / 1e12,
                    bk.c_str(), (tdq + best) / T * 1e6);
    }
    return 0;
}
