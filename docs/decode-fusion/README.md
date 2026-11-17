# decode-fusion: measurements first (ain-soph fork)

Where one Metal decode step's time could go by fusing kernels, measured before writing any fused kernel.
`STRATA_METAL_STEPPROF=T STRATA_METAL_FUSEPROF=out.tsv`: the window's own dispatch list is re-run with a part (or one
pipeline) left out, or with the barriers in front of its dispatches removed, each variant interleaved with the unchanged
list (15 pairs, medians). Left out = the most that part could shrink; no barriers = what its synchronization costs.
Outputs are wrong by design (profiling only).

Setup: Apple M5 Ultra 96 GB, Qwen3.8-Flash-Next AD-4.27 (experts IQ4_NL/IQ2_S/IQ3_S, dense Q8_0), resident, INT8 KV,
`--batch 2`, MTP `--spec 3`. Decode runs T1..T3 windows; ~117 tok/s single stream.

| variant (time saved per step) | T1 (9.64 ms) | T3 (12.88 ms) |
|---|---|---|
| every barrier removed (611 of 707) - the ceiling of all synchronization | 2.87 ms (29.8%) | 3.43 ms (26.6%) |
| hc: gr_down + gr_up joined to gr_norm2's group (194 barriers) | 0.76 ms (7.9%) | 0.82 ms (6.4%) |
| hc: only gr_down's barrier (= gr_norm2 fused into gr_down) | 0.39 ms (4.0%) | 0.33 ms (2.6%) |
| router: moe_route joined to the router GEMV (48) | 0.23 ms (2.4%) | 0.29 ms (2.3%) |
| hc + router fusions together | 0.99 ms (10.3%) | 1.12 ms (8.7%) |
| hc kernels left out entirely | 1.89 ms (19.6%) | 2.47 ms (19.1%) |
| gr_down left out | 0.73 ms | 1.12 ms |
| gr_up left out | 0.77 ms | 0.87 ms |

Reading:
- Synchronization is real but bounded: ~4.7 us per barrier, 30% of the step at most, and only part of it removable.
- gr_down / gr_up read 3.5 MB each but take 7.5-11.6 us (T1-T3), about 2-3x the bandwidth floor (~3.5 us at
  ~1 TB/s), and gr_down grows with T. Its grid is 324 rows / 8 per threadgroup = 41 threadgroups, tuned on an M5 Max;
  the Ultra has twice the GPU cores, so occupancy is the first suspect (the 16-way K-split variants are compiled but
  not selectable).
- gr_norm2 -> gr_down fusion is feasible (each gr_down threadgroup recomputes the 4 x 2560 norm); gr_down -> gr_up is
  not (gr_up needs all 320 `lo`, a grid-wide dependency).

## Step 1: gr_down grid sweep (negative)

`STRATA_METAL_GRIDPROF=SG_KS,...` swaps every layer's gr_down_q8_8_8 for gr_down_q8_<SG>_<KS> (RS = SG/KS rows per
threadgroup, grid ceil(324 / RS)), timed in situ against the unchanged window (15 interleaved pairs). `8_8` is the
control. Files: gridprof-T1.tsv, gridprof-T3.tsv.

| SG_KS (rows/TG) | 8_8 (1) | 8_4 (2) | 4_4 (1) | 16_16 (1) | 16_8 (2) | 8_2 (4) | 16_4 (4) | 32_8 (4) | 8_1 (8) | 4_1 (4) |
|---|---|---|---|---|---|---|---|---|---|---|
| T1 step change | 0.000 | 0.000 | -0.013 | -0.019 | +0.025 | +0.146 | +0.082 | +0.135 | +0.159 | +0.161 ms |
| T3 step change | -0.003 | +0.044 | +0.026 | -0.001 | +0.173 | +0.334 | +0.167 | +0.362 | +0.934 | +1.054 ms |

The shipped 8_8 is already the best; more rows per threadgroup (L1 reuse of xn) is slower, most at T3. The earlier
reading was wrong: of gr_down's 7.5 us per call at T1, ~4 us is its barrier (nobar 0.386 ms / 97), so the kernel body
is ~3.5 us, at the bandwidth floor. At T3 the body is ~8 us (drop 11.6 - nobar 3.4), so a T>1 body is the only
kernel-side gain left there (~0.4 ms at T3, none at T1). The hc cost is mostly synchronization.

Revised plan: (2) gr_norm2 fused into gr_down (~0.35 ms, removes 97 barriers); (3) router GEMV + moe_route (~0.25 ms);
(4) maybe a T>1 gr_down body (~0.4 ms at T3 only). Realistic total ~0.6-1.0 ms (~6-9%, ~117 -> ~125 tok/s).

## Original plan (before step 1)

Plan, in order of expected gain: (1) gr_down / gr_up grid and K-split for the Ultra (up to ~0.6-1.2 ms); (2) gr_norm2
fused into gr_down (~0.35 ms); (3) router GEMV + moe_route (~0.25 ms). Realistic total ~1.2-1.8 ms of 9.6-12.9
(~12-15%, ~117 -> ~130 tok/s), every change checked bit-exact or by KL against the current kernels.
