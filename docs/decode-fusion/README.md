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

Plan, in order of expected gain: (1) gr_down / gr_up grid and K-split for the Ultra (up to ~0.6-1.2 ms); (2) gr_norm2
fused into gr_down (~0.35 ms); (3) router GEMV + moe_route (~0.25 ms). Realistic total ~1.2-1.8 ms of 9.6-12.9
(~12-15%, ~117 -> ~130 tok/s), every change checked bit-exact or by KL against the current kernels.
