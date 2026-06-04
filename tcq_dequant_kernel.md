# TCQ Dequant / Kernel Optimization Scratchpad

Date: 2026-06-03

## Goal

Exhaust practical TCQ weight dequant/kernel optimization options for the
promoted TCQ GGUF. Speed is measured on the A100, but PPL/model quality is a
hard gate.

Target model:

- `/workspace/runs/analysis/promoted_best_full_remaining_rebuild.gguf`

## Quality Gate

For math/storage changes, use:

- corpus: `/workspace/data/wiki.test.raw`
- command shape: `llama-perplexity -c 2048 --chunks 8`
- promoted reference from prior queue: `PPL = 5.6374 +/- 0.15535`
- current accepted q8_1 activation baseline from prior queue:
  `PPL = 5.6496 +/- 0.15580`
- review any absolute PPL movement greater than `0.02`

Bit-identical or storage-cache-only changes should keep PPL within normal chunk
noise. Every speed candidate still needs a smoke generation sanity check before
promotion.

## Prior Closed Work

Source scratchpads:

- `/home/blice/cuda-llama/tcq-math-sweep/TCQ_DEQUANT_EFFICIENCY_EXPERIMENTS_20260603.md`
- `/home/blice/cuda-llama/tcq-math-sweep/TCQ_DECODE_PERF_RESEARCH.md`
- `fusion.md`

Closed or rejected paths:

| Path | Decision |
| --- | --- |
| MTP speculative decoding | Not profitable for TCQ; sidecar and inline both bottleneck on target verify. |
| DFlash IQ4_XS drafting | Not profitable for TCQ; target verify dominates. |
| Procedural codebook replacement | Out of scope; keep trained TCQ codebook. |
| Constant-memory codebook lookup | Rejected; divergent access serialized. |
| Wide shared/XOR/scrambled codebook replicas | Rejected; bank conflicts improved but throughput regressed. |
| `cp.async` staging | Rejected; no improvement in current load pattern. |
| Rowpair `<64>` | Rejected; CTA too large. |
| Row-quad | Rejected; worse despite more activation reuse. |
| Global fused 19-bit window to cpack LUT | Rejected; global lookup much slower. |
| Codebook word-locality trick | Skipped; measured locality is essentially random. |

Important prior measurements:

- q8_1 activation + `dp4a` rowpair path is already the deployed baseline.
- Prior A100 baseline in matching checkout:
  - `pp64 = 155.99 +/- 4.23 t/s`
  - `tg32 = 25.18 +/- 0.15 t/s`
  - `PPL = 5.6496 +/- 0.15580`
- Prior all-FFN decoded int8 cache:
  - env: `GGML_CUDA_WQ3_DECODE_CACHE=ffn_up,ffn_gate,ffn_down`
  - cache footprint: about `16.44 GiB`
  - `pp64 = 157.02 +/- 3.46 t/s`
  - `tg32 = 28.64 +/- 0.18 t/s`
  - `PPL = 5.6415 +/- 0.15556`
  - smoke generation reported `29.3 t/s`

## Current Branch Features To Verify

This branch already contains:

- opt-in decoded int8 cache via `GGML_CUDA_WQ3_DECODE_CACHE`;
- per-tensor TCQ profile hooks via `GGML_CUDA_WQ3_PROFILE`;
- graph-level gate/up/GLU fusion in the CUDA backend;
- `GGML_CUDA_DISABLE_FUSION=1` to isolate fusion from cache;
- `GGML_CUDA_WQ3_FUSION_LOG=1` to confirm the fused launcher activates.

## Experiment Matrix

### E0. Current Branch Baseline Reconfirmation

Status: complete

Measure this exact branch on the A100 before further edits:

| Run | Env | Purpose |
| --- | --- | --- |
| baseline | no cache, fusion default | branch default speed |
| no-fusion baseline | `GGML_CUDA_DISABLE_FUSION=1` | isolate graph fusion benefit |
| all-FFN cache | `GGML_CUDA_WQ3_DECODE_CACHE=ffn_up,ffn_gate,ffn_down` | validate cache in this branch |
| all-FFN cache, no fusion | cache env + `GGML_CUDA_DISABLE_FUSION=1` | isolate cache benefit |
| all-FFN cache with fusion log | cache env + `GGML_CUDA_WQ3_FUSION_LOG=1` | confirm fused launcher usage |

Bench command shape:

- `llama-bench -m MODEL -ngl 99 -fa 1 -ctk f16 -ctv f16 -p 64 -n 32 -r 5`

Quality/sanity:

- Run PPL chunks8 for the fastest candidate.
- Run one deterministic smoke generation for the fastest candidate.

Result:

- Initial branch matrix showed the cache env was not taking effect:
  - baseline: `pp64 = 157.33 +/- 3.18`, `tg32 = 26.30 +/- 0.07`
  - no fusion: `pp64 = 156.92 +/- 3.82`, `tg32 = 25.55 +/- 0.12`
  - all-FFN cache env: `pp64 = 157.47 +/- 3.61`, `tg32 = 26.47 +/- 0.08`
  - all-FFN cache env, no fusion: `pp64 = 156.52 +/- 4.33`, `tg32 = 25.29 +/- 0.18`
  - fusion log showed `cache_up=0 cache_gate=0`, so the cache pointers were null.
- Root cause: this branch read decoded caches from `tensor->extra`, but the current CUDA backend either leaves `extra` null for normal CUDA tensors or uses it for split tensor GPU metadata. The old cache prototype's pointer path was stale.
- Fix: normal CUDA buffer contexts now own decoded-cache allocations, and WQ3 TCQ matvec/fused call sites lazily create/cache the decoded int8 tensor on first use. Split tensors remain unsupported for this experiment path, matching the single-A100 target.
- Confirmed cache activation:
  - `GGML_CUDA_WQ3_DECODE_CACHE=ffn_up,ffn_gate,ffn_down`
  - `GGML_CUDA_WQ3_FUSION_LOG=1`
  - logs show all `blk.*.ffn_{gate,up,down}.weight` caches created at `87.66 MiB` each.
  - fused launcher reports `cache_up=1 cache_gate=1`.
- Post-fix A100 r5:
  - baseline: `pp64 = 157.39 +/- 3.48`, `tg32 = 26.49 +/- 0.07`
  - all-FFN cache + fusion: `pp64 = 157.45 +/- 3.34`, `tg32 = 29.68 +/- 0.11`
  - all-FFN cache, fusion disabled: `pp64 = 156.97 +/- 3.71`, `tg32 = 29.08 +/- 0.17`
  - cache-only gain over baseline: `+2.59 t/s` on tg32 (`+9.8%`)
  - cache + fusion gain over baseline: `+3.19 t/s` on tg32 (`+12.0%`)
  - fusion still contributes about `+0.60 t/s` after cache.
- Quality gate:
  - command used comparable prior recipe: `llama-perplexity ... -c 2048 --chunks 8`
  - `PPL = 5.6415 +/- 0.15556`
  - default `n_ctx=512` PPL run fails an existing WQ3 TCQ MMQ assertion for batched src1 (`ne12/ne13 > 1`); not a cache correctness failure.
- Smoke:
  - `llama-cli` confirms cache allocation and reported `Prompt: 31.7 t/s`, `Generation: 26.8 t/s`.
  - Caveat: `llama-cli` stayed interactive and spammed `>` after completion; use server/bench/perplexity for future non-interactive checks.

### E1. If Cache + Fusion Is Positive

Status: complete

Promote the best env-gated deployable path and document:

- tg/pp delta;
- PPL;
- smoke generation;
- VRAM/cache footprint;
- load-time cost;
- exact env flags.

Current best deployable path:

- branch: `feat/tcq-wq3-ffn-fusion`
- env: `GGML_CUDA_WQ3_DECODE_CACHE=all`
- default fusion enabled
- expected extra VRAM: about `23.28 GiB`
- target hardware: A100 80GB / high-VRAM deployments
- decoded-cache layout: `norm,pad,qs[128]` so cached dp4a matvecs can issue aligned 32-bit `qs` loads.
- cached large-matrix row grouping: `BR=2` on A100 after alignment; override/debug env `GGML_CUDA_WQ3_CACHE_BLOCK_ROWS={2,4,8,16,32,64}`.
- lazy cache-build cost: first decoded token pays the cache construction work; minimal wall-clock probe was `4.512s` no-cache vs `4.599s` all-cache, but `tg1` is not representative (`12.52` no-cache vs `2.13` all-cache) because cache creation happens inside that one-token timed section.

High-VRAM result:

- pre-alignment r5: `pp64 = 157.50 +/- 3.44`, `tg32 = 30.96 +/- 0.14`
- aligned-cache r5: `pp64 = 157.36 +/- 3.67`, `tg32 = 41.64 +/- 0.41`
- aligned-cache + cached `BR=2` r5: `pp64 = 157.53 +/- 3.45`, `tg32 = 42.14 +/- 0.46`
- final default no-override r5: `pp64 = 157.38 +/- 3.62`, `tg32 = 42.13 +/- 0.42`
- PPL: `5.6415 +/- 0.15556`
- gain vs no-cache branch baseline `tg32 = 26.49 +/- 0.07`: `+15.65 t/s` (`+59.1%`)
- gain vs pre-alignment all-cache r5 `tg32 = 30.96 +/- 0.14`: `+11.18 t/s` (`+36.1%`)
- gain vs FFN-only cache r5 `tg32 = 29.68 +/- 0.11` from E0: `+12.46 t/s` (`+42.0%`)

Lower-VRAM alternatives:

- FFN-only cache:
  - env: `GGML_CUDA_WQ3_DECODE_CACHE=ffn_up,ffn_gate,ffn_down`
  - extra cache: about `16.44 GiB`
  - best r5: `tg32 = 29.68 +/- 0.11`
  - PPL: `5.6415 +/- 0.15556`
- Exact targeted cache:
  - env: `GGML_CUDA_WQ3_DECODE_CACHE=ffn_up,ffn_gate,ffn_down,attn_qkv,ssm_out,attn_gate`
  - extra cache: about `21.75 GiB`
  - r5: `pp64 = 157.36 +/- 3.38`, `tg32 = 30.23 +/- 0.12`
  - not recommended vs `all` unless the extra `~1.5 GiB` matters.

Implementation notes:

- Fixed stale cache wiring in `ggml-cuda.cu`: normal CUDA buffer contexts now own decoded-cache allocations, and WQ3 TCQ matvec/fusion sites lazily attach caches to normal CUDA tensors.
- Extended `GGML_CUDA_WQ3_DECODE_CACHE` parsing in `wq3-tcq.cu`:
  - `1` or `all` caches every WQ3 TCQ tensor.
  - existing aliases `up`, `gate`, `down`, `ffn_up`, `ffn_gate`, `ffn_down` still select FFN tensors.
  - additional comma/space-separated tokens now match exact tensor suffixes like `.ssm_out.weight`, `.attn_qkv.weight`, `.attn_gate.weight`.
- Selector bug found/fixed: broad substring matching made `attn_q` also match `attn_qkv`. Exact suffix matching avoids this for non-FFN tokens.
- Cache layout bug/perf issue found/fixed: original decoded-cache block was `norm,qs[128],pad`, putting `qs` at byte offset 2. The cached kernel rebuilt cpack from four byte loads per lane. Moving the pad before `qs` keeps the block size at 132 bytes but aligns `qs` to byte offset 4, enabling one 32-bit load per lane. This is the current biggest single speedup.
- Cached block-row sweep after aligned loads found `BR=2` fastest on A100 for large cached matrices. The env override is retained for diagnostics, but default cached large-matrix launch now uses `BR=2`.
- Load-cost probe: `/workspace/runs/tcq_dequant_kernel_20260603/e16_cache_load_cost_20260603T222924`
  - no-cache minimal bench: `tg1 = 12.52`, `WALL_SECONDS=4.512`
  - all-cache minimal bench: `tg1 = 2.13`, `WALL_SECONDS=4.599`
  - interpretation: cache creation is lazy and charged to first decode; process wall time increases only slightly in this minimal probe, but first-token latency is much worse if all caches are cold.
- Split-buffer guard smoke after making cache lookup reject CUDA split buffers before reading `tensor->extra`:
  - run: `/workspace/runs/tcq_dequant_kernel_20260603/e17_split_guard_smoke_20260603T223233`
  - default all-cache r3: `pp64 = 156.45 +/- 4.20`, `tg32 = 41.73 +/- 0.38`
  - interpretation: safety guard preserved the promoted single-A100 path.

### E2. If Cache + Fusion Is Flat Or Regresses

Status: complete

Use `GGML_CUDA_WQ3_PROFILE=1` on a short decode to identify which tensor classes
remain dominant, then choose the next kernel experiment from remaining evidence.

Profile result after FFN cache:

- Direct profile with CUDA graphs crashed in `ggml_cuda_wq3_tcq_profile_end`; rerun with `GGML_CUDA_DISABLE_GRAPHS=1`.
- Decode128 profile with FFN cache showed remaining TCQ matvec categories:
  - `ffn_down.weight`: `942.348 ms`
  - `attn_qkv.weight`: `423.873 ms`
  - `ssm_out.weight`: `348.695 ms`
  - `attn_gate.weight`: `306.022 ms`
  - `attn_q.weight`: `172.855 ms`
  - `attn_output.weight`: `114.885 ms`
  - `attn_v.weight`: `66.355 ms`
- This motivated the cache-scope experiments.

### E3. Cache Scope Experiments

Status: complete

Goal: determine whether non-FFN decoded caches are worth the extra VRAM.

Exact-selector r3:

| Env suffix beyond FFN | Extra cache | tg32 |
| --- | ---: | ---: |
| none | `16.44 GiB` | `29.58 +/- 0.03` |
| `ssm_out` | `17.89 GiB` | `29.94 +/- 0.22` |
| `attn_gate` | `17.89 GiB` | `29.88 +/- 0.14` |
| `attn_qkv,ssm_out,attn_gate` | `21.75 GiB` | `30.51 +/- 0.16` |
| `all` | `23.28 GiB` | `30.96 +/- 0.18` |

Final r5:

| Env | Extra cache | pp64 | tg32 | PPL |
| --- | ---: | ---: | ---: | ---: |
| `ffn_up,ffn_gate,ffn_down,attn_qkv,ssm_out,attn_gate` | `21.75 GiB` | `157.36 +/- 3.38` | `30.23 +/- 0.12` | not rerun |
| `all` | `23.28 GiB` | `157.50 +/- 3.44` | `30.96 +/- 0.14` | `5.6415 +/- 0.15556` |

Conclusion:

- `all` is the fastest measured path so far.
- The extra non-FFN cache over FFN-only costs about `6.84 GiB` and, after aligned cache loads, buys about `+11.96 t/s`.
- The targeted combined subset saves only about `1.53 GiB` vs `all` but loses about `0.73 t/s`, so it is not the best A100 setting.

### E4. Cached Dual Gate/Up GLU Kernel

Status: rejected

Hypothesis:

- With decoded caches active, the fused gate/up launcher currently falls back to:
  - cached up matvec into `dst`;
  - cached gate matvec into a temporary vector;
  - separate GLU apply kernel.
- A single cached dual rowpair GLU kernel might avoid one temporary vector and one extra launch.

Implementation tested:

- Added `k_wq3_tcq_mmvq_q8_1_dual_rowpair_glu_cached`.
- It consumed decoded int8 caches for up/gate and performed both dp4a reductions plus GLU in one rowpair kernel.
- Activated only when both `decoded_cache_up` and `decoded_cache_gate` were present.

Result:

- Run: `/workspace/runs/tcq_dequant_kernel_20260603/e6_cached_dual_glu_20260603T220734`
- Env: `GGML_CUDA_WQ3_DECODE_CACHE=all GGML_CUDA_WQ3_FUSION_LOG=1`
- Log confirmed: `WQ3_TCQ fusion: cached dual rowpair GLU kernel active`
- r5:
  - cached dual GLU: `pp64 = 157.34 +/- 3.66`, `tg32 = 28.03 +/- 0.11`
  - all-cache fusion-disabled in same build: `pp64 = 157.37 +/- 3.44`, `tg32 = 27.85 +/- 0.18`

Conclusion:

- Rejected. The single-kernel form regressed sharply versus the previous all-cache best `tg32 = 30.96 +/- 0.14`.
- Likely reason: combining two cached matvecs in one warp doubles cache/activation load pressure and register pressure enough to lose occupancy or memory efficiency. The existing two-matvec cached path is better despite the temporary gate vector.
- Patch was backed out; do not reattempt this exact shape without first validating with lower-level occupancy/memory profiling.

### E5. Aligned Decoded-Cache Loads

Status: promoted

Hypothesis:

- The decoded cache block used `norm,qs[128],pad`, so `qs` started at a 2-byte offset.
- Cached matvec rebuilt `cpack` from four byte loads per lane:
  - `qs[lane*4+0]`, `qs[lane*4+1]`, `qs[lane*4+2]`, `qs[lane*4+3]`.
- Moving the pad before `qs` keeps the same 132-byte footprint but lets each lane issue a single aligned 32-bit load.

Implementation:

- Changed `block_wq3_tcq_i8_cache` to `norm,pad,qs[128]`.
- Replaced four byte loads with:
  - `*reinterpret_cast<const uint32_t *>(blk->qs + lane * 4)`

Result:

- Run: `/workspace/runs/tcq_dequant_kernel_20260603/e9_aligned_cache_loads_20260603T221309`
- Env: `GGML_CUDA_WQ3_DECODE_CACHE=all`
- r5: `pp64 = 157.36 +/- 3.67`, `tg32 = 41.64 +/- 0.41`
- PPL run: `/workspace/runs/tcq_dequant_kernel_20260603/e9_aligned_cache_ppl_20260603T221351`
- PPL: `5.6415 +/- 0.15556`

Conclusion:

- Promote. This is numerically identical because only the cache memory layout and load pattern changed.
- Current best A100 TCQ setting is `GGML_CUDA_WQ3_DECODE_CACHE=all`, default fusion enabled, aligned decoded-cache layout.

### E6. Cached Norm Warp Broadcast

Status: rejected

Hypothesis:

- In the cached rowpair kernel every lane loads the same fp16 norm for each cached row.
- Loading norm in lane 0 and broadcasting with `__shfl_sync` might reduce redundant memory traffic.

Result:

- Run: `/workspace/runs/tcq_dequant_kernel_20260603/e11_norm_broadcast_20260603T221725`
- Env: `GGML_CUDA_WQ3_DECODE_CACHE=all`
- r5: `pp64 = 156.90 +/- 4.29`, `tg32 = 38.61 +/- 0.28`

Conclusion:

- Rejected. It regressed versus aligned-cache baseline `tg32 = 41.64 +/- 0.41`.
- Likely reason: the norm load is cheap/cache-friendly, while the extra shuffle/dependency sits directly on the accumulation path.
- Patch was backed out.

### E7. Cached Block-Row Geometry Sweep

Status: promoted

Hypothesis:

- After aligned decoded-cache loads, cached matvec occupancy/memory pressure may prefer a different `BLOCK_ROWS` than the older uncached/procedural path.
- Added temporary diagnostic env:
  - `GGML_CUDA_WQ3_CACHE_BLOCK_ROWS={2,4,8,16,32,64}`

Results:

- Sweep r3: `/workspace/runs/tcq_dequant_kernel_20260603/e12_cache_block_rows_20260603T222132`
  - `BR=32`: `tg32 = 41.60 +/- 0.28`
  - `BR=16`: `tg32 = 40.63 +/- 0.28`
  - `BR=8`: `tg32 = 41.15 +/- 0.46`
  - `BR=4`: `tg32 = 41.37 +/- 0.53`
- Extremes r3: `/workspace/runs/tcq_dequant_kernel_20260603/e13_cache_block_rows_extremes_20260603T222402`
  - `BR=64`: `tg32 = 40.75 +/- 0.65`
  - `BR=2`: `tg32 = 42.14 +/- 0.29`
- Direct r5: `/workspace/runs/tcq_dequant_kernel_20260603/e14_br2_vs_br32_r5_20260603T222432`
  - `BR=32`: `pp64 = 157.44 +/- 3.56`, `tg32 = 41.62 +/- 0.44`
  - `BR=2`: `pp64 = 157.53 +/- 3.45`, `tg32 = 42.14 +/- 0.46`
- Final no-override confirmation: `/workspace/runs/tcq_dequant_kernel_20260603/e15_final_default_br2_20260603T222704`
  - `pp64 = 157.38 +/- 3.62`, `tg32 = 42.13 +/- 0.42`
- Final default PPL gate: `/workspace/runs/tcq_dequant_kernel_20260603/e15_final_default_ppl_20260603T222723`
  - `PPL = 5.6415 +/- 0.15556`

Conclusion:

- Promote `BR=2` for large decoded-cache matrices on A100.
- Keep `GGML_CUDA_WQ3_CACHE_BLOCK_ROWS` as a diagnostic override.
- Uncached/procedural paths retain previous geometry order.

### E8. 24GB / 3090-Oriented Partial Cache Sweep

Status: measured

Goal:

- Find useful partial-cache choices for users who cannot afford the full `all`
  decoded cache.
- Measurements are on A100 using the final aligned-cache/`BR=2` kernel, so they
  are a ranking proxy for 3090 users rather than a guaranteed 3090 result.

Clean run:

- `/workspace/runs/tcq_dequant_kernel_20260603/e19_3090_cache_budget_sweep_clean_20260603T233720`

| Selector | Extra cache | tg32 | Delta vs no cache |
| --- | ---: | ---: | ---: |
| none | `0.00 GiB` | `26.47 +/- 0.10` | `+0.00` |
| `attn_v` | `0.08 GiB` | `26.40 +/- 0.11` | `-0.07` |
| `attn_output` | `0.48 GiB` | `26.59 +/- 0.11` | `+0.12` |
| `attn_q` | `0.97 GiB` | `26.85 +/- 0.10` | `+0.38` |
| `ssm_out` | `1.45 GiB` | `27.06 +/- 0.18` | `+0.59` |
| `attn_qkv` | `2.42 GiB` | `27.09 +/- 0.16` | `+0.62` |
| `ffn_up` | `5.48 GiB` | `28.35 +/- 0.12` | `+1.88` |
| `ffn_gate` | `5.48 GiB` | `28.41 +/- 0.10` | `+1.94` |
| `attn_gate` | `6.93 GiB` | `28.75 +/- 0.15` | `+2.28` |
| `ffn_down` | `5.48 GiB` | `29.81 +/- 0.11` | `+3.34` |
| `attn_qkv,ssm_out,attn_gate` | `10.80 GiB` | `30.60 +/- 0.14` | `+4.13` |
| `ffn_down,attn_qkv` | `7.90 GiB` | `30.63 +/- 0.11` | `+4.16` |
| `ffn_down,ssm_out` | `6.93 GiB` | `30.65 +/- 0.17` | `+4.18` |
| `ffn_up,ffn_gate` | `10.96 GiB` | `30.95 +/- 0.17` | `+4.48` |
| `attn_qkv,ssm_out,attn_gate,attn_q,attn_output,attn_v` | `12.33 GiB` | `31.64 +/- 0.10` | `+5.17` |
| `ffn_down,attn_gate` | `12.41 GiB` | `32.86 +/- 0.29` | `+6.39` |
| `ffn_up,ffn_gate,attn_qkv,ssm_out,attn_gate` | `16.27 GiB` | `34.35 +/- 0.25` | `+7.88` |
| `ffn_down,attn_qkv,ssm_out,attn_gate` | `16.27 GiB` | `35.24 +/- 0.26` | `+8.77` |

Recommendations:

- Best small-cache starting point: `GGML_CUDA_WQ3_DECODE_CACHE=ffn_down`.
- Best sub-8 GiB candidate: `GGML_CUDA_WQ3_DECODE_CACHE=ffn_down,ssm_out`.
- Best sub-12 GiB candidate: `GGML_CUDA_WQ3_DECODE_CACHE=ffn_up,ffn_gate`.
- Fastest measured partial cache below the full `all` mode:
  `GGML_CUDA_WQ3_DECODE_CACHE=ffn_down,attn_qkv,ssm_out,attn_gate`.
- 24GB users probably need to start with `ffn_down` and add one class at a time,
  because even `+5.48 GiB` extra may be tight depending on context length, KV
  cache, CUDA graphs, and any mmproj.
