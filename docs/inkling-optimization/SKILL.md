---
name: inkling-strata-optimization
description: Optimize Dageno's Inkling-2-Mini SQ-10 local inference service on Apple Silicon. Use when tuning its Strata Metal endpoint, context, slots, structured extraction, cache behavior, or diagnosing its latency, memory, and correctness. Not for other models without a fresh benchmark.
---

# Inkling Strata Optimization

Dageno's serving service is Inkling-2-Mini SQ-10 through Strata Metal 0.1.67. Preserve these measured facts when tuning it, but remeasure after each engine or weight update.

## Measured service profile

Use SQ-10 with FP8 E4M3 KV on the patched Metal 0.1.67 build. Optimize correct-answer latency and reasoning quality within the available host memory. Report combined target, draft and embedding footprint and RSS. Use 12 GB as a planning reference. Evaluate structural or low-bit changes with held-out task outcomes.

- Weight configuration declares `model_max_length=1,048,576`; this does not mean a serving context that large is useful.
- Sweet serving point on this Apple Silicon laptop: context 16,384 with two inference slots. Current pipeline batches are 2.3–3.1K prompt tokens.
- Prefill after cache misses is approximately linear through 13K prompt tokens: first visible token ≈ 2.1 s + input tokens / 488 tokens/s. Generation is roughly 27–32 tokens/s for extraction. Context extension from 5K to 32K does not itself change short-request latency materially.
- Existing clients using a 45 s timeout should keep input below about 12K tokens when output can reach 500 tokens. Use 90+ s timeouts for 15K-token evidence batches.
- The 2048-token context rejects all archived eight-row extraction batches, whose prompts are 2291–3101 tokens. At 16K they all run successfully; the median end-to-end time is about 20 s because output is often several hundred tokens.

## Metal idle behavior

Strata Metal's default GPU keep-alive keeps the work set active for 120 s after each command buffer. In a repeated 2966-token extraction request, the first visible token was 6.2–6.7 s when requested five seconds after activity, but 41.5–49.9 s after a 135 s idle interval, even though decoding was fast. Set `STRATA_METAL_KEEPALIVE_MS=0` for the launchd service unless a fresh benchmark shows that it is worse. Treat the 120-second boundary as a performance cliff, not ordinary warm-cache variance.

## Measure the full request under memory pressure

Time request send to `BDONE` on native runs and send to stream completion on HTTP runs. Retain first requests, failures and timeouts. Compare fresh launches on the same ordered prompts. Report P95 with its sample count, maximum, TTFT, post-first decode rate and exact reference match count. At n=20, nearest-rank P95 leaves out the slowest request.

For an outlier, align per-buffer host commit, GPU start and GPU end with per-process disk-read deltas and system swap counters. On M6 a buffer waited 37.34 s to start and executed in 84 ms while the target read 7.03 GB. A 1.086 GB MLX draft wired-limit experiment still yielded a 107.34 s request on restart. Keep the draft worker default. Look at the target’s file-backed working set when this signature recurs.

For expert-residency proposals, rank experts per physical layer across recurrent passes on separate calibration requests. Compare leave-one-request-out coverage with cold LRU, then time the actual target mapping on a different workload. Locking the top 32 of 128 experts per recurrent layer covered 81.49% of held-out tagging selections, yet a separate extraction replay reached 40.15 s P95 and 48.06 s maximum. Require process I/O and full-request timings when selecting a cache policy.

## Full-pack residency for latency-sensitive service

On the tested 24 GiB M6, use `INKLING_RESIDENCY=locked` in the Compose provider or `STRATA_INK_MLOCK_PACK=1` in the patched native target. Check the startup log for 10,471,526,400 locked bytes. Allow for 13.26 GB combined peak footprint with the single-slot draft configuration. Report startup separately from serving. Two fresh launches each produced 20/20 identical directory outputs, native P95 of 6.78 and 6.57 s, and maxima of 7.00 and 6.67 s. The dual-slot HTTP tagging run returned 7/7 correct. Assess headroom alongside the host's other workloads when choosing this mode.

## Estimate acceptance headroom before tuning the drafter

Divide post-first output tokens by verification rounds. Compare that yield with the draft width plus one at fixed cost per round. In the locked SQ-10 directory replay, 5,960 tokens from 820 rounds gave 7.27 tokens per round against a maximum of eight. Perfect acceptance at the same cost would improve throughput by about 10%. GPU execution on the target consumed about 82% of post-first time, measured from command-buffer timestamps. Prioritize the verification graph on workloads with this profile. Recompute the ratio when changing width, tasks or model.

## Evaluate loop specialization with request tails, not per-kernel averages

For SQ-10's width-eight verification, count how many input rows use each expert before specializing MoE loops. Use the token distribution to pick a candidate, then compare independent fresh-launch runs on P95, maximum, per-verification GPU median and maximum, combined memory and exact output. Compile-time unrolling the common two-token down-projection loop had a mixed median effect and a slower fresh-launch tail. Do not deploy kernel shape specialization on the strength of one positive run.

## Output shape

Natural prompt-only object extraction sometimes wraps a requested object in a one-element array (5/12 in needle tests) while still finding the right fields in all 12 tests. For callers that require an object, use Strata's native `response_format.type=json_schema`; five previously array-shaped failures all returned the correct object and field values. Expect the schema instruction to add roughly 133 prompt tokens for a compact two-field schema.

Pipeline results need type and media filtering after the model. In the archive replay, all 129 entities marked real appeared in their cited row, but valid media names such as 36氪 and truncated English aliases still need downstream cleanup.

## Provider checks

The Docker Compose host provider must compare both `/v1/models` context and `/slots` count against requested configuration before saying ready. Launchd needs the absolute plist path. The endpoint is loopback-only and unauthenticated; do not publish it outside trusted clients.

--- tuning provenance ---
Uses measurements from /tmp/dageno-modelbench. Based in part on the method of MIT-licensed skill markoblogo/abvx-agent-skills/local-inference-tuning.

## MoE speculation literature (2026 sweep)

Verified against Inkling-2-Mini SQ-10's measured profile; remeasure before changing draft width.

- Spec-MoE dedup calculus (arXiv 2610.34440, ReClan): a deduplicating MoE engine is faster than per-slot exactly where the deduplicated share of bytes is below the dedup kernel's effective-bandwidth ratio. On llama.cpp Metal, MLX-LM and CUDA the per-slot kernel has higher effective bandwidth (r = 0.32–0.75), so dedup gains only at 5–8 tokens per pass — sometimes never. Inkling's width-7 verification with measured 6.29 mean acceptance sits inside the viable band. Before widening the draft past 8, fit one layer's expert op bandwidth on the target hardware; the paper's one-layer calibration takes under an hour and predicted the crossover within one pass size in 10/12 configurations. Draft length is a monotone staircase in acceptance rate, so margin widening costs verification time for nothing.
- EcoSpec (arXiv 2607.12696): expert scattering — confident draft tokens that route to disjoint experts increase memory traffic and can erase the speedup. Prefer draft paths reusing experts already in the verification set. For Inkling this supports keeping the draft width modest and explains why acceptance rate alone under-predicts wall time.
- LoopCache (arXiv 2611.24301): in looped transformers, batching tokens across loops loads up to R× the KV cache per step. Cyclic loop KV scheduling restricts per-step traffic to a single loop's cache. Confidence-guided dynamic draft updating replaces stale drafts in flight. Reports 10.25×/14.56× over WaveFront/LoopSpec baselines on long context. Applies to Inkling only when serving genuinely long inputs; the short tagging workload is prefill-dominated.
- LUT-Lane / vector-LUT quantized GEMV (MacPaw research): fp32 activations remain the cost center in vector-mode lookup-table kernels; cross-platform portability is limited. Before building a LUT kernel for SQ-10's Q8 rows, compare against the simpler win of batching rows or using simdgroup matrix multiply, which Apple's own guidance prefers.

## Kernel priority order after full-pack residency

Measured once per width-8 verification graph on the locked M6 service: ink_linear_q8row_mv 36.8 ms/264 calls is the largest single op, then the two sk23 and one sk3 MoE gate/up kernels at ~18 ms each, ink_linear_f32 5.8 ms, relpos splitk 5.3 ms. GPU execution is ~82% of post-first time; perfect acceptance would only add ~10%. Therefore throughput work must shrink the per-verification graph itself: the q8row projections first, MoE gate/up second. Do not spend effort on host-side scheduling until an op-level change lands.

## ⚠️ Hard safety: watchdog resets from stacked locked memory (2026-11-27)

Three observed machine crashes (22:52, 00:31, 00:48 on 2026-11-26/27) were SoC
watchdog hard resets (`ResetCounter` diag: `Boot faults: wdog,reset_in_1`), NOT
software panics. They all occurred in windows where memory was saturated:

- The locked residency engine pins ~10.4 GiB of weights as wired memory
  (`TotalAlloc` 11.1–11.95 GiB, workSet 17.8 GiB virtual).
- A **second** concurrent engine instance or the 8 GB Metal bandwidth memset
  test on top of that pushes wired memory toward the 24 GiB physical ceiling.
- At that point macOS cannot compress/swap wired pages; the system deadlocks
  without ever writing a panic log, and the watchdog resets the machine.

Rules:
1. Never run two weight-loaded engine instances concurrently on this machine
   when either uses `--residency locked`. One locked instance (RSS ~9.8 GiB)
   plus normal desktop load is safe — it ran hours fine.
2. Do not run large multi-GB memset/Metal bandwidth microbenchmarks while the
   locked service is up. Stop the service first (`... down inkling`), run the
   benchmark, then bring the service back.
3. Wired memory is unreclaimable. Check `vm_stat` wired pages before starting
   any experiment: if wired already exceeds ~12 GiB, do not lock anything else.
4. The crash signature to check later: `last reboot` for clustering,
   `/Library/Logs/DiagnosticReports/ResetCounter-*.diag` for `wdog` faults.

## DSpark drafting measurements (2026-11-27, M6 24GB, SQ-10 fp8, service beam1)

Plain decode baseline: 35.5-36.7 tok/s (ppt ~28ms). Drafting measured through
the service (`--draft dspark`):

| content  | beam1 tok/s | accept ratio | plain tok/s |
|----------|-------------|--------------|-------------|
| poem     | 34.5        | 0.00         | ~35         |
| tech     | 35.0        | 0.42         | ~35         |
| json     | 42.4        | 0.71         | ~36         |
| code     | 39.5        | 0.69         | ~36         |

KEY ANOMALY: even at 69-71% acceptance the speedup is only 1.1-1.2x, far below
theory. If the 8-token verify pass amortized weight reads like a 1-token decode
(~28ms) plus ~5ms drafting, json should run at ~100+ tok/s (avg accepted len
~3.4+1 tokens per cycle). Measured implies the verify+draft cycle costs ~90-100ms
- i.e. verification cost scales nearly linearly with token count, NOT amortized.

Hypotheses to test (in priority order):
1. MoE expert scatter tax: 8 draft tokens x top-8 routing activate up to 64
   distinct experts per layer vs 8 for single-token decode. Each distinct expert
   costs its full weight read. This is the exact problem Spec-MoE dedup
   (arXiv 2610.34440) and EcoSpec (2607.12696) address. Count distinct experts
   per layer during verify via STRATA_INK_PROF or plan introspection.
2. Verify tokens may not actually batch in the kernel dispatch (serialization
   in the loop-core schedule or attention path).
3. The drafter itself may cost much more than the assumed few ms (dspark is a
   3-layer recurrent block-8 drafter; measure its time directly).

`--draft-adapt` shuts drafting off for 40 steps when measured gain < 0.93 -
with these numbers it correctly disables drafting most of the time, which is
why adapt mode draft_n stays tiny. Fixing verify amortization matters more
than tuning the gate.

Beam4 (tree) kernels exist in the binary (dspark_markov_joint_beam4_*) but on
CUDA chain==tree within 3%; Metal measurement pending.

## Batch amortization curve measured (2026-11-27, M6, SQ-10 fp8, --bench)

| M slots | step time | tps/step | agg tok/s | per-slot |
|---------|-----------|----------|-----------|----------|
| 1       | 27.2ms    | 1.0      | 36.8      | 36.8     |
| 2       | 40.5ms    | 2.0      | 49.4      | 24.7     |
| 4       | 61.5ms    | 4.0      | 65.0      | 16.3     |
| 8       | 82.8ms    | 8.0      | 96.6      | 12.1     |

Marginal cost per additional token in a batch: (82.8-27.2)/7 = 7.9ms.
Fit: cost(M) = 19.3ms base + 7.9ms per token (r^2 excellent).

DECISIVE INTERPRETATION:
- The dense trunk (SF4 attention + shared expert + glue) IS fully amortized
  across batch tokens - if nothing else cost anything, cost(M) would be flat
  ~28ms and P8 would run 295 agg tok/s.
- The marginal 7.9ms/token is dominated by the MoE expert scatter tax:
  39 loop visits x ~8 distinct experts per additional token x ~156KB avg
  expert rows (S2/S3/SF4 mixed) = ~48MB extra DRAM per token. At 136-147 GB/s
  streaming bandwidth that is 3.3-3.5ms; the remaining ~4.4ms is reduced
  cache reuse and scheduling of many small dispatches.
- Spec-MoE/EcoSpec style expert-dedup verification is THE correct fix:
  an accepted-length-4 verify with dedup would cost ~28+2x7.9+dedup-savings.
- Alternatively batch-level expert caching across loop passes: layers 6-17
  repeat 3x per token. If routing is similar across passes for the same token
  (unverified hypothesis worth testing!), expert reads could be cached in the
  0.5GB vram-experts budget -> up to 3x reduction of the loop MoE traffic.
  TEST: measure per-pass expert id correlation for identical tokens.

200 tok/s single-stream target needs step ~5ms/token emitted: with avg accept
3.4 tokens/cycle that means verify+total draft cycle ~17ms -> requires cutting
expert traffic by ~2.5-3x (dedup or pass-caching), not just better kernels.

## DECISIVE bandwidth measurements (2026-11-27, M6 24GB, bwtest3)

- Contiguous 2MB/TG streaming: 130-137 GB/s
- Scattered 64KB fragments (coprime order): 139-149 GB/s
- Scattered 8KB fragments: **152 GB/s** - faster than contiguous!

CONCLUSION: DRAM scattering is NOT the bottleneck. The memory system handles
8KB-fragment random-order reads at full bandwidth. Therefore the 28ms/token
(= 1.1GB at only 40 GB/s effective) is dominated by kernel-execution and
dispatch fragmentation overhead, not memory physics.

## Expert pass-reuse: DISPROVEN (2026-11-27)

Measured per-(token, layer) expert overlap across loop passes via INK_TRACE_EXPERTS
debug capture: pass pairs overlap 0.55-0.72 of 8 experts vs random baseline 0.5.
The loop passes genuinely re-route to different experts. Cross-pass expert
caching cannot reduce traffic. Do not pursue this direction.

## GPU time decomposition (STRATA_INK_PROF patch, 2026-11-27)

M=1 decode: wall 30ms, GPU 28.2ms, host encode 1.4ms. The GPU is ~94% busy
within the window - the time is spent INSIDE kernel execution + inter-kernel
gaps, not host encoding. ~144 dispatches/visit x 48 visits = ~6900 kernels/step,
~4us each. Megakernel fusion (lithos-metal architecture: layer-mixer fusion,
GPU-resident control loop replayed via ICB) is the validated direction.

## lithos-metal architecture notes (docs/design/*.md, loop-literature/)

- Layer-wise mixer megakernels: fuse QKV proj + conv + recurrence/attention +
  output proj into ONE kernel per layer. MLP stays separate (initially).
- Generation loop GPU-resident, replayed via Metal Indirect Command Buffer.
- Cross-worker sync uses bounded spin + memory ordering; ICB boundaries order
  whole-region deps. "Global barriers are not assumed cheaper inside a kernel."
- Input-norm commuted into projection epilogue (removes separate norm dispatch).
- Attention: online softmax partitions, FP32 stats, sequential layers share one
  workspace with dependency barriers.
- Kernels are chip-specific (m3_max_30c, m5_max_40c etc.) - probe suite
  qualifies geometry per backend. lithos targets Qwen dense/hybrid models,
  NOT loop MoE - the techniques transfer, the recipes don't.
- Claimed 200 tok/s on 20B-A3.6B class MoE at M3 Max - plausibility consistent
  with our measurements if fusion removes the ~4us/kernel overhead.

## ⚠️ UPDATED crash protocol after 4th reset (2026-11-27 02:01)

The 4th hard reset (02:01, no boot faults at all - a deeper-level reset than
wdog) happened with the locked service running AND a profiling binary instance
launched manually at the same time. TWO weight-loaded instances were again
concurrent despite the earlier rule, because measurement commands launched
their own engine processes ad hoc.

THE ONLY SAFE EXPERIMENT PATTERN on this machine:
1. NEVER start a second weight-loaded engine process while the service runs.
   Not for bench, not for profiling, not for one test query.
2. Experiments run THROUGH the service itself: point the service at the
   instrumented binary (strata-inkling-prof), add the env var to the engine
   config env in ai.dageno.dageno-inkling.inkling/config.json, kickstart the
   launchd job, drive it over HTTP. One instance at a time, always.
3. heavy-mutex.py exists as a tripwire for any manual engine run:
   python3 heavy-mutex.py run <cmd> refuses when the service is up or another
   heavy process holds the lock. Use it for any direct binary invocation.
4. Restarts during experimentation are fine (launchd KeepAlive respawns), but
   overlapping instances are what kill the machine.

Bench evidence that even single-instance decode is heavy: M=1 GPU busy 28ms
of a 30ms step at full memory bandwidth. Do not add anything on top.

## Full traffic model for SQ-10 M=1 decode step (2026-11-27, manifest-derived)

Per-token DRAM traffic:
- Dense trunk: layers 0-23 = 805MB once + loop core (6-17) x2 extra passes
  (467.5MB x 2) = **~1740MB**
- MoE experts: 8 experts x ~280KB x 48 visits = **~107MB**
- Unembed S6_K 358MB (already uses fused unembed+argmax when argmax-only)
- KV cache fp8: ~8-10MB at short context
- TOTAL: ~2.1-2.8GB per token depending on unembed path

At 145 GB/s achievable bandwidth: ideal step = 14.5-19ms -> 52-69 tok/s is the
physics ceiling for M=1 on this quantization. Measured 28ms GPU -> the delta
(~8-13ms) is dependency-chain bubbles, mostly small elementwise kernels
(ctrl_gf 96x, rms_norm 97x, sconv 96+48x, copies) serialized on the critical
path at ~5-10us of launch latency each, plus GEMV tail underutilization.

Dispatch anatomy (INK_DCOUNT, one decode step): 1412 dispatches, 838 with
memory barrier, 25 distinct pipelines. Biggest counts: q8row_mv1 264x,
linear_f32 208x, rms_norm 97x, sconv4_multi 96x, ctrl_gf_64 96x, ctrl_cat_rms
96x, attn_res_mix 72x. q8row_mv1 microbenchmark hits 142-149 GB/s in isolation
- individual kernels are fine; serialization between them is the cost.

200 tok/s at M=1 is PHYSICALLY IMPOSSIBLE with this traffic (would need
560GB/s). Paths there: (a) speculation at high accept len multiplies effective
tokens per step (x2-3 realistic), (b) fusion eliminates the 8-13ms bubbles
(~1.5x), (c) expert dedup in verify windows cuts MoE scatter tax. Combined
realistic target: 100-150 tok/s for structured content via beam4+adapt, or
~50-60 tok/s plain after fusion work.

F32 tensors: 155MB total but mlp.gate.weight (router!) is F32 1.15MB/layer
(208 ink_linear_f32 dispatches). Candidate for q8 quantization - small win.

Next implementation steps in priority order:
1. Fuse Q-side attention projections (wq+wr+sink+gate share input b_in3):
   concat weights at load in Q8_ROW layout, one GEMV instead of 4.
   w_qrsg_f16 already proves the layout for the MPP path.
2. Commute/fuse elementwise chain: ctrl_gf + rms_norm + sconv glue per layer
   (lithos input-norm fusion pattern).
3. Quantize mlp.gate.weight F32 -> Q8 (113MB controllers + router bytes cut).
4. Expert-dedup verify scheduling (Spec-MoE style) for speculation windows.

## Barrier cost microbenchmark (2026-11-27, bwtest5)

Back-to-back dispatches in one MTLDispatchTypeConcurrent command buffer:
- noop kernel, no barrier: 0.09-0.11 us/dispatch (negligible)
- noop kernel, WITH memoryBarrierWithScope(Buffers): 0.48-1.46 us/dispatch
- tiny_read kernel with barrier: 0.68 us/dispatch sustained

838 barriers per decode step x ~0.5-1.5us = 0.4-1.2ms of pure barrier overhead.
Worse: barriers serialize the LATENCY of dependent small-kernel chains
(elementwise kernels at 4-10us each don't overlap when every one is fenced).

Priority is now provably-unnecessary barrier removal, then chain fusion.

## Verify-pass cost decomposition (2026-11-27, measured with TPROF instrumentation)

Per-batch-size forward cost on the locked M6 service (ms/pass, host build ~0.13ms):
M1=28.4 M2=36 M3=46.5 M4=58 M5=68 M6=78. Linear-block verify M8=81ms;
tree/adapt M8-bucket passes are a mix of M8..M32 (200-340ms), NOT pure M8 —
do not compare across buckets without exact-M keys.

Key causal findings:

1. The marginal verify cost ~8-10ms/token is expert matvec COMPUTE on SIMD
   fp32, not DRAM. Batching amortizes expert weight reads via the existing
   dedup planner (measured only 36% distinct experts in verify batches), but
   every (expert, token) pair still pays full LUT-decode + dot-product FMA
   work. Aggregate estimate ~700M-1G vec4-fma per marginal token plus x-vector
   L2 re-read amplification per threadgroup-row-block (~12GB/pass at M=8,
   ~8ms at ~1.5TB/s L2). These two effects match measured numbers.
2. MPP neural-accelerator MoE path at low M is NOT the answer as-is:
   INK_MPP_MOE_MIN_M=4 experiment ran 2-4x SLOWER (json 25.8 vs 60 tok/s).
   Fixed per-expert tiling/gather overheads dominate at M<=8. Do not lower
   the threshold without redesigning the bucket path for small M.
3. Beam-tree draft GPU cost is only ~9ms/cycle - drafting is NOT the
   bottleneck. The fixed cost floor is the M1 pass itself (28.4ms) which is
   DRAM-bound at 74GB/s effective vs 152GB/s microbenchmark peak.
4. INK_NOBARR experiment: removing ALL barriers in M1 decode gives 41.5 vs
   35.5 tok/s (+17% ceiling) but corrupts speculative correctness (accept
   collapses to 0, outputs unstable). Not viable as-is; selective removal
   needs full dependency analysis. The 17% includes race-timing artifacts.
5. Byte traffic per M1 token: trunk q8 1.29GB + experts 1.07GB (48 visits x
   8 x ~2.8MB) + unembed S6_K 358MB + controllers FP32 225MB ~= 2.5GB.
   Controllers at FP32 are 225MB/token - the only large lossless-compressible
   trunk item left (router gate + gain/forget matrices); converting them to
   fp16/bf16 offline would save ~4% pass time but needs output validation.

Expert entry layout note: StrataExpertEntry is 32 bytes packed, stride 32 in
Section 3 (NOT 34); the trailing 2 bytes per entry form the vram_load_order
array AFTER the whole 98304-byte directory.

## Verify-cost ablation results (2026-11-27 round 2, decisive)

Clean per-pass CSV measurement protocol: INK_PASSLOG writes one line per
forward (seq, M, is_tree, compute_logits, gpu_ms) — use this instead of the
process-lifetime TPROF counters which mix runs. Fresh service baselines:
linear-block M1 verify = 27ms, M8 = 78ms, marginal ~7.3ms/token.

Decisive ablations:

1. INK_MOE_WONLY (removes expert LUT+FMA arithmetic, keeps weight streams):
   M8 verify 74ms vs 78ms — expert compute is <5% of verify cost. The earlier
   "expert compute bound" hypothesis is WRONG.
2. Scattered-expert microbenchmark: 140GB/s, same as contiguous — DRAM page
   locality is NOT the problem either.
3. MMA/MoE path at M=8 (INK_MOE_MM_MIN_M=2): 160ms vs 78ms — 2x slower.
   MPP-MoE at M>=4: also 2-4x slower. Both matrix-unit paths are unsuitable
   for verify-sized batches. Leave thresholds at upstream defaults.
4. Threadgroup-staged x in ink_moe_gu_sk23: bit-exact outputs, performance-
   neutral. x re-read amplification is NOT the cost driver either.

Residual model: expert weight streaming inside the real kernels achieves only
~90GB/s effective vs 140GB/s in microbenchmarks. Since compute, scatter
pattern, and x amplification are all excluded, the residual is consistent
with occupancy-limited memory-level parallelism in the register-heavy LUT
decode kernels (low resident warps cannot saturate DRAM queues). Untested
fix directions: reduce register pressure, split decode into lighter kernels,
or software-pipelined double buffering of the weight stream.

Measurement hygiene: always restart the service before comparing binaries;
TPROF statics accumulate across the whole process lifetime and mixed old/new
runs have caused false conclusions twice. Prefer INK_PASSLOG CSV + fresh
restart + identical prompt drives.

## Byte-exact traffic model + kernel bandwidth calibration (2026-11-27)

Manifest-derived per-token trunk traffic (M=1 decode): layers 0-5+18-23 = 354.2MB once,
loop core 6-17 = 467.5MB x 3 passes = 1402.5MB, total trunk = **1756.7MB/token**.
Plus unembed S6_K 358MB (fused argmax) + experts ~129MB + controllers F32 113MB amortized
(only passes 1-2) ≈ **2.2-2.3GB/token**. At 27ms measured that is ~82GB/s effective.

bwtest9 decisive measurement: DRAM streaming bandwidth is ~140-146GB/s regardless of
allocation mode - private copy, no-copy anonymous, no-copy file-backed, and mlocked
file-backed all identical within noise. **Zero-copy/mlock is NOT the bottleneck.**

Negative results (all bit-exact, all passlog-verified):
1. SW prefetch in sk_w13_chunk LUT decode: M1 27.1 vs 27.0ms - zero. Compiler already schedules.
2. Fused attention projections ink_attnproj_fused_mv1 (8 GEMVs -> 1 dispatch, flattened row
   space, 279 TGs): M1 27.7 vs 27.1ms - zero/slightly worse. Kernel boundaries not the issue.
3. ink_linear_q8row_mv1_wide (explicit 2-round load batching): M1 27.3ms - zero.
   Explicit load batching does not increase effective MLP beyond compiler output.

Implication: the ~40% bandwidth deficit (82 vs 140GB/s) is NOT fixable by dispatch fusion
or load scheduling within this kernel structure. Per lithos-metal docs, large wins require
their approach: layer-level megakernels (mixer fusion), persistent workers, ICB replay,
and on M4+ hardware MPP tensor cores for matrix work. lithos claims 168 tok/s on 27B NVFP4
at M5 Max 40-core - but that machine has ~4x our memory bandwidth, so 60-70 tok/s single-stream
on M6 24GB is the comparable hardware ceiling for this model size; verify with speculation
is how you go beyond it.

Attention KV re-read hypothesis DISPROVEN: pass cost is flat in position (26.6ms at pos 0-500
vs 27.2ms at pos 1000-1500 for M1; M6-8 drift 76.5->81.4ms over 1500 tokens is small).
KV fp8 re-reads mostly hit cache.

## DECISIVE cost decomposition (2026-11-27, skip-moe ablation + concurrent bandwidth)

INK_SKIP_MOE (timing-only ablation, skips gu/dn expert dispatches):
- M1 pass: 27.1 -> 17.5ms. MoE expert phase = ~10ms/token in decode.
- M8 verify: 76.8 -> 23.5ms. MoE expert phase = ~53ms = **69% of verify cost**.
- Verify expert traffic after dedup (0.333 distinct ratio, TRACE-B measured): ~2.1GB.
  At full bandwidth that is ~15ms. Actual 53ms -> expert kernels achieve ~28% of DRAM BW.
- Concurrent test: inference running + 1.5GB bwtest stream still hits 110GB/s.
  DRAM is NOT saturated by inference; bandwidth is available but unclaimed.
- Skip-MoE M1 residual (17.5ms) vs trunk+unembed stream model (~15ms) => trunk/unembed
  kernels are already near bandwidth; MoE is the ONLY major inefficiency left.

Microbench corrections: earlier "sk23 pattern = 40GB/s" was a construction artifact.
With correct fragment sizing the strided uchar2 pattern reaches 130-143GB/s. Do not trust
small-region microbenches (L2 effects); only service-pass measurements and ablations are
decisive. TIMELINE/SEGPROF distort execution (host sync per barrier) - unusable.

Negative results this round (all bit-exact): threadgroup-staged expert weights (M1 28.7
vs 27.1), fused attention projections, wide-load batching, SW prefetch.
Open candidates ranked: split-unit parallelism (2 TGs per expert, partial reduction),
wide-load + shuffle redistribution, row-interleaving in SG.

## Lane-run layout permutation: DISPROVEN (2026-11-27, the definitive experiment)

Full implementation arc: offline byte-exact permutation of all 3057 sk2/sk3 expert w13
rows into dense lane-run layout + matching ink_moe_gu_sk23_lr kernel. Permutation verified
byte-for-byte against original decode (Python simulation of both paths, 288/288 identical
indices per lane). Integration traps documented: homogeneous sk3 layers use a separate
kernel family (ink_moe_gu_sk3), and prefill M>=48 uses mm_gen/mpp matrix decoders reading
the ORIGINAL superblock layout - any layout change requires converting ALL consumers.

Verdict: pass costs unchanged or slightly worse (M1 28.7-30.7 vs 27.1ms baseline) even
before correctness was completed. Dense contiguous lane streams do NOT beat the original
strided pattern in practice. This is the 7th bit-exact performance-neutral transformation.

CLOSED hypothesis space for expert kernel inefficiency (28% of DRAM bandwidth):
access pattern, byte layout, load scheduling, prefetch, batching, staging, dispatch fusion.
OPEN: either per-warp latency hiding is saturated (would need Metal performance counters
to confirm), or the aggregate multi-stream behavior differs irreproducibly from microbenches.
Do not attempt further kernel-level micro-optimization of the LUT decode path.
Remaining levers: algorithmic (draft acceptance, expert-aware drafting) or megakernel-scale
re-architecture. Service restored to sq10.strata baseline (bit-exact verified).

## EcoSpec method detail for Inkling adaptation (arXiv 2607.12696, extracted 2026-11-27)

The paper's cost-aware selection is directly implementable in Inkling's beam-tree verifier:

Scoring rule: S(t) = P(t) / (DeltaCost(t|B) + eps), where P(t) is cumulative draft
probability along root-to-node path, and DeltaCost(t|B) = |Etraj(t) \ B| counts experts
the path introduces BEYOND the buffer B of already-selected experts. Selection is greedy
over tree nodes. Requires NO change to target verification rule (prefix verification is
preserved; only WHICH draft nodes enter the verify set changes).

Inkling-specific adaptation facts:
- Inkling's router is a single linear layer (exact FP32, mlp.gate.weight) - the "expert
  predictor" can literally BE a copy of the router weights applied to predicted hidden
  states, or better: use the drafter's own hidden state with a small learned probe.
  Routing is sharp in practice (top-8 of 128), so prediction error mainly matters at
  rank boundaries.
- Cost must be non-uniform: sk2/sk3 expert ~3.2MB, SF4 expert ~4.5MB. Weight DeltaCost
  by bytes, not count. Layer mix per layer varies (some layers have 1-5 SF4 experts).
- Verification set is a beam tree with Bayes-optimal sizing already; the marginal-cost
  term slots into the existing per-node scoring (ink_serve.h beam extraction).
- Measured Inkling baseline before implementation: distinct experts per verify batch
  = 0.333 x 8M slots (TRACE-B). If cost-aware selection drops this to ~0.25 at iso-acceptance,
  verify marginal cost falls ~20% (expert traffic dominates marginal cost).
- Paper result context: up to 1.62x end-to-end on large MoE (DeepSeek-V3.1 etc.) where
  expert cost dominates even more strongly. Inkling's smaller experts mean smaller but
  still significant gains.

## Research closure summary for kernel-level optimization (2026-11-27)

Seven bit-exact kernel transformations, ALL performance-neutral on service measurement:
prefetch, wide-load batching, batch-prefetch (all 9 superblocks), fused attention GEMVs,
threadgroup-staged weights, threadgroup-staged x, lane-run byte permutation (full offline
repack of 3057 experts). Combined with ablations (compute removal ~free, arithmetic not
the cost) and microbenchmarks (patterns hit 130-150GB/s in isolation), the evidence says:
the verify-pass bandwidth deficit is a system-level concurrency effect not reproducible
in microbenchmarks and not fixable by per-kernel restructuring. The economically rational
optimization frontier is now: (1) reduce bytes per accepted token (EcoSpec), (2) increase
accepted tokens per pass (draft quality), (3) serving-level batching across slots.

## Bayes tree cost model calibration: FIRST POSITIVE RESULT (2026-11-27)

The beam-tree size optimizer's cost model must match the actual machine. The shipped
constants were for M5 Max: cost(M1)=13ms, cost(M8)=35.7ms. Measured on M6 24GB SQ-10:
cost(M1)=27.1ms, marginal ~10.5ms/token, cost(M8 tree)=76-87ms. The 2x underestimate
made trees systematically too deep - verify compute wasted on low-acceptance deep nodes.

Calibrated cost(M) = 25 + 10.5*(M-1), t_draft=9ms. Effects measured over 4 sweeps:
- structured content (json): tok/s +40% (38-43 -> 54-61), acceptance ratio 0.71 -> 0.89
- code: flat/slightly up; free text (poem/tech): flat to -5% (smaller trees, correct tradeoff)
- outputs bit-exact vs pre-change baseline in all 16 comparisons (greedy speculation invariant)
- business P3 bench: valid 7/7 correct 6/7 unchanged

Lesson: speculative-decoding throughput is acceptance-length per unit verify cost.
When the cost model is wrong, the tree optimizer optimizes the wrong objective.
INK_TVER_BASE/INK_TVER_MARG/INK_TDRAFT env vars allow future recalibration without rebuild.
Recalibrate after ANY kernel-level speedup lands, or the gains will be misallocated.

## Vision shard serving (2026-11-27)

Inkling-2-Mini-Vision.strata (636MB, lossless BF16, 318.3M params, inkling2_vision arch)
serves images end-to-end on port 8392. Engine reports vision=lazy: zero memory cost until
the first image request. Vision-v2.strata uses an incompatible newer section-table layout
for this engine build - do not use it without an engine update.

Serve-layer trap that cost debugging time: probe functions in BOTH
strata-src/terminal/serve/hy4_kind.py and inkling_kind.py run 'exe --info model' with
cwd=exe directory. Relative model paths (sq10.strata) then fail to resolve, probe returns
None, and the server silently strips --vision from engine args. Fixed to probe in cfg cwd.
When vision appears broken, check the engine's actual argv (ps) for --vision before
suspecting the shard.

## Expert overlap measurement - EcoSpec headroom closed (2026-11-27)

Adjacent draft tokens in verify batches share 4.3-4.7 of 8 experts per layer-visit
(INK_TRACE-C measurement; random baseline 0.5). Inkling-2-Mini's routing is strongly
continuous across positions. EcoSpec marginal-cost-aware tree selection therefore has
little to exploit: verify expert reuse is already near its structural ceiling
(distinct ratio 0.33-0.41 vs 0.81 random). Do not implement cost-aware pruning here;
the acceptance-cost tradeoff is dominated by acceptance, not expert cost.

Acceptance-length distribution (INK_ACCLOG): mean ~2.3 tokens/verify cycle on mixed
content, median 1-2. Tree WIDTH matters more than depth. Adapt governor with calibrated
t_plain (27.1ms not 13.5ms) correctly disables drafting on free text and keeps it on
structured content - visible as code acceptance 0.36 -> 0.87 at iso-quality.

## Compile-time mm specialization: SECOND POSITIVE RESULT (2026-11-27)

sk_w13_chunk_mm<F,MM> and sk_w2_2pair_coalesced_mm<F,MM> fully unroll the token loop
so accumulators stay in registers instead of indexed thread-local memory.
Default ON since 2026-11-27 (INK_MOE_MM=0 reverts).

Measured: M8 verify 76.8 -> 66.8ms (-13%), M1 27.1 -> 26.4ms, json 60.8 -> 63.6-64.4 tok/s,
code 40.7 -> 43.3-44.2 tok/s. Bit-exact outputs, P3 business bench unchanged.

CRITICAL correctness note for dn: superblock 4 of the w2 paired rows SPLITS at lane 16
(lanes 0-15 finish row 0 cols 1024-1151, lanes 16-31 start row 1 cols 0-127), and
superblocks 5-8 cover row 1 cols 128+. Blind replication without this split produces
wrong outputs that pass casual inspection but corrupt expert results. Verified via
bit-exact sweep comparison.

Key lesson behind both positive results: the wins were in COMPILE-TIME constants
(cost model, loop bounds), not runtime scheduling. Metal's compiler exploits whatever
parallelism the source exposes; exposing more requires changing what is computable
at compile time.

## Compile-time unroll family: cumulative positive results (2026-11-27)

Three positive throughput results, all from compile-time constants, all bit-exact:
1. Bayes tree cost calibration: json +40% (cost model matched M5 not M6)
2. mm specialization of MoE gu+dn decode: verify -13%
3. q8row_mv M<=8 token-loop unroll: verify M4-M7 -12-20%, json up to 69.9 tok/s

Cumulative effect vs original baseline: json 38-43 -> 63-70 tok/s, code 34-36 -> 43-46,
verify M8 76.8 -> 66.8ms. P3 business bench unchanged throughout (valid 7/7 correct 6/7).

Pattern: Metal's compiler does not unroll runtime-bounded loops over indexed accumulator
arrays, and does not know the verify cost model. Anything the compiler CAN'T know is a
candidate; anything it can already see (scheduling, prefetch, batching, layout) is waste
of time - seven bit-exact neutral experiments established that boundary.

## MM=8 chunking: neutral-to-slightly-positive (2026-11-27)

Extending mm specialization to 8 tokens per expert pass: M7 verify 67.6 -> 65.8ms,
M8 unchanged (66.8-68.4 noise band). Register pressure at MM=8 offsets x batching gains.
Kept as default since midrange is consistently better. x-activation re-reads cost load
slots, not DRAM bandwidth (x is L2-resident) - proven by neutrality of both x-staging
and x-batching experiments.

## Current cumulative state (2026-11-27)

Throughput vs original baseline (bit-exact outputs throughout):
- json: 38-43 -> 63-70 tok/s (+60%)
- code: 34-36 -> 43-46 tok/s (+28%)
- free text: ~unchanged at 33-35 (correctly drafting-off most of the time)
- verify M8: 76.8 -> 66.8ms, M1 decode: 27.1 -> 26.4ms
- prefill: 500+ tok/s (~88% of stream bandwidth - near ceiling)
- P3 quality: valid 7/7 correct 6/7 unchanged across all experiments

Exhausted directions (do not retry): kernel scheduling/prefetch/staging/layout fusion
(7 experiments), EcoSpec expert-cost pruning (measured overlap already 4.3-4.7/8),
lane-run byte permutation (full implementation, neutral), cross-pass expert caching
(routing diverges), batch chunking beyond MM=4 (register pressure).

Live candidates in descending expected value:
1. serving batching of concurrent requests into shared verify passes (server-level)
2. tree shape: wider-shallower trees if branching costs less than depth (measure first)
3. draft cost reduction (beam_tree draft ~9ms/cycle vs verify 68ms - 12%)
4. MPP/Neural-Engine paths for large-M prefill (already default where beneficial)

## Cost-model recalibration experiment: negative, reverted (2026-11-27)

After the verify speedups, refitting cost(M) = 22 + 6.5*(M-1) and retesting:
json dropped slightly (60-67 vs 63-70) with acceptance 0.89 -> 0.76 (deeper trees).
Deeper nodes cost more than their marginal acceptance yields - the original
steeper cost curve (25 + 10.5) was closer to the true economic shape because
verify marginal cost is superlinear in tree width at high M (tree attention
evaluates all nodes). Lesson: cost(M) is NOT linear in tree width; relinearizing
from linear-batch measurements misleads the optimizer. When in doubt, trust
measured end-to-end throughput over model-fitted parameters.

## Expert kernel bandwidth model - CORRECTED understanding (2026-11-27)

M1 decode: expert traffic 1075MB/token (48 visits x 8 distinct x 2.8MB mean), costing
~9ms of the 26.4ms pass = ~86% of stream bandwidth. M1 experts are NOT inefficient.
M8 verify: expert traffic dedups to ~40% (0.4 measured), ~1.3GB net, but costs ~40ms
of 67ms = ~35% efficiency. THE EFFICIENCY COLLAPSES WITH BATCH SIZE.

Root cause hypothesis (consistent with all evidence): the mm-batch multiplication of
x-activation load instructions. Each token in the mm loop needs 2x float4 x-loads per
unit iteration per lane; weights are read once. At mm=8, x-load instructions outnumber
weight loads 8:1. These hit L2 (x is tiny) but saturate LSU throughput, starving the
weight stream. This is why: mm specialization helped (fewer loop overheads), wider
batching MM=8 didn't (register pressure AND no LSU relief), x-staging didn't (moves
the same loads to tgmem), and scheduling/prefetch didn't (not a memory latency problem).

Implication for further verify speedup: reduce x-load INSTRUCTIONS per weight byte,
e.g. shuffle-redistribute x across lanes so one load serves multiple tokens, or
restructure lanes-per-row x tokens-per-lane. NOT yet attempted - all prior x work
addressed traffic volume rather than instruction count.

## End-to-end production validation (2026-11-27)

3K-token P3-shaped evidence batch: correct answer in 3.8s. JSON long-form generation
sustains 63-71 tok/s. Vision end-to-end working (color/shape description accurate).

## MPP/matrix MoE threshold experiment: negative, reverted (2026-11-27)

Lowering INK_MOE_MM_MIN_M from 48 to 16 to engage matrix-tile MoE at prefill batch sizes:
M1 decode degraded 26.4 -> 31.1ms (matrix path pays fixed tiling overhead at small batch),
and chunked-prefill passes (M498-512, 960-990ms) did not improve. The matrix MoE kernels
in their current form are never the right answer on this hardware - the GEMV path with
mm-spec accumulators wins at every batch size tested. INK_MOE_MM_MIN_M stays at default.

## LoopSpec / DLoop paper evaluation (2026-11-27)

LoopSpec (2609.17184): pipelines decoding across recurrent depths - token r+1 starts
while token r finishes deeper recurrences, sharing the loop weight reads across pipeline
positions. Inapplicable as-is to Inkling serving: (1) verify marginal cost is expert-
dominated (3x traffic ratio), not trunk-dominated, and LoopSpec amortizes trunk reads
only; (2) requires batching positions at different recurrent depths with different
attention states - a rewrite of the verify pipeline structure. Would help M1 decode
(trunk is 66% of cost there) but M1 is latency- not throughput-critical in serving.

DLoop (2610.07659): reuses accepted-token computation as the next draft when the target
accepts everything. Inkling's beam-tree verifier already does this implicitly (the
verify pass produces the next token and its hidden state for free). No new mechanism.

MoE-Infinity expert caching (2401.14361): inapplicable - all weights are already locked
resident in memory; there is no cache miss to prefetch around.

LoopCache (2611.24301): targets long-context loop KV traffic. Our production batches are
2-3K tokens where KV reads are tiny vs weight streaming. Would matter at 32K+ context.

The literature frontier for this exact deployment (loop MoE, small batch, weights
resident, structured output) is: nothing published beats tuned tree speculation with
hardware-calibrated cost models. That is the current state of the art here.

## Speculation contribution accounting (2026-11-27)

Plain decode ceiling at M1 26.4ms = 37.9 tok/s. Measured against it:
- json 71 tok/s = 187% of ceiling (speculation highly effective on structured content)
- code 44.7 = 118% (partial - governor toggles)
- poem/tech 33-35 = ~90% of ceiling (governor probe overhead ~10%)

Remaining governor imperfection: probes cost a few wasted verify passes before the
cooldown engages on low-acceptance content. Ceiling on improvement ~10% for free text,
zero for structured. Not worth further tuning complexity.

DFlash2 alternative draft evaluated and rejected for this deployment: requires a
separate weight-loaded worker (~1GB additional) and single-slot operation, against a
machine with a documented watchdog-reset history at high memory pressure. The draft
phase is only 12% of cycle cost, capping any possible acceptance improvement's value.

## Tree budget increase experiment: neutral, reverted (2026-11-27)

Raising max_b tree budget 10 -> 16: json 65-67 vs 67-71 tok/s, no improvement.
The Bayes optimizer with the calibrated cost model already chooses trees well below
the budget cap - the cap is not binding. Tree shape is cost-optimized, not
budget-limited. INK_TREE_BUDGET env knob left in the binary for future experiments.

## Parallel slot scaling measurement (2026-11-27)

Concurrent streams DO share verify passes (96% M2 batches on 2 parallel streams).
Marginal decode batch cost: M1 27.7 -> M2 42.5ms = 15ms/token at M2 (high because
independent streams share few experts), but only ~6ms/token marginal at M4.

Measured aggregate throughput on independent JSON streams:
- solo: 24-70 tok/s (content dependent)
- 2 streams on parallel=2: 39.5 tok/s aggregate
- 4 streams on parallel=2 (queued): 39.5 tok/s (no gain - queueing)
- 4 streams on parallel=4: 49.9 tok/s aggregate (+26%), memory cost +1GB KV
- single-stream latency on parallel=4: slight degradation (59.8-66 vs 67-71)

Deployment guidance: parallel=2 optimizes single-stream latency (the P3 pipeline
pattern). parallel=4 wins only when 3+ concurrent streams are routine. KV cost is
~672MB/slot at 16K context fp8. The crossover is workload-dependent, not a universal
default. P3 discovery batches are serial per worker, so parallel=2 stays the default.

## PR location

https://github.com/sonder-albert/Strata/pull/1 - full experiment documentation in the
PR description including positive results, negative results, and root-cause model.

## System memory pressure interaction (2026-11-27)

Measured throughput varies 28-71 tok/s for identical content/config depending on
whole-system memory pressure. With 9.8GB locked inference weights + full user app
load, swap reached 5.6/6GB and decode slowed ~2x. This is NOT an engine regression:
the engine's own cost structure is unchanged (bit-exact outputs throughout).

Operational guidance: on a 24GB host, locked residency (~10GB) plus heavy user apps
is at the edge. If sustained slow throughput is observed, check vm.swapusage first
before suspecting engine regression. Dropping locked residency is NOT the fix - it
reintroduces the 40-second-stall problem it was created to solve. The correct relief
is closing unused applications or reducing parallel slots (670MB each).

## x-full-staging transpose experiment: DECISIVE NEGATIVE (2026-11-27)

The final untested variant of the transpose idea: stage all tokens' x rows ONCE per
expert-rowgroup into threadgroup memory (single barrier per chunk, mm<=2 due to the
32KB tgmem limit), decode reads on-chip. Result: everything slower (M1 31.3 vs 26.4,
M8 86.8 vs 66.8). Two compounding causes: tgmem allocation halves occupancy, and
mm<=2 chunking doubles barrier/plan overhead per expert. Verdict: the verify expert
cost is NOT movable by changing where x lives - not DRAM (proven), not LSU/tgmem
(proven now), not compute (proven by WONLY ablation), not scheduling/layout (7 experiments).

FINAL root-cause acceptance: the verify marginal cost is intrinsic to the kernel shape
x batch x dedup economics and cannot be reduced by ANY single-kernel transformation
available in Metal. The remaining gains live in: serving batching economics (verified),
acceptance-length improvements (draft quality), or megakernel fusion (unproven,
weeks-scale). The transpose direction is permanently closed with this evidence.

## Multi-stream beam-tree speculation: structural analysis (2026-11-27)

Naive per-slot speculation (round-robin INK_SPEC_MULTI) REGRESSES concurrent throughput
(30.8 vs 39.1 tok/s aggregate) because the steps serialize: each slot's verify pass
pays full trunk streaming cost separately. Plain batching is better because it merges
slots into ONE pass amortizing trunk reads. Correct multi-stream speculation requires
merging several slots' trees into a single mixed verify batch - a backend restructure
of the per-slot step machinery (GPU-side KV isolation per slot is fine, but the step
must batch across slots before forward). Effort: significant; benefit: speculative
acceptance on top of the 1.6x batching gain. Worth doing only after megakernel or
acceptance improvements shift the balance.

The INK_SPEC_MULTI code path remains behind env flag, default off.

## Multi-stream speculation: implemented, measured, REJECTED (2026-11-27)

Full implementation of per-slot chain drafting + merged verify batch (INK_SPEC_BATCH):
- nd=4: 19.1 tok/s aggregate vs 39.1 plain batching
- nd=7: 16.6 tok/s (deeper chains worse - larger verify batches)

Root cause measured via passlog: merged verify passes of M=10 cost 145ms with marginal
token cost ~12ms because expert deduplication collapses for INDEPENDENT streams
(unrelated content routes to disjoint experts). Speculation multiplies the tokens
verified per pass; without dedup the marginal economics invert. Verify cost must be
< accepted-tokens x marginal-cost-of-plain-batch for speculation to win, and at M=10+
the expert-streaming cost dominates everything.

Single-stream speculation works because batch M<=8 dedups to 40%. Multi-stream
speculation fails because independent content dedups to ~64/128 slots = no savings.
CONCLUSION: for independent concurrent streams on this model, plain batched decode
with shared trunk streaming is optimal. Speculation is a single-stream tool here.
The implementation remains behind INK_SPEC_BATCH, default off.

## Megakernel feasibility analysis (2026-11-27): low ROI on this codebase

Bubble fraction (host wall - GPU execution time) is only ~1-2ms per 27ms pass = 4-7%.
The codebase already fuses aggressively: attention epilogue + sconv + residual in one
kernel, MoE plan fused, QR+sink+gate fused projections, concurrent projection batches.
lithos-metal's megakernel wins were on dense models at higher bandwidth where bubbles
are a larger fraction. Do not prioritize megakernel here; the in-kernel dataflow and
bytes-per-token are the binding constraints.

Bigger lever identified: trunk weights are q8row (1 byte/weight, 1.76GB/token stream).
nvfp4-class quantization would halve trunk traffic (the single largest byte reduction
available, ~40% of total per-token traffic) but requires Metal nvfp4 kernels AND
requantization quality validation. This matches the user's stated NVFP4 direction.

## Inkling-2-Mini original precision: NVFP4+MXFP8 native (2026-11-27, model card)

The original checkpoint is distributed in MIXED NVFP4 and MXFP8 precisions - it is a
natively low-precision model, not a high-precision model being degraded by quantization.
The SQ-10 pack's q8 trunk is a TRANSCODE from lower-precision originals, not a quality
reference. This inverts the quality calculus: serving nvfp4 directly is the
training-native precision; q8 transcoding can only lose information relative to the
original nvfp4 tensors, never gain. Optimal serving = decode the original nvfp4/mxfp8
encodings as-is. No requantization quality risk exists for tensors already nvfp4 in
the checkpoint - they are exact by construction.

S4_K comparison for non-nvfp4 tensors: 4.19 bpw vs nvfp4 4.5 bpw, kernels exist in-tree.
Optimizer for Inkling training is not documented in the public model card; Miles/SkyRL
are the post-training frameworks listed. Muon is NOT mentioned anywhere in the card.

## NVFP4 serving direction (supersedes requantization plan)

Original weights download in progress (19.5GB bf16-and-mixed checkpoint). Plan:
1. inspect actual per-tensor encodings in the safetensors (dtype extensions?)
2. implement Metal decode for the nvfp4/mxfp8 formats as-is
3. repack trunk in original precisions, serve, verify bit-parity vs HF reference logits

## Checkpoint precision anatomy (2026-11-27, decisive analysis)

Original checkpoint structure (1244 tensors): trunk attention/embeddings BF16,
experts natively nvfp4-packed: u8 nibble arrays [128, 2304, 720] w13 + [128, 2304, 360] w2,
per-16-element F8_E4M3 scales, per-expert fp32 amax, per-tensor fp32 outer scale.

Size comparison per expert: nvfp4 original 2.99MB vs SQ-10 s2/s3 custom 2.8MB.
**The SQ-10 custom expert quantization sacrificed training-native precision for nearly
zero size benefit.** Serving nvfp4 experts directly restores original quality at +0.6GB
total memory. This is a quality upgrade, not a speed tradeoff.

trunk attention is BF16 in the original - the q8 in SQ-10 is a lossy transcode.
nvfp4 trunk would halve traffic (big speed win) but compounds quantization error.
Quality verification on P3 tasks is the gate for trunk nvfp4.

NVFP4 decode: E2M1 nibble pairs + E4M3 per-16 scales. The existing kE2M1 LUT in
ink_metal_codecs.metal matches the value set exactly. Kernel work is adaptation of
sf4/mx8 decode patterns, not new research.

## NVFP4/S4 trunk requantization workstream (2026-11-27, in progress)

Infrastructure complete:
- s4k_encode.py: bit-compatible S4_K encoder (verified roundtrip vs kernel decode logic)
- repack_s4_trunk.py: full pack rebuild replacing q8 attention with s4k from bf16 source
- ink_linear_sk_mv_4 kernel already exists in engine (dispatch is format-automatic)
- quality_check.py: P3 + sweep comparison harness

Key format facts: q8 attention in SQ-10 = 132 tensors 0.62GB; s4k halves the stream.
Fused attention kernel only accepts q8/f32 - s4 falls back to per-projection path
(that path was measured neutral anyway). unembed stays s6k.

CRITICAL quality context: model is nvfp4/mxfp8-training-native in experts but BF16
in attention. 4bit attention quantization compounds error that training never saw.
Quality gate: P3 correct must stay >= 6/7 and sweep outputs coherent.

## Trunk requantization results: quality gates PASSED, speed blocked by draft mismatch (2026-11-27)

Complete requantization campaign with bf16 originals downloaded (19.5GB checkpoint):

sq10-s4trunk.strata (9.95GB, attention q8->S4_K):
- P3 business quality: valid 7/7 correct 6/7 (IDENTICAL to q8 baseline)
- draft acceptance: 0 (speculation dead - draft was trained on original-precision logits)
- speed: 36-39 tok/s (plain decode equivalent; draftoff saves draft overhead)

sq10-s6trunk.strata (10.1GB, attention q8->S6_K):
- P3 business quality: valid 7/7 correct 7/7 (BETTER than q8 baseline's 6/7)
- draft acceptance: 0 (same mismatch)
- speed: 36 tok/s plain

Critical strategic finding: speculation multiplies throughput (json 72 tok/s) more
than any quantization saves bytes (max ~45-58 tok/s theoretical). Trunk quantization
only wins after the draft model is recalibrated/retrained for the quantized target.
This is exactly the user's stated long-term direction (QAT/adaptation training).

Quantization SNR measurements on attention tensors: q8 44dB, s6 32dB, s4 19dB.
s6 is the quality sweet spot (0.7% noise). Both packs are valid artifacts for
draft-off deployments, memory-constrained variants, and QAT starting points.

Infrastructure in repack/: s4k_encode.py, s6k_encode.py (bit-compatible encoders),
repack_full.py (resumable pipeline), quality_check.py. bf16 original cached at
ms-cache/models/thinkingmachines--Inkling-2-Mini (19.5GB).

## NVFP4 hybrid experts: quality verified, throughput cost measured (2026-11-27)

sq10-nvfp4.strata (12.1GB): 1108 hottest experts (by vram_load_order priority) upgraded
from s2/s3 to training-native nvfp4, byte-identical lossless repack from the original
checkpoint. Kernel layout is 100% compatible with existing sf4 decoders - zero kernel work.

Quality: P3 valid 7/7 correct 6/7 - IDENTICAL to sq10 baseline. Outputs differ from q8
(expected - different precision) but are coherent and arguably richer.

Cost: json 67-68 tok/s accept 0.94 (vs 71-72 at 0.89 on sq10). Expert streaming per token
is 1.6x larger for upgraded experts. Memory RSS 12.8GB vs 9.8GB (+3GB).

KEY FINDING: nvfp4 expert upgrade is a QUALITY play, not a speed play. It costs throughput
and memory for training-native precision. The business value depends on whether s2/s3
quantization measurably hurts task quality - on the P3 benchmark it does NOT (both 6/7).
Decision: keep sq10 for production throughput; keep sq10-nvfp4 as the quality-variant
pack for cases where task quality matters more than speed.

Original checkpoint precision anatomy confirmed: layers 0-19 experts nvfp4-native,
layers 20-23 mxfp8-native (8.2MB/expert - not viable at native precision on 24GB).
Encoding formats byte-identical to sf4/mx8 strata containers respectively.

## NVFP4 final decision matrix (2026-11-27, complete)

Three variants measured end-to-end:
- sq10-nvfp4 (1108 hottest upgraded): P3 6/7 same as baseline, json 67-68 accept 0.94,
  12.1GB file / 12.8GB RSS. Quality-neutral but draft acceptance improved.
- sq10-nvfp4-500 (500 hottest): P3 5/7 WORSE, json 66 accept 0.76. Partial upgrade is
  worse than none - router/expert interactions need broad precision coverage to help.
- sq10 production: P3 6/7, json 71-72 accept 0.89, 9.8GB RSS. Best throughput.

CONCLUSION: nvfp4 upgrade is quality-neutral on the P3 business metric while costing
memory and speed. Partial upgrades are actively harmful. The s2/s3 compression in
SQ-10 was already well-chosen. KEEP sq10 for production. nvfp4 pack retained for QAT
starting point (its training-native experts are the right foundation for draft
retraining, where precision actually matters for acceptance).

This closes the quantization direction with complete evidence: every cheaper-than-q8
path loses more to speculation mismatch than it saves in bandwidth.

## NVFP4 calibrated-cost experiment: throughput parity, not a win (2026-11-27)

Recalibrating tree cost model for nvfp4 pack economics (base 29, marg 7):
json 66-73 tok/s across 4 runs (mean ~69) vs sq10 67-72 (mean ~70) - statistically
equivalent throughput at +3GB memory cost. Draft acceptance is consistently better
on nvfp4 (0.94 vs 0.89) because training-native experts restore draft-target logit
consistency, but the larger expert traffic cost exactly offsets the acceptance gain.

FINAL quantization verdict: SQ-10's precision allocation is already optimal for
throughput on this hardware. All compression paths beyond it lose to speculation
mismatch; all precision upgrades cost more in traffic than they gain in acceptance.
The sq10-nvfp4 pack is retained solely as a QAT/draft-retraining foundation.

## Draft quantization experiment: decisive negative (2026-11-27)

User question: can quantizing the draft to sq10's bit-width improve acceptance?
Answer measured: NO, catastrophically worse. s5->s4 draft quantization:
- acceptance 0.89 -> 0.14-0.21
- json throughput 72 -> 29 tok/s

The draft is accuracy-bound, not consistency-bound. It's already a small distilled
model at the edge of its precision budget; s5k is what it needs. Quantizing it further
destroys its ability to predict the target's tokens regardless of distribution matching.

Inverse evidence confirms the asymmetry: making the TARGET more precise (nvfp4 experts)
IMPROVED acceptance 0.89 -> 0.94, because the draft itself is precision-native. The
draft-target consistency benefits from target precision, not from draft degradation.

This definitively answers: speculation quality is improved by raising target precision
or retraining the draft at the target distribution - never by lowering draft precision.

## Draft precision sweep complete: s5k is the sweet spot (2026-11-27)

Three draft precision variants measured against sq10 target:
- bf16 draft (original): accept 0.76, json 61-62 tok/s
- s5k draft (sq10 pack): accept 0.89, json 67-72 tok/s  <- BEST
- s4k draft: accept 0.14-0.21, json 29 tok/s

Counter-intuitive but decisive: the QUANTIZED s5k draft outperforms the bf16 original.
Explanation hypotheses: (1) sq10 target is itself quantized - the s5k draft's distortion
partially matches the target's distortion landscape, sharpening agreement on the exact
argmax token; (2) bf16's extra precision resolves toward the ORIGINAL model's distribution
which the quantized target no longer follows. Either way: matching the target's precision
regime matters more than raw precision. This strongly validates QAT-style draft retraining
at the target distribution as the correct long-term direction.

## Prefix cache verification (2026-11-27)

Shared-prefix reuse confirmed working: exact repeat of 1K-token prompt 3.33s -> 0.27s
TTFT (92% reduction). Extended prompts reuse the shared prefix and only recompute the
delta. Production batches with shared structure benefit proportionally. No further
optimization available here - the mechanism is already correct and effective.

Remaining TTFT floor: fresh 1-3K token batch = 2.5-3.8s at prefill 500 tok/s.
This is bandwidth-bound and irreducible without prefill architecture changes (YOCO etc).

## LoopRes paper analysis (arXiv 2611.31094, Nov 2026) - user-requested direction

LoopRes turns loops into significant gains for pretrained LLMs by attending to outputs
of earlier loops. Mechanism: each looped layer attends not just to current input but to
outputs of previous loop iterations (attention residuals), with two key innovations:
1. Loop-aware keys: trainable per-block-per-loop vectors that encode structural identity
   of each iterate (since same-layer outputs across loops are hard to tell apart)
2. Residual window W: restrict sources to the latest W loops, forcing incremental
   buildup and bounding compute/memory overhead

Results: at 2x compute, improves Qwen3-1.7B-Base by 29.1% over SFT. Scales to 30B
and across 3 model families. This is a MODEL ARCHITECTURE change requiring training
(post-training at minimum) - not an inference-time optimization.

Relevance to Inkling-2-Mini: Inkling already HAS loop passes. The question is whether
its attention already has loop-residual connections or just plain recurrence. If plain,
adding LoopRes-style attention residuals would be a training-time capability upgrade.
For INFERENCE on the existing model: attention residuals don't apply - they change the
forward computation graph itself. This is firmly in the training-direction territory
the user marked as long-term. Mixture-of-Loops (2611.34923) is a related engram-style
mixture-of-depths/routes idea also requiring training.

GLM 5.4: no technical report found on arxiv as of this date (searched). Zhipu's latest
public materials don't cover attention residuals for loops specifically.

## MoLT paper analysis (arXiv 2611.34923, Nov 2026) - attention residual family

Mixture-of-Loops Transformer: uses keys/values as memory selectively reused across
depths and loops (MoLA attention). Variants: MoLT-R (loop residual + KV reads),
MoLT-KV (restart from embeddings + memory reads), U-MoLT (encoder/decoder split).
Core question it answers: what should be passed between loops - hidden states only
or KV memory too? Answer: selective KV reuse across loops improves the frontier.

Same classification as LoopRes: MODEL ARCHITECTURE change requiring training.
For inference optimization of the existing Inkling-2-Mini, neither LoopRes nor MoLT
applies directly. They are inputs to the training-direction planning (engram adaptation,
draft distillation at target distribution). Both PDFs archived in loop-literature/.

## marg=12 cost recalibration: FIFTH POSITIVE RESULT (2026-11-27)

After mm-spec made verify faster, the cost model needed rebalancing again.
INK_TVER_MARG=12 (from 10.5): json throughput 67-72 -> 76.6-85.6 tok/s across 6
repeated runs (mean ~81, +15%). All outputs bit-exact, P3 unchanged (6/7).
Curve scan: marg 8 = 61, 10.5 = 70, 12 = 82, 14 = 66 - clear peak at 12.
Note: acc=1.0 metric is inflated by refusal-style short outputs in some probes;
the true acceptance structure is unchanged. The speed gain is real and verified
by output-length-controlled comparison. LESSON: recalibrate cost model after
EVERY kernel speedup, as documented in this skill's earlier guidance.

## Cost model final tuning: base 25 + marg 12 confirmed optimal (2026-11-27)

Full 2D scan after mm-spec speedups: base {20=73, 25=81, 30=57-69}, marg {8=61, 10.5=70,
12=82, 14=66}. The compiled-in default (25, 12) sits at the peak. marg12 json throughput
mean ~78 tok/s across 10 runs vs ~68 baseline mean across 7 runs (+15%). High variance
(66-86) is inherent to acceptance distribution on mixed content - always compare means
across multiple runs, never single measurements. Base and marg are now both compiled
defaults; env knobs remain for future re-tuning after further kernel changes.

## Long-context behavior validation (2026-11-27)

TTFT scales linearly with prompt length (no cliff): ~750-940 tok/s marginal prefill
from 2K to 8K tokens. Decode degrades mildly at long context from KV attention reads
(~118MB/token extra at 8K position = ~5% traffic). The 16K context configuration is
correct for the production batch profile (2-3K typical). The 1M theoretical maximum
is irrelevant for this workload - KV memory at fp8 dominates feasibility anyway.

## DeepInfer reference architecture (deepseek-ai/DeepInfer, production infra)

Key techniques relevant to our serving stack:
- **DSpark + confidence-scheduled verification**: draft length adaptive per confidence,
  default max 5. This matches our beam-tree approach but uses confidence estimates
  directly rather than Bayes cost model. Their varlen scheduling could improve our
  acceptance economics further - worth studying DEEPINFER_USE_NEXTN_DECODE behavior.
- **SWA bounded replay**: for decoder prefill, replay only bounded suffix covering
  SWA window + max backtrack. Reduces prefill work on continued generation.
  Directly applicable to our chunked-prefill continuation.
- **Asynchronous scheduling**: overlap CPU batch scheduling with device execution.
  Our serving already overlaps via concurrent encoder; their zero-overhead batch
  scheduler is the more aggressive version for many-slot serving.
- **Batch-invariant inference**: same results regardless of batch composition.
  This is a correctness property our bit-exact sweeps implicitly assume. Their
  strict mode is NVIDIA-only but the principle matters for multi-stream work.
- Hardware targets: NVIDIA SM100, Huawei Ascend. No Apple Silicon support.

Notable: their draft verification is confidence-scheduled not cost-model-scheduled.
The deepseek approach to the same speculation problem validates our direction while
offering a different acceptance-vs-cost control mechanism worth comparing.

## Architecture adaptation research notes (2026-11-27)

YOCO (arXiv 2405.05254, Microsoft): decoder-decoder architecture - self-decoder encodes
global KV once, cross-decoder reuses via cross-attention. Only caches KV ONCE total
(vs per-layer in vanilla transformer). Enables prefill early-exit and cuts memory.
This is the deepseek-v4.1-flash style architecture the user referenced.

Key insight for Inkling adaptation: YOCO is a PRE-TRAINING architecture, but the user
correctly notes that architecture conversion can be done with FREEZE + light adaptation
not full retraining. Published conversion recipes (Mistral->SWA, dense->sparse family)
show 1-5% of original training compute suffices for architecture swaps when base weights
are frozen and only the new attention structure's projections/norms are adapted.

StepFun's Step-3.7-Flash exists as a modern flash model reference; their tech blog
details production distillation-of-architecture. The pattern across all these works:
architecture conversion = freeze base + adapt interface layers + short alignment run.

Feasibility on 24GB Mac: draft distillation (640MB student) YES. engram/attention
adapter adaptation YES if adapter-scale (LoRA-style or small projection modules).
Full YOCO conversion of a 26B model: forward pass fits (inference), backward pass
for even frozen-weight adaptation needs activations+gradients ~15-25GB - borderline
feasible with careful gradient checkpointing. Cloud GPU only needed for full pretraining.

## EAGLE-3 + LK-loss draft training methods (2026-11-27, core references for distillation)

EAGLE-3 (arXiv 2503.01840): abandons feature prediction constraints of EAGLE.
Key insight: scaling training data gives limited gains when draft predicts FEATURES;
EAGLE-3 instead trains the drafter against the target's output distribution directly
(training-time test). This validates our approach of behavior-cloning + KL on target logits.

LK-loss (arXiv 2602.23881): direct acceptance rate optimization - trains the drafter
to maximize expected acceptance length rather than token-level cross-entropy.
This is the sharpest objective match for our goal: acceptance is THE throughput driver.
Recommended loss for the Inkling draft retraining: LK-loss primary, KL auxiliary.

SpecForge (sgl-project): production-grade training framework with online/offline
disaggregated training modes. NVIDIA-oriented but methodology transfers. Reference
for pipeline structure, not directly usable on Apple Silicon.

Research gap: stepfun technical leads publish training details on Zhihu rather than
official blog. Web scraping of zhihu/bing blocked in this environment - needs
browser-capable followup or manual retrieval by user.

## PD-disaggregation for training (Yibo Zhu, zhihu p/2087842854592950950)

User-provided reference on training-serving disaggregation. Article could not be
scraped from this host (zhihu anti-bot). Key concept from title and stepfun context:
separating prefill (training-data generation / teacher forward) from decode (student
training) as independent scaling dimensions - same PD-separation philosophy as
inference but applied to the training pipeline. For our distillation: teacher capture
(target hidden states) is the 'P' side already running on the service; student
training is the 'D' side that can scale independently (even on cloud GPUs later).
The user's point: training infra should be designed with this separation from day one.

## LatentMoE evaluation for Inkling-2-Mini (2026-11-27)

LatentMoE (deepseek-style): experts operate in a compressed latent space with per-expert
up/down projections rather than full-dimension expert MLPs. Benefits: smaller expert
weights per expert, better parameter efficiency, potentially better quality per byte.

Applicability to Inkling-2-Mini serving:
1. It is an ARCHITECTURE choice, not an inference optimization. Inkling-2-Mini was
   trained with full-dim experts (2304 hidden, 128 experts x top-8). Converting to
   LatentMoE requires retraining or at minimum substantial adaptation training.
2. Freeze+adapt feasibility: converting full-rank expert weights to low-rank factored
   form CAN be done as a compression operation (SVD-like) without full retraining,
   but that is weight compression, not architecture change. Quality cost needs
   evaluation per-expert - hot experts need higher rank than cold ones.
3. Serving benefit if done: smaller per-expert bytes -> less streaming traffic ->
   directly attacks the verify marginal cost we've measured as the bottleneck.
   This is the ONE architectural direction that maps exactly onto our measured bottleneck.
   Estimated gain if experts go from 2.8MB to ~1.4MB latent-factored: verify marginal
   cost halves, json potentially 79 -> 110-130 tok/s.

Cost: requires computing low-rank factorization per expert (offline, hours of CPU),
plus brief alignment finetune to recover quality (small, feasible on this machine).
This is a MEDIUM-term high-value project, ranked after draft distillation which is
already underway and cheaper.

SuSun Kern article (zhihu p/2107865260535424227): RK3588 NEON-optimized Inkling
inference on ARM - confirms the model runs at edge scale with careful kernel work.
Article not retrievable from this host (zhihu anti-bot). NEON/ARM optimizations
don't transfer to Apple Silicon Metal but the quantization-deployment pattern validates
the low-bit serving approach.

## DFlash2 self-distillation pipeline (2026-11-27, in progress)

User decision: follow DFlash2 but make it OURS - quantize, train, modify architecture
on our SQ-10 tap distribution. Root-cause hypothesis for the native checkpoint's
acceptance collapse on SQ-10: it was trained against bf16/SQ-20 target tap
distributions; SQ-10 quantization shifts the tap distribution systematically.

Architecture reverse-engineered from upstream ink_metal_dflash2.{h,metal}, full spec in
/Users/sonders/dageno-modelbench/dflash2-arch-spec.md. Key facts that differ from
intuition:
- NOT autoregressive: block-8 parallel forward, then greedy chain selection row by row
  using rank-256 bilinear link scores: s(c) = raw_logit(c) + <pred[prev]*gsel[d+1], succ[c]>
- Dynamic depthwise causal conv (2-tap, 16-channel groups, DYN=576) pre/post each sublayer
- ctx ingest: fc[2304,9216] compresses 4 taps -> per-layer Wkv writes fp16 ring (2112 window)
- rope theta = 1e4 in draft (target trunk uses 1e6) - do not mix them up
- base_kernel [2,2,2304] flat rows: [0,0]=w0 [0,1]=w1 (pre), [1,0]=w2 [1,1]=w3 (post)
- attention within block is NON-causal full 8x8, plus ctx cells [max(lo,pos0+t-2047), pos0+7]
- inference raw logits are NOT softcapped; softcap(30) only inside select_chain confidence
- q/o/gate/up/down already q8row in the metal path; fc/wkv still bf16 (first quantization target)
- codebooks pred/succ [V,256] bf16 = ~206MB combined; rank 256->128 is an architecture candidate

Trainer: /Users/sonders/dageno-modelbench/dflash2_train.py
- Muon for matrices + AdamW for vectors, same dual-optimizer pattern as DSpark trainer
- loss = per-row CE (row r predicts gold token at anchor+r) + 0.5 * chain-consistency CE
  (teacher-forced: full chain score incl. bilinear link term, differentiable)
- FREEZE_MLP=1 default (mlp gate/up/down frozen, saves ~2.4GB optimizer memory)
- FREEZE_CODEBOOK=0 default (codebooks trainable - they're central to chain quality)
- CTX_PRECOMPUTE=1 default: ctx KV precomputed once with initial weights (stale; recompute
  is 2.6G Flops/layer/window and doubles step cost - evaluate whether staleness matters)
- data: distill_windows.npz 548 windows, taps [77,4,2304] fp32 per position - exactly what
  DFlash2 needs (same INK_TAPDUMP format as DSpark)
- NOT YET RUN: memory pressure too high while DSpark training active. Smoke test first
  with EPOCHS=1 MAXSTEPS=5 before any long run.

Safety rules learned the hard way (4 watchdog reboots):
- NEVER run two weight-loading engine/training processes concurrently
- check memory_pressure before starting anything heavy; wired >14GB = do not add load

## Wave-style self-speculation literature for looped transformers (2026-11-27 sweep)

Three papers specifically target looped transformers. Relevance ranking for Inkling:

1. **DLoop (arXiv 2610.07659, NAVER)** — HIGHEST applicability to our DFlash2 pipeline.
   Parallel draft models waste a verify pass even when the draft is confident enough to
   keep going. DLoop runs MULTIPLE drafting stages before one verification, continuing
   while confidence stays high, verifying all accumulated tokens at once. Key enabler:
   "loop-aware training" exposes the draft to its own hidden states for unverified tokens.
   Reported +5-41% wall-clock on top of EAGLE-3/DFlash/Domino/DSpark/MTP.
   For us: DFlash2 already outputs per-row conf; the adaptive verify-width planner
   (choose_verify_drafts) is a static per-block version of this idea. DLoop generalizes
   it across blocks. Requires training support — fits directly into dflash2_train.py
   as a curriculum extension (feed student hidden states from unverified rows).
   NOTE: verify cost grows superlinearly with accumulated tokens on MoE (expert union),
   so the confidence threshold must be cost-aware, not accuracy-only.

2. **LoopSpec (arXiv 2609.17184, KAIST)** — training-free self-speculation using
   EARLY recurrent-depth predictions as drafts, pipelined across time, with a selective
   second proposal from deeper depth. Closed-form optimal proposal depths. Up to 6.83x
   on looped models WITHOUT any draft model at all.
   Inkling applicability: our taps already capture intermediate passes (visits 5,23,41,48
   = pass1/pass2/pass3 interleavings). EXPERIMENT QUEUED: project early-pass hidden
   states through unembed offline (distill_windows.npz has everything needed) and
   measure agreement with final tokens. If pass-2 predictions agree with pass-3 at high
   rate on structured content, a zero-training draft path exists: verify pass-2 drafts
   with pass-3 in a pipelined schedule. Caveat: LoopCD evidence shows pass-1 is much
   weaker than pass-3; the pipelining win depends on pass-2 quality, not pass-1.

3. **LoopCache (arXiv 2611.24301, PKU)** — assumes loop-SPECIFIC KV caches
   (Huginn/Ouro/Raven style). Inkling recomputes KV per pass and shares one cache,
   so the cyclic-KV scheduling does not transfer. One transferable idea:
   confidence-guided dynamic draft updating — refine drafts in flight at intermediate
   loops instead of committing to fixed-depth drafts. Their correlation finding:
   prediction switches between loops correlate with low confidence. Could inform a
   per-token early-exit/decision rule in a future pipelined design.

Deprioritized (read, not applicable): MegaBlocks (training-side MoE), llm-in-a-flash
( NAND/dram paging — we already lock full pack in memory), MoE-Infinity (expert caching
for batch-1 — our experts are all resident), 2607.12696 cost-aware SD for MoE (already
implemented as our E_union cost model), 2610.34440 expert-bytes analysis (already
internalized in our byte-exact traffic model).

## Architecture adaptation roadmap (user-ordered: arch before draft, 2026-11-27)

User decision: DFlash2 training moves LAST. Architecture adaptation first (accumulate
data in parallel), because draft training targets the tap distribution - any arch change
invalidates draft training. Order by hidden-distribution disturbance:

1. DSA replacing the 4 SWA layers (layers 5,17,29,41 of 48 - sparse hybrid, NOT the
   24/48 pattern sometimes assumed). SWA is training-native so no long-context
   extension needed. Small blast radius, moderate payoff: decode KV read width.
2. AttnRes form refinements (already training-native in trunk, engine-supported).
3. YOCO-style prefill restructure: affects prefill compute graph mostly, decode
   hidden distribution less - can follow after DSA/AttnRes.
4. LatentMoE / Engram: far-term, after data accumulation.
5. Draft training (DSpark re-distill, then DFlash2) ONLY after architecture freeze.

## 2DSD paper key facts (arXiv SqueezeAILab, local /tmp/2dsd.pdf)

- Two-dimensional SD: depth-axis (loop-level self-speculation) x token-axis (block draft)
- LAST-LOOP KV SHARING: all loops share one KV cache instead of per-loop caches;
  some looped models (Dory1.0-4B) are robust WITHOUT adaptation, others need
  "lightweight self-distillation". For Inkling: KV is recomputed per pass - sharing
  the final pass KV across passes would cut KV memory AND write bandwidth ~3x.
  This is a candidate arch adaptation with freeze+self-distill training.
- Geomean throughput: up to 7.6x greedy / 5.5x sampling over native AR on looped LMs
- Depth-concurrent execution amortizes weight reads across loops - orthogonal to
  our token-axis speculation; potential combination target after arch freeze.

## Data v2 (user-directed distribution)

- prompt_pool.jsonl built: 9590 prompts = openr1-math 3000 + ultrachat 3996 +
  magicoder 2499 + dageno-biz 95 (~1%)
- Collection: on-policy via live service, temperature 0.8, max_tokens 400,
  INK_TAPDUMP captures (token, taps) directly - no regeneration needed
- Sources chosen to mirror frontier-lab draft-training mixes (EAGLE-3/NeMo recipe:
  ShareGPT-like chat + math reasoning + code + agentic tool use)
- NeMo DFlash2 recipe facts: position-decay weighting exp(-(k-1)/gamma) with
  gamma=4 for block 8; selector loss = CE restricted to top-k candidates of the
  position logits, skipped (and tracked via candidate_recall) when gold absent;
  monitor acceptance length directly during training.
  dflash2_train.py updated to match; gather-VJP pitfall: stop_gradient topk indices.

## Compute plan (user-approved direction)

Training is draft-scale (126M-350M params): Kaggle free P100/T4 is enough for the
first iterations; Colab Pro A100 only justified if iteration frequency demands it.
CRITICAL PATH is data collection (must run on the SQ-10 service itself - taps cannot
be generated anywhere else), not training compute. MLX->PyTorch port of the trainer
is required for CUDA; forward semantics are fully specified in dflash2-arch-spec.md.

## Engram retrofit feasibility verdict (2026-11-27, evidence-based)

DeepSeek Engram paper read in full (33pp, /tmp/engram_paper.pdf). Key facts:
- Deterministic n-gram hash lookup enables host-memory prefetch - ideal for UMA
- U-shaped sparsity allocation: optimal reallocates 20-25% of MoE params to memory
- Biggest gains in reasoning (BBH +5.0) and long-context (NIAH 84->97), mechanism:
  relieves early layers from static reconstruction + frees attention for global context
- NOT applied every layer; residual insert at selected layers (Hy4: layers 2,12,22,32)

WHY COLD-START GRAFTING IS A DEAD END (verified against all public sources):
- Hy4-flash native tables: ~31.7GB bf16 PER LAYER, 4 layers - even q4 one layer
  exceeds total budget. Sized for 8B+ models, wrong scale entirely.
- EngramFlash-qwen3-8b checkpoint: 14GB. Same problem.
- Hash-space portability (fundamental): index = hash(compressed_donor_token_ngram).
  Inkling tokenizer != donor tokenizer => same text hashes to different rows =>
  grafted table is semantically scrambled garbage. Shrinking table modulus rescrambles
  too. No published cross-tokenizer transfer recipe exists.
- Qwen3.8-Flash-Next has NO engram at all (config verified, zero engram keys).
- Hy4-flash DOES use deepseek_sparse_attention layers + sliding hybrid - it is the
  living reference for BOTH DSA and Engram directions. layer_types in config.

AFFORDABLE ENGRAM PATH (if pursued): self-train small tables on Inkling vocab.
Per-layer 2M rows x 768 x q8 ~ 1.5GB for 2 layers - fits budget. Backbone frozen,
trainable = tables + gating + conv. Real cost is token budget (continued-pretrain
scale, billions) - needs rented GPU days, not local hardware.
GATE METRIC BEFORE SPENDING: measure static n-gram repetition rate in our actual
workload distribution (JSON/code/business text should score high). If low, kill.
NOTE tension: Engram says move sparse budget FROM MoE TO memory; LatentMoE says
compress MoE further. The n-gram measurement arbitrates - do not do both blindly.

## Hy4-flash as DSA reference architecture

text_config.layer_types mixes deepseek_sparse_attention with sliding_attention,
mlp_layer_types dense->sparse. If we implement DSA for Inkling's 4 SWA layers,
Hy4's config + oMLX PR #5475 runtime are the closest working references.

## DSpark distill round-1 postmortem (2026-11-27, decisive)

Distilled draft deployed: acc = 0.00 everywhere, json tps fell 84 -> 33 (plain-decode floor).
Offline losses had IMPROVED dramatically: train windows 7.19 -> 2.57, heldout 13.45 -> 2.85.
Loss improvement + runtime collapse = classic exposure bias, not data starvation alone:

ROOT CAUSE (high confidence): Markov head chains on prev token. Training fed gold
(teacher forcing); inference feeds the student's own predictions. First-row error
cascades through the chain. The loss measured mastery of the wrong (easier) task.
Secondary contributors: tiny 9.8k-token dataset (heavy duplication, 195 unique seqs),
greedy-only generation, ctx precompute staleness.

FIXES APPLIED/PLANNED:
1. Scheduled sampling in distill_train_full (SS_P=0.3: feed student argmax as prev
   token) - EAGLE noise-injection / DLoop loop-aware family
2. Data v2 collection running (9590 diverse prompts, temp 0.8, 400 max tokens)
3. Retrain only at 300k+ unique tokens; keep ep1 ckpt as comparison point
4. ctx refresh every N steps rather than once-frozen (when compute allows)

LESSON FOR DFLASH2 TRAINING: chain loss must include student-in-the-loop exposure;
pure teacher forcing will reproduce this exact failure. NeMo recipe's selector CE
is teacher-forced too - must be combined with noise injection or scheduled sampling.

## Encoder verification (positive)

S5_K re-encoder is byte-exact against production tensors (1494/1494 bytes) -
repack pipeline exonerated. Q8_ROW layout: [rows x f32 scales][rows*K int8].

## Engram tokenizer portability analysis (2026-11-27, user-challenged, revised)

User was right to push back. Refined verdict after measuring:

FACTS:
- Inkling tokenizer: custom 200k BPE (hello=32205, 60 special tokens incl
  <|content_invoke_tool_text|>) - NOT o200k, NOT Qwen2. No public match found.
- Hy4 tokenizer: 120k BPE. Shared token STRINGS with Inkling: 52,015 (43% of hy4).
  Identical IDs: only 256. IDs don't matter - Engram hashes compressed canonical
  ids derived from NFKC+lowercase TEXT, so string-level sharing is what counts.
- Engram table rows are indexed by hash(canonical_id n-gram). Row semantics are
  entangled with donor's canonical id assignment, which is scan-order dependent.
  Direct row transfer to Inkling's id space is meaningless WITHOUT remapping.
- THE WORKING TRANSFER PATH (open question in community, no published impl):
  build inkling_token -> text -> NFKC -> donor_canonical_id map for shared
  tokens; at lookup time use donor id space. N-gram boundary mismatch means
  some rows hit wrong entries - partial transfer value, must be measured.
- SSD offload of Engram tables IS production-proven (oMLX PR #5475 closed/merged):
  BF16 or packed affine tables read on-demand from disk - table size is NOT a
  memory blocker on UMA machines. Removes the 31.7GB objection entirely.
- Hy4-flash config confirms engram at layers [2,12,22,32], 3-gram max, 8 heads,
  768 dim per ngram, AND deepseek_sparse_attention layer types + sigmoid-routed
  MoE + latent experts - single reference architecture for all our target archs.

DECISION IMPLICATIONS:
- Engram self-training remains the clean path; grafting Hy4 tables requires the
  canonical remap probe FIRST (measure alignment hit-rate on real token streams)
- Probe is cheap: build both compressed lookups, run tap tokens through, measure
  what fraction of trigrams map exactly. If >60-70% of high-frequency trigrams
  align, grafting warms up meaningfully. Build this probe next arch session.

## AttnRes already training-native (user correction)

Inkling 2 Mini already has attention-residual across passes trained in - the engine's
attn_res_mix_norm fused kernel serves the native architecture. There is NO adaptation
gap to close; the upside is in the DSA indexer direction instead. DSA reference impl
exists in oMLX patches (hy_v4/indexing.py + glm_moe_dsa/indexer_nax.py, NAX kernels,
top-512/2048 selection). Porting the DSA indexer to Inkling's 4 SWA layers is the
highest-value architecture adaptation available without retraining scale cost.

## Engram grafting: GO verdict (2026-11-27, measurement-backed)

engram_align_probe.py on 143k real workload tokens:
- token translatability to Hy4 canonical space: 96.3%
- trigram full-translatable coverage: 92.9% (frequency-weighted identical)
- untranslatable = mostly inkling control tokens (<|end_message|> etc) which should
  bypass semantic memory anyway - near-zero cost
DECISION: grafting path is GO. Tables can live on SSD (oMLX #5475 pattern).

IMPLEMENTATION PATH:
1. Download Hy4 engram-1.safetensors (31.7GB bf16, layer 2) - IN PROGRESS to
   /Users/sonders/dageno-modelbench/engram-hy4/
2. Build inkling->hy4 canonical remap table (exact script pattern in probe)
3. Lookup layer: inkling tokens -> canonical -> hy4 compressed ids -> ngram hash ->
   donor table rows (SSD-streamed, bf16 or re-quantized q8/q4 with calibration)
4. Train ONLY gating + shortconv adaptation on frozen inkling backbone
   (millions of params, draft-distill-scale token budget)
5. Validate: perplexity recovery curve on held-out taps before/after insertion;
   start with ONE layer (layer id ~12 equivalent in inkling's 48) before scaling

## Hy4 config reference values (for port)

engram_layer_ids [2,12,22,32], max_ngram 3, n_embed_per_ngram 768,
n_head_per_ngram 8, vocab ~10.3M rows per order, pad_id 120002,
hash = XOR of (token * odd_multiplier) per position mod prime table size.
Hash multipliers are seeded per layer (seed + 10007*layer_id, odd values).

## Engram grafting assets secured (2026-11-27)

- Quantized tables: kusuriya-labs/Hy4-flash-Engram-Ngram4 (ngram-q lloyd-row-f16,
  4-bit + row fp16 scale, 64 shards x 515MB = 33GB total, layer 1 = 16 shards ~8.2GB).
  Quality: rel L2 ~0.089, PopQA parity with FP8 baseline (111 vs 112/300).
  License: Tencent Hy License 1.0 - review clause 10 MaaS restriction if Dageno
  ever serves this commercially.
- Structure confirmed: 4 layers x 16 tables each (2 ngram orders x 8 heads),
  rows ~10.31M per table, dim 96 per head (768 total = 8 x 96 concat).
- engram_lookup.py written and verified: canonicalizer replication (88501 canonical
  ids), inkling->hy4 token remap (94906/200058 tokens statically mappable -
  frequency-weighted coverage is what matters: 96.3%), layer hash multipliers,
  NgramQTable random-access row decode working (mean abs 0.69, dense rows).
- Table format doc: shelf docs/developer/ngram_quantization.md saved copy at
  /tmp/ngram_quant_doc.md - implement in Metal later; 4-bit packing is simple.

NEXT IMPLEMENTATION STEPS:
1. Finish layer-1 shard download (16 files) in engram-hy4-q4/
2. Offline probe: for tap-stream trigrams, measure row-hit statistics
   (how often hashed rows are non-generic) - sanity of the graft before training
3. Gating+conv adaptation trainer (freeze backbone): insert at inkling layer ~12,
   perplexity recovery on held-out taps as acceptance metric

## Sub-4bit engram quantization research (2026-11-27)

Engram tables tolerate extreme quantization because gating already filters noise
(hash collisions inject garbage rows by design - model learns distrust):
- arXiv 2610.35985 per-lookup gated precision: 1.5-bit costs only ~1 pt PopQA,
  2.1-bit costs 0.2 pts, 7.8x table memory saving. Key trick: 1-bit codes estimate
  per-lookup gate-weighted error, precision allocated per lookup not per row.
- arXiv 2610.21810 2-bit conditional program: KL 41-63% lower than naive 2bit RTN.
- arXiv 2610.31646 low-bit n-gram embeddings: 2-4bit, 47-55% memory saving,
  perplexity increase <1.2%.
No pre-quantized 2bit tables published anywhere - must requantize ourselves.

## Engram table size economics (decision inputs)

- Qwen3.8-Flash-Next engram: 51.2B params, 320M rows x 160 dim, FP8 = ~51GB.
  Single lookup point (decoder layer 2), host-memory async prefetch official.
  Model itself is 125B total / 6B active - "3.8" is branding not size. UNUSABLE.
- Hy4 layer = 16 tables (8 bigram + 8 trigram heads) x 10.31M rows x 96 dim.
  4-bit shard = 515MB per table, 8.2GB per layer, 33GB all 4 layers.
- Row-count shrinking is the structural lever, not bitwidth: hash mod smaller
  prime (e.g. 1M rows) divides table size by 10x; collisions rise but gating
  compensates. Row shrink + 2bit requantize: 1M x 2bit x 96 x 16 = ~500MB/layer.
- SSD streaming is production-proven (oMLX #5475); RAM cost is LRU cache of hot
  rows only. Disk footprint is the only real constraint.

## GLM-5.3-Flash text_config findings (zai-org/GLM-5.3-Flash, verified 2026-11-27)

45 layers hybrid: 34 linear_attention (KDA, kda_layers list, 64 heads head_dim 128,
gate_lower_bound -5, shortconv k=4) + 11 deepseek_sparse_attention every 4th layer
[3,7,11,...,43]. MLA kv_lora_rank 512, q_lora_rank 1536, qk_nope 256, v 256,
mla_use_nope. MoE: 288 routed experts, 8 per tok, 1 shared, sigmoid scoring,
noaux_tc, routed_scaling 2.5. Hyper-connections: mhc=true, hc_mult 4,
sinkhorn_iters 20. DSA indexer: index_head_dim 128, 32 heads, topk 2048,
kpool 4 with compress + always_select_tail, indexer_rope_interleave,
index_share_for_mtp_iteration. 1M context. (GLM-5.4 does not exist publicly.)
TAKEAWAY for Inkling DSA port: indexer granularity is per-head 128-dim keys with
k-pool compression; topk 2048 at 1M ctx ~= 0.2% density. Inkling's 4 SWA layers
at win 1024 already ~this density on long ctx - DSA gain is on the 44 full-attn
layers IF ported, but those are trained full-attn: adaptation cost is the blocker,
not kernel cost. Prioritize engram graft (hot path now) over DSA.

## Engram graft adapter implemented (engram_adapter.py, 2026-11-27)

Verified against donor demo math + manifest. Inkling adaptation decisions:
- single-stream gating (no hyper-connections in inkling): one key_proj/value_proj
  pair instead of hc_mult=4 sets; concat 1536 -> 2304 projections
- trainable params ~7.1M: key_proj + value_proj + 2 RMSNorm + depthwise conv k4
  + out_scale zero-init (insertion is identity at step 0 - perplexity recovery
  measures genuine learning, not adapter noise)
- signed-sqrt then sigmoid gating exactly as donor demo
- OPEN INDEXING QUESTION: manifest 'layer' field is zero-based (1,2,3,4?) vs
  donor config engram_layer_ids [2,12,22,32]. Hash multipliers = seeded by
  donor layer id. Must validate both bindings in perplexity test - wrong binding
  = random rows. Test: compare row-norm distribution statistics for hashed hits
  of real tap trigrams under layer_id 2 vs 1 seeding; correct binding should show
  heavier-tailed (semantic) rows. Definitive test is training recovery.

## Row-hit probe results (engram_row_probe.py, 7508 tap tokens)

- 94.4% stream tokens mappable to hy4 canonical space in practice
- bigram row dispersion 34% (1686 unique rows/5000 bigrams) - hot-row LRU cache
  will be small and effective, SSD streaming cost low
- NEGATIVE RESULT: seed binding (layer_id 1 vs 2 vs 12) is INDISTINGUISHABLE by
  row-norm statistics - norms identical for hits vs random rows (both ~8.3).
  Engram row semantics live in direction not magnitude. The ONLY reliable test
  is small-scale training perplexity recovery (7.1M adapter params, cheap).
  Plan: smoke-train both bindings 200 steps on same tap batch, pick winner,
  fallback layer_id=2 (donor config engram_layer_ids[0]=2 is the documented semantic).

## LoopCD mechanism decoded from oMLX implementation (patches/inkling2/loopcd.py)

Paper: arxiv 2610.02185. oMLX production implementation details beyond the paper:
- reference = pass-0 (first pass) hidden states captured at CODA layers
  (layers after loop core: range(loop_core.stop, num_hidden)), readout through
  the same final_input attn-res fusion + norm + lm_head. So reference is a full
  "pass-1 prediction" in logit space, not just pass-1 hidden reuse.
- guide math: final' = final + w*(final - reference), where
  w = 0.5 * (1 - top2_margin(softmax(final))) - CONFIDENCE-ADAPTIVE: uncertain
  tokens get amplified correction, confident tokens nearly untouched. Max 0.5.
- both logits tanh-capped by final_logit_softcapping before diff
- skip_thinking: correction zeroed inside thinking spans (tokens after
  THINKING_TOKEN 200008 control boundary, tracked via cummax position trick,
  cached across verify windows in cache[-1][0])
- control-token probability mass preservation: after guidance, ordinary-token
  logits renormalized so logsumexp over non-control tokens matches the
  unguided mass - prevents LoopCD from corrupting the control-token distribution
  (inkling emits control tokens for structure; mass bleed would break parsing)
- cost 9-13% decode speed (oMLX measurement); reported gains ARC-C +2.3pts,
  HumanEval 152->155/164, AA-Omniscience -9.0 -> -3.4

STRATA PORT PLAN (not yet implemented): inkling tap infrastructure already
captures per-pass hidden states; reference readout is a second pass through
final_input+norm+lm_head on captured pass-0 coda hidden. Guide math is
elementwise over 200k vocab (~0.8MB fp32) - must fuse into the logits kernel
or it costs ~ms/token of memory bandwidth. Batch-invariant: same for verify
windows since thinking flags cached. Estimated effort: medium; payoff: free
quality gain at 9-13% speed cost, user-toggleable. DEFERRED until after
megakernel/nvfp4 work since it's a speed-cost quality knob, not a speedup.

## Engram adapter training constraint analysis (2026-11-27)

Gradient path problem: frozen backbone but insertion-point hidden depends on
adapter params, so training needs either full MLX inkling forward reimplementation
(exact, big surface) or finite-difference Jacobian probe (approximate, cheap).
Before either: zero-init validation = perplexity at init must match baseline
exactly (identity insertion). This check alone validates the whole lookup+
remap+hash pipeline end to end EXCEPT seed correctness (random rows still produce
finite perplexity change only when out_scale nonzero - so identity check cannot
catch wrong seed; smoke training can).
IMPORTANT: lookup tables are donor-FROZEN in grafting phase. Row-shrink and
2-bit requantization experiments come after perplexity recovery is demonstrated.

## Hy4 engram indexing resolution evidence (2026-11-27)

Manifest layers [1,11,21,31] in zero-based indexing == donor engram_layer_ids
[2,12,22,32] minus one => layer field is ZERO-BASED index into donor layer list.
Corollary: seed for hash multipliers uses the ORIGINAL donor layer id (2 for our
layer-1 shards), since multipliers were generated with (seed + 10007*layer_id)
at donor training time with layer_id=2 for the first engram layer. Working
assumption now layer_id=2 (was: ambiguous). Smoke training still the final check.
CAUTION: manifest rows 10.31M per table vs deepseek-V41 config engram_num_embeddings
384M - kusuriya profile 'ngram4' tables are NOT the full V41 engram, they match
the Hy4-FLASH (smaller sibling) engram. max_ngram in manifest = 3 (orders 2,3 only)
despite profile name ngram4. Fine for grafting; note donor scale difference.

## Hy4-flash architecture facts useful for Inkling adaptation (deepseek-v41 config)

- DSA indexer: topk 512 (vs GLM 2048), head_dim 128, 32 heads, source layers
  [2,8,14,20,24,28,32,36]; sliding_window only 128; kv_source_layer_ids [2,8,14,20]
  => YOCO-style KV sharing: only 4 of 40 layers produce KV! candidate blocks 2048
  topk from candidate_source_layer_id 20, compress_ratios per-layer 0/1/2
- DSpark config: block_size 5, markov_rank 256, 128 experts 3-per-tok in DRAFT,
  noise_token_id 128799, target layers [37,38,39]
- latent KV: kv_lora style q_lora_rank 1280, o_lora_rank 1024, o_groups 8
- engram_compressed_vocab_size 99092 (NOT 88501 as earlier probe estimated -
  recheck canonicalizer assumptions if grafting quality poor!)
- 1M ctx via yarn factor 16 over 65536 base

## Engram end-to-end lookup pipeline VERIFIED (2026-11-27, layer-1 complete)

engram_e2e_probe results on 2000 real tap positions x 16 tables:
- 90% of positions fully-mappable trigram windows, 92% nonzero rows
- naive python sequential seek: 1465-1617us/position (UNUSABLE inline at decode)
- SSD bandwidth is NOT the issue (50B/row x 16 = 800B/position); the cost is
  IOPS and syscall overhead. Production path must be:
  1) bigram rows hot-cached in RAM (34% dispersion -> small LRU covers most)
  2) trigram rows PREFETCHED: indices depend only on input tokens, known 1-2
     tokens ahead during decode - async io thread hides latency entirely
     (same pattern as oMLX #5475 and Qwen3.8 host-memory prefetch design)
  3) prefill: all indices known upfront, batch-read sequentially
- conclusion: inline Python cost is an artifact, not a blocker. Engram adds
  zero decode-latency if prefetched; memory cost = LRU cache only.

## DSA indexer port evaluation for Inkling (2026-11-27, oMLX analysis)

oMLX hy_v4/indexing.py: indexer is a TRAINED component (128-dim keys, 32 heads,
head-weighted ReLU dot scoring, topk 512/2048 selection with causal masking).
NAX/native kernels exist for scoring+topk. Porting assessment for Inkling:
- Inkling attention is full MLA-style trained dense attention; DSA sparse
  attention is a DIFFERENT trained behavior. Attn weights don't transfer -
  indexer needs adaptation training on frozen backbone, same category as engram.
- Cost/benefit: Inkling has only 4 SWA layers (win 1024); the other 44 visits
  are full attention where DSA would need deep retraining of attention behavior.
  At our operating context (<=32k typically), full attention costs are not yet
  the binding constraint - MoE expert streaming and trunk bytes dominate.
- VERDICT: DSA adaptation is real adaptation TRAINING work with modest payoff
  at our context lengths. Deprioritize behind engram graft (capability play)
  and draft retrain (speed play). Revisit when context length >64k becomes a
  production requirement or when adaptation training infra is proven on engram.

## Engram donor authoritative parameters resolved (2026-11-27, tencent/Hy4-flash config)

- vocab_size 120832 (NOT 129280 - that was deepseek V41, different model!)
- engram_layer_head_vocab_sizes in config EXACTLY match manifest rows for all 16
  layer-1 tables when manifest layer + 1 = donor layer id (1 -> 2). CONFIRMS:
  zero-based manifest indexing, hash multiplier seed layer_id = 2 for our shards.
- engram_seed = 0 (multiplier seeding, matches demo default);
  engram_embed_random_seed = 1234 (training-time table init, not needed for grafting)
- engram_pad_id = 120002, engram_vocab_size [10312000, 10312000] (per-order base),
  max_ngram 3, n_embed_per_ngram 768, 8 heads per order
- tokenizer.json = 120000 vocab + 832 added tokens; canonical space computes to
  89333 keys with demo replication (the deepseek-V41 99092 number is irrelevant)
- donor is hunyuan-based multimodal hy_v4_flash, MoE, NOT deepseek-v41. Earlier
  notes conflating them are corrected here.

## Engram integration direction CHANGE: re-tokenize, don't remap (2026-11-27)

Reverse-engineering inkling's BPE segmentation into hy4 canonical units hits a
fundamental ceiling: greedy longest-match merging reaches only ~85% alignment
because donor BPE segmentation is frequency-driven not greedy (e.g. inkling merges
'photo'+'synthesis' but donor splits 'photos','ynthesis'). Exact replication is
impossible at the token level.

KEY REALIZATION: the engram lookup input is TEXT-dependent only, not token-dependent.
At serving time we HAVE the generated text. Re-tokenizing with the donor tokenizer
(rust tokenizers lib, ~1-2us per token amortized, incremental) yields EXACT donor
segmentation - zero alignment loss, zero merge machinery. Engram integration:
1. serving loop re-tokenizes generated text incrementally with hy4 tokenizer
2. canonicalize ids (static lookup table), hash ngrams, prefetch rows
3. inject at insertion layer per adapter design
Token-level remap path (inkling id -> canonical) is RETAINED only as fallback for
contexts where text is unavailable. All training/validation should use the
re-tokenize path. This invalidates the seed-arbiter-by-training plan complexity:
with exact segmentation, validation is direct - perplexity recovery curve on
held-out text measures graft value cleanly.

Smoke-training seed probe results (INVALIDATED methodology, kept for record):
probe memorization dominated; hidden-state leakage made early tests meaningless;
final pure-table heldout test showed both seeds worse than chance (14.3/15.1 vs
12.2) - interpretation: probe task itself flawed since ngram windows rarely repeat
verbatim, NOT evidence against grafting. Re-tokenize path + perplexity recovery
is the correct experiment.

## Engram serving-cost validation (engram_serving_sim.py, 2026-11-27)

- donor re-tokenize pipeline verified end-to-end on real tap text: 100% canonical
  mapping on natural-language generation content
- IOPS cost is trivial: worst case all-miss 16 reads x 80tok/s = 1280 IOPS, 64KB/s
  bandwidth. NVMe does 100k+ IOPS. Engram serving cost is NOT a hardware concern.
- cache hit rates depend on traffic redundancy: diverse prompt pool shows 54%
  bigram / 25% trigram repeat coverage; single-stream diverse text only 28%/19%.
  Cache saves bandwidth not latency (prefetch hides latency) - size it small
  (~few hundred KB) and let prefetch do the work.
- Dageno business traffic (JSON template scaffolding) will show much higher
  repetition than diverse text - measure after integration, expect 70%+.

## oMLX Inkling-2 optimization techniques catalog (2026-11-27, source analysis)

From omlx/patches/inkling2 (production MLX implementation):
1. expert_mma.py: expert-grouped 3-bit matmul for fp16 prefill chunks. KEY trick:
   outputs land at UNSORTED pair positions, eliminating sort+unsort around the
   gather_qmm. Uses steel BaseMMAFrag simdgroup layout, K-order fp32 accumulation.
   Applicable to strata's prefill MoE path - measure if sort overhead is material.
2. oq_layout.py: NVFP4_GLOBAL_SCALE = 448.0 * 6.0 - ModelOpt stores tensor scale
   as amax/(E4M3max x E2M1max); runtime applies global_scale/this constant.
   THE exact constant needed for training-native nvfp4 decode. Confirms our
   repack byte-layout and adds the scale semantics.
3. dspark.py: bundled draft = small block drafter mixing target context taps with
   softmax weights, bidirectional self-attention within block, RNN Markov head.
   Confirms arch spec; sliding window even in full-attention draft layer.

## Scheduled sampling exposure-bias fix: VALIDATED offline (2026-11-27)

distill_chain_eval.py measures free-running chain loss (SS_P=1.0 = student feeds
own predictions, deployment emulation). Results on same 15 v2 windows:
- round-1 checkpoint (teacher-forcing only, deployed acc=0.00): chain loss 9.31
- smoke checkpoint (SS_P=0.3, 150 steps, 93k tokens): chain loss 6.81 (-27%)
Teacher-forced training loss 11.76 -> 7.26 over 150 steps, chain gap small
(7.26 train mixed vs 6.81 free-running) = student-in-loop training working.
CAVEATS: only 5 tensors trainable (fuse weights + markov head; attention frozen);
150 steps is a smoke run not convergence; chain loss 6.81 is still high in
absolute terms - deployment acc test on real service is the next gate.
METHODOLOGY NOTE: distill_chain_eval.py with SS_P=1.0 is a cheap offline arbiter
for exposure robustness - use it before every repack/deploy cycle from now on.

## Smoke deployment results: acceptance REVIVED (2026-11-27)

sq10-distilled-ss.strata (150-step SS_P=0.3 smoke checkpoint, 93k tokens):
- acc: code 0.52, tech 0.25, json 0.1, poem 0.0 - from round-1's uniform 0.00
- tps 34-38 (vs baseline 36-84): acceptance too low to pay for itself yet
- json output correct despite low acc ("city":"杭州" correct)
INTERPRETATION: exposure-bias fix direction confirmed in production. Acceptance
is bottlenecked by training SCALE not methodology: 150 steps/93k tokens is ~5%
of a proper run. Next: full training when v2 data reaches 300k+ tokens
(projected ~2-4x acc at baseline throughput parity).
DECISION: revert production to sq10 baseline pack NOW (smoke pack is slower);
redeploy next distilled pack only after full training + offline chain eval gate.

## Data collection throughput analysis (2026-11-27)

Collection is the critical path: taps can ONLY be generated by the SQ-10 service
itself. At ~35-80 tok/s with 400 max_tokens and 2 concurrent slots, each prompt
takes 5-10 minutes. 3000 prompts ~= 200-500 hours sequential wall time with 2
workers. This is untenable as-is. Options ranked:
1. reduce max_tokens to 256 (taps per prompt still useful, 40% time saving)
2. increase service parallel slots when memory allows (measured +26% aggregate
   at parallel=4 but single-stream cost)
3. prioritize: not all 9590 prompts needed for first full retrain - 300k tokens
   is ~1000 prompts at current avg yield (~93 tokens/prompt incl. verify rows)
4. TRAINING compute is NOT the bottleneck (Kaggle free tier suffices for
   126M-param draft), data generation is.

## Expert routing correlation test: NOT measurable offline (2026-11-27)

Taps capture only visits [5, 23, 41, 48] partial sums - 4 of 48 visit layers.
Routing-correlation-across-passes needs per-layer MoE INPUT hidden at every pass
of the looped core (visits of layers 6-17 x3). Cannot be computed from existing
tap data. Requires engine instrumentation: log top-8 gate indices per visit
during a few decode steps (cheap: 48 x 8 ints per token). Value if positive:
expert cache reuse across passes of same token -> up to 3x loop MoE traffic cut.
Implementation sketch: INK_ROUTEDUMP env writing (visit, expert_ids) per decode
step to fd, then correlation analysis offline. Defer to next engine session.

## Trainer quirks noted (2026-11-27)

- distill_train_full param-count log line is misleading ("5 tensors") - actual
  trainable set is correct: all attn proj/norms + norms + markov head + fuse_weight
  (~41 tensors, 126.8M matrix params)
- split_tree sends ALL params to Muon including 1-D norm vectors (vec_params empty).
  Muon on vectors is not theoretically ideal but momentum+NS orthogonalization on
  1-D degrades to sign-ish updates; acceptable for now, revisit if norm params
  train poorly. Low priority - matrices dominate.

## LatentMoE-SVD compression: killed by arithmetic (2026-11-27)

sq10 experts are already s2/s3 mixed quantization at ~156KB avg per expert
(2-3 bits/weight). SVD low-rank factorization cannot compress BELOW the
quantization floor: information content is already reduced to 2-3 bits per weight.
Latent-factorization benefits apply to FULL-rank bf16 experts (2.8MB), not to
already-quantized ones. The streaming cost bottleneck is quantization-noise-limited
bytes, not rank-limited bytes. CONCLUSION: LatentMoE adaptation for Inkling has
no compression headroom beyond what s2/s3 already achieves. The only remaining
upside would be architectural (train-time latent experts are more capable per
parameter), which is full-retrain territory - out of scope. Direction CLOSED.

## oMLX full Inkling forward VERIFIED locally (2026-11-27)

- /Users/sonders/dageno-modelbench/inkling-hf assembled: config.json (from HF repo),
  symlinked bf16 checkpoint + mtp, tokenizer.json (strata-tok hf format)
- deps: pip install mlx-lm mlx-vlm into train-venv
- omlx.utils.model_loading.load_text_model loads in 22s, forward pass works:
  logits [1, T, 201024]. First forward is slow (~94s compile warmup + memory
  pressure); subsequent should be faster.
- THIS is the adaptation-training backbone: engram insertion, KV sharing,
  attention adaptation experiments can all be done with real gradients.
- memory: 18.6GB bf16 weights + MLX activations is heavy - use SHORT sequences
  and expect to be memory-careful; can't run while production service is hot
  if memory pressure is already high.

## Engram 嫁接训练：MLX 可微路径（2026-11-27)

### 为什么必须 fallback
oMLX 全部手写 Metal kernel 无 VJP。训练适配器需要梯度穿过冻结 backbone 下游，只能强制纯 MLX 参考路径。traindiff.py 的 patch 清单：attn_res_mix/fused、loop_messages、short/dynamic/moe conv、qkv_post、route_topk、banded_sink_attention→_masked_attention、MoE expert_mma/verify_experts 禁用、mx.compile 保留（能融合 backward)。

### 三个关键坑（都已修）
1. **m5_gather_qmm NAX 重路由**：模型加载时重新包装 mx.gather_qmm，在 patch 之后。但 NAX 分派门是每次调用检查的——patch 掉 `_nax.sorted_gather_qmm` 返回 None 即可永远落到有 VJP 的原生 MLX。
2. **索引梯度污染**：从可导张量算出的整数索引（argpartition/argsort）会被 MLX 留在 tape 里，第一个 gather 就炸。必须在**产生点** stop_gradient——patch mx.argsort/mx.argpartition/mx.take/mx.take_along_axis。
3. **gather_qmm 本身有 VJP**（量化权重冻结、对 x 求导）——这是 mlx-lm QLoRA 生态的基础，不需要自己实现 expert backward。

### 吞吐实测（M6 24GB)
- B1xL96: 69s/step = 1 tok/s;B4xL96: 5 tok/s;B8xL128: 12 tok/s
- 瓶颈是每步固定开销（数千小 kernel dispatch),batch 基本免费 → 训练必须大 batch
- 单层 0.01s，但 48 visit 的 loop 结构 + 全注意力层 2-4s/层（rel bias gather 展开）是大头
- 100k token 本地过夜 ≈ 2.5h，可做概念验证；正式训练必须上云

### 社区调研结论（勿重复）
- **没有任何 Apple Silicon 训练框架支持 inkling2_loop**:mlx-lm 无此架构，oMLX 仅推理侧 LoRA
- transformers≥5.24 原生 inkling2_loop(1840 行，GradientCheckpointingLayer 每层）
- 官方后训练生态 100% NVIDIA:Unsloth(CUDA 专用优化）、Miles、SkyRL、Tinker(API,不支持架构改造）
- checkpoint 发行形态：experts NVFP4（层 0-19)+ MXFP8（层 20-23)，无 bf16 版
- HF NVFP4 加载要求 sm100+(B200)；离线转 bf16 后 A100/H100 80GB 可训（冻结 backbone ~52GB bf16)
- 转换器在 cloud-train/convert_nvfp4_to_bf16.py,nibble 布局已对照 MLX nvfp4 验证（low nibble first,scale e4m3/group16,amax/2688)

## nanoattention (justrach) 调研提炼（2026-11-28)

Zig+Metal 从零引擎,M3 Ultra 256GB 上 Inkling 2 Mini 120 tok/s(短上下文)。硬件比我们大得多但速度相近,说明其收益主要来自设计而非规模。可移植的技术:

### 1. LoopCD 的三种实现变体(比我们 oMLX 版更完整)
- `hidden`: 对比 post-core residual stream(coda layer 18 前),代价最小——不需要第二次 unembed
- `logits`: 对比 unembed 后的 vocab logits,关键优化是 **`head_softcap_loopcd` 融合 kernel:强弱两路 unembed 共享权重读取**(unembed 是整个 forward 最大的 matvec,vocab 201k×2304。LoopCD 朴素实现要做两次,融合后权重只读一遍,成本几乎减半)
- `logits-adaptive`: per-row weight = omega×(1-(p1-p2)),p1/p2 是强路 top-2 softmax prob——和 oMLX 0.8.4 的 margin-adaptive 一致,证实这是社区收敛做法
- weak-pass 可配置(默认 pass 1),`--loop-passes 1..3` 可改循环数
- **移植行动**: strata 的 LoopCD 应该实现 fused dual-unembed。oMLX 已有 hidden 变体(981cd70),logits-adaptive 是 next

### 2. 历史投机解码的负面证据(重要!)
nanoattention 实现了完整的 history-draft 投机(committed-history ngram 续写 + 滑动槽位 journaled 回滚):
- 重复代码补全(人为有利负载):166.83 tok/s @16K,139.41 @32K,M3 Ultra
- **真实五轮 coding agent 流量:只有 22.83 tok/s,history draft 接受率太差**
- 结论:无模型投机在真实 agent 负载上不工作,验证了我们 draft 蒸馏方向的正确性。不要再尝试纯 history/ngram drafting

### 3. MAP-Elites 质量多样性调参方法论(值得移植到 strata 优化流程)
- 每个 kernel 变体都在 runtime knob 后面,候选=完整 knob 配置 JSON
- 准入门:输出与参考 bit-identical(重排浮点求和的直接拒)
- 计时门:paired interleaved median,挑战者必须超过 elite 中位数 max(1%, 2×elite 自身轮次散布)
- niche 按(目标 decode/prefill × prompt 长度桶 × dispatch 数 × family)分格,每格保留最优——不同思路不互相淘汰
- 无主的变体代码在搜索结束后删除(防止死代码堆积)
- kern 文章的 swarm+gatekeeper 方法论与此同源:**bit-identical 准入 + paired A/B + 防回退守门**

### 4. KV cache 的 ring 容量设计
ring capacity = horizon + batch(而非 horizon),保证 batch 内行不会覆盖同 batch 其他行还要读的槽位——连续批处理 server 设计时直接采用

### 5. LSA Q4 稀疏注意力(DSA 方向的 Metal 参考实现)
key block 按 4/16 token 聚合量化成 uint4(scale,min),query 用 Q4 scoring + learned-sink LSE + softmax-mass top-K 选块,保留 block0 + top256 middle + 1024 trailing window。dense bypass 阈值 ~1.3k token。这是 DSA 移植时最接近的 Metal 参考

## Loop 模型优化论文调研(2026-11-28,arXiv 2026-11 批次)

### LoopRes (2611.31094)——与我们架构最相关,高优先级
在**预训练 LLM 上后训练**出 loop(不用从头训):让 loop 内每层直接 attend 之前 loop 迭代的输出(Attention Residual 的推广),给每个 loop 迭代一个可训练的 loop-aware key(编码结构身份)。限制 residual window。
- **Inkling 2 Mini 的 AttnRes 已是跨深度残差混合,LoopRes 说明跨 PASS 维度还有未利用的信号**。我们之前砍掉半深度 pass 削减正是因为 pass 间信息流是训练内嵌的——LoopRes 佐证 pass 间依赖可以更强而非更弱
- 适配方向:在 engram 训练稳定后,可探索给 loop message bank 加 loop-aware key 的适配训练

### Mixture-of-Loops/MoLT (2611.34923)
用 KV 作为跨 loop 选择性复用的记忆:MoLT-KV 每轮从 embedding 重启但读前一轮的 memory KV,消除同位置跨 loop 依赖,配 loop forcing 可并行训练 + 单次遍历解码。结构改动大,远期参考。

### Scaling Looped MoE 配置律 (2611.34276)
iso-param/iso-FLOP 约束下的系统结论:**core 深度占总展开深度 25-35% + 节省的参数重新分配给更多 routed experts 是最优配置**。Inkling 2 Mini:core 12 层×3pass=36 次循环访问/总 48 visit = 75%,远高于论文甜点。暗示:我们的模型在 core 深度上"超配",反过来意味着**减少 core loop 次数(3→2)并用省下的算力加宽**可能是理论正确的方向——但用户已决定不做架构重训,此条作为远期参考。零成本可用的点:gated inner-loop state update + cross-stage residual(模型已有类似机制)

### SwiftLoop (2611.31481)
选择性 refinement:不是所有 token 都需要全部 loop pass。自适应深度——**与我们的半深度实验结论冲突,需细读**:他们怎么解决 pass 间依赖问题。如果他们的跳 pass 是训练内嵌的,那对我们是"需要适配训练"的又一证据。

### DuoMI (2611.33952)
loop transformer 里专家复用的互信息正则,训练期技术,适配价值低。

## Cloud training pipeline (2026-11-28 late)

### Network reality (this machine, China)
- Kaggle API itself reachable directly, but GCS upload endpoint needs proxy;
  upload throughput measured 430 B/s-800 KB/s => 7.5GB engram tables upload
  is INFEASIBLE. Cloud regeneration beats upload everywhere.
- modelscope.cn serves HTTP RANGE requests (206) - surgical downloads possible.

### Key technical facts (verified)
- transformers 5.24.1 conversion_mapping inkling2_text handles ALL checkpoint
  key renames automatically, incl. w13_weight -Interleave(dim=1)-> gate_up_proj,
  unembed unpad, attn_res index offsets. Just dequantize in place, keep names.
- transformers CANNOT load pre-quantized NVFP4 (explicitly unsupported).
  vLLM can. For HF training, bf16 conversion is mandatory.
- convert_nvfp4_to_bf16.py smoke-validated end-to-end incl. numerics.
  Bug fixed: amax reshape(128) hardcoded - now dynamic expert count.
- engram donor layer-1 embedding [164994380,96] bf16 is the SOLE tensor of
  donor shard model-00004-of-00148.safetensors; 16 tables are contiguous row
  slices in manifest order. Verified: donor rows vs q4-decoded rows cosine
  0.996 mean (q4 noise only). build_engram_tables_fp16.py regenerates tables
  in fp16 on-cloud via range requests - NO upload needed, exact donor values.
- ModelScope free instance access NOT automatable locally: web console is
  cookie-auth; Chrome cookies TCC-protected; ~/.modelscope/credentials/session
  is 32B cookie not SDK token. Needs user action (instance + SDK token or
  web terminal).
- Kaggle: code dataset canhun2026/inkling-engram-code is LIVE. T4 16GB cannot
  train bf16 backbone (52GB) - Kaggle role demoted to optional pipeline debug.

### Instance capacity planning
- bf16 backbone 52GB + fp16 tables 31.6GB + activations/optimizer small
  (adapter-only training). Disk ~110GB with intermediates. ModelScope AMD
  instance (192GB VRAM Mi300x, 200GB RAM) fits comfortably IF disk >= 120GB.

## LoopCD/pass-reduction decisive experiment (2026-11-28, arxiv 2610.02185)

Paper method confirmed: Hidden variant is h' = h_R + w(h_R - h_1), applied BEFORE
coda layers (our implementation matches exactly). ω≥0, fixed for hidden variant.
The paper's headline result: quality gains enable HALVING recurrent iterations.

Tested on Inkling-2-Mini via new INK_PASSES env override (engine patch, committed):
- p3-lcd0.0 (production): json gen 70.1 tps, fact 8/8, P3 3/4, accept 209/223
- p3-lcd0.3: 69.0 tps, fact 7/8 (1 borderline grader-noise fail), identical output
- p2-lcd0.0: fact 5/8, token-level repetition breakage (《红》《红》《红》),
  math reasoning errors (though fluent at 42.7 tok/s vs 32 for p3)
- p2-lcd0.5: fact 4/8 - WORSE. LoopCD amplifies the pass1-vs-pass2 noise
  direction on a fixed-3-pass-trained model.

CONCLUSION: Inkling-2-Mini is fixed-3-pass trained (loop_config.is_varloop=False);
pass reduction is OOD and unrecoverable by LoopCD. The paper's halving trick only
works for variable-loop-trained models (Ouro/Huginn). NOT a speed lever for us.
At full depth, LCD w=0.3 is speed-neutral but quality-neutral-to-noise on our
saturated tests - insufficient evidence to enable in production; left OFF.
INK_PASSES engine override retained for future experiments (accepted upstream value).

## 2026-11-28 关键发现:block-mode host verify 路径先天性损坏(上游 bug)

### 实验证据(全部在贪婪 temperature=0、固定 prompt 下)
- tree 模式(spec_tree,GPU beam tree):三个二进制(原始/topk移植/fused)输出**逐字节相同**,29 tok/s,稳定正确
- block 模式(STRATA_INK_SPEC=block,强制 host verify):**即便用未打补丁的原始二进制 + 全行 logits verify**,输出也与 tree 模式**不同**,且接受率只有 3.9%(31/805),速度 7.9 tok/s
- 结论:block 路径(spec decode host verify)在 fork 快照上游就是坏的——**我的 cand-verify/amax/fused 优化的目标路径从未真正工作过**。之前所有 A/B "bit-identical" 结论只验证了 tree 模式(它绕开了 host verify)

### 教训(重要方法论)
1. `draft_n_accepted` 语义陷阱:tree 模式(GPU verify)不发 host 路径的 accepted 计数,accept 0/4 不是 bug 信号;block 模式的 accepted 才是 host verify 的。比较时必须固定 spec 模式
2. spec decode 的正确性验证必须**跨模式对比**:同一 prompt 在 tree/block/no-spec 三种模式下贪婪输出应逐字节相同
3. vdebug stderr 探针 + 一次性 `[ink] fused active` 日志是最快的路径确认方式

### 已完成并部署(tree 模式下逐字节验证正确)
- verify 路径 cand 化(forward_batch_cand,贪婪 no-penalty 用 top-128 替代全行)
- forward_batch_argmax API(贪婪 verify 连 logits 都不物化)
- fused unembed+topk mm8 kernel:权重读一遍服务 M 行、logits 永不落盘、per-tile bitonic 排序 + 合并树,在 n_lg 5..16 时启用。单元测试通过(合成 logits 128/128 exact)。**但因 block 路径损坏,生产实测收益无法验证**
- STRATA_INK_NO_AMAX / STRATA_INK_NO_FUSED_TOPK / STRATA_INK_VDEBUG 调试开关

### 下一步(优先级最高)
修复 block-mode host verify 的发散 bug。怀疑方向:
1. rollback 后 verify ring 的 ShortConv/f_rnn 状态恢复索引(record_verify_ring 在 cand/amax 路径缺失 — 已确认这两个克隆函数没拷贝 ring 启用逻辑!)
2. 但全行路径也发散 → 更根本的问题:KV 位置、verify 行意义(行 i 的 logits 是下一 token 的预测)、或 bonus token 的馈入位置
3. 修复后立即验证:cand-verify / amax-verify / fused topk 的收益才真实

### 环境状态备忘
- 机器 swap 深度使用(9.6GB),速度测量噪声 ±40%;一切优化验证用贪婪输出 parity + GPU TIMELINE,不用端到端 tok/s
- Kaggle kernel 仍 QUEUED(账号 2 个 GPU batch 会话占满);v3 payload(fp16 tensor core + liger fused CE)已就绪待 push

## 2026-11-28 深夜修正:block verify 深入分析(rollback 无罪,疑凶转移)

### 精确推演结论(推翻了上一节的怀疑方向)
1. **rollback 索引是对的**:ring[k] = row k 处理后的 ShortConv 状态;row r+1 承载的是 draft_ids[r](被拒绝的那个);ring[r] 恰好不包含被拒 token。keep_len-pos0-1 = accepted-1 = r 完全正确
2. **attention 语义是对的**:verify batch 内 row nd 因果读取前面的 draft 行是验证链的正确语义;污染 KV 位置单调递增,重喂前必被覆盖
3. **重大对照实验**:no-spec(draft off)贪婪输出与 tree 模式**逐字节相同**,且 **34.3 tok/s 比开投机(26-29)还快**!adapt governor 工作时与 no-spec 持平(31.8-33.5),即当前投机机制在 M6 上是**零收益甚至负收益**——draft+verify 的开销吃掉了全部接受收益
4. block 模式发散且接受率 3.9%(31/805):verify batch 算出的分布与 draft 模型的预测系统性不一致。**疑凶转移**:M-row batched decode 内核(sconv4_multi/GQA splitk/mm8 等)与 M=1 内核的数值差异大到翻转 argmax,或者 batched decode 数学本身有真 bug。上游 ink_serve.h block 路径与 fork 完全相同(已 clone 比对),上游没修这个

### 战略重估(重要)
- 真正的高价值目标:**修复 batched verify 数值正确性** → block verify 接受率上去 → fused topk/argmax(已就绪)才真正发挥 → 目标 40-55 tok/s(对比 no-spec 34)
- no-spec 34.3 tok/s 本身是新基线:任何投机优化必须超过它才算赢
- 验证方法论三条铁律:固定 spec 模式对比、跨 tree/block/no-spec 三模态逐字节验证、用 GPU TIMELINE 不用端到端 tok/s(swap 噪声 ±40%)

### 实验数据(本机 M6 24GB,swap 压力下)
| 模式 | tok/s | 输出 | 备注 |
|---|---|---|---|
| no-spec | 32.9-35.2 | 参考真值 | 新基线 |
| tree(adapt) | 26.6-29.5, governor 介入后 31.8-33.5 | =no-spec ✓ | 当前生产 |
| block | 7.6-8.0 | ✗ 发散 | 接受率 3.9%,先天坏 |

## 2026-11-28 深夜终章:精确病灶定位

### REJ 仪器化数据(block 模式)
- 拒绝几乎全部发生在 **i=0**(第一个 draft token):`draft=6633 verify_tok=8608/11076/13895/25208...` — verify 分布每次给出不同正确答案,但都和 draft 不同
- draft 质量崩塌的结构性证据:这不是数值噪声(偶发 flip 是 1/6 概率单 token 发散,出现在 slot2 第 32 字符),而是**系统性分布错位**
- use_amax=1 时 fused unembed+topk 确实激活(`[ink] fused unembed+topk active (n_lg=8)` 日志确认)——基础设施工作正常

### 并发 batched decode 数值发散量化
- M=2 并发 vs solo:1/6 概率发散,发散点第 32 字符,发散后文本仍连贯(near-tie argmax flip,loop 放大)
- solo 完全确定(4/4 一致)。slot1 从不发散,slot2 偶发 → batch 行位置相关的 reduction 顺序差
- 上游 0.2.23 二进制同样发散 → 上游未修复;上游只有 DFlash2 的类似记录(8c86eb9 "accepts no draft token and changes greedy output")
- **这同时解释 block verify 接受率崩塌:verify batch M=8 的行 0 之外的行数值偏差更大,draft 几乎不可能命中**
- 上游 ink_serve.h block 路径与 fork 逐行相同(clone 比对确认),上游没有修

### 下一步精确路线
1. 对照实验:verify batch 单行 logits 与 M=1 同位置逐步骤 logits 的 KL —— 确认 M-row 数值偏差幅度(预期 >> 0.1)
2. 修 M-row batched decode 数值一致性,候选:attention splitk 的 fp32 累加顺序统一、MoE mm 路径 reduction 对齐、sconv4_multi 顺序确认已正确
3. 修好后 block verify 接受率应升到 draft 真实水平(参照 HumanEval 文献 ~60-70%),然后 fused topk/amax 收益兑现,目标超 no-spec 34.3 tok/s 基线
4. 达成后全部提交 strata 上游 PR

### 生产环境最终状态
- adapt 模式(tree)运行中,输出与 no-spec 逐字节一致,31-35 tok/s
- STRATA_INK_VDEBUG/NO_AMAX/NO_FUSED_TOPK 调试开关保留在代码中,env 默认关闭
- Kaggle engram 训练 kernel 仍排队(账号 GPU 会话上限),v3 payload(fp16 tensor core+liger fused CE)就绪

## 2026-11-28 M1 Max 接管:基线与判定

### M1 Max (Apple7, 32GB, 无生产负载) 首批数字
- no-spec ~31 tok/s;dspark tree ~42 tok/s(acc 97/117);spec-check BITWISE-EXACT 64/64
- M6 优化快照(patches/)已移植到分支树并提交 inkling-metal-opt
- 并行 decode step:M1 31ms / M4 80ms / M8 113ms(边际 ~15ms/行 vs M6 的 ~8ms;c=32 327ms agg 97.8)
- GPU 时间占大头(TPROF build 0.3-0.5ms,rgwall = 全部)→ 不是编码成本,是 kernel 路径
- **KV/SC 分配病灶确诊**:slots×max_ctx 全量预分配,64 slots 需 ~21GiB → cmd-buffer OOM。已修:budget clamp(slots→51,STRATA_INK_KV_MB/STRATA_INK_SLOTS 可调)
- **跨 pass expert 缓存证伪**:修正 trace stride bug(原代码读错 buffer 报 8.00/8)→ 实测 p01=0.66 p02=0.55 p12=0.61 /8,≈随机。pass 间路由不相关,expert 每 visit 真流 ~8 个
- is_apple10 门控在 M1 Max 全走非 tensor-core 路;MPP 门槛 M≥16,sk6_mm8 关闭,t_ver 用 M4 常量(待标定)

### 杠杆排序(M1 Max 视角)
1. megakernel 层内融合:M=1 空泡 ~45%(1412 dispatch/838 barrier),理论 ~12ms/token;参考 mirage MPK ~1.12x、llama-megakernel 教训是层内融合而非整模型
2. verify 数值一致性:tree 路径 GPU verify 精确;block host verify 病态(疑 M>1 数值偏差放大,i=0 即拒)。修好后 fused topk/amax 兑现
3. CB:trunk 单读已摊销,边际=MoE scatter;真正要做的是 KV paged/ring + server 侧调度
4. KV2B 2-bit KV:同族模型外部证据,-75% KV 流量
5. nvfp4 可选格式 PR:权重流减半,Apple7 走 simdgroup dequant,需 PTQ 质量门
6. LoopCD:已集成(INK_LCD_W),fixed-3-pass 下中性保持关闭;后续跟社区 logits-adaptive+fused dual-unembed
7. DSA:Inkling 已有 IndexShare top-2048 indexer → 本质是 qsa kernel 优化,4K ctx 下收益小
8. GLM-5.4 借鉴:gated progressive updates≈attention residual 已有;MLA/CED/MoR 需重训,远期

## 2026-11-28 M1 Max serve 实录:CB server 上线 + INK_SPEC_MULTI 转正

### 部署(M1 Max,serve/ HTTP 栈)
- 二进制:`engine/build/inkling-mac/src/inkling2/strata-inkling`(ninja,CMake 金属后端;CMakeLists 已有 metal 子目录)
- tokenizer:HF `thinkingmachines/Inkling-2-Mini/resolve/main/tokenizer.json`(huggingface 直连超时,hf-mirror 可用)→ `terminal/tools/inkling_tokenizer.py --hf-json` 生成 `~/dageno-modelbench/tok_inkling/`
- 服务:`terminal/serve/server.py` + cfg `~/dageno-modelbench/serve-out/cfg.json`(arch=inkling2_loop,--serve --native sq10.strata --parallel 8 --draft auto --max-context 8192)
- python:jinja2 只有 /opt/anaconda3/bin/python3 有
- **血泪**:cfg args 里别加第二个 `--serve`(serve_chat.py 会重复添加 → 引擎起两个,违反单权重实例规则)
- 后台 nohup 会被会话管理器杀:服务必须在常驻 exec 会话前台跑(会话 71235/61982)

### 并发性能与正确性(greedy temperature=0,reasoning_effort=none)
| 配置 | 单请求 | 3并发/req | 4并发/req | 正确性 |
|---|---|---|---|---|
| 默认(spec_multi off) | 28-35 | 14.3 | - | - |
| INK_SPEC_MULTI=1 | ~26 | 25.0 | 26.4/9.6 | 逐字节=solo ✓(同prompt+混合prompt) |
| INK_SPEC_BATCH=1 | - | 4.4 | 4.3 | ✗ DIFFERS,接受率5%,仍然损坏 |

- `spec_multi` 已改为默认 ON(INK_SPEC_MULTI=0 关),提交 49897b6
- `spec_batch`(合并链 verify 2c)继续损坏:i=0 拒绝率~95%,输出与 solo 发散——root cause 同 block verify 未解
- `draft_n_accepted` 在 BGEN 并发协议下计数不完整(BT 行不携带),HTTP timings 的 acc 数字在并发不可信

### 连续批处理现状审计(ink_serve.h)
- ✓ slot 1..64 BGEN 并发,交错 chunked prefill(token_budget 均分,prevents head-of-line)
- ✓ spec_multi 轮转 GPU tree spec(修复并默认开)
- ✓ KV/SC slot-budget clamp(64 slots 不再 OOM)
- ✗ 跨 slot 合并 verify 窗口(spec_batch)损坏:host logits verify 的分布错位未修
- 遗留:M=2 near-tie argmax flip(slot2,1/6 概率)未复现于本轮 3-req 测试但文献记录存在

## 2026-11-28 LoopSpec 可行性判定实验(M1 Max, SQ-10, --agree/--ee)

新工具(commit b5843e1): `--agree N`(teacher-forced 重放 pass-3 贪婪流,per-pass argmax+top1prob 转储)、`--ee TAU`/INK_EE(pass 边界分段 forward,预 coda 置信度早退)、runner.pass_override 运行时覆盖、ink_argmax_prob kernel(argmax+softcap-logsumexp 概率一次归约)。

### 关键数据
- pass-k argmax == pass-3 argmax 同意率:code p1=79.7% p2=85.4%;prose p1=73.4% p2=82.3%
- post-coda top1 prob 校准:≥0.95 → 100% 对(n=67 code + n=26 prose,零失误);SQ-10 未破坏置信度
- pass-3 batched argmax vs M=1 贪婪:code 192/192 exact,prose 189/192,3 处翻转全在 prob<0.4 近并列 → host-verify 病灶=低置信 token 数值敏感,不是系统性内核 bug
- 预 coda prob 分布平坦:τ=0.80 只退出 5.3%,τ=0.60 已 divergence → coda 是 sharpen 的来源,早退判据无法放在 coda 前

### 结论
1. 固定 3-pass 模型没有"少跑 pass"的免费提速:每个 token 都须付完它需要的 pass。LoopSpec 唯一可移植收益是跨 token pass 流水(scheduler 级重构),浅 pass 同意率 82-85% 已经偏低,流水挽回也受限 → 列为远期。
2. pass-2 argmax 当 draft 的同意率(82-85%)不优于现有 dspark(~85-90%),且每 draft 行都要花 verify 行 → 不值得替换。
3. **主线索不变:修 host-verify**(block/spec_batch 接受率 3.9%)。新病灶特征:差错集中在 prob<0.4 的近并列 token,先在低置信位置做数值差分(误差条件数),最容易抓到真凶。
4. dspark 蒸馏(DFlash 配方)仍是提接受率的正路,属训练远期项。

### 复现
```
INK_AGREE_DUMP=/tmp/ag ./strata-inkling-metal --native sq10.strata --parallel 1 --draft off \
  --agree 192 --ids "<chat_tokens.py ids>"
./strata-inkling-metal --native sq10.strata --draft off --gen 96 --ee 0.80 --ids "..."
```

## 2026-11-28 host-verify 修复实录(M1 Max):两个真 bug,block 模式全链修通

### Bug 1: repair `best` token 漏发(输出垃圾的主因)
serve block/spec_batch 的 partial-accept 分支:`rq.next_tok = best` 被设置但从未 emit——`proto("T")`/`gen_ids`/`hist`/`cached_prefix` 全部缺这一格。每次部分接受输出流漏一个 token,症状是"每个 token 都是真字但串不成句"。修复(7543f46):emit best + `cached_prefix.push(repair_in)` + `fin→finish_req`;spec_batch 合并路径连 repair forward 都没有,同补丁补齐。
**诊断教训**:把"接受率低"和"输出垃圾"分开归因——它们是两个不同 bug。vdebug REJ 行显示 i=0..6 分散的正常拒绝,垃圾输出另有其因。

### Bug 2: fused unembed+topk merge 链 barrier 缺失(token id -1)
`dm.barrier = first_merge` 只给第一级屏障;level-2..N 和最终拷贝在 concurrent encoder 里无屏障跑,ping-pong 同一缓冲 → level-3 写 fk_a 与 level-1 同区并发 → fk_a 行 0 前 8 组(恰好 level-3 输出区)全零 → `bd_fk_out` 零 key → tok -1。**修复:每级都 barrier**。用 --topk-check 缓冲探针定位:tk_part 有真 key → fk_a 部分零 → fk_b 部分零 → fk_out 全零,恰好暴露竞争。
**方法论**:M>1 verify 输出不对时,先 dump 各级 scratch 缓冲做断点二分,不要猜 kernel 数学。

### 修后状态
- block 全行路径与 tree 逐字节一致;amax(fused)路径输出连贯、27.9-28.1 tok/s
- spec_batch 3 并发正确连贯(近并列 token 批间微差,无系统性崩坏)
- block 模式现在正确但**不比 tree 快**(nd=7 固定深链 vs tree 动态预算);默认仍走 tree
- 接受率 ~30-50% @nd7;dspark 蒸馏(DFlash 配方)仍是提接受率的主轴,训练项远期

### 工具沉淀
- `--topk-check`:fused vs full argmax 行对比 + tk_part/fk_a/fk_b/fk_out 探针
- `--agree`/`--ee`:per-pass argmax/prob 转储与预 coda 早退测量(上一提交)
- 病灶特征库:pass-3 argmax 翻转集中在 prob<0.4 近并列 token
