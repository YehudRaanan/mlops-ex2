# Report: LLM inference + observability (text-to-SQL on BIRD-bench)

Model: **Qwen3-30B-A3B-Instruct-2507** (MoE, ~3B active) on **1x H100 80GB** (Nebius).
All numbers are from the real 30B endpoint.

## 1. Serving configuration (Phase 1)

vLLM 0.10.2, flags chosen for this workload (~1.5-3K-token prompts, short SQL outputs,
2-3 dependent calls/request, target P95 < 5s @ 10 RPS):

| Flag | Value | Rationale |
|---|---|---|
| `--max-model-len` | 4096 | Prompt envelope is ~3K + short output; a small window frees KV-cache slots -> more concurrency. |
| `--gpu-memory-utilization` | 0.90 | bf16 weights ~60GB of 80GB; leave the rest for KV without OOM. |
| `--max-num-seqs` | 512 | Deep batching; the A3B MoE decodes cheaply (KV sat ~40% at 256, so headroom existed). |
| `--enable-prefix-caching` | on | The schema is a large prefix shared across the 2-3 calls/request and same-DB questions -> skip re-prefill. |
| `--dtype` | bf16 (auto) | H100-native; faithful to the released checkpoint. |

Sanity: 30B loads and a manual query returns correct SQL (`screenshots/vllm_manual_query.png`).

## 2. Observability dashboard (Phase 2)

`infra/grafana/.../serving.json`: **Latency** (e2e / TTFT / time-per-output-token / queue,
percentiles), **Throughput** (prompt vs generation tokens, completion-by-reason,
running/waiting), and **KV-cache** usage with thresholds. Answers "is it slow, and *where*
in the request lifecycle?" Panels react under load - `screenshots/grafana_serving.png`,
`grafana_eval_run.png`.

## 3. Agent design (Phase 3)

`generate_sql -> execute -> verify -> (revise -> execute -> verify)*`, capped at
`MAX_ITERATIONS`. `verify` is an LLM call returning `{ok, issue}` (parsed defensively,
fail-safe to `ok=false`); the router loops to `revise` on failure. Prompts target SQL
errors, zero rows when rows are implied, and irrelevant columns. At least one question
triggers a revise (`langfuse_trace.png` shows a multi-iteration waterfall).

## 4. Agent tracing (Phase 4)

Langfuse `CallbackHandler` in `agent/server.py`; each request's tags are surfaced as both
Langfuse trace tags (chips) and filterable metadata. Traces show the
`generate_sql / verify / (revise)` waterfall - `screenshots/langfuse_trace.png`,
`langfuse_tags.png`.

## 5. Baseline eval (Phase 5)

Execution accuracy (canonicalized row-set match), 30 BIRD-dev questions.
`results/eval_baseline.json`:

- Overall: **36.7%** (11/30)
- Per-iteration (carry-forward): **33.3 -> 36.7 -> 36.7%**

Commentary: the verify->revise loop recovers **+1 question** over single-shot, then plateaus
- real but marginal at this model size. Execution accuracy is strict (exact rows; near-misses
count as wrong). 36.7% on schema+question only is in line with no-hint baselines (BIRD dev
average ~43%, top single models ~54-57%, humans ~92%).

## 6. Hitting the SLO (Phase 6)

Target: **P95 end-to-end agent latency < 5s at 10+ RPS over 5 min**
(`load_test/driver.py --rps 10 --duration 300`). Baseline vs SLO: **P95 119s, 38% failures**
- far off. Iteration log (*saw -> hypothesized -> changed -> result*):

| Step | Change | Dashboard signal -> result |
|---|---|---|
| v2 | 8 uvicorn workers | vLLM idle (running pinned **40**, KV 13%, no queue) -> the *sync agent* (40-thread pool) was the cap, not the GPU -> **P95 119->60s**, errors 38->0.5% |
| v3 | schema pruning (BM25+dense+RRF retrieval of relevant tables) | prefill dominated per-call latency -> shrink the prompt -> **P95 60->21s** (quality held) |
| v4 | verify-gating | accept when rows>0, verify/revise only on error/empty -> calls/req 2.6->~1.4 -> **P95 21->7.6s** |
| v5 | `--max-num-seqs` 512 + cached schema embeddings + cap revise to 1 | used idle KV headroom, trimmed the loop tail -> **P95 7.6->5.27s** |
| v6 | retrieval-augmented few-shot (leakage-guarded) | better first-try SQL -> avg iters 1.2->1.07 -> **P95 5.27->4.55s AND accuracy 36.7->46.7%** |

**Final: P95 4.55s @ 10 RPS, p50 2.38s, 99.9% success - SLO met.** Quality improved, not
regressed (`results/eval_after_tuning.json` = **46.7%**). One bug surfaced *only* under load:
embeddings defaulted to CUDA -> 8 workers OOM'd vLLM (`EngineDeadError`) -> pinned embeddings
to CPU. Trajectory: P95 **119 -> 60 -> 21 -> 7.6 -> 5.27 -> 4.55s**.

## 7. Agent value & what I'd do with more time

**Did the loop help?** On its own, modestly: per-iteration **33.3 -> 36.7%** (+1 question). Its
bigger payoff is as a *substrate* - gating `verify` cut latency ~3x with no quality loss, and
retrieval few-shot lifted accuracy **+10 pts**. So the architecture adds measurable value
mainly by enabling targeted, cheap optimizations.

**More time (specific):** (1) feed BIRD's per-question `evidence` hint (the scaffold's
`load_data.py` strips it; worth ~+10 EX, toward the 54-57% range); (2) schema-value retrieval
(sample distinct column values for literal matching); (3) LoRA fine-tune on the BIRD *train*
split (higher ceiling, but outside this assignment's fixed-model / inference+o11y scope);
(4) an async agent endpoint to replace the worker x threadpool concurrency model.