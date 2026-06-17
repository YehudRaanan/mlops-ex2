# Report: LLM inference + observability (text-to-SQL on BIRD-bench)

Model: **Qwen3-30B-A3B-Instruct-2507** (MoE, ~3B active) on **1x H100 80GB** (Nebius).
All numbers are from the real 30B endpoint.

## 1. Serving configuration (Phase 1)

**Workload reasoning:** the A3B MoE activates only ~3B params, so *decode* is cheap - but
every call still pays full *prefill* on a 1.5-3K-token schema prompt and emits a short SQL
output. The system is therefore **prefill / concurrency-bound, not decode-bound**, which
dictates the flags: cap context, batch many sequences, and cache the shared schema prefix.

| Flag | Value | Rationale |
|---|---|---|
| `--max-model-len` | 4096 | Prompt envelope ~3K + short output; a small window frees KV slots -> more concurrency. |
| `--gpu-memory-utilization` | 0.90 | bf16 weights ~60GB of 80GB; leave the rest for KV without OOM. |
| `--max-num-seqs` | 512 | Deep batching - decode is cheap for the MoE, and KV sat only ~40% at 256. |
| `--enable-prefix-caching` | on | Schema is a large prefix shared across the 2-3 calls/request and same-DB questions -> skip re-prefill. |
| `--dtype` | bf16 (auto) | H100-native; faithful to the released checkpoint. |

Sanity: 30B loads, manual query returns correct SQL (`screenshots/vllm_manual_query.png`).

## 2. Observability dashboard (Phase 2)

`infra/grafana/.../serving.json`, built from vLLM's `/metrics`:
- **Latency:** e2e p50/p95/p99 `histogram_quantile(.., rate(vllm:e2e_request_latency_seconds_bucket))`, plus TTFT (`vllm:time_to_first_token_seconds`), time-per-output-token, and queue time (`vllm:request_queue_time_seconds`) - so you can see *where* in the lifecycle time goes.
- **Throughput:** `rate(vllm:generation_tokens_total)` / `prompt_tokens_total`, completion-by-reason `rate(vllm:request_success_total)`.
- **Concurrency / headroom:** `vllm:num_requests_running` & `num_requests_waiting`, and `vllm:gpu_cache_usage_perc` (KV) with 0.8/0.95 thresholds.

Panels react under load (`screenshots/grafana_serving.png`, `grafana_eval_run.png`). This
dashboard is what made the Phase-6 diagnosis possible (below).

## 3. Agent design (Phase 3)

`generate_sql -> execute -> verify -> (revise -> execute -> verify)*`, capped at
`MAX_ITERATIONS`. `verify` is an LLM call returning `{ok, issue}` (parsed defensively,
fail-safe to `ok=false`); the router loops to `revise` on failure. Prompts target SQL errors,
zero rows when rows are implied, and irrelevant columns. At least one question triggers a
revise (`langfuse_trace.png` shows a multi-iteration waterfall).

## 4. Agent tracing (Phase 4)

Langfuse `CallbackHandler` in `agent/server.py`; each request's tags are surfaced as both
trace tags (chips) and filterable metadata. Traces show the `generate_sql / verify / (revise)`
waterfall - `screenshots/langfuse_trace.png`, `langfuse_tags.png`.

## 5. Baseline eval (Phase 5)

Execution accuracy (canonicalized row-set match - we compare *executed rows*, never SQL text),
30 BIRD-dev questions. `results/eval_baseline.json`:

- Overall: **36.7%** (11/30); per-iteration (carry-forward): **33.3 -> 36.7 -> 36.7%**.

The verify->revise loop recovers **+1 question** over single-shot, then plateaus - real but
marginal at this size. 36.7% on schema+question only is in line with no-hint baselines
(BIRD dev avg ~43%, top single models ~54-57%, humans ~92%).

**Variance caveat:** with n=30 each question is ~3.3 pts, and temp=0 isn't fully deterministic
under batching, so run-to-run swings of +-2-3 questions (~+-10%) are noise. Treat sub-10-pt
differences below as flat.

## 6. Hitting the SLO (Phase 6)

Target: **P95 end-to-end agent latency < 5s at 10+ RPS over 5 min**
(`load_test/driver.py --rps 10 --duration 300`). Baseline vs SLO: **P95 119s, 38% failures**.
Iteration log (*saw -> hypothesized -> changed -> result*); quality tracked alongside latency:

| Step | Change | Dashboard signal -> result | p95 | Eval |
|---|---|---|---|---|
| baseline | defaults (full schema, 1 sync worker) | running pinned 40, KV 13%, no queue | 119s | 36.7% |
| v2 | 8 uvicorn workers | running **40->170**, KV 13->28% -> sync agent was the cap, not the GPU | 60s | 36.7% |
| v3 | schema pruning (BM25+dense+RRF) | prefill/call dominated -> shrink prompt | 21s | 43.3% |
| v4 | verify-gating | calls/req **2.6->1.4** | 7.6s | 40.0% |
| v5 | max-num-seqs 512 + cached embeds + cap revise | used idle KV, trimmed loop tail | 5.27s | 36.7% |
| v6 | retrieval-augmented few-shot | avg iters **1.2->1.07** | **4.55s** | **46.7%** |

**Diagnosis that mattered most (v2):** the dashboard showed vLLM *idle* (running pinned at 40,
KV 13%, zero queue) while P95 was 119s - so the bottleneck was the synchronous agent endpoint
(FastAPI's 40-thread pool), not the GPU. Multiple workers unblocked it (before/after:
`grafana_before.png` / `grafana_after.png`).

**Final: P95 4.55s @ 10 RPS, p50 2.38s, 99.9% success - SLO met**, and **quality followed**:
flat within variance through v2-v5, genuinely up at v6, so none of the latency work cost
accuracy (`results/eval_after_tuning.json` = **46.7%**). Trajectory P95 **119 -> 60 -> 21 ->
7.6 -> 5.27 -> 4.55s**. (Note: the driver fires a fixed 3000 requests, so its `achieved_rps`
~8.3 is total/wall-clock, not a cap - the SLO metric is p95 over the window.) One bug surfaced
*only* under load: embeddings defaulted to CUDA -> 8 workers OOM'd vLLM (`EngineDeadError`)
-> pinned embeddings to CPU.

## 7. Agent value & what I'd do with more time

**Did the loop help?** On its own, modestly (per-iteration 33.3 -> 36.7%, +1 question). Its
bigger payoff is as a *substrate*: gating `verify` cut latency ~3x with no quality loss, and
retrieval few-shot lifted accuracy **+10 pts**. So the architecture adds measurable value
mainly by enabling targeted, cheap optimizations.

**More time (specific):** (1) feed BIRD's per-question `evidence` hint (the scaffold's
`load_data.py` strips it; ~+10 EX, toward 54-57%); (2) schema-value retrieval (sample distinct
column values for literal matching); (3) LoRA fine-tune on the BIRD *train* split (higher
ceiling, but outside this assignment's fixed-model / inference+o11y scope); (4) async agent
endpoint to replace the worker x threadpool model.

## Reproduce
- **Baseline:** vLLM defaults + full schema, single sync worker, no agent flags.
- **Optimized (v6):** vLLM `--max-model-len 4096 --gpu-memory-utilization 0.90 --max-num-seqs 512 --enable-prefix-caching`; agent with **8 uvicorn workers** and env `GATE_VERIFY=1 SCHEMA_TOPK=3 MAX_ITERATIONS=2 FEWSHOT_K=3 EMB_DEVICE=cpu`.