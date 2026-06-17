# Report: LLM inference + observability (text-to-SQL on BIRD-bench)

> Skeleton. Numbers marked `<TBD: H100>` must come from the real
> `Qwen/Qwen3-30B-A3B-Instruct-2507` on the H100. Everything else is built and
> validated locally on CPU (`Qwen3-0.6B`). Target ≤ 3 pages.

## 1. Serving configuration (Phase 1)

Model: `Qwen/Qwen3-30B-A3B-Instruct-2507` (MoE, ~3B active). Hardware: 1× H100 80GB.
Workload: 1.5–3K-token prompts, short structured (SQL) outputs, ~2–3 dependent vLLM
calls per agent run. SLO: P95 end-to-end agent latency < 5s at 10+ RPS over 5 min.

| Flag | Value | One-line justification |
|---|---|---|
| `--max-model-len` | `<TBD>` | Cap context to the real prompt envelope (~3K in + small out) to free KV cache for concurrency. |
| `--gpu-memory-utilization` | `<TBD>` | Maximize KV headroom on the 80GB card without OOM. |
| `--max-num-seqs` | `<TBD>` | Concurrency ceiling tuned to hit 10+ RPS without queue blowup. |
| `--dtype` / quant | `<TBD>` | Precision vs throughput tradeoff for an MoE on H100. |
| `<others>` | `<TBD>` | … |

`<TBD: H100>` — confirm load + a manual query returning sensible SQL.
Screenshot: `screenshots/vllm_manual_query.png`.

## 2. Observability dashboard (Phase 2)

Dashboard `infra/grafana/provisioning/dashboards/serving.json` covers:
- **Latency:** end-to-end request latency (p50/p95/p99), time-to-first-token, time-per-output-token, queue time — answers "is it slow, and *where* in the request lifecycle?"
- **Throughput:** prompt vs generation tokens/sec, request completion rate by finished_reason, running/waiting requests.
- **KV cache:** `vllm:gpu_cache_usage_perc` with 0.8/0.95 thresholds — the headroom gauge for scaling RPS.

Validated locally (panels react under load on CPU vLLM; absolute numbers
unrepresentative). Screenshot under load: `screenshots/grafana_serving.png`.

## 3. Agent design (Phase 3)

`generate_sql → execute → verify → (revise → execute → verify)*` capped at
`MAX_ITERATIONS=3`. `verify` (vLLM call) judges plausibility and emits
`{ok, issue}`; `route_after_verify` loops to `revise` on `ok=false` until the cap.
Prompts target the obvious failures: SQL error, zero rows when rows are implied,
columns that don't answer the question. At least one eval question triggers a revise
(`<TBD: cite a question>`).

## 4. Agent tracing (Phase 4)

Langfuse `CallbackHandler` wired in `agent/server.py`; every `/answer` call passes
request `tags` as trace metadata. Traces show the `generate_sql / verify / (revise)`
waterfall with prompts, responses, latency, token counts.
Screenshots: `screenshots/langfuse_trace.png`, `screenshots/langfuse_tags.png`.

## 5. Baseline eval results (Phase 5)

Execution accuracy via canonicalized row-set comparison (sort rows, stringify,
None→""). 30 questions. `<TBD: H100>` `results/eval_baseline.json`.

- Overall pass rate: `<TBD>`
- Per-iteration pass rate (carry-forward): iter0 `<TBD>` → iter1 `<TBD>` → iter2 `<TBD>`
- Commentary: does iterN > iter0 (loop earns its keep) or not? `<TBD>`

Screenshot of dashboard during the run: `screenshots/grafana_eval_run.png`.

## 6. Hitting the SLO (Phase 6)

Baseline vs SLO (P95 < 5s @ 10+ RPS / 5 min): `<TBD: H100>`.

Iteration log:
- saw `<X>` → hypothesized `<Y>` → changed `<Z>` → result `<W>`
- … (3–4 iterations typical)

Before/after the change that moved the needle: `screenshots/grafana_before.png`,
`screenshots/grafana_after.png`. Post-tuning eval `results/eval_after_tuning.json` —
did quality survive? `<TBD>`. Verdict: SLO hit, or missed with gap quantified `<TBD>`.

## 7. Agent value & what I'd do with more time

- **Did the loop help?** `<TBD: cite per-iteration pass rate from §5>`.
- **More time (specific):** `<TBD — e.g. few-shot schema-linking, self-consistency voting over n samples, prefix-caching the schema prompt, speculative decoding for the short SQL outputs>`.
