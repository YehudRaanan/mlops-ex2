# Report: LLM inference + observability (text-to-SQL on BIRD-bench)

Model: `Qwen/Qwen3-30B-A3B-Instruct-2507` (MoE, ~3B active). Hardware: 1x H100 80GB
(Nebius, eu-north1). All numbers below are from the real 30B on the H100.

## 1. Serving configuration (Phase 1)

vLLM 0.10.2, launched with:

| Flag | Value | Justification |
|---|---|---|
| `--max-model-len` | 4096 | Prompts are ~1.5-3K tokens (schema + question), outputs are short SQL; capping context maximizes KV-cache slots -> higher concurrency. |
| `--gpu-memory-utilization` | 0.90 | bf16 weights ~60GB on the 80GB card; 0.90 leaves the rest for KV without OOM. |
| `--max-num-seqs` | 256 | High batch ceiling - the A3B MoE decodes cheaply, so deep batching is the throughput lever. |
| `--enable-prefix-caching` | on | The DB schema is a large prefix shared across the 2-3 calls per request and across questions on the same DB; caching it avoids repeated prefill. |
| `--dtype` | bf16 (auto) | H100-native; faithful to the released checkpoint for the eval numbers. |

Sanity: model loads, a manual query returns correct SQL
(`SELECT COUNT(*) FROM circuits WHERE country = 'Italy';`).
See `screenshots/vllm_manual_query.png`.

## 2. Observability dashboard (Phase 2)

`infra/grafana/provisioning/dashboards/serving.json` covers Latency (e2e / TTFT /
time-per-output-token / queue percentiles), Throughput (prompt vs generation tokens,
completion-by-reason, running/waiting), and KV-cache usage with thresholds. Panels react
under load - see `screenshots/grafana_serving.png`. The KV-cache and GPU panels are only
meaningful on the H100 (verified here).

## 3. Agent design (Phase 3)

`generate_sql -> execute -> verify -> (revise -> execute -> verify)*`, capped at
`MAX_ITERATIONS=3`. `verify` is an LLM call returning `{ok, issue}` (parsed defensively,
fail-safe to ok=false); `route_after_verify` loops to `revise` on failure until the cap.
Prompts target SQL errors, empty results, and irrelevant columns. The revise loop fires on
real cases (see per-iteration lift in S5).

## 4. Agent tracing (Phase 4)

Langfuse `CallbackHandler` in `agent/server.py`; each `/answer` request's tags are surfaced
as both Langfuse trace tags (chips) and filterable metadata. Traces show the
`generate_sql / verify / (revise)` waterfall - `screenshots/langfuse_trace.png`,
`screenshots/langfuse_tags.png`.

## 5. Baseline eval (Phase 5)

Execution accuracy (canonicalized row-set match), 30 BIRD-dev questions, 30B.
`results/eval_baseline.json`:

- Overall pass rate: **36.7%** (11/30)
- Per-iteration (carry-forward): iter0 **33.3%** -> iter1 **36.7%** -> iter2 **36.7%**
- Avg iterations: 1.63; agent errors: 0

Read: the verify->revise loop earns its keep but **marginally** - it recovers exactly one
additional question (+3.3 pts) over single-shot, then plateaus. The architecture adds
real-but-small value at this model size.

## 6. Hitting the SLO (Phase 6)

Target: **P95 end-to-end agent latency < 5s at 10+ RPS over 5 min.** Load via
`load_test/driver.py --rps 10 --duration 300`. (Note: the driver fires a fixed 3000
requests, so its "achieved_rps" ~8.3 is total/wall-clock, not a throughput cap - read
latency + error counts.)

Iteration log (saw -> hypothesized -> changed -> result):

1. **Baseline.** Saw P95 **118.7s**, P50 88.3s, 38% failures (256 timeouts / 373 HTTP-500 /
   504 disconnects). Dashboard: vLLM e2e p95 only **2.3s**, `num_requests_running` pinned at
   **40**, `num_requests_waiting`=0, KV **13%**. -> Hypothesis: vLLM is starved; the
   bottleneck is the **synchronous agent endpoint** capped at FastAPI's default 40-thread
   pool. -> Changed: ran the agent with **8 uvicorn workers**. -> Result: P50 88->**6.1s**,
   P95 119->**53.7s**, timeouts 256->1, disconnects 504->10. vLLM now batched **170**
   concurrent (KV 28%). Big win, SLO still missed.

2. **Correctness.** Saw a stubborn **~12.7% HTTP-500** rate; captured the body:
   `AttributeError: 'NoneType' object has no attribute 'replace'`. -> Root cause: `render_schema`
   in `agent/schema.py` calls `_q(fk[4])` but `PRAGMA foreign_key_list."to"` is **NULL** for
   FKs referencing an implicit PK -> crashes deterministically for several BIRD DBs. ->
   Changed: render the FK without a column list when `to` is None. -> Result: failures
   **38% -> 0.5%** (2984/3000 ok). But P95 **60.2s** (P50 9.2s, P99 74.2s): the
   previously-instantly-failing big-schema questions now actually run, adding real load.

Final dashboard (valid iter-2 run): vLLM `num_requests_running` **247** (~max-num-seqs 256),
`num_requests_waiting` **0**, KV **41%**, **25.6 calls/s**, vLLM e2e p95 **6.8s**, TTFT p95 0.24s.

**Verdict: SLO MISSED.** Final P95 **~60s** vs 5s target. Diagnosis: each agent request is
**2-3 sequential 30B calls**; at the concurrency needed for 10 RPS each call's e2e is ~2.6s
(247 seqs sharing the GPU), so a request floors at ~7s regardless of queueing - latency is
bounded by the **multi-call agent shape**, not the serving flags. vLLM is concurrency-capped
(247/256) with KV headroom (41%), i.e. compute/scheduler-bound, not memory-bound. Quality
survived: post-tuning eval `results/eval_after_tuning.json` = **36.7%**, identical to baseline (serving precision unchanged; tuning was concurrency + the schema-render fix). `results/load_after.json` holds the final latency distribution.

## 7. What I'd do with more time (specific)

- **Schema pruning / linking**: send only the tables the question needs (retrieval over the
  schema) - the 2-3K-token schema prefill dominates per-call latency.
- **Cut the call count**: skip `verify` when the SQL executes and returns non-empty plausible
  rows; only spend the extra calls on suspect results. Roughly halves calls/request.
- **Parallelize**: self-consistency via batched parallel generations instead of a sequential
  loop, picking by execution agreement.
- **Serving**: raise `--max-num-seqs` (KV is only 41%), and try `--quantization fp8` to ~halve
  weights -> far more KV + higher decode throughput, trading a little quality (re-measure S5).
- **Smaller draft + speculative decoding** for the short, structured SQL outputs.