# Text-to-SQL Agent - vLLM serving + full observability

An internal **"ask your data in English"** proof-of-concept: an analyst asks a question in
plain English, the system writes SQL, runs it against a SQLite warehouse
([BIRD-bench](https://bird-bench.github.io/)), and returns the rows - served on a **single
H100** with production-style monitoring and tracing.

Built for the Nebius MLOps assignment (original brief preserved in
[ASSIGNMENT.md](ASSIGNMENT.md)). Full writeup with all numbers: [REPORT.md](REPORT.md).

## System

```
              Grafana :3000  <--  Prometheus :9090  <--  vLLM /metrics
                                                            |
 user --> Agent :8001  --(OpenAI API)-->  vLLM :8000  (Qwen3-30B-A3B, 1x H100)
(question)    |  LangGraph: generate_sql -> execute -> verify -> (revise)*
              `-->  Langfuse :3001  (per-run traces)
```

- **Serving** - vLLM, OpenAI-compatible, the 30B MoE on one H100 80GB.
- **Agent** ([agent/](agent/)) - LangGraph `generate_sql -> execute -> verify -> (revise)*`,
  plus two optional accelerators: schema-pruning retrieval and retrieval-augmented few-shot.
- **Serving o11y** - Prometheus scrapes vLLM `/metrics`; Grafana dashboard
  ([serving.json](infra/grafana/provisioning/dashboards/serving.json)) for latency /
  throughput / KV-cache.
- **Agent o11y** - Langfuse captures the generate/verify/revise waterfall per request.
- **Eval** ([evals/run_eval.py](evals/run_eval.py)) - execution accuracy (executed rows vs gold).
- **Load test** ([load_test/driver.py](load_test/driver.py)) - drives RPS for the SLO.

## Results (real, 30B on H100)

| Metric | Baseline | Final |
|---|---|---|
| Execution accuracy (BIRD, n=30) | 36.7% | **46.7%** |
| P95 end-to-end latency @ 10 RPS | 119 s | **4.55 s** (SLO < 5 s, met) |
| Error rate under load | 38% | 0.1% |

## How we got latency under 5 seconds

The SLO was **P95 end-to-end agent latency < 5 s at 10+ RPS**. The baseline was **119 s** -
about 24x off. Nothing here was guesswork: every change came from reading the Grafana
dashboard, forming a hypothesis, changing one thing, and confirming the targeted metric moved.

**The key diagnosis (v2).** At baseline the dashboard showed vLLM essentially *idle* -
`num_requests_running` pinned at exactly **40**, KV cache **13%**, zero queue - yet P95 was
119 s. That "40" is FastAPI's default thread-pool size: the **synchronous agent server was the
bottleneck, not the GPU.** Running the agent with **8 uvicorn workers** unblocked it (vLLM
jumped to ~170 concurrent) -> **119 s -> 60 s**.

After that, every remaining second came from *work per request* - the agent makes 2-3
sequential model calls, and each pays full **prefill** on a 2-3K-token schema prompt:

| Step | Change | Why it helped | P95 |
|---|---|---|---|
| baseline | full schema, 1 sync worker | - | 119 s |
| **v2** | 8 uvicorn workers | agent was thread-capped while the GPU sat idle | 60 s |
| **v3** | **schema pruning** - retrieve only the relevant tables (BM25 + embeddings) | prefill dominates latency; smaller prompt = less prefill | 21 s |
| **v4** | **verify-gating** - skip verify/revise when the SQL already returned rows | cuts calls/request 2.6 -> ~1.4 | 7.6 s |
| **v5** | `--max-num-seqs 512` + cache schema embeddings + cap revise to 1 | use idle KV headroom; trim the looping tail | 5.27 s |
| **v6** | **retrieval-augmented few-shot** - inject similar solved examples | better *first-try* SQL -> avg iterations 1.2 -> 1.07 -> fewer calls **and** higher accuracy | **4.55 s** |

Two things worth calling out:

- **v6 got faster *and* more accurate** - counter-intuitive, because few-shot makes the prompt
  *bigger*. But better first-try SQL means the agent rarely needs to verify/revise, so with
  gating most requests collapse to a **single** model call. Accuracy 36.7 -> 46.7% and P95
  5.27 -> 4.55 s, together.
- **A bug only load testing could find:** the retriever's embedding model defaulted to the
  **GPU**, so under concurrency the 8 workers each loaded it onto the H100 alongside vLLM ->
  **CUDA OOM -> vLLM crash -> 56% errors**. Fix: pin embeddings to **CPU**. Sequential eval
  never hit it; the load test did.

Net trajectory: **P95 119 -> 60 -> 21 -> 7.6 -> 5.27 -> 4.55 s.** Full diagnosis log in
[REPORT.md](REPORT.md).

## Configuration (env-gated; the default is faithful baseline behavior)

| Flag | Default | Optimized |
|---|---|---|
| vLLM `--max-num-seqs` | 256 | 512 |
| `SCHEMA_TOPK` (schema pruning) | 0 (full schema) | 3 |
| `GATE_VERIFY` (verify-gating) | 0 (off) | 1 |
| `MAX_ITERATIONS` (revise cap) | 3 | 2 |
| `FEWSHOT_K` (few-shot examples) | 0 (off) | 3 |
| `EMB_DEVICE` (retriever device) | cpu | cpu |
| agent uvicorn workers | 1 | 8 |

## Run it

See [RUNBOOK.md](RUNBOOK.md) (local CPU dev -> one-shot H100) and
[H100_CHECKLIST.md](H100_CHECKLIST.md). Short version:

```bash
uv sync
uv run python scripts/load_data.py        # BIRD subset -> data/bird, eval_set.jsonl
docker compose up -d                       # Prometheus + Grafana + Langfuse
bash scripts/start_vllm.sh                 # vLLM (H100; add your Phase-1 flags)

# optimized agent:
GATE_VERIFY=1 SCHEMA_TOPK=3 MAX_ITERATIONS=2 FEWSHOT_K=3 \
  uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001 --workers 8

uv run python evals/run_eval.py --out results/eval_baseline.json
uv run python load_test/driver.py --rps 10 --duration 300
```

## Layout

- `agent/` - `graph.py` (the loop), `prompts.py`, `schema.py` + `schema_index.py` (pruning),
  `fewshot_index.py` (few-shot retrieval), `server.py` (FastAPI + Langfuse)
- `evals/run_eval.py` - execution-accuracy eval
- `load_test/driver.py` - load generator
- `infra/` - Prometheus config + Grafana dashboard
- `results/` - `eval_baseline.json`, `eval_after_tuning.json` (`archive/` = per-version evidence)
- `screenshots/` - dashboard + trace captures
- `REPORT.md` - full writeup; `RUNBOOK.md` / `H100_CHECKLIST.md` - run guides; `ASSIGNMENT.md` - original brief