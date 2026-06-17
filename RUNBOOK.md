# Runbook: local (WSL/CPU) build → one-shot H100 run

Helper notes for executing this repo. Not a graded deliverable. The only
local↔H100 delta is the **vLLM launch + `.env` `VLLM_MODEL`**; the docker-compose
o11y stack is identical in both places (vLLM runs on the host, not in compose).

## A. Local on WSL (no GPU) — everything except real 30B numbers

> Work inside the WSL-native filesystem (`~/mlops-assignment`), **not** `/mnt/g/...`
> — Docker bind mounts + uv are slow/flaky on the Windows mount. The Windows copy
> under `G:\...\Ex2` is for editing; `rsync`/`git pull` it into `~/mlops-assignment` to run.

```bash
# 0. Toolchain
curl -LsSf https://astral.sh/uv/install.sh | sh      # uv
sudo apt-get install -y python3-dev build-essential   # vLLM torch.compile headers
cd ~/mlops-assignment && uv sync                       # agent/evals/load_test deps
cp .env.example .env                                   # then edit (see repo .env)

# 1. Data (~500 MB BIRD dev) -> data/bird/, evals/eval_set.jsonl, load_test/perf_pool.jsonl
uv run python scripts/load_data.py

# 2. o11y stack: Prometheus :9090, Grafana :3000 (admin/admin), Langfuse :3001
docker compose up -d

# 3. CPU vLLM stand-in serving Qwen3-0.6B on :8000 (the one piece that needs a GPU on H100)
#    Build the CPU image once (heavy), then run it. See:
#    https://docs.vllm.ai/en/latest/getting_started/installation/cpu.html
#    git clone https://github.com/vllm-project/vllm && cd vllm
#    docker build -f docker/Dockerfile.cpu -t vllm-cpu .
docker run --rm -p 8000:8000 vllm-cpu \
  --model Qwen/Qwen3-0.6B --max-model-len 4096 --dtype bfloat16
#    Sanity: curl localhost:8000/v1/models ; curl localhost:8000/metrics | head
#    Fallback for Phases 3/4/5 only (no /metrics): set the hosted-API block in .env.

# 4. Agent server on :8001
uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001

# 5. Smoke-test the agent (use a real db_id from data/bird/ and a question from eval_set.jsonl)
curl -s -X POST localhost:8001/answer \
  -H 'Content-Type: application/json' \
  -d '{"question":"...","db":"<db_id>","tags":{"phase":"smoke"}}' | jq

# 6. Phase 4: sign up at localhost:3001, create project, paste keys into .env, restart agent,
#    fire ~10 questions, confirm generate_sql/verify/(revise) waterfall + tags.

# 7. Phase 5 harness check (pass rate will be poor on 0.6B - expected; real numbers come from H100)
uv run python evals/run_eval.py --out results/eval_baseline.json

# 8. Phase 2: open Grafana dashboard "vLLM serving", drive load, confirm panels react.
uv run python load_test/driver.py --rps 2 --duration 60
```

Local acceptance gate: all 5 services up, agent answers, Langfuse traces appear,
Grafana panels react, `run_eval.py` writes valid JSON. Then commit/push.

## B. One-shot H100 run — the reportable numbers + screenshots

```bash
# SSH with 5 port-forwards: 3000 9090 3001 8000 8001
ssh -L 3000:localhost:3000 -L 9090:localhost:9090 -L 3001:localhost:3001 \
    -L 8000:localhost:8000 -L 8001:localhost:8001 <user>@<vm>

git clone <repo> && cd mlops-assignment && uv sync
nvidia-smi                                            # confirm the GPU
huggingface-cli download Qwen/Qwen3-30B-A3B-Instruct-2507   # ~60 GB: the long pole, do FIRST
uv run python scripts/load_data.py
# .env: VLLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 (hosted-API block commented)
docker compose up -d
bash scripts/start_vllm.sh                            # add your Phase-1 tuned flags here

# Deliverables on the real model:
#  P1: screenshots/vllm_manual_query.png + flags+rationale in REPORT.md
#  P5: uv run python evals/run_eval.py --out results/eval_baseline.json + grafana_eval_run.png
#  P6: uv run python load_test/driver.py --rps <n> --duration 300
#      iterate flags; grafana_before.png / grafana_after.png; eval_after_tuning.json
#  Capture: grafana_serving.png, langfuse_trace.png, langfuse_tags.png
# Finalize REPORT.md (Phase 7).
```

Reportable numbers (P5 pass rate, P6 SLO) **must** come from the real 30B on the H100.
