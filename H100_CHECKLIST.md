# H100 one-shot checklist

Pre-flight for the real run. Items that are local-CPU-only are marked **STRIP**.

## 0. Code + deps + weights (on the H100 VM)
- `git clone <your repo> && cd <repo> && git checkout ex2-local-build`  (or merge to main)
- `uv sync`   # installs the REAL GPU vllm wheel - correct here; do NOT use the local `--no-sync` hack
- `nvidia-smi`  # confirm the H100
- `huggingface-cli download Qwen/Qwen3-30B-A3B-Instruct-2507`  # ~60 GB, do FIRST (long pole); set HF_TOKEN if prompted
- `uv run python scripts/load_data.py`  # regenerates data/bird/, eval_set.jsonl, perf_pool.jsonl

## 1. .env - flip these
| Key | Local (now) | H100 |
|---|---|---|
| VLLM_MODEL | Qwen/Qwen3-0.6B | **Qwen/Qwen3-30B-A3B-Instruct-2507** |
| VLLM_NO_THINK | 1 | **remove / comment out** (the 30B Instruct is non-thinking) |
| LLM_MAX_TOKENS | 512 | 512 ok (raise only if SQL gets truncated) |
| VLLM_BASE_URL | http://localhost:8000/v1 | same |
| OPENAI_API_KEY | not-needed | not-needed |
| HF_TOKEN | (empty) | set if the model download requires it |
| LANGFUSE_* | seeded keys | keep as-is (compose seeds the same keys) |

## 2. vLLM launch - GPU path, NOT the CPU image
**STRIP all CPU bits:** the `vllm-cpu*` Docker image, `--enforce-eager`, `VLLM_CPU_KVCACHE_SPACE`,
and the transformers/fastapi/torchaudio pins (those only fixed the from-source CPU build; `uv.lock`
handles versions on the H100).

Run real vLLM via `scripts/start_vllm.sh` (set MODEL + add your Phase-1 flags), e.g.:
```
--model Qwen/Qwen3-30B-A3B-Instruct-2507 --max-model-len 4096 \
--gpu-memory-utilization 0.90 --max-num-seqs <tune> --dtype auto
```
Iterate these for the SLO (P95 < 5s @ 10+ RPS); record each flag + rationale in REPORT.md.

## 3. o11y stack
- `docker compose up -d` (same file). Prometheus already scrapes host:8000 -> your GPU vLLM.
- `LANGFUSE_INIT_*` seeds org/project/keys/user on first boot (login `admin@example.com` / `langfuse123`),
  OR delete that block and sign up manually for a clean compose (see section 6).

## 4. Local-only workarounds that DON'T apply on the H100
- The keepalive process / WSL idle-shutdown handling - local only.
- "Don't run concurrent load" was a CPU limit. On the H100 the load test IS Phase 6:
  `uv run python load_test/driver.py --rps 10 --duration 300`

## 5. Capture the REAL numbers + screenshots (must come from the 30B)
- Phase 1: `screenshots/vllm_manual_query.png` + flags/rationale in REPORT.md
- Phase 5: `results/eval_baseline.json` + `screenshots/grafana_eval_run.png`
- Phase 6: iterate; `screenshots/grafana_before.png` / `grafana_after.png`;
  `results/eval_after_tuning.json`; SLO verdict in REPORT.md
- Re-capture `screenshots/grafana_serving.png` vs the 30B (KV-cache / GPU panels now meaningful)
- `langfuse_trace.png` / `langfuse_tags.png` may stay local (structure is identical)

## 6. Optional cleanup for submission
- Revert the `docker-compose.yml` `LANGFUSE_INIT_*` seed (drops the hardcoded `langfuse123`) for a
  clean compose, then get keys via the Langfuse UI.
- `.gitignore` excludes `screenshots/*.png` - keep using `git add -f` for the deliverable screenshots.
---

## Discovered during the real H100 run (2026-06-17) - fold these in

1. **transformers must be < 5** (`uv.lock` pulls 5.x; vLLM 0.10.2 needs the 4.5x tokenizer
   API). Fix: `uv pip install "transformers==4.55.4"` after `uv sync` (or pin in pyproject
   and re-lock). Symptom: `Qwen2Tokenizer has no attribute all_special_tokens_extended`.
2. **python3-dev + build-essential required** (the prerequisites' "torch.compile needs
   headers"). Without them vLLM engine init dies in Triton: `gcc ... cuda_utils.c ... exit 1`.
   Fix: `sudo apt-get install -y python3-dev build-essential`.
3. **Run the agent with multiple uvicorn workers** for the SLO. The sync `/answer` endpoint
   caps at FastAPI's 40-thread pool -> vLLM starves (running pinned 40, KV 13%). `--workers 8`
   took P95 119s -> 53s. (Restart cleanly: kill old workers + free port 8001 first, else
   `Errno 98 Address already in use` leaves the agent down.)
4. **schema.py None-FK fix is now in the code** (PRAGMA foreign_key_list "to" is NULL for
   implicit-PK FKs) - this removed a deterministic ~12% HTTP-500 rate on big-schema DBs.
5. Image used: `ubuntu24.04-cuda13.0-serverless` (driver 580); platform `gpu-h100-sxm`,
   preset `1gpu-16vcpu-200gb`, 300 GB network_ssd boot disk.