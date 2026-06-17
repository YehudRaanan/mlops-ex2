"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from pathlib import Path

import httpx

# Agent calls can be slow on a CPU stand-in (several vLLM calls per question).
# Override for fast H100 runs if desired.
AGENT_TIMEOUT = float(os.environ.get("EVAL_AGENT_TIMEOUT", "300"))

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implement these (Phase 5) ----------------------------------

def eval_one(question: dict, agent_url: str) -> dict:
    """Score one question. Return a dict capturing per-iteration correctness.

    We ask the agent once, then re-run *each* SQL it produced (one per
    generate/revise step, taken from the returned history) against the gold DB
    and compare canonicalized row sets to the gold query's rows. This gives a
    correctness signal for every iteration, which `summarize` rolls up into a
    per-iteration pass rate.
    """
    db_id = question["db_id"]
    gold_sql = question["gold_sql"]
    gold_ok, gold_rows, gold_err = run_sql(db_id, gold_sql)

    payload = {"question": question["question"], "db": db_id}
    t0 = time.monotonic()
    agent_error: str | None = None
    history: list[dict] = []
    final_sql = ""
    try:
        resp = httpx.post(agent_url, json=payload, timeout=AGENT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        history = data.get("history", [])
        final_sql = data.get("sql", "")
        if not data.get("ok", False):
            agent_error = data.get("error")
    except Exception as e:  # noqa: BLE001
        agent_error = f"{type(e).__name__}: {e}"
    latency = time.monotonic() - t0

    # One SQL per generate/revise step; verify entries carry no "sql".
    per_iteration_sql = [h["sql"] for h in history if h.get("sql")]

    per_iteration_correct: list[bool] = []
    for sql in per_iteration_sql:
        ok, rows, _err = run_sql(db_id, sql)
        per_iteration_correct.append(ok and matches(gold_rows, rows))

    final_correct = per_iteration_correct[-1] if per_iteration_correct else False

    return {
        "db_id": db_id,
        "question": question["question"],
        "gold_sql_ok": gold_ok,
        "gold_sql_error": gold_err,
        "final_sql": final_sql,
        "iterations": len(per_iteration_sql),
        "per_iteration_correct": per_iteration_correct,
        "final_correct": final_correct,
        "agent_error": agent_error,
        "latency_seconds": latency,
    }


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results.

    Per-iteration carry-forward: if the agent terminated at iteration j < k
    (verify said ok at j, or it hit MAX_ITERATIONS at j < k), treat the
    question's iteration-k result as identical to its iteration-j result.
    The agent stopped emitting; whatever it had at termination is what
    would have been served had we polled at iteration k.
    """
    n = len(results)
    max_iters = max((len(r["per_iteration_correct"]) for r in results), default=0)

    def correct_at(r: dict, k: int) -> bool:
        seq = r["per_iteration_correct"]
        if not seq:
            return False
        return seq[k] if k < len(seq) else seq[-1]  # carry forward last emitted

    pass_rate_at_iteration = [
        (sum(1 for r in results if correct_at(r, k)) / n if n else 0.0)
        for k in range(max_iters)
    ]

    overall_correct = sum(1 for r in results if r["final_correct"])
    return {
        "n_questions": n,
        "overall_pass_rate": (overall_correct / n) if n else 0.0,
        "overall_correct": overall_correct,
        "pass_rate_at_iteration": pass_rate_at_iteration,
        "avg_iterations": (sum(r["iterations"] for r in results) / n) if n else 0.0,
        "n_agent_errors": sum(1 for r in results if r["agent_error"]),
    }


# ---------- Main (provided) --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
