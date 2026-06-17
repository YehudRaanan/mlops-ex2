"""Retrieval-augmented few-shot: inject the most similar solved
`question -> SQL` examples into the generate prompt.

Pool = BIRD `dev.json` **excluding the eval questions** (no answer leakage; other
questions on the same DB are legitimate demonstrations). Retrieval reuses the
sentence-transformers model from `schema_index` - dense cosine over a cached pool
embedding matrix (the pool is ~1.5k items, so dense-only is the perf-sane choice;
schema retrieval stays hybrid because that corpus is tiny).

Enable with FEWSHOT_K>0 (see agent/graph.py). Default off.
"""
from __future__ import annotations

import json
from functools import lru_cache

from agent.schema import DB_DIR, ROOT
from agent.schema_index import _get_model

EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"


def _dev_json():
    cands = sorted(DB_DIR.rglob("dev.json"))
    return cands[0] if cands else None


@lru_cache(maxsize=1)
def _pool() -> tuple:
    """(db_id, question, sql) triples from dev.json, minus the eval questions."""
    dev = _dev_json()
    if dev is None:
        return ()
    rows = json.loads(dev.read_text())
    exclude = set()
    if EVAL_FILE.exists():
        for line in EVAL_FILE.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                exclude.add((r["db_id"], r["question"].strip()))
    return tuple(
        (r["db_id"], r["question"], r["SQL"])
        for r in rows
        if (r["db_id"], r["question"].strip()) not in exclude
    )


@lru_cache(maxsize=1)
def _pool_embeddings():
    pool = _pool()
    if not pool:
        return None
    try:
        return _get_model().encode([q for _, q, _ in pool], normalize_embeddings=True)
    except Exception:
        return None


def retrieve_examples(db_id: str, question: str, k: int = 3) -> str:
    """Return a formatted few-shot block of the k most similar solved examples.

    Same-DB examples are preferred (same schema => most transferable patterns),
    then filled with the most globally-similar others. Returns "" if disabled or
    embeddings are unavailable.
    """
    pool = _pool()
    embs = _pool_embeddings()
    if not pool or embs is None or k <= 0:
        return ""
    try:
        qv = _get_model().encode([question], normalize_embeddings=True)[0]
        cos = (embs @ qv).tolist()
    except Exception:
        return ""
    order = sorted(range(len(pool)), key=lambda i: cos[i], reverse=True)
    same = [i for i in order if pool[i][0] == db_id][:k]
    chosen = (same + [i for i in order if i not in same])[:k]
    blocks = []
    for n, i in enumerate(chosen, 1):
        dbid, q, sql = pool[i]
        blocks.append(f"-- Example {n} (db: {dbid})\n-- Q: {q}\n{sql}")
    return "Similar solved examples:\n" + "\n\n".join(blocks) + "\n\n"
