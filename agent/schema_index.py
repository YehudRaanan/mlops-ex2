"""Schema-linking retrieval: prune a DB schema to the tables relevant to a
question, to cut prompt size (prefill latency).

Per table -> one chunk = its CREATE TABLE text, enriched with keywords:
  - tokenized table name (camelCase/snake split + crude singular, weighted x3)
  - tokenized column names
  - foreign-key-neighbor table names
Retrieval fuses BM25 (sparse) and sentence-transformers cosine (dense) via
Reciprocal Rank Fusion, then expands one hop along foreign keys so join tables
are not dropped. Falls back to BM25-only if sentence-transformers is unavailable.

Enable in the agent with env SCHEMA_TOPK > 0 (see agent/graph.py).
"""
from __future__ import annotations

import math
import os
import re
import sqlite3
from dataclasses import dataclass
from functools import lru_cache

from agent.schema import _q, db_path


def _tokens(name: str) -> list[str]:
    """Split an identifier or NL phrase into lowercase tokens (+ crude singular)."""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)  # camelCase -> spaced
    s = re.sub(r"[_\W]+", " ", s)                       # snake_case / separators
    toks = [t.lower() for t in s.split() if t]
    sing = [t[:-1] for t in toks if len(t) > 3 and t.endswith("s")]
    return toks + sing


@dataclass(frozen=True)
class _Chunk:
    table: str
    ddl: str
    fk_tables: tuple[str, ...]
    doc_tokens: tuple[str, ...]  # enriched keyword document for retrieval


def _introspect(db_id: str) -> list[_Chunk]:
    path = db_path(db_id)
    if not path.exists():
        raise FileNotFoundError(f"DB {db_id} not found at {path}.")
    chunks: list[_Chunk] = []
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        for t in tables:
            cols: list[str] = []
            lines: list[str] = []
            for _cid, name, ctype, notnull, _dflt, pk in conn.execute(
                f"PRAGMA table_info({_q(t)})"
            ):
                cols.append(name)
                line = f"  {_q(name)} {ctype}"
                if pk:
                    line += " PRIMARY KEY"
                elif notnull:
                    line += " NOT NULL"
                lines.append(line)
            fks: list[str] = []
            for fk in conn.execute(f"PRAGMA foreign_key_list({_q(t)})"):
                ref = _q(fk[2]) if fk[4] is None else f"{_q(fk[2])}({_q(fk[4])})"
                lines.append(f"  FOREIGN KEY ({_q(fk[3])}) REFERENCES {ref}")
                fks.append(fk[2])
            ddl = f"CREATE TABLE {_q(t)} (\n" + ",\n".join(lines) + "\n);"
            doc = _tokens(t) * 3
            for c in cols:
                doc += _tokens(c)
            for nb in fks:
                doc += _tokens(nb)
            chunks.append(_Chunk(t, ddl, tuple(fks), tuple(doc)))
    return chunks


@lru_cache(maxsize=16)
def _chunks(db_id: str) -> tuple[_Chunk, ...]:
    return tuple(_introspect(db_id))


def _bm25(docs: list[list[str]], query: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    n = len(docs)
    avgdl = (sum(len(d) for d in docs) / n) if n else 0.0
    df: dict[str, int] = {}
    for d in docs:
        for w in set(d):
            df[w] = df.get(w, 0) + 1
    scores = [0.0] * n
    for w in set(query):
        if w not in df:
            continue
        idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
        for i, d in enumerate(docs):
            tf = d.count(w)
            if tf:
                dl = len(d)
                scores[i] += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
    return scores


_model = None


def _dense_rank(chunk_texts: list[str], question: str) -> list[int] | None:
    """Cosine ranking via sentence-transformers; None if unavailable."""
    global _model
    try:
        if _model is None:
            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer(
                os.environ.get("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            )
        emb = _model.encode(chunk_texts + [question], normalize_embeddings=True)
        qv = emb[-1]
        cos = [float(sum(cv_i * qv_i for cv_i, qv_i in zip(cv, qv))) for cv in emb[:-1]]
        return sorted(range(len(cos)), key=lambda i: cos[i], reverse=True)
    except Exception:
        return None


def _rrf(rankings: list[list[int]], c: int = 60) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (c + rank + 1)
    return fused


def select_tables(db_id: str, question: str, k: int = 5, fk_expand: bool = True) -> list[str]:
    chunks = _chunks(db_id)
    if len(chunks) <= k:
        return [c.table for c in chunks]  # too small to prune

    docs = [list(c.doc_tokens) for c in chunks]
    rankings = [sorted(range(len(chunks)), key=lambda i, s=_bm25(docs, _tokens(question)): s[i], reverse=True)]
    dense = _dense_rank([" ".join(c.doc_tokens) for c in chunks], question)
    if dense is not None:
        rankings.append(dense)

    fused = _rrf(rankings)
    top = sorted(fused, key=lambda i: fused[i], reverse=True)[:k]
    chosen = {chunks[i].table for i in top}
    if fk_expand:
        for i in top:
            chosen.update(chunks[i].fk_tables)
    return [c.table for c in chunks if c.table in chosen]  # stable (alpha) order


def render_pruned_schema(db_id: str, question: str, k: int = 5) -> str:
    by_name = {c.table: c for c in _chunks(db_id)}
    chosen = select_tables(db_id, question, k)
    header = (
        f"-- Database: {db_id} "
        f"(schema pruned to {len(chosen)}/{len(by_name)} tables relevant to the question)"
    )
    parts = [header]
    for t in chosen:
        parts.append("\n" + by_name[t].ddl)
    return "\n".join(parts)
