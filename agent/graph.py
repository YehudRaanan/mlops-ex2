"""LangGraph agent: text-to-SQL with verify+revise loop.

Graph shape:

    START -> attach_schema -> generate_sql -> execute -> verify
                                                          |
                                              ok=true ----+----> END
                                                          |
                                              ok=false ---+----> revise -> execute -> verify (loop)

Loop is capped at MAX_ITERATIONS total generate/revise calls.

The execute node and the graph wiring are provided. `generate_sql_node` is
filled in as a worked example; you implement `verify`, `revise`, and the
conditional router following the same shape.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from agent import prompts
from agent.execution import ExecutionResult, execute_sql
from agent.schema import render_schema

# Total generate + revise calls before the loop is forced to stop.
# 3-5 is a reasonable range; tune it as part of Phase 3. Env-overridable so the
# loop depth can be tuned for the SLO (e.g. MAX_ITERATIONS=2 caps revise to 1).
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "3"))

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
# vLLM ignores the key, but a hosted OpenAI-compatible provider needs a real one.
# Lets you point the agent at e.g. OpenAI while iterating without a running vLLM.
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")

# SQL replies are short; cap output to bound latency (helps the Phase 6 SLO too).
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "1024"))
# Qwen3 *thinking* variants emit <think> blocks that waste tokens and latency. The
# H100 target (Qwen3-30B-A3B-Instruct-2507) is non-thinking; set VLLM_NO_THINK=1 to
# normalize a thinking CPU stand-in (e.g. Qwen3-0.6B) to the same behavior. Left
# unset on the H100 so the real model's behavior is untouched.
_NO_THINK = os.environ.get("VLLM_NO_THINK") == "1"

# Schema pruning: K>0 retrieves only the top-K question-relevant tables
# (+ FK neighbors) via agent/schema_index.py to cut prompt size / prefill
# latency; K=0 sends the full schema. Default 3 = same accuracy, ~13-22% lower
# latency on BIRD (see REPORT section 8).
SCHEMA_TOPK = int(os.environ.get("SCHEMA_TOPK", "3"))

# Verify-gating: when 1, skip the verify (and revise) calls if the SQL executed
# and returned >=1 row - accept it as a single-call fast path. verify+revise then
# only fire on the suspicious cases (error / zero rows), which is where the loop
# earns its keep. Cuts calls-per-request from ~2-3 to ~1 for the common case.
# Default 0 keeps the always-verify behavior.
GATE_VERIFY = os.environ.get("GATE_VERIFY") == "1"


@dataclass
class AgentState:
    """State threaded through the graph. Extend with fields you need."""

    question: str
    db_id: str
    schema: str = ""
    sql: str = ""
    execution: ExecutionResult | None = None
    verify_ok: bool = False
    verify_issue: str = ""
    iteration: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


def llm() -> ChatOpenAI:
    """Chat client pointed at VLLM_BASE_URL (your local vLLM by default)."""
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}} if _NO_THINK else None
    return ChatOpenAI(
        model=VLLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=0.0,
        max_tokens=LLM_MAX_TOKENS,
        extra_body=extra_body,
    )


# ---- Nodes ------------------------------------------------------------

def _attach_schema(state: AgentState) -> dict:
    """Render the DB schema once at the start of the run.

    Full schema by default; if SCHEMA_TOPK>0, prune to the question-relevant
    tables (hybrid BM25 + dense retrieval) to shrink the prompt.
    """
    if SCHEMA_TOPK > 0:
        from agent.schema_index import render_pruned_schema

        return {"schema": render_pruned_schema(state.db_id, state.question, SCHEMA_TOPK)}
    return {"schema": render_schema(state.db_id)}


def _extract_sql(text: str) -> str:
    """Pull a SQL statement out of an LLM reply, stripping markdown fences/prose.

    Intentionally simple: take the first ```sql ... ``` block if there is one,
    otherwise the whole reply. You may need to harden this for your prompts.
    """
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    # Small models often emit an opening fence with no closing one - strip a
    # leading ```sql / ``` and any dangling trailing fence.
    text = re.sub(r"^\s*```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def _parse_verdict(text: str) -> dict[str, Any]:
    """Defensively parse a {"ok": bool, "issue": str} verdict from an LLM reply.

    The model may wrap the JSON in prose or fences, so grab the first {...} block.
    On any parse failure we fail safe to ok=False so the loop revises rather than
    silently accepting a bad answer.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            return {
                "ok": bool(obj.get("ok", False)),
                "issue": str(obj.get("issue", "") or ""),
            }
        except (json.JSONDecodeError, AttributeError):
            pass
    return {"ok": False, "issue": f"could not parse verifier reply: {text[:200]}"}


def generate_sql_node(state: AgentState) -> dict:
    """Worked example - the other LLM nodes follow this same shape.

    Build messages from the prompts, call the shared llm(), extract the SQL,
    and return only the state fields you changed. `iteration` is bumped here
    (and in revise) so route_after_verify can enforce MAX_ITERATIONS.

    This node is wired and ready; fill in GENERATE_SQL_SYSTEM / GENERATE_SQL_USER
    in prompts.py to make it produce real queries.
    """
    response = llm().invoke([
        ("system", prompts.GENERATE_SQL_SYSTEM),
        ("user", prompts.GENERATE_SQL_USER.format(
            schema=state.schema,
            question=state.question,
        )),
    ])
    sql = _extract_sql(response.content)
    return {
        "sql": sql,
        "iteration": state.iteration + 1,
        "history": state.history + [{"node": "generate_sql", "sql": sql}],
    }


def execute_node(state: AgentState) -> dict:
    """Provided. Runs the SQL and stores the result."""
    return {"execution": execute_sql(state.db_id, state.sql)}


def verify_node(state: AgentState) -> dict:
    """Decide whether state.execution plausibly answers state.question.

    Follow the generate_sql_node pattern: build messages from the VERIFY_*
    prompts, call llm(), parse the reply. Ask the model for a small JSON object
    like {"ok": bool, "issue": str} and parse it defensively - the model may
    wrap it in prose or fences. state.execution.render() gives you a compact
    view of the rows or error to feed into the prompt.

    Return: {"verify_ok": <bool>, "verify_issue": <str>}.
    What counts as "not plausible" is yours to define - see the Phase 3 targets
    in the README.
    """
    result_text = state.execution.render() if state.execution is not None else "ERROR: no execution result"
    response = llm().invoke([
        ("system", prompts.VERIFY_SYSTEM),
        ("user", prompts.VERIFY_USER.format(
            question=state.question,
            sql=state.sql,
            result=result_text,
        )),
    ])
    verdict = _parse_verdict(response.content)
    return {
        "verify_ok": verdict["ok"],
        "verify_issue": verdict["issue"],
        "history": state.history + [{
            "node": "verify",
            "ok": verdict["ok"],
            "issue": verdict["issue"],
        }],
    }


def revise_node(state: AgentState) -> dict:
    """Produce a revised SQL query given state.verify_issue and the prior attempt.

    Same shape as generate_sql_node, but the prompt should include the failing
    SQL, its execution result, and the verifier's complaint so the model can fix
    it. Bump the iteration counter the same way generate_sql_node does so the
    loop terminates.

    Return: {"sql": <str>, "iteration": state.iteration + 1, ...}.
    """
    result_text = state.execution.render() if state.execution is not None else "ERROR: no execution result"
    response = llm().invoke([
        ("system", prompts.REVISE_SYSTEM),
        ("user", prompts.REVISE_USER.format(
            schema=state.schema,
            question=state.question,
            sql=state.sql,
            result=result_text,
            issue=state.verify_issue,
        )),
    ])
    sql = _extract_sql(response.content)
    return {
        "sql": sql,
        "iteration": state.iteration + 1,
        "history": state.history + [{"node": "revise", "sql": sql}],
    }


def route_after_verify(state: AgentState) -> str:
    """Conditional router: return "revise" to loop, "end" to terminate.

    Two reasons to end: the verifier was happy (state.verify_ok), or you've hit
    the iteration cap (state.iteration >= MAX_ITERATIONS). Otherwise, revise.
    """
    if state.verify_ok or state.iteration >= MAX_ITERATIONS:
        return "end"
    return "revise"


def route_after_execute(state: AgentState) -> str:
    """Gating router (used only when GATE_VERIFY=1).

    Accept immediately (skip verify+revise) when the SQL ran and returned rows -
    the common, plausible case. Only spend a verify call on the suspicious cases
    (execution error or zero rows), and stop once the iteration cap is hit.
    """
    ex = state.execution
    if ex is not None and ex.ok and ex.row_count > 0:
        return "end"
    if state.iteration >= MAX_ITERATIONS:
        return "end"
    return "verify"


# ---- Graph wiring -----------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("attach_schema", _attach_schema)
    g.add_node("generate_sql", generate_sql_node)
    g.add_node("execute", execute_node)
    g.add_node("verify", verify_node)
    g.add_node("revise", revise_node)

    g.add_edge(START, "attach_schema")
    g.add_edge("attach_schema", "generate_sql")
    g.add_edge("generate_sql", "execute")
    if GATE_VERIFY:
        # execute -> (accept | verify) instead of always verifying
        g.add_conditional_edges(
            "execute",
            route_after_execute,
            {"verify": "verify", "end": END},
        )
    else:
        g.add_edge("execute", "verify")
    g.add_conditional_edges(
        "verify",
        route_after_verify,
        {"revise": "revise", "end": END},
    )
    g.add_edge("revise", "execute")
    return g.compile()


graph = build_graph()
