"""
cemg/api.py
-----------
FastAPI app -- the CEMG REST endpoints.

FIX LOG v2 (this pass):
  - All endpoints now accept task_namespace (defaults to "default") to
    prevent cross-task memory contamination.
  - store_experience now accepts observed_error separately from
    reasoning, tool/params (used to build the action_signature), and
    cost_tokens.
  - New POST /memory/prune -- decay-triggered deletion, dry_run=True by
    default. This is the mechanism for both the unbounded-graph-growth
    problem and the PII/retention problem.
  - POST /memory/check_compliance now accepts decision_snapshots
    directly (captured client-side by the agent at decision time via
    peek_signature_status) instead of a bare list of signatures the
    server would have had to re-query after the fact. This is the fix
    for the compliance-timing bug: a live re-query after the run could
    miss a violation if the signature's state changed later in the
    same run.
  - GET /memory/peek_status -- new endpoint exposing peek_signature_status
    so non-Python clients can snapshot status before acting too.
  - /health now reports memory subsystem status distinctly, so a
    caller can tell "Neo4j is down, degrade gracefully" from "everything
    is fine."

Run with:
    uvicorn cemg.api:app --reload --port 8100
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from cemg.graph import bootstrap_schema, get_driver, is_healthy, DEFAULT_NAMESPACE
from cemg.memory import (
    store_experience,
    recall_relevant,
    get_causal_path,
    build_memory_block,
    evaluate_compliance,
    peek_signature_status,
    prune,
)

load_dotenv()

_driver = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _driver
    _driver = get_driver()
    bootstrap_schema(_driver)
    yield
    _driver.close()


app = FastAPI(
    title       = "CEMG -- Causal Experience Memory Graph",
    description = (
        "A temporal causal memory API for long-horizon LLM agents. "
        "Store action-outcome pairs, retrieve them with class-aware decay "
        "and live verification status, and reconstruct causal chains."
    ),
    version  = "0.2.0",
    lifespan = lifespan,
)


OutcomeType = Literal["success", "failure", "partial"]

class StoreRequest(BaseModel):
    agent_id:       str
    session_id:     str
    action:         str
    outcome:        OutcomeType
    reasoning:      str  = Field("", description="Agent's SELF-REPORTED explanation -- not trusted as ground truth")
    observed_error: str  = Field("", description="RAW tool/error text -- classification is based on this, not reasoning")
    context_hint:   str  = ""
    tool:           str  = Field("", description="Tool name -- used with params to build the action_signature")
    params:         dict = Field(default_factory=dict)
    task_namespace: str  = DEFAULT_NAMESPACE
    cost_tokens:    int  = 0
    parent_exp_id:  Optional[str] = None


class StoreResponse(BaseModel):
    exp_id:           Optional[str]
    action_signature: Optional[str]
    failure_class:    Optional[str]
    message:          str = "Experience stored"


class RecallResponse(BaseModel):
    agent_id:    str
    count:       int
    experiences: list[dict]


class CausalPathResponse(BaseModel):
    exp_id: str
    depth:  int
    chain:  list[dict]


class ComplianceRequest(BaseModel):
    decision_snapshots: list[dict] = Field(
        ..., description="Snapshots captured at decision time, each "
        "{action_signature, status_before, ...} -- from peek_signature_status "
        "calls made BEFORE each action executed, not re-derived after the run."
    )


class PeekRequest(BaseModel):
    agent_id:       str
    tool:           str
    params:         dict = Field(default_factory=dict)
    task_namespace: str  = DEFAULT_NAMESPACE


class PruneRequest(BaseModel):
    agent_id: Optional[str] = None
    dry_run:  bool = True


# -- Endpoints ----------------------------------------------------------------
@app.post("/memory/store_experience", response_model=StoreResponse)
def api_store(req: StoreRequest):
    """
    Store one action-outcome experience after a tool call.
    Pass observed_error (raw failure text) separately from reasoning
    (the agent's own explanation) -- classification and decay-class
    selection are based on observed_error only.
    """
    if _driver is None:
        raise HTTPException(503, "Neo4j driver not ready")
    try:
        result = store_experience(
            driver         = _driver,
            agent_id       = req.agent_id,
            session_id     = req.session_id,
            action         = req.action,
            outcome        = req.outcome,
            reasoning      = req.reasoning,
            observed_error = req.observed_error,
            context_hint   = req.context_hint,
            tool           = req.tool,
            params         = req.params,
            task_namespace = req.task_namespace,
            cost_tokens    = req.cost_tokens,
            parent_exp_id  = req.parent_exp_id,
        )
        return StoreResponse(**result)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/memory/recall_relevant", response_model=RecallResponse)
def api_recall(
    agent_id:         str  = Query(...),
    query_action:     str  = Query("", description="Current task -- makes recall relevance-ranked"),
    task_namespace:   Optional[str] = Query(None, description="Filter to one task namespace; omit to search all"),
    include_failures: bool = Query(True),
    top_k:            int  = Query(int(os.getenv("CEMG_TOP_K", "10"))),
):
    """
    Retrieve the most relevant past experiences. Each result includes a
    live-computed verification_status (CLEAN/ACTIVE_FAILURE/PROBATION/
    CONFIRMED_BROKEN/RESOLVED) -- PROBATION means the failure is past its
    cooldown and may be worth retrying, not a permanent block.
    """
    if _driver is None:
        raise HTTPException(503, "Neo4j driver not ready")
    try:
        exps = recall_relevant(
            driver           = _driver,
            agent_id         = agent_id,
            query_action     = query_action,
            task_namespace   = task_namespace,
            include_failures = include_failures,
            top_k            = top_k,
        )
        return RecallResponse(agent_id=agent_id, count=len(exps), experiences=[e.to_dict() for e in exps])
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/memory/causal_path/{exp_id}", response_model=CausalPathResponse)
def api_causal_path(exp_id: str, max_depth: int = Query(10)):
    if _driver is None:
        raise HTTPException(503, "Neo4j driver not ready")
    try:
        chain = get_causal_path(_driver, exp_id, max_depth)
        return CausalPathResponse(exp_id=exp_id, depth=len(chain), chain=[e.to_dict() for e in chain])
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/memory/context_block", response_class=PlainTextResponse)
def api_context_block(
    agent_id:       str = Query(...),
    query_action:   str = Query(""),
    task_namespace: Optional[str] = Query(None),
    top_k:          int = Query(int(os.getenv("CEMG_TOP_K", "10"))),
):
    """Formatted memory block ready to paste into an LLM system prompt."""
    if _driver is None:
        raise HTTPException(503, "Neo4j driver not ready")
    try:
        block = build_memory_block(_driver, agent_id, query_action=query_action, task_namespace=task_namespace, top_k=top_k)
        return block or "(no memory yet)"
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/memory/check_compliance")
def api_check_compliance(req: ComplianceRequest):
    """
    Pure evaluation over decision-time snapshots -- does NOT re-query
    Neo4j. Pass the snapshots your agent captured via calls to
    GET /memory/peek_status made BEFORE each action executed. This
    ensures the compliance result reflects what the agent actually knew
    at each decision point, not the signature's state after the fact.
    """
    try:
        return evaluate_compliance(req.decision_snapshots)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/memory/peek_status")
def api_peek_status(
    agent_id:       str  = Query(...),
    tool:           str  = Query(...),
    params_json:    str  = Query("{}", description="JSON-encoded params dict"),
    task_namespace: str  = Query(DEFAULT_NAMESPACE),
):
    """
    Look up verification status for a (tool, params) call BEFORE
    executing it. Call this immediately before running a tool, then
    hang onto the result -- pass it back into /memory/check_compliance
    later as part of decision_snapshots.
    """
    if _driver is None:
        raise HTTPException(503, "Neo4j driver not ready")
    try:
        import json as _json
        params = _json.loads(params_json)
        return peek_signature_status(_driver, agent_id, tool, params, task_namespace)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/memory/prune")
def api_prune(req: PruneRequest):
    """
    Delete experiences whose decay weight has fallen below the floor
    AND are not under active verification tracking. dry_run=True by
    default -- always dry-run first in any environment with real data.
    """
    if _driver is None:
        raise HTTPException(503, "Neo4j driver not ready")
    try:
        return prune(_driver, agent_id=req.agent_id, dry_run=req.dry_run)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/health")
def health():
    if _driver is None:
        return {"status": "no driver", "memory_available": False}
    healthy = is_healthy(_driver)
    return {
        "status":           "ok" if healthy else "degraded",
        "neo4j":             "connected" if healthy else "unreachable",
        "memory_available":  healthy,
    }
