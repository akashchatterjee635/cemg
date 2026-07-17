"""
cemg/graph.py
-------------
Neo4j connection + schema bootstrap + raw Cypher helpers.

FIX LOG v2 (this pass):
  - observed_error (raw tool error text) is now a separate field from
    reasoning (agent's self-reported explanation). Ranking/classification
    reads observed_error; reasoning is stored for context but never
    trusted as ground truth. Fixes the confabulation problem.
  - Every write computes failure_class (transient/structural) from
    observed_error via cemg.classify, and maintains an ActionSignature
    aggregate node per distinct (agent_id, tool, params) combination so
    verification status (ACTIVE_FAILURE/PROBATION/CONFIRMED_BROKEN/
    RESOLVED) can be computed live at read time -- never stored frozen.
  - task_namespace is now a required-by-convention field on every
    Experience, so cross-task memory contamination in benchmark runs
    can be ruled out by filtering, instead of being silently possible.
  - External tool content is sanitised via cemg.security before storage.
  - Added prune_stale_experiences() -- decay-triggered deletion, which
    solves both the unbounded-graph-growth problem and the PII/retention
    problem with one mechanism, since both need "forget it once it's
    definitely no longer useful or sensitive."
  - get_driver() now supports a health check the agent can call before
    trusting the connection, so a Neo4j outage can degrade gracefully
    instead of crashing agent startup (see cemg/agent.py).

FIX LOG v1 (carried over, still in effect):
  - Decay recomputed LIVE at read time from raw timestamps.
  - query_action actually used via keyword_overlap for relevance ranking.
  - CAUSED_BY write split into two explicit statements for testability.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import uuid
from typing import Optional

from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver

from cemg.classify import classify_failure, compute_verification_status, LAMBDA_BY_CLASS, generalize_params
from cemg.security import sanitize_external_content, is_external_source
from cemg.embeddings import EmbeddingProvider, TfidfCosineProvider

_DEFAULT_EMBEDDING_PROVIDER = TfidfCosineProvider()


load_dotenv()

# -- tunables from env --------------------------------------------------------
LAMBDA            = float(os.getenv("CEMG_LAMBDA",            "0.03"))
TOP_K             = int(os.getenv("CEMG_TOP_K",               "10"))
FAIL_BOOST        = float(os.getenv("CEMG_FAILURE_BOOST",     "2.0"))
RELEVANCE_WEIGHT  = float(os.getenv("CEMG_RELEVANCE_WEIGHT",  "1.5"))
FETCH_WINDOW      = int(os.getenv("CEMG_FETCH_WINDOW",        "500"))
DEFAULT_NAMESPACE = os.getenv("CEMG_DEFAULT_NAMESPACE",       "default")
PRUNE_FLOOR       = float(os.getenv("CEMG_PRUNE_FLOOR",       "0.02"))  # decay weight below this is eligible for deletion


# -- decay (class-aware) ------------------------------------------------------
def decay(ts_unix: float, lam: float = LAMBDA) -> float:
    """w(t) = exp(-lambda * delta_t_days). Recent -> 1.0. Old -> ->0."""
    delta_days = (time.time() - ts_unix) / 86_400.0
    return math.exp(-lam * max(delta_days, 0.0))


def decay_for_class(ts_unix: float, failure_class: Optional[str]) -> float:
    """
    Decay using the failure-class-specific lambda when the experience is
    a classified failure; falls back to the global LAMBDA for successes
    and unclassified rows. This is what makes a transient (server-hiccup)
    failure fade out of memory in days, while a structural (wrong
    reasoning) failure stays flagged for much longer.
    """
    lam = LAMBDA_BY_CLASS.get(failure_class, LAMBDA) if failure_class else LAMBDA
    return decay(ts_unix, lam=lam)


# -- relevance -----------------------------------------------------------------
_WORD_RE = re.compile(r"[a-z0-9]+")

def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall((text or "").lower()))


def keyword_overlap(query: str, *fields: str) -> float:
    """Jaccard-style overlap between query and one or more text fields, 0..1."""
    q_tokens = _tokenize(query)
    if not q_tokens:
        return 0.0
    combined = _tokenize(" ".join(f for f in fields if f))
    if not combined:
        return 0.0
    overlap = q_tokens & combined
    union   = q_tokens | combined
    return len(overlap) / len(union) if union else 0.0


# -- action signature ------------------------------------------------------------
def make_action_signature(tool: str, params: dict) -> str:
    """
    Deterministic identity for an action signature. Applies generic parameter
    normalization (removing specific numbers, timestamps, and UUIDs) and
    tool-specific rules (like extracting the folder path for files) before
    hashing. This groups structurally identical actions to avoid signature explosion.
    """
    generalized = generalize_params(tool, params)
    canonical = json.dumps({"tool": tool, "params": generalized}, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]



# -- connection ------------------------------------------------------------------
from cemg.storage import BaseStorage, get_storage_provider

def get_driver() -> Driver:
    uri  = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER",     "neo4j")
    pwd  = os.getenv("NEO4J_PASSWORD", "neo4j")
    return GraphDatabase.driver(uri, auth=(user, pwd))


def is_healthy(driver: Driver | BaseStorage, timeout_s: float = 2.0) -> bool:
    if isinstance(driver, BaseStorage):
        return driver.is_healthy()
    from cemg.neo4j_storage import Neo4jStorage
    return Neo4jStorage(driver).is_healthy(timeout_s)


def bootstrap_schema(driver: Driver | BaseStorage) -> None:
    if isinstance(driver, BaseStorage):
        return
    from cemg.neo4j_storage import Neo4jStorage
    Neo4jStorage(driver)._bootstrap_schema()


# -- write -----------------------------------------------------------------------
def write_experience(
    driver:          Driver | BaseStorage,
    agent_id:        str,
    session_id:      str,
    action:          str,
    outcome:         str,
    reasoning:       str  = "",
    observed_error:  str  = "",
    context_hint:    str  = "",
    tool:            str  = "",
    params:          Optional[dict] = None,
    task_namespace:  str  = DEFAULT_NAMESPACE,
    cost_tokens:     int  = 0,
    parent_exp_id:   Optional[str] = None,
    ts:              Optional[float] = None,
) -> dict:
    if isinstance(driver, BaseStorage):
        return driver.write_experience(
            agent_id=agent_id, session_id=session_id, action=action, outcome=outcome,
            reasoning=reasoning, observed_error=observed_error, context_hint=context_hint,
            tool=tool, params=params, task_namespace=task_namespace, cost_tokens=cost_tokens,
            parent_exp_id=parent_exp_id, ts=ts
        )
    from cemg.neo4j_storage import Neo4jStorage
    return Neo4jStorage(driver).write_experience(
        agent_id=agent_id, session_id=session_id, action=action, outcome=outcome,
        reasoning=reasoning, observed_error=observed_error, context_hint=context_hint,
        tool=tool, params=params, task_namespace=task_namespace, cost_tokens=cost_tokens,
        parent_exp_id=parent_exp_id, ts=ts
    )


# -- read: ranked recall -----------------------------------------------------
def read_relevant(
    driver:           Driver | BaseStorage,
    agent_id:         str,
    query_action:     str  = "",
    task_namespace:   Optional[str] = None,
    include_failures: bool  = True,
    top_k:            int   = TOP_K,
    fail_boost:       float = FAIL_BOOST,
    relevance_weight: float = RELEVANCE_WEIGHT,
    embedding_provider: Optional[EmbeddingProvider] = None,
) -> list[dict]:
    if isinstance(driver, BaseStorage):
        return driver.read_relevant(
            agent_id=agent_id, query_action=query_action, task_namespace=task_namespace,
            include_failures=include_failures, top_k=top_k, fail_boost=fail_boost,
            relevance_weight=relevance_weight, embedding_provider=embedding_provider,
            fetch_window=FETCH_WINDOW
        )
    from cemg.neo4j_storage import Neo4jStorage
    return Neo4jStorage(driver).read_relevant(
        agent_id=agent_id, query_action=query_action, task_namespace=task_namespace,
        include_failures=include_failures, top_k=top_k, fail_boost=fail_boost,
        relevance_weight=relevance_weight, embedding_provider=embedding_provider,
        fetch_window=FETCH_WINDOW
    )


# -- read: causal chain -------------------------------------------------------
def read_causal_path(driver: Driver | BaseStorage, exp_id: str, max_depth: int = 10) -> list[dict]:
    if isinstance(driver, BaseStorage):
        return driver.read_causal_path(exp_id, max_depth)
    from cemg.neo4j_storage import Neo4jStorage
    return Neo4jStorage(driver).read_causal_path(exp_id, max_depth)


# -- lookup: single action signature status (for pre-decision compliance checks) --
def read_signature_status(driver: Driver | BaseStorage, agent_id: str, signature: str, task_namespace: str) -> Optional[dict]:
    if isinstance(driver, BaseStorage):
        return driver.read_signature_status(agent_id, signature, task_namespace)
    from cemg.neo4j_storage import Neo4jStorage
    return Neo4jStorage(driver).read_signature_status(agent_id, signature, task_namespace)


# -- maintenance: prune stale experiences --------------------------------------
def prune_stale_experiences(
    driver:      Driver | BaseStorage,
    agent_id:    Optional[str] = None,
    floor:       float = PRUNE_FLOOR,
    dry_run:     bool  = True,
) -> dict:
    if isinstance(driver, BaseStorage):
        return driver.prune_stale_experiences(agent_id, floor, dry_run)
    from cemg.neo4j_storage import Neo4jStorage
    return Neo4jStorage(driver).prune_stale_experiences(agent_id, floor, dry_run)
