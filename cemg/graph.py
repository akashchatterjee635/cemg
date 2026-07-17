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
def get_driver() -> Driver:
    uri  = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER",     "neo4j")
    pwd  = os.getenv("NEO4J_PASSWORD", "neo4j")
    return GraphDatabase.driver(uri, auth=(user, pwd))


def is_healthy(driver: Driver, timeout_s: float = 2.0) -> bool:
    """
    Cheap connectivity check the agent can call before trusting memory.
    Returns False (never raises) so callers can degrade gracefully
    instead of crashing on a database outage.
    """
    try:
        with driver.session() as s:
            s.run("RETURN 1", timeout=timeout_s).consume()
        return True
    except Exception:
        return False


def bootstrap_schema(driver: Driver) -> None:
    """
    Idempotent -- safe to run on every startup.

    FIX (this pass): the old cemg_sig_id constraint enforced
    `s.signature IS UNIQUE` GLOBALLY across every ActionSignature node --
    which is wrong on two counts. First, it ignored task_namespace,
    so a failure in one task's namespace could silently attach to (or
    collide with) an identical call in a different namespace. Second,
    and more seriously, it enforced uniqueness of the signature value
    across ALL agents, not just within one agent -- two different
    agents calling the same tool with the same params would collide
    on write. Replaced with a composite index over the three properties
    that actually define identity here (signature, agent_id,
    task_namespace); correctness now comes from the MERGE pattern
    matching all three properties together, not from a constraint.
    """
    stmts = [
        "CREATE CONSTRAINT cemg_exp_id IF NOT EXISTS "
        "FOR (e:Experience) REQUIRE e.id IS UNIQUE",

        "CREATE INDEX cemg_sig_composite IF NOT EXISTS "
        "FOR (s:ActionSignature) ON (s.signature, s.agent_id, s.task_namespace)",

        "CREATE INDEX cemg_agent_ts IF NOT EXISTS "
        "FOR (e:Experience) ON (e.agent_id, e.timestamp_unix)",

        "CREATE INDEX cemg_agent_ns IF NOT EXISTS "
        "FOR (e:Experience) ON (e.agent_id, e.task_namespace)",

        "CREATE INDEX cemg_outcome IF NOT EXISTS "
        "FOR (e:Experience) ON (e.outcome)",

        "CREATE FULLTEXT INDEX cemg_text IF NOT EXISTS "
        "FOR (e:Experience) ON EACH [e.action, e.reasoning, e.observed_error, e.context_hint]",
    ]
    with driver.session() as s:
        for stmt in stmts:
            try:
                s.run(stmt)
            except Exception:
                pass
    print("[CEMG] Neo4j schema ready")


# -- write -----------------------------------------------------------------------
def write_experience(
    driver:          Driver,
    agent_id:        str,
    session_id:      str,
    action:          str,
    outcome:         str,             # "success" | "failure" | "partial"
    reasoning:       str  = "",       # agent's SELF-REPORTED explanation -- not trusted as ground truth
    observed_error:  str  = "",       # RAW tool/error text -- this is what classification reads
    context_hint:    str  = "",
    tool:            str  = "",
    params:          Optional[dict] = None,
    task_namespace:  str  = DEFAULT_NAMESPACE,
    cost_tokens:     int  = 0,
    parent_exp_id:   Optional[str] = None,
    ts:              Optional[float] = None,
) -> dict:
    """
    Write one experience node to the graph, plus update the aggregate
    ActionSignature node used for verification-status tracking.

    Returns {"exp_id": ..., "action_signature": ..., "failure_class": ...}
    so callers can pass action_signature forward for the current-run
    compliance check (did the agent repeat something flagged this run).

    Sanitisation: if context_hint indicates the action pulled in external
    content (web_search, read_url, etc.), the observed_error / reasoning
    text is passed through cemg.security.sanitize_external_content()
    before storage -- this is the mitigation for stored prompt injection.
    """
    exp_id  = str(uuid.uuid4())
    ts      = ts or time.time()
    params  = params or {}
    tool    = tool or context_hint or "unknown_tool"
    sig     = make_action_signature(tool, params)

    if is_external_source(context_hint):
        reasoning      = sanitize_external_content(reasoning)
        observed_error = sanitize_external_content(observed_error)

    failure_class = classify_failure(observed_error) if outcome == "failure" else None
    w_at_write    = decay_for_class(ts, failure_class)

    create_node = """
    MERGE (a:Agent {id: $agent_id})
    MERGE (s:Session {id: $session_id})
    CREATE (e:Experience {
        id:               $exp_id,
        agent_id:         $agent_id,
        session_id:       $session_id,
        task_namespace:   $task_namespace,
        action:           $action,
        outcome:          $outcome,
        reasoning:        $reasoning,
        observed_error:   $observed_error,
        context_hint:     $context_hint,
        action_signature: $sig,
        failure_class:    $failure_class,
        cost_tokens:      $cost_tokens,
        timestamp_unix:   $ts,
        temporal_weight_at_write: $weight
    })
    MERGE (e)-[:ATTEMPTED_IN]->(s)
    MERGE (e)-[:PERFORMED_BY]->(a)
    """

    link_parent = """
    MATCH (e:Experience {id: $exp_id})
    MATCH (p:Experience {id: $parent_id})
    MERGE (e)-[:CAUSED_BY {timestamp_unix: $ts}]->(p)
    """

    # Aggregate ActionSignature update -- one transaction so a crash
    # mid-write can't leave the signature aggregate half-updated relative
    # to the Experience node it summarises.
    #
    # FIX (this pass): MERGE key now includes task_namespace alongside
    # signature and agent_id. Before this fix, the same exact tool+params
    # call in two different task namespaces under one agent_id shared a
    # single ActionSignature aggregate -- meaning a failure recorded in
    # Task A's namespace could mark that call CONFIRMED_BROKEN inside
    # Task B's namespace too, even though Task B never saw the failure.
    # Raw experience recall already respected task_namespace; this fixes
    # the aggregate status (the AVOID/PROBATION label) to match.
    update_signature = """
    MERGE (sig:ActionSignature {signature: $sig, agent_id: $agent_id, task_namespace: $task_namespace})
      ON CREATE SET sig.failure_count = 0, sig.success_count = 0, sig.tool = $tool
    SET sig.last_outcome  = $outcome,
        sig.last_ts       = $ts,
        sig.failure_class = coalesce($failure_class, sig.failure_class),
        sig.failure_count = sig.failure_count + CASE WHEN $outcome = 'failure' THEN 1 ELSE 0 END,
        sig.success_count = sig.success_count + CASE WHEN $outcome = 'success' THEN 1 ELSE 0 END
    WITH sig
    MATCH (e:Experience {id: $exp_id})
    MERGE (e)-[:INSTANCE_OF]->(sig)
    """

    with driver.session() as sess:
        with sess.begin_transaction() as tx:
            tx.run(create_node, {
                "agent_id":       agent_id, "session_id": session_id,
                "task_namespace": task_namespace, "exp_id": exp_id,
                "action":         action, "outcome": outcome,
                "reasoning":      reasoning, "observed_error": observed_error,
                "context_hint":   context_hint, "sig": sig,
                "failure_class":  failure_class, "cost_tokens": cost_tokens,
                "ts": ts, "weight": w_at_write,
            })
            tx.run(update_signature, {
                "sig": sig, "agent_id": agent_id, "task_namespace": task_namespace,
                "tool": tool, "outcome": outcome, "ts": ts,
                "failure_class": failure_class, "exp_id": exp_id,
            })
            if parent_exp_id:
                tx.run(link_parent, {"exp_id": exp_id, "parent_id": parent_exp_id, "ts": ts})
            tx.commit()

    return {"exp_id": exp_id, "action_signature": sig, "failure_class": failure_class}


# -- read: ranked recall -----------------------------------------------------
def read_relevant(
    driver:           Driver,
    agent_id:         str,
    query_action:     str  = "",
    task_namespace:   Optional[str] = None,
    include_failures: bool  = True,
    top_k:            int   = TOP_K,
    fail_boost:       float = FAIL_BOOST,
    relevance_weight: float = RELEVANCE_WEIGHT,
    embedding_provider: Optional[EmbeddingProvider] = None,
) -> list[dict]:
    """
    Retrieve the most relevant past experiences for the agent.

    task_namespace, if given, filters to that namespace only -- ruling
    out cross-task contamination in benchmark runs. Leave None to search
    across all namespaces for this agent (the "personal assistant"
    use case, where cross-task memory is wanted).

    Scoring (all computed live in Python, every call):
        score = decay_for_class(ts, failure_class)      <- class-aware, live
                x (fail_boost if outcome=='failure' else 1.0)
                x (1 + relevance_weight x semantic_similarity)

    Each returned row also carries its live verification status
    (CLEAN/ACTIVE_FAILURE/PROBATION/CONFIRMED_BROKEN/RESOLVED) computed
    from its ActionSignature aggregate -- this is what lets a failure
    that's past its cooldown surface as "worth retrying" instead of a
    permanent block.
    """
    ns_filter = "AND e.task_namespace = $task_namespace" if task_namespace else ""
    outcome_filter = "" if include_failures else "AND e.outcome <> 'failure'"

    cypher = f"""
    MATCH (e:Experience {{agent_id: $agent_id}})
    WHERE true {ns_filter} {outcome_filter}
    OPTIONAL MATCH (e)-[:INSTANCE_OF]->(sig:ActionSignature)
    RETURN
        e.id              AS id,
        e.action          AS action,
        e.outcome         AS outcome,
        e.reasoning       AS reasoning,
        e.observed_error  AS observed_error,
        e.context_hint    AS context_hint,
        e.action_signature AS action_signature,
        e.failure_class   AS failure_class,
        e.cost_tokens     AS cost_tokens,
        e.timestamp_unix  AS ts,
        sig.failure_count AS sig_failure_count,
        sig.success_count AS sig_success_count,
        sig.last_outcome  AS sig_last_outcome,
        sig.last_ts       AS sig_last_ts,
        sig.failure_class AS sig_failure_class
    ORDER BY e.timestamp_unix DESC
    LIMIT $fetch_window
    """
    params = {"agent_id": agent_id, "fetch_window": FETCH_WINDOW}
    if task_namespace:
        params["task_namespace"] = task_namespace

    with driver.session() as sess:
        result = sess.run(cypher, params)
        raw_rows = [dict(r) for r in result]

    provider = embedding_provider or _DEFAULT_EMBEDDING_PROVIDER
    if query_action and raw_rows:
        relevances = provider.compute_similarity(query_action, raw_rows)
    else:
        relevances = [0.0] * len(raw_rows)

    scored: list[dict] = []
    for r, rel in zip(raw_rows, relevances):
        w_now = decay_for_class(r["ts"], r.get("failure_class"))
        boost = fail_boost if r["outcome"] == "failure" else 1.0
        score = w_now * boost * (1 + relevance_weight * rel)

        vstatus = compute_verification_status(
            last_outcome  = r.get("sig_last_outcome")  or r["outcome"],
            last_ts       = r.get("sig_last_ts")       or r["ts"],
            failure_class = r.get("sig_failure_class") or r.get("failure_class"),
            failure_count = r.get("sig_failure_count")  or (1 if r["outcome"] == "failure" else 0),
            success_count = r.get("sig_success_count")  or (1 if r["outcome"] == "success" else 0),
        )

        scored.append({
            **r,
            "weight":              w_now,
            "relevance":           rel,
            "score":               score,
            "verification_status": vstatus.status,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]



# -- read: causal chain -------------------------------------------------------
def read_causal_path(driver: Driver, exp_id: str, max_depth: int = 10) -> list[dict]:
    cypher = f"""
    MATCH path = (start:Experience {{id: $exp_id}})
                 -[:CAUSED_BY*0..{max_depth}]->(ancestor:Experience)
    WITH nodes(path) AS chain
    UNWIND chain AS e
    RETURN DISTINCT
        e.id AS id, e.action AS action, e.outcome AS outcome,
        e.reasoning AS reasoning, e.observed_error AS observed_error,
        e.timestamp_unix AS ts
    ORDER BY e.timestamp_unix ASC
    """
    with driver.session() as sess:
        result = sess.run(cypher, {"exp_id": exp_id})
        return [dict(r) for r in result]


# -- lookup: single action signature status (for pre-decision compliance checks) --
def read_signature_status(driver: Driver, agent_id: str, signature: str, task_namespace: str) -> Optional[dict]:
    """
    Look up the live verification status for one specific action
    signature, scoped to a task_namespace.

    task_namespace is now REQUIRED (not optional) -- this is the fix for
    the namespace leak found in review: without it, the same tool+params
    signature used in two different tasks under one agent_id would share
    one verification status, so a failure in Task A could silently mark
    an identical, never-attempted-in-Task-B call as CONFIRMED_BROKEN
    inside Task B too.

    Used by agent.py BEFORE executing a tool call, to snapshot the
    status as it existed at decision time -- this is what makes the
    compliance metric measure "did the agent ignore what it knew at the
    moment it acted," rather than "what does the signature look like
    now, after the whole run has already finished."
    """
    cypher = """
    MATCH (sig:ActionSignature {signature: $signature, agent_id: $agent_id, task_namespace: $task_namespace})
    RETURN sig.failure_count AS failure_count, sig.success_count AS success_count,
           sig.last_outcome AS last_outcome, sig.last_ts AS last_ts,
           sig.failure_class AS failure_class
    """
    with driver.session() as sess:
        result = sess.run(cypher, {"agent_id": agent_id, "signature": signature, "task_namespace": task_namespace})
        row = result.single()
        if row is None:
            return None
        vstatus = compute_verification_status(
            last_outcome  = row["last_outcome"],
            last_ts       = row["last_ts"],
            failure_class = row["failure_class"],
            failure_count = row["failure_count"],
            success_count = row["success_count"],
        )
        return {**dict(row), "verification_status": vstatus.status}


# -- maintenance: prune stale experiences --------------------------------------
def prune_stale_experiences(
    driver:      Driver,
    agent_id:    Optional[str] = None,
    floor:       float = PRUNE_FLOOR,
    dry_run:     bool  = True,
) -> dict:
    """
    Delete experiences whose LIVE decay weight has fallen below `floor`
    AND whose action signature is not in an active-attention state
    (ACTIVE_FAILURE / PROBATION / CONFIRMED_BROKEN) -- i.e. it is safe
    to forget: old, faded, and not something the system is still
    actively tracking for re-verification.

    This is the single mechanism that answers two separate problems:
      - unbounded graph growth (nothing was ever deleted before)
      - PII / data-retention exposure (raw tool params/text living
        forever with no erasure path)

    dry_run=True (default) returns what WOULD be deleted without
    deleting anything -- always run this first before dry_run=False
    in any environment holding real user data.
    """
    agent_filter = "AND e.agent_id = $agent_id" if agent_id else ""

    cypher = f"""
    MATCH (e:Experience)
    WHERE true {agent_filter}
    OPTIONAL MATCH (e)-[:INSTANCE_OF]->(sig:ActionSignature)
    RETURN e.id AS id, e.agent_id AS agent_id, e.timestamp_unix AS ts,
           e.failure_class AS failure_class,
           sig.failure_count AS failure_count, sig.success_count AS success_count,
           sig.last_outcome AS last_outcome, sig.last_ts AS last_ts,
           sig.failure_class AS sig_failure_class
    """
    params = {"agent_id": agent_id} if agent_id else {}

    with driver.session() as sess:
        rows = [dict(r) for r in sess.run(cypher, params)]

    to_delete = []
    for r in rows:
        w = decay_for_class(r["ts"], r.get("failure_class"))
        if w >= floor:
            continue
        vstatus = compute_verification_status(
            last_outcome  = r.get("last_outcome")  or "success",
            last_ts       = r.get("last_ts")       or r["ts"],
            failure_class = r.get("sig_failure_class") or r.get("failure_class"),
            failure_count = r.get("failure_count")  or 0,
            success_count = r.get("success_count")  or 0,
        )
        if vstatus.status in ("ACTIVE_FAILURE", "PROBATION", "CONFIRMED_BROKEN"):
            continue   # still actively tracked -- do not forget yet
        to_delete.append(r["id"])

    if not dry_run and to_delete:
        with driver.session() as sess:
            sess.run(
                "MATCH (e:Experience) WHERE e.id IN $ids DETACH DELETE e",
                {"ids": to_delete},
            )

    return {"eligible_count": len(to_delete), "deleted": (not dry_run), "ids": to_delete}
