"""
cemg/memory.py
---------------
The public CEMG operations, cleanly separated from Neo4j internals.

FIX LOG v3 (this pass):
  - read_signature_status now requires task_namespace -- fixes the
    namespace leak where an aggregate ActionSignature status could be
    shared across unrelated tasks under one agent_id.
  - check_compliance() is REPLACED by evaluate_compliance(), a pure
    function operating on decision-time snapshots the agent captures
    BEFORE each action executes (see cemg/agent.py). The old design
    re-queried live status AFTER the whole run finished, which could
    miss a genuine violation if the same signature's state changed
    later in the same run (e.g. failed once, then somehow succeeded
    on a retry within the same session -- the old check would see the
    final RESOLVED state and miss that the first use was a real
    violation at the time it happened). peek_signature_status() is the
    new pre-decision lookup agent.py calls to build each snapshot.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from neo4j import Driver

from cemg.graph import (
    write_experience,
    read_relevant,
    read_causal_path,
    read_signature_status,
    prune_stale_experiences,
    make_action_signature,
    DEFAULT_NAMESPACE,
)
from cemg.embeddings import EmbeddingProvider


load_dotenv()

TOP_K = int(os.getenv("CEMG_TOP_K", "10"))


# -- Data model ---------------------------------------------------------------
@dataclass
class ExperienceRecord:
    id:                  str
    action:              str
    outcome:             str
    reasoning:           str  = ""    # agent's self-report -- NOT trusted as ground truth
    observed_error:      str  = ""    # raw tool error -- what classification is based on
    context_hint:        str  = ""
    action_signature:    str  = ""
    failure_class:       Optional[str] = None
    cost_tokens:         int  = 0
    timestamp_unix:      float = 0.0
    temporal_weight:     float = 0.0
    relevance:           float = 0.0
    score:               float = 0.0
    verification_status: str  = "CLEAN"

    @property
    def timestamp_str(self) -> str:
        return datetime.fromtimestamp(self.timestamp_unix, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def to_prompt_line(self) -> str:
        """
        One-line serialisation for the LLM system prompt. Surfaces
        verification_status prominently -- this is what lets a stale,
        past-cooldown failure read as "worth retrying" instead of a
        permanent block, and lets the LLM see cost alongside reliability
        instead of only a binary avoid/allow signal.
        """
        status_tag = {
            "ACTIVE_FAILURE":   "AVOID (recent failure)",
            "CONFIRMED_BROKEN": "AVOID (repeated failure, confirmed)",
            "PROBATION":        "UNCERTAIN (past cooldown -- may be fixed, verify before trusting)",
            "RESOLVED":         "OK (previously failed, since succeeded)",
            "CLEAN":            "OK",
        }.get(self.verification_status, self.verification_status)

        cost_tag = f" | ~{self.cost_tokens} tok" if self.cost_tokens else ""
        cause = f" | cause: {self.observed_error}" if self.observed_error else \
                (f" | agent's stated reason: {self.reasoning}" if self.reasoning else "")

        return (
            f"[{self.timestamp_str} | w={self.temporal_weight:.2f} rel={self.relevance:.2f} "
            f"score={self.score:.2f}{cost_tag}] {status_tag}: {self.action}{cause}"
        )

    def to_dict(self) -> dict:
        return asdict(self)


# -- Public API -----------------------------------------------------------------
def store_experience(
    driver:         Driver,
    agent_id:       str,
    session_id:     str,
    action:         str,
    outcome:        str,
    reasoning:      str  = "",
    observed_error: str  = "",
    context_hint:   str  = "",
    tool:           str  = "",
    params:         Optional[dict] = None,
    task_namespace: str  = DEFAULT_NAMESPACE,
    cost_tokens:    int  = 0,
    parent_exp_id:  Optional[str] = None,
) -> dict:
    """
    Store one action-outcome pair. Returns
    {"exp_id", "action_signature", "failure_class"} -- pass action_signature
    forward if you want to check compliance later (did the agent avoid
    what it was told to avoid), and exp_id as parent_exp_id on the next
    call to build the causal chain.

    IMPORTANT: pass observed_error (the raw tool error text) separately
    from reasoning (the agent's own explanation). Classification and
    decay-class selection are based on observed_error, never on
    reasoning, because self-reported reasoning can be wrong about the
    actual cause.
    """
    if outcome not in ("success", "failure", "partial"):
        raise ValueError(f"outcome must be success|failure|partial, got: {outcome!r}")

    return write_experience(
        driver         = driver,
        agent_id       = agent_id,
        session_id     = session_id,
        action         = action,
        outcome        = outcome,
        reasoning      = reasoning,
        observed_error = observed_error,
        context_hint   = context_hint,
        tool           = tool,
        params         = params,
        task_namespace = task_namespace,
        cost_tokens    = cost_tokens,
        parent_exp_id  = parent_exp_id,
        ts             = time.time(),
    )


def recall_relevant(
    driver:           Driver,
    agent_id:         str,
    query_action:     str  = "",
    task_namespace:   Optional[str] = None,
    include_failures: bool  = True,
    top_k:            int   = TOP_K,
    embedding_provider: Optional[EmbeddingProvider] = None,
) -> list[ExperienceRecord]:
    """Retrieve the most relevant past experiences, ranked and status-annotated."""
    rows = read_relevant(
        driver             = driver,
        agent_id           = agent_id,
        query_action       = query_action,
        task_namespace     = task_namespace,
        include_failures   = include_failures,
        top_k              = top_k,
        embedding_provider = embedding_provider,
    )
    return [
        ExperienceRecord(
            id                   = r["id"],
            action               = r["action"],
            outcome              = r["outcome"],
            reasoning            = r.get("reasoning") or "",
            observed_error       = r.get("observed_error") or "",
            context_hint         = r.get("context_hint") or "",
            action_signature     = r.get("action_signature") or "",
            failure_class        = r.get("failure_class"),
            cost_tokens          = r.get("cost_tokens") or 0,
            timestamp_unix       = r["ts"],
            temporal_weight      = r["weight"],
            relevance            = r.get("relevance", 0.0),
            score                = r["score"],
            verification_status  = r.get("verification_status", "CLEAN"),
        )
        for r in rows
    ]


def get_causal_path(driver: Driver, exp_id: str, max_depth: int = 10) -> list[ExperienceRecord]:
    rows = read_causal_path(driver, exp_id, max_depth)
    return [
        ExperienceRecord(
            id=r["id"], action=r["action"], outcome=r["outcome"],
            reasoning=r.get("reasoning") or "", observed_error=r.get("observed_error") or "",
            timestamp_unix=r["ts"],
        )
        for r in rows
    ]


def build_memory_block(
    driver:         Driver,
    agent_id:       str,
    query_action:   str = "",
    task_namespace: Optional[str] = None,
    top_k:          int = TOP_K,
    embedding_provider: Optional[EmbeddingProvider] = None,
) -> str:
    """
    Build the formatted memory context block for the LLM system prompt.

    Groups by verification_status rather than raw outcome, so a
    past-cooldown failure (PROBATION) is shown distinctly from a
    firmly-avoid one (ACTIVE_FAILURE / CONFIRMED_BROKEN) -- this is
    what prevents a permanently stale blacklist.
    """
    experiences = recall_relevant(
        driver, agent_id, query_action=query_action,
        task_namespace=task_namespace, top_k=top_k,
        embedding_provider=embedding_provider,
    )
    if not experiences:
        return ""

    hard_avoid = [e for e in experiences if e.verification_status in ("ACTIVE_FAILURE", "CONFIRMED_BROKEN")]
    probation  = [e for e in experiences if e.verification_status == "PROBATION"]
    ok         = [e for e in experiences if e.verification_status in ("CLEAN", "RESOLVED") and e.outcome != "failure"]

    lines = ["=== CAUSAL EXPERIENCE MEMORY (CEMG) ==="]
    if query_action:
        lines.append(f"(ranked for relevance to: \"{query_action[:80]}\")")

    if hard_avoid:
        lines.append("AVOID -- recent or repeated failures:")
        for e in hard_avoid:
            lines.append(f"  {e.to_prompt_line()}")
    else:
        lines.append("AVOID -- none currently flagged.")

    if probation:
        lines.append("UNCERTAIN -- past cooldown, may be fixed now, verify before trusting fully:")
        for e in probation:
            lines.append(f"  {e.to_prompt_line()}")

    if ok:
        lines.append("SUCCESSES -- patterns that worked:")
        for e in ok[:5]:
            lines.append(f"  {e.to_prompt_line()}")

    lines.append("=========================================")
    return "\n".join(lines)


def peek_signature_status(
    driver:         Driver,
    agent_id:       str,
    tool:           str,
    params:         dict,
    task_namespace: str = DEFAULT_NAMESPACE,
) -> dict:
    """
    Look up the verification status for a (tool, params) call BEFORE it
    executes -- call this immediately before running a tool, not after.

    This is the fix for the compliance-timing bug: checking status
    retroactively, after a full run has finished, can miss a genuine
    violation if the signature's state changed later in the same run.
    By snapshotting status at the actual moment of decision, the record
    reflects what the agent actually knew when it chose to act.

    Returns {"action_signature": str, "status_before": str} -- status_before
    is "CLEAN" if no prior record exists for this signature in this
    namespace (a first-ever attempt is never a violation).
    """
    sig = make_action_signature(tool, params)
    record = read_signature_status(driver, agent_id, sig, task_namespace)
    status = record["verification_status"] if record else "CLEAN"
    return {"action_signature": sig, "status_before": status}


def evaluate_compliance(decision_snapshots: list[dict]) -> dict:
    """
    Pure function -- no driver, no re-query. Operates only on snapshots
    already captured at decision time via peek_signature_status(),
    stored on the agent as it runs (see CEMGAgent.decision_snapshots
    in cemg/agent.py).

    A violation is any snapshot where status_before was ACTIVE_FAILURE
    or CONFIRMED_BROKEN and the agent went ahead and used that action
    anyway -- i.e., the agent had clear prior warning and ignored it.
    PROBATION is deliberately NOT counted as a violation: retrying a
    past-cooldown failure is the intended, encouraged behaviour, not
    non-compliance.

    Returns {"total_used", "violations", "violation_rate", "violating_signatures"}
    """
    violations = [
        s["action_signature"] for s in decision_snapshots
        if s.get("status_before") in ("ACTIVE_FAILURE", "CONFIRMED_BROKEN")
    ]
    total = len(decision_snapshots)
    return {
        "total_used":           total,
        "violations":           len(violations),
        "violation_rate":       (len(violations) / total) if total else 0.0,
        "violating_signatures": violations,
    }


def prune(driver: Driver, agent_id: Optional[str] = None, dry_run: bool = True) -> dict:
    """Thin wrapper over graph.prune_stale_experiences -- see that docstring."""
    return prune_stale_experiences(driver, agent_id=agent_id, dry_run=dry_run)
