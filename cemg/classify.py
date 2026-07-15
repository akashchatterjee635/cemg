"""
cemg/classify.py
-----------------
Failure classification and the verification state machine.

This module answers the two questions raised as open doubts:

  1. "A failure might be transient (server hiccup) vs structural
     (agent's approach was actually wrong) -- these shouldn't decay
     at the same rate."
  2. "A failure might get fixed later -- there needs to be a mechanism
     to re-check, not a permanent blacklist."

Design choice, stated plainly: the SAME decay constant that ranks
memories also determines how long a failure sits in "assume still
broken" state before the system considers re-testing it. A transient
failure (lambda=0.3, ~3.3 day characteristic time) clears the cooldown
fast; a structural failure (lambda=0.01, ~100 day characteristic time)
stays flagged for a long time, because the reasoning that caused it
doesn't fix itself just because time passed. This is a genuine, defensible
research contribution on top of plain temporal decay, not just an
engineering nicety -- it's the answer to "what happens when the
environment changes and your memory is stale?"
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

# -- Failure classes and their decay constants --------------------------------
LAMBDA_BY_CLASS = {
    "transient":  0.30,   # characteristic time ~3.3 days -- infra blips fade fast
    "structural": 0.01,   # characteristic time ~100 days -- wrong reasoning persists
    "unknown":    0.01,   # unclassifiable -- treat conservatively, like structural
}

# Regex signatures for infra/transient failures. Deliberately simple and
# auditable -- this is pattern matching on the RAW tool error message
# (observed_error), never on the LLM's self-reported reasoning, because
# self-reported reasoning can be wrong about the actual cause (see
# cemg/memory.py's separation of reasoning vs observed_error).
_TRANSIENT_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\btimeout\b", r"\btimed out\b", r"\bconnection (reset|refused|aborted)\b",
        r"\b5\d{2}\b.{0,20}\berror\b", r"\bservice unavailable\b",
        r"\btemporarily unavailable\b", r"\brate.?limit(ed)?\b", r"\b429\b",
        r"\bnetwork error\b", r"\bECONNRESET\b", r"\bECONNREFUSED\b",
        r"\bgateway timeout\b", r"\btry again later\b", r"\bthrottl",
    ]
]


def classify_failure(observed_error: str) -> str:
    """
    Classify a failure as transient or structural based on the RAW error
    text -- not the agent's self-reported reasoning about why it failed.

    Returns "transient", "structural", or "unknown" (treated as structural
    for decay purposes, but tracked separately so you can audit how often
    the classifier can't decide).
    """
    if not observed_error:
        return "unknown"
    for pattern in _TRANSIENT_PATTERNS:
        if pattern.search(observed_error):
            return "transient"
    return "structural"


def cooldown_days(failure_class: str) -> float:
    """
    The characteristic time (1/lambda) for a failure class -- how long
    before the system considers a failure 'old enough to be worth
    re-testing' rather than 'assume still broken.'

    Deliberately reuses the exact same lambda used for ranking decay,
    so there's one tunable per class, not two independent knobs that
    could drift out of sync.
    """
    lam = LAMBDA_BY_CLASS.get(failure_class, LAMBDA_BY_CLASS["unknown"])
    return 1.0 / lam


@dataclass
class VerificationStatus:
    status:        str     # CLEAN | ACTIVE_FAILURE | PROBATION | CONFIRMED_BROKEN | RESOLVED
    age_days:      float
    cooldown_days: float
    failure_count: int
    success_count: int


def compute_verification_status(
    last_outcome:   str,
    last_ts:        float,
    failure_class:  str,
    failure_count:  int,
    success_count:  int,
    now:            float = None,
) -> VerificationStatus:
    """
    Compute the current verification status of an action signature.

    This is evaluated LIVE at read time from raw facts -- never stored
    as a frozen verdict -- for the same reason decay() is recomputed
    live: a status computed once and cached would go stale the moment
    time passes, exactly the bug that was fixed in graph.py's decay
    handling.

    States:
      CLEAN             -- no failure on record at all
      ACTIVE_FAILURE     -- failed once recently, still within cooldown -- avoid
      PROBATION          -- cooldown (possibly escalated) has passed --
                            worth retrying to verify, not a hard block
      CONFIRMED_BROKEN   -- failed 2+ times, still within the ESCALATED
                            cooldown for this failure count -- avoid,
                            with stronger evidence than a single failure
      RESOLVED           -- most recent attempt succeeded, after a prior
                            failure history -- the old failure is
                            neutralised, path is trusted again

    FIX (found by running eval/simulate_convergence.py, not by
    inspection): the original version of this function had no escape
    path out of CONFIRMED_BROKEN -- once failure_count reached 2 and
    the base cooldown had passed once, the status stayed
    CONFIRMED_BROKEN forever, no matter how much more time passed.
    Verified directly: 200 simulated days after a 2nd failure, status
    was still CONFIRMED_BROKEN. That made repeat failures behave
    exactly like the permanent blacklist CEMG is meant to improve on.

    Fixed with standard exponential backoff: each additional failure
    doubles the required wait before the system is willing to try
    again (effective_cooldown = base_cooldown * 2^(failure_count-1)).
    This preserves the useful signal (repeat failures ARE treated more
    cautiously than a single one) while guaranteeing every failure,
    however repeated, eventually becomes worth one more probe --
    it just takes exponentially longer to earn that probe.
    """
    now = now or time.time()

    if last_outcome != "failure":
        status = "RESOLVED" if failure_count > 0 else "CLEAN"
        return VerificationStatus(status, 0.0, 0.0, failure_count, success_count)

    base_cd       = cooldown_days(failure_class)
    escalation     = 2 ** max(failure_count - 1, 0)   # 1st failure: x1, 2nd: x2, 3rd: x4, ...
    effective_cd  = base_cd * escalation
    age_days      = max((now - last_ts) / 86_400.0, 0.0)

    if age_days < effective_cd:
        status = "ACTIVE_FAILURE" if failure_count < 2 else "CONFIRMED_BROKEN"
    else:
        status = "PROBATION"   # escalated cooldown has passed -- worth another look,
                                # regardless of how many times it failed before

    return VerificationStatus(status, age_days, effective_cd, failure_count, success_count)
