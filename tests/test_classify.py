"""
tests/test_classify.py
-----------------------
Unit tests for cemg/classify.py -- the failure-classification and
verification state machine that answers the two open doubts:

  1. Transient (server hiccup) vs structural (agent was actually wrong)
     failures should decay at different rates.
  2. A failure should be able to move from "assume still broken" to
     "worth re-testing" once enough time has passed -- not stay a
     permanent blacklist entry.

Zero setup required -- pure functions, no Neo4j, no API key.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from cemg.classify import (
    classify_failure,
    cooldown_days,
    compute_verification_status,
    LAMBDA_BY_CLASS,
)


# -- classify_failure() -----------------------------------------------------
class TestClassifyFailure:

    def test_timeout_is_transient(self):
        assert classify_failure("Connection timed out after 30s") == "transient"

    def test_503_is_transient(self):
        assert classify_failure("HTTP 503 Service Unavailable") == "transient"

    def test_rate_limit_is_transient(self):
        assert classify_failure("429 Too Many Requests -- rate limited") == "transient"

    def test_file_not_found_is_structural(self):
        """
        A missing file is not an infra blip -- it reflects a wrong
        assumption the agent made, so it should NOT be classified
        transient (which would make it decay away too fast).
        """
        assert classify_failure("FileNotFoundError: no such file 'config.json'") == "structural"

    def test_wrong_parameter_is_structural(self):
        assert classify_failure("TypeError: unexpected keyword argument 'foo'") == "structural"

    def test_empty_string_is_unknown(self):
        assert classify_failure("") == "unknown"

    def test_case_insensitive(self):
        assert classify_failure("CONNECTION TIMEOUT") == "transient"


# -- cooldown_days() ---------------------------------------------------------
class TestCooldownDays:

    def test_transient_cooldown_is_short(self):
        """Transient failures should clear their cooldown quickly (~days)."""
        cd = cooldown_days("transient")
        assert 1 <= cd <= 10

    def test_structural_cooldown_is_long(self):
        """Structural failures should stay flagged much longer (~months)."""
        cd = cooldown_days("structural")
        assert cd >= 50

    def test_structural_outlasts_transient(self):
        assert cooldown_days("structural") > cooldown_days("transient")

    def test_cooldown_derived_from_same_lambda_as_decay(self):
        """
        This is the design property worth testing explicitly: cooldown
        is 1/lambda using the EXACT SAME lambda values used for ranking
        decay -- one tunable per class, not two independent knobs that
        could silently drift out of sync.
        """
        for cls, lam in LAMBDA_BY_CLASS.items():
            assert abs(cooldown_days(cls) - (1.0 / lam)) < 1e-9


# -- compute_verification_status() -------------------------------------------
class TestVerificationStatus:

    def test_no_failure_history_is_clean(self):
        v = compute_verification_status(
            last_outcome="success", last_ts=time.time(),
            failure_class=None, failure_count=0, success_count=1,
        )
        assert v.status == "CLEAN"

    def test_success_after_failures_is_resolved(self):
        """
        This is the fix for the 'permanent blacklist' doubt: if the most
        recent attempt succeeded, the path should be trusted again, even
        if it failed before.
        """
        v = compute_verification_status(
            last_outcome="success", last_ts=time.time(),
            failure_class="structural", failure_count=2, success_count=1,
        )
        assert v.status == "RESOLVED"

    def test_recent_transient_failure_is_active(self):
        recent = time.time() - 3600  # 1 hour ago
        v = compute_verification_status(
            last_outcome="failure", last_ts=recent,
            failure_class="transient", failure_count=1, success_count=0,
        )
        assert v.status == "ACTIVE_FAILURE"

    def test_old_transient_failure_moves_to_probation(self):
        """
        This is the core regression test for the second doubt: a
        transient failure old enough to be past its cooldown should
        move to PROBATION (worth retrying), not stay blocked forever.
        """
        old = time.time() - 10 * 86400  # 10 days ago -- well past transient cooldown (~3.3 days)
        v = compute_verification_status(
            last_outcome="failure", last_ts=old,
            failure_class="transient", failure_count=1, success_count=0,
        )
        assert v.status == "PROBATION"

    def test_recent_structural_failure_is_active_even_after_10_days(self):
        """
        The complement of the above: a STRUCTURAL failure at the same
        10-day age should still be ACTIVE_FAILURE, because structural
        cooldown (~100 days) is much longer than transient (~3.3 days).
        This is the whole point of class-aware decay.
        """
        ten_days_ago = time.time() - 10 * 86400
        v = compute_verification_status(
            last_outcome="failure", last_ts=ten_days_ago,
            failure_class="structural", failure_count=1, success_count=0,
        )
        assert v.status == "ACTIVE_FAILURE"

    def test_repeated_failures_past_cooldown_is_confirmed_broken(self):
        """
        A path that failed multiple times AND is past cooldown should
        be treated as stronger evidence (CONFIRMED_BROKEN), not just
        PROBATION -- repeated failure is more damning than a single one.
        """
        old = time.time() - 10 * 86400
        v = compute_verification_status(
            last_outcome="failure", last_ts=old,
            failure_class="transient", failure_count=3, success_count=0,
        )
        assert v.status == "CONFIRMED_BROKEN"

    def test_single_failure_past_cooldown_is_only_probation_not_confirmed(self):
        """Distinguishes PROBATION (1 failure, past cooldown) from
        CONFIRMED_BROKEN (2+ failures, past cooldown)."""
        old = time.time() - 10 * 86400
        v = compute_verification_status(
            last_outcome="failure", last_ts=old,
            failure_class="transient", failure_count=1, success_count=0,
        )
        assert v.status == "PROBATION"

    def test_confirmed_broken_eventually_escapes_to_probation(self):
        """
        Regression test for a real gap found by running
        eval/simulate_convergence.py: the original implementation had
        no escape path out of CONFIRMED_BROKEN -- verified directly
        that 200 simulated days after a 2nd failure, status was still
        CONFIRMED_BROKEN, forever. That made repeat failures behave
        exactly like a permanent blacklist, defeating the whole point
        of a verification state machine.

        Fixed with exponential backoff: 2nd failure requires 2x the
        base cooldown, 3rd requires 4x, etc. -- but there is always
        SOME amount of elapsed time that moves it to PROBATION.
        """
        transient_base_cooldown = cooldown_days("transient")  # ~3.33 days
        # 2nd failure -> effective cooldown = base * 2^1 = ~6.67 days.
        # At 5 days: still within the escalated cooldown -> CONFIRMED_BROKEN.
        five_days_ago = time.time() - 5 * 86400
        v_still_broken = compute_verification_status(
            last_outcome="failure", last_ts=five_days_ago,
            failure_class="transient", failure_count=2, success_count=0,
        )
        assert v_still_broken.status == "CONFIRMED_BROKEN"

        # At 200 days: FAR past the escalated cooldown, however large --
        # must have transitioned to PROBATION. This is the exact scenario
        # that was broken before the fix.
        two_hundred_days_ago = time.time() - 200 * 86400
        v_eventually_ok = compute_verification_status(
            last_outcome="failure", last_ts=two_hundred_days_ago,
            failure_class="transient", failure_count=2, success_count=0,
        )
        assert v_eventually_ok.status == "PROBATION"

    def test_cooldown_escalates_with_repeated_failures(self):
        """The effective cooldown reported in the status object should
        grow with failure_count -- this is what to check/log/tune if
        recovery feels too slow or too fast in a real deployment."""
        ts = time.time() - 1000  # recent, doesn't matter for this check
        v1 = compute_verification_status("failure", ts, "transient", 1, 0)
        v2 = compute_verification_status("failure", ts, "transient", 2, 0)
        v3 = compute_verification_status("failure", ts, "transient", 3, 0)
        assert v1.cooldown_days < v2.cooldown_days < v3.cooldown_days
        # Exactly 2x per additional failure (exponential backoff, base 2)
        assert abs(v2.cooldown_days / v1.cooldown_days - 2.0) < 1e-9
        assert abs(v3.cooldown_days / v2.cooldown_days - 2.0) < 1e-9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
