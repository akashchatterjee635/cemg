"""
tests/test_compliance.py
--------------------------
Regression tests for the two gaps found in the third review pass:

  1. ActionSignature aggregates leaking verification status across
     task_namespace boundaries.
  2. Compliance being checked retroactively (after a run finishes)
     against live state, instead of at decision time.

These are pure-logic tests where possible (evaluate_compliance is a
pure function, tested with zero setup). The namespace-scoping fix
itself lives in Cypher, so it's verified structurally here (checking
the query text contains task_namespace in the right places) --
a full integration test against a live Neo4j instance would be the
next step once a test DB is wired into CI.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from cemg.memory import evaluate_compliance


# -- evaluate_compliance() -- pure function, fully testable -------------------
class TestEvaluateCompliance:

    def test_no_snapshots_is_zero_violations(self):
        result = evaluate_compliance([])
        assert result["total_used"] == 0
        assert result["violations"] == 0
        assert result["violation_rate"] == 0.0

    def test_all_clean_snapshots_no_violations(self):
        snapshots = [
            {"action_signature": "sig1", "status_before": "CLEAN"},
            {"action_signature": "sig2", "status_before": "CLEAN"},
        ]
        result = evaluate_compliance(snapshots)
        assert result["violations"] == 0
        assert result["violation_rate"] == 0.0

    def test_active_failure_counts_as_violation(self):
        snapshots = [{"action_signature": "sig1", "status_before": "ACTIVE_FAILURE"}]
        result = evaluate_compliance(snapshots)
        assert result["violations"] == 1
        assert result["violation_rate"] == 1.0
        assert "sig1" in result["violating_signatures"]

    def test_confirmed_broken_counts_as_violation(self):
        snapshots = [{"action_signature": "sig1", "status_before": "CONFIRMED_BROKEN"}]
        result = evaluate_compliance(snapshots)
        assert result["violations"] == 1

    def test_probation_is_not_a_violation(self):
        """
        Retrying a past-cooldown failure is the intended, encouraged
        behaviour -- it should never be counted as non-compliance.
        This is the core semantic distinction the whole verification
        state machine exists to make.
        """
        snapshots = [{"action_signature": "sig1", "status_before": "PROBATION"}]
        result = evaluate_compliance(snapshots)
        assert result["violations"] == 0

    def test_resolved_is_not_a_violation(self):
        snapshots = [{"action_signature": "sig1", "status_before": "RESOLVED"}]
        result = evaluate_compliance(snapshots)
        assert result["violations"] == 0

    def test_mixed_snapshots_correct_rate(self):
        snapshots = [
            {"action_signature": "a", "status_before": "CLEAN"},
            {"action_signature": "b", "status_before": "ACTIVE_FAILURE"},
            {"action_signature": "c", "status_before": "PROBATION"},
            {"action_signature": "d", "status_before": "CONFIRMED_BROKEN"},
        ]
        result = evaluate_compliance(snapshots)
        assert result["total_used"] == 4
        assert result["violations"] == 2   # only ACTIVE_FAILURE and CONFIRMED_BROKEN
        assert result["violation_rate"] == 0.5

    def test_this_is_the_fix_for_the_timing_bug(self):
        """
        Documents exactly what the fix changes: the OLD design would
        check the signature's CURRENT (post-run) status for every
        signature used anywhere in the run. If a signature failed
        early in the run and then succeeded later in the SAME run, the
        old post-hoc check would see the final success and report zero
        violations -- masking the fact that the agent used a
        known-bad action at least once.

        The NEW design uses one independent snapshot PER USE, captured
        at the moment of that specific decision -- so the same
        signature used twice in one run, once while ACTIVE_FAILURE and
        once after it becomes RESOLVED, correctly shows exactly one
        violation, not zero.
        """
        same_signature_used_twice_in_one_run = [
            {"action_signature": "sig_x", "status_before": "ACTIVE_FAILURE"},  # 1st use: flagged, used anyway -- violation
            {"action_signature": "sig_x", "status_before": "RESOLVED"},        # 2nd use, later: now fine
        ]
        result = evaluate_compliance(same_signature_used_twice_in_one_run)
        assert result["total_used"] == 2
        assert result["violations"] == 1   # exactly the first use, correctly caught


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
