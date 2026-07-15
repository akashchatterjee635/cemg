"""
tests/test_core.py
-------------------
Unit tests for the three functions most central to the post-review fixes:

  1. decay()              -- must actually decrease over time (this was broken)
  2. keyword_overlap()     -- must actually differentiate relevant vs irrelevant text
  3. check_task_success()  -- must check the real artifact, not string-match chatter

These run with zero setup -- no Neo4j connection, no API key required --
because they test pure functions in isolation. This was flagged as a gap
in the earlier review (requirements.txt listed pytest, but no test files
existed); this file is the fix for that specific gap.

Run:
    pytest tests/ -v
"""

import math
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from cemg.graph import decay, keyword_overlap, LAMBDA
from eval.baselines import check_task_success


# ── decay() ────────────────────────────────────────────────────────────────
class TestDecay:

    def test_now_is_near_one(self):
        """A timestamp from right now should decay to ~1.0."""
        w = decay(time.time())
        assert 0.99 <= w <= 1.0

    def test_decreases_with_age(self):
        """
        This is the regression test for the critical bug: two experiences
        at different ages MUST produce different weights. The old code
        stored weight once at write time, so this would have failed --
        both values would have been ~1.0 regardless of age.
        """
        now      = time.time()
        recent   = now - 3600            # 1 hour ago
        old      = now - 30 * 86400      # 30 days ago
        very_old = now - 365 * 86400     # 1 year ago

        w_recent   = decay(recent)
        w_old      = decay(old)
        w_very_old = decay(very_old)

        assert w_recent > w_old > w_very_old, (
            f"Decay must be strictly decreasing with age: "
            f"got recent={w_recent:.4f} old={w_old:.4f} very_old={w_very_old:.4f}"
        )

    def test_matches_closed_form(self):
        """Sanity check against the exact exponential formula."""
        ts = time.time() - 10 * 86400   # 10 days ago
        expected = math.exp(-LAMBDA * 10)
        assert abs(decay(ts) - expected) < 1e-6

    def test_future_timestamp_clamped_to_one(self):
        """A timestamp in the future shouldn't produce weight > 1.0."""
        w = decay(time.time() + 100_000)
        assert w <= 1.0

    def test_custom_lambda_decays_faster(self):
        """Higher lambda should decay a fixed-age timestamp faster."""
        ts = time.time() - 20 * 86400
        w_default = decay(ts, lam=0.03)
        w_fast    = decay(ts, lam=0.10)
        assert w_fast < w_default


# ── keyword_overlap() ─────────────────────────────────────────────────────
class TestKeywordOverlap:

    def test_empty_query_returns_zero(self):
        assert keyword_overlap("", "some text here") == 0.0

    def test_no_overlap_returns_zero(self):
        score = keyword_overlap("weather forecast tomorrow", "database migration script")
        assert score == 0.0

    def test_identical_text_returns_one(self):
        score = keyword_overlap("read the config file", "read the config file")
        assert score == 1.0

    def test_partial_overlap_is_between_zero_and_one(self):
        """
        This is the regression test for the second critical bug:
        query_action was accepted as a parameter but never used, so
        relevant and irrelevant memories scored identically. This test
        would have failed against the old code path (which had no
        relevance signal at all).
        """
        score = keyword_overlap(
            "read the config file for the API",
            "read the settings file", "config was missing"
        )
        assert 0.0 < score < 1.0

    def test_more_shared_words_scores_higher(self):
        low  = keyword_overlap("read config file", "write log entry")
        high = keyword_overlap("read config file", "read the config settings file")
        assert high > low

    def test_is_case_insensitive(self):
        a = keyword_overlap("Read Config File", "read config file")
        assert a == 1.0


# ── check_task_success() ──────────────────────────────────────────────────
class TestCheckTaskSuccess:

    def test_missing_file_is_failure(self):
        result = check_task_success(output_path="/tmp/definitely_not_here_12345.txt")
        assert result["success"] is False
        assert "does not exist" in result["reason"]

    def test_file_with_enough_bullets_succeeds(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Summary:\n- point one is here\n- point two is here\n- point three is here\n")
            path = f.name
        try:
            result = check_task_success(output_path=path, min_bullets=3)
            assert result["success"] is True
            assert result["n_bullets"] == 3
        finally:
            os.remove(path)

    def test_file_with_too_few_bullets_fails(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("- only one point\n")
            path = f.name
        try:
            result = check_task_success(output_path=path, min_bullets=3)
            assert result["success"] is False
            assert result["n_bullets"] == 1
        finally:
            os.remove(path)

    def test_empty_file_fails(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            path = f.name
        try:
            result = check_task_success(output_path=path)
            assert result["success"] is False
        finally:
            os.remove(path)

    def test_numbered_list_counts_as_bullets(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("1. first point\n2. second point\n3. third point\n")
            path = f.name
        try:
            result = check_task_success(output_path=path, min_bullets=3)
            assert result["success"] is True
        finally:
            os.remove(path)

    def test_string_match_hack_would_have_falsely_passed(self):
        """
        Documents the exact bug that was fixed: the old success check was
        `"session" not in answer.lower()`. This test shows why that was
        wrong -- a completely empty/failed output with no artifact would
        have passed that check, but correctly fails check_task_success().
        """
        fake_agent_answer = "I have finished the task successfully."
        old_buggy_check = "session" not in fake_agent_answer.lower()
        assert old_buggy_check is True   # the old check says "success"

        # but no real artifact was ever created:
        result = check_task_success(output_path="/tmp/definitely_not_here_98765.txt")
        assert result["success"] is False   # the real check correctly disagrees


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
