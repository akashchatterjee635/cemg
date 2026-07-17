import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from cemg.embeddings import cosine_similarity, TfidfCosineProvider


class TestCosineSimilarity:

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_identical_vectors(self):
        assert abs(cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) - 1.0) < 1e-9

    def test_opposite_vectors(self):
        assert abs(cosine_similarity([1.0, 1.0], [-1.0, -1.0]) - (-1.0)) < 1e-9

    def test_zero_vector_returns_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


class TestTfidfCosineProvider:

    def test_empty_inputs(self):
        provider = TfidfCosineProvider()
        assert provider.compute_similarity("", [{"action": "test"}]) == [0.0]
        assert provider.compute_similarity("test", []) == []

    def test_perfect_match(self):
        provider = TfidfCosineProvider()
        candidates = [
            {"action": "read the config file from directory"},
            {"action": "write a summary of the analysis report"}
        ]
        scores = provider.compute_similarity("read the config file from directory", candidates)
        assert len(scores) == 2
        assert abs(scores[0] - 1.0) < 1e-9
        assert scores[1] < 0.2

    def test_idf_weighting(self):
        """
        Verify that rare terms weigh more than common/frequent terms.
        "common" appears in both docs, but "rare_term_a" and "rare_term_b" are unique.
        A query for "common rare_term_a" should score higher on doc 1 than doc 2,
        because rare_term_a is rare in the corpus.
        """
        provider = TfidfCosineProvider()
        candidates = [
            {"action": "this is common and contains rare_term_a"},
            {"action": "this is common and contains rare_term_b"}
        ]
        # Query for "common rare_term_a"
        scores = provider.compute_similarity("common rare_term_a", candidates)
        assert scores[0] > scores[1]

    def test_unrelated_query(self):
        provider = TfidfCosineProvider()
        candidates = [
            {"action": "read the config file"},
            {"action": "write a summary"}
        ]
        scores = provider.compute_similarity("xyz abc", candidates)
        assert all(s == 0.0 for s in scores)
