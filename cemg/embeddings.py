from __future__ import annotations

import math
import re
from typing import List, Dict, Any

_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _WORD_RE.findall((text or "").lower())


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot_product = sum(x * y for x, y in zip(v1, v2))
    norm_v1 = math.sqrt(sum(x * x for x in v1))
    norm_v2 = math.sqrt(sum(y * y for y in v2))
    if norm_v1 == 0.0 or norm_v2 == 0.0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)


class EmbeddingProvider:
    """
    Abstract base class for pluggable semantic relevance providers.
    """
    def compute_similarity(self, query: str, candidates: List[Dict[str, Any]]) -> List[float]:
        """
        Compute similarity scores between the query and each candidate dict.
        Each candidate is a dict containing Experience fields.
        Returns a list of float scores in the same order as candidates.
        """
        raise NotImplementedError


class TfidfCosineProvider(EmbeddingProvider):
    """
    Pure Python TF-IDF Cosine Similarity provider. Runs locally with zero
    external network dependencies. TF-IDF is computed dynamically over the
    FETCH_WINDOW candidate set, prioritizing rare/distinctive query terms.
    """
    def compute_similarity(self, query: str, candidates: List[Dict[str, Any]]) -> List[float]:
        if not query or not candidates:
            return [0.0] * len(candidates)

        # Extract corpus documents
        documents = []
        for c in candidates:
            parts = [
                c.get("action") or "",
                c.get("reasoning") or "",
                c.get("observed_error") or "",
                c.get("context_hint") or ""
            ]
            documents.append(" ".join(parts))

        # Tokenize documents and query
        doc_tokens = [tokenize(doc) for doc in documents]
        query_tokens = tokenize(query)

        # Build vocabulary
        vocab = sorted(list(set(term for tokens in doc_tokens for term in tokens) | set(query_tokens)))
        if not vocab:
            return [0.0] * len(candidates)

        term_to_idx = {term: idx for idx, term in enumerate(vocab)}

        # Compute Document Frequency
        N = len(candidates)
        df = {}
        for term in vocab:
            df[term] = sum(1 for tokens in doc_tokens if term in tokens)

        # Compute Inverse Document Frequency (smoothed)
        idf = {}
        for term in vocab:
            idf[term] = math.log(1.0 + N / (1.0 + df[term])) + 1.0

        # Helper to convert tokens to a TF-IDF vector
        def get_tfidf_vec(tokens: List[str]) -> List[float]:
            if not tokens:
                return [0.0] * len(vocab)
            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            doc_len = len(tokens)
            vec = [0.0] * len(vocab)
            for t, count in tf.items():
                if t in term_to_idx:
                    idx = term_to_idx[t]
                    vec[idx] = (count / doc_len) * idf[t]
            return vec

        # Generate vectors
        query_vec = get_tfidf_vec(query_tokens)
        doc_vecs = [get_tfidf_vec(tokens) for tokens in doc_tokens]

        # Calculate cosine similarity for each document
        return [cosine_similarity(query_vec, doc_vec) for doc_vec in doc_vecs]
