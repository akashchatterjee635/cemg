from __future__ import annotations

import math
import os
import sqlite3
import time
import uuid
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

from cemg.classify import classify_failure, compute_verification_status, generalize_params
from cemg.embeddings import TfidfCosineProvider, EmbeddingProvider
from cemg.security import sanitize_external_content, is_external_source


# -- Base Interface -----------------------------------------------------------
class BaseStorage(ABC):
    """
    Abstract interface for pluggable storage providers (Neo4j, SQLite, In-Memory).
    """
    @abstractmethod
    def write_experience(
        self,
        agent_id: str,
        session_id: str,
        action: str,
        outcome: str,
        reasoning: str = "",
        observed_error: str = "",
        context_hint: str = "",
        tool: str = "",
        params: Optional[dict] = None,
        task_namespace: str = "default",
        cost_tokens: int = 0,
        parent_exp_id: Optional[str] = None,
        ts: Optional[float] = None,
    ) -> dict:
        pass

    @abstractmethod
    def read_relevant(
        self,
        agent_id: str,
        query_action: str = "",
        task_namespace: Optional[str] = None,
        include_failures: bool = True,
        top_k: int = 10,
        fail_boost: float = 2.0,
        relevance_weight: float = 1.5,
        embedding_provider: Optional[EmbeddingProvider] = None,
        fetch_window: int = 500,
    ) -> list[dict]:
        pass

    @abstractmethod
    def read_causal_path(self, exp_id: str, max_depth: int = 10) -> list[dict]:
        pass

    @abstractmethod
    def read_signature_status(self, agent_id: str, signature: str, task_namespace: str) -> Optional[dict]:
        pass

    @abstractmethod
    def prune_stale_experiences(self, agent_id: Optional[str] = None, floor: float = 0.02, dry_run: bool = True) -> dict:
        pass

    @abstractmethod
    def is_healthy(self) -> bool:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


# -- Helper functions for local scoring ----------------------------------------
def _local_decay(ts_unix: float, failure_class: Optional[str], lam_default: float = 0.03) -> float:
    from cemg.classify import LAMBDA_BY_CLASS
    lam = LAMBDA_BY_CLASS.get(failure_class, lam_default) if failure_class else lam_default
    delta_days = (time.time() - ts_unix) / 86400.0
    return math.exp(-lam * max(delta_days, 0.0))


# -- In-Memory Storage Provider ------------------------------------------------
class InMemoryStorage(BaseStorage):
    """
    Volatile, in-memory graph store fallback. Ideal for unit tests, local CLI
    simulations, and ephemeral runs. Requires zero dependencies and zero setup.
    """
    def __init__(self):
        self.experiences: Dict[str, Dict[str, Any]] = {}
        self.signatures: Dict[str, Dict[str, Any]] = {}  # key: (sig, agent_id, ns)

    def write_experience(
        self,
        agent_id: str,
        session_id: str,
        action: str,
        outcome: str,
        reasoning: str = "",
        observed_error: str = "",
        context_hint: str = "",
        tool: str = "",
        params: Optional[dict] = None,
        task_namespace: str = "default",
        cost_tokens: int = 0,
        parent_exp_id: Optional[str] = None,
        ts: Optional[float] = None,
    ) -> dict:
        from cemg.graph import make_action_signature

        exp_id = str(uuid.uuid4())
        ts = ts or time.time()
        params = params or {}
        tool = tool or context_hint or "unknown_tool"
        sig = make_action_signature(tool, params)

        if is_external_source(context_hint):
            reasoning = sanitize_external_content(reasoning)
            observed_error = sanitize_external_content(observed_error)

        failure_class = classify_failure(observed_error) if outcome == "failure" else None

        # Store experience
        exp = {
            "id": exp_id,
            "agent_id": agent_id,
            "session_id": session_id,
            "task_namespace": task_namespace,
            "action": action,
            "outcome": outcome,
            "reasoning": reasoning,
            "observed_error": observed_error,
            "context_hint": context_hint,
            "action_signature": sig,
            "failure_class": failure_class,
            "cost_tokens": cost_tokens,
            "ts": ts,
            "parent_exp_id": parent_exp_id
        }
        self.experiences[exp_id] = exp

        # Update signature
        sig_key = (sig, agent_id, task_namespace)
        sig_rec = self.signatures.setdefault(sig_key, {
            "signature": sig,
            "agent_id": agent_id,
            "task_namespace": task_namespace,
            "tool": tool,
            "failure_count": 0,
            "success_count": 0,
            "last_outcome": None,
            "last_ts": None,
            "failure_class": None
        })
        sig_rec["last_outcome"] = outcome
        sig_rec["last_ts"] = ts
        if failure_class:
            sig_rec["failure_class"] = failure_class
        if outcome == "failure":
            sig_rec["failure_count"] += 1
        elif outcome == "success":
            sig_rec["success_count"] += 1

        return {"exp_id": exp_id, "action_signature": sig, "failure_class": failure_class}

    def read_relevant(
        self,
        agent_id: str,
        query_action: str = "",
        task_namespace: Optional[str] = None,
        include_failures: bool = True,
        top_k: int = 10,
        fail_boost: float = 2.0,
        relevance_weight: float = 1.5,
        embedding_provider: Optional[EmbeddingProvider] = None,
        fetch_window: int = 500,
    ) -> list[dict]:
        # Filter candidate experiences
        candidates = []
        for e in self.experiences.values():
            if e["agent_id"] != agent_id:
                continue
            if task_namespace and e["task_namespace"] != task_namespace:
                continue
            if not include_failures and e["outcome"] == "failure":
                continue
            candidates.append(e)

        # Sort candidate experiences (newest first)
        candidates.sort(key=lambda x: x["ts"], reverse=True)
        candidates = candidates[:fetch_window]

        if not candidates:
            return []

        # Calculate semantic similarity
        provider = embedding_provider or TfidfCosineProvider()
        if query_action:
            relevances = provider.compute_similarity(query_action, candidates)
        else:
            relevances = [0.0] * len(candidates)

        scored = []
        for r, rel in zip(candidates, relevances):
            w_now = _local_decay(r["ts"], r["failure_class"])
            boost = fail_boost if r["outcome"] == "failure" else 1.0
            score = w_now * boost * (1 + relevance_weight * rel)

            # Get signature stats
            sig_key = (r["action_signature"], agent_id, r["task_namespace"])
            sig_rec = self.signatures.get(sig_key)
            if sig_rec:
                vstatus = compute_verification_status(
                    last_outcome=sig_rec["last_outcome"],
                    last_ts=sig_rec["last_ts"],
                    failure_class=sig_rec["failure_class"],
                    failure_count=sig_rec["failure_count"],
                    success_count=sig_rec["success_count"]
                )
            else:
                vstatus = compute_verification_status(
                    last_outcome=r["outcome"],
                    last_ts=r["ts"],
                    failure_class=r["failure_class"],
                    failure_count=1 if r["outcome"] == "failure" else 0,
                    success_count=1 if r["outcome"] == "success" else 0
                )

            scored.append({
                **r,
                "weight": w_now,
                "relevance": rel,
                "score": score,
                "verification_status": vstatus.status,
                # compatibility mappings
                "sig_failure_count": sig_rec["failure_count"] if sig_rec else 0,
                "sig_success_count": sig_rec["success_count"] if sig_rec else 0,
                "sig_last_outcome": sig_rec["last_outcome"] if sig_rec else None,
                "sig_last_ts": sig_rec["last_ts"] if sig_rec else None,
                "sig_failure_class": sig_rec["failure_class"] if sig_rec else None
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def read_causal_path(self, exp_id: str, max_depth: int = 10) -> list[dict]:
        chain = []
        curr_id = exp_id
        for _ in range(max_depth):
            e = self.experiences.get(curr_id)
            if not e:
                break
            chain.append(e)
            curr_id = e.get("parent_exp_id")
            if not curr_id:
                break
        # Return chronologically (oldest first)
        chain.reverse()
        return chain

    def read_signature_status(self, agent_id: str, signature: str, task_namespace: str) -> Optional[dict]:
        sig_key = (signature, agent_id, task_namespace)
        row = self.signatures.get(sig_key)
        if not row:
            return None
        vstatus = compute_verification_status(
            last_outcome=row["last_outcome"],
            last_ts=row["last_ts"],
            failure_class=row["failure_class"],
            failure_count=row["failure_count"],
            success_count=row["success_count"]
        )
        return {**row, "verification_status": vstatus.status}

    def prune_stale_experiences(self, agent_id: Optional[str] = None, floor: float = 0.02, dry_run: bool = True) -> dict:
        to_delete = []
        for eid, r in list(self.experiences.items()):
            if agent_id and r["agent_id"] != agent_id:
                continue
            w = _local_decay(r["ts"], r["failure_class"])
            if w >= floor:
                continue

            # Verify status is not actively tracked
            sig_key = (r["action_signature"], r["agent_id"], r["task_namespace"])
            sig_rec = self.signatures.get(sig_key)
            if sig_rec:
                vstatus = compute_verification_status(
                    last_outcome=sig_rec["last_outcome"],
                    last_ts=sig_rec["last_ts"],
                    failure_class=sig_rec["failure_class"],
                    failure_count=sig_rec["failure_count"],
                    success_count=sig_rec["success_count"]
                )
                if vstatus.status in ("ACTIVE_FAILURE", "PROBATION", "CONFIRMED_BROKEN"):
                    continue

            to_delete.append(eid)

        if not dry_run and to_delete:
            for eid in to_delete:
                del self.experiences[eid]

        return {"eligible_count": len(to_delete), "deleted": (not dry_run), "ids": to_delete}

    def is_healthy(self) -> bool:
        return True

    def close(self) -> None:
        pass


# -- SQLite Storage Provider ---------------------------------------------------
class SqliteStorage(BaseStorage):
    """
    Standard, file-based SQLite database. Persists experiences locally to
    cemg_memory.db with zero dependencies. Runs out-of-the-box on any computer.
    """
    def __init__(self, db_path: str = "cemg_memory.db"):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        # Return a fresh connection to handle multi-threaded FastAPI contexts safely
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiences (
            id TEXT PRIMARY KEY,
            agent_id TEXT,
            session_id TEXT,
            task_namespace TEXT,
            action TEXT,
            outcome TEXT,
            reasoning TEXT,
            observed_error TEXT,
            context_hint TEXT,
            action_signature TEXT,
            failure_class TEXT,
            cost_tokens INTEGER,
            ts REAL,
            parent_exp_id TEXT
        )
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS action_signatures (
            signature TEXT,
            agent_id TEXT,
            task_namespace TEXT,
            tool TEXT,
            failure_count INTEGER,
            success_count INTEGER,
            last_outcome TEXT,
            last_ts REAL,
            failure_class TEXT,
            PRIMARY KEY (signature, agent_id, task_namespace)
        )
        """)
        # Indexes for fast retrieval
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_exp_agent_ts ON experiences (agent_id, ts)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_exp_agent_ns ON experiences (agent_id, task_namespace)")
        conn.commit()
        conn.close()

    def write_experience(
        self,
        agent_id: str,
        session_id: str,
        action: str,
        outcome: str,
        reasoning: str = "",
        observed_error: str = "",
        context_hint: str = "",
        tool: str = "",
        params: Optional[dict] = None,
        task_namespace: str = "default",
        cost_tokens: int = 0,
        parent_exp_id: Optional[str] = None,
        ts: Optional[float] = None,
    ) -> dict:
        from cemg.graph import make_action_signature

        exp_id = str(uuid.uuid4())
        ts = ts or time.time()
        params = params or {}
        tool = tool or context_hint or "unknown_tool"
        sig = make_action_signature(tool, params)

        if is_external_source(context_hint):
            reasoning = sanitize_external_content(reasoning)
            observed_error = sanitize_external_content(observed_error)

        failure_class = classify_failure(observed_error) if outcome == "failure" else None

        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            # Insert experience
            cursor.execute("""
            INSERT INTO experiences (
                id, agent_id, session_id, task_namespace, action, outcome,
                reasoning, observed_error, context_hint, action_signature,
                failure_class, cost_tokens, ts, parent_exp_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                exp_id, agent_id, session_id, task_namespace, action, outcome,
                reasoning, observed_error, context_hint, sig,
                failure_class, cost_tokens, ts, parent_exp_id
            ))

            # Upsert ActionSignature aggregate
            cursor.execute("""
            INSERT INTO action_signatures (
                signature, agent_id, task_namespace, tool, failure_count, success_count,
                last_outcome, last_ts, failure_class
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(signature, agent_id, task_namespace) DO UPDATE SET
                last_outcome = excluded.last_outcome,
                last_ts = excluded.last_ts,
                failure_class = coalesce(excluded.failure_class, action_signatures.failure_class),
                failure_count = action_signatures.failure_count + CASE WHEN excluded.last_outcome = 'failure' THEN 1 ELSE 0 END,
                success_count = action_signatures.success_count + CASE WHEN excluded.last_outcome = 'success' THEN 1 ELSE 0 END
            """, (
                sig, agent_id, task_namespace, tool,
                1 if outcome == "failure" else 0,
                1 if outcome == "success" else 0,
                outcome, ts, failure_class
            ))
            conn.commit()
        finally:
            conn.close()

        return {"exp_id": exp_id, "action_signature": sig, "failure_class": failure_class}

    def read_relevant(
        self,
        agent_id: str,
        query_action: str = "",
        task_namespace: Optional[str] = None,
        include_failures: bool = True,
        top_k: int = 10,
        fail_boost: float = 2.0,
        relevance_weight: float = 1.5,
        embedding_provider: Optional[EmbeddingProvider] = None,
        fetch_window: int = 500,
    ) -> list[dict]:
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            query_parts = [
                "SELECT e.*, ",
                "  sig.failure_count AS sig_failure_count, ",
                "  sig.success_count AS sig_success_count, ",
                "  sig.last_outcome AS sig_last_outcome, ",
                "  sig.last_ts AS sig_last_ts, ",
                "  sig.failure_class AS sig_failure_class ",
                "FROM experiences e ",
                "LEFT JOIN action_signatures sig ON e.action_signature = sig.signature ",
                "  AND e.agent_id = sig.agent_id AND e.task_namespace = sig.task_namespace ",
                "WHERE e.agent_id = ? "
            ]
            params = [agent_id]

            if task_namespace:
                query_parts.append("AND e.task_namespace = ? ")
                params.append(task_namespace)

            if not include_failures:
                query_parts.append("AND e.outcome <> 'failure' ")

            query_parts.append("ORDER BY e.ts DESC LIMIT ?")
            params.append(fetch_window)

            cursor.execute(" ".join(query_parts), params)
            rows = [dict(r) for r in cursor.fetchall()]
        finally:
            conn.close()

        if not rows:
            return []

        # Local scoring via Python similarity and decay
        provider = embedding_provider or TfidfCosineProvider()
        if query_action:
            relevances = provider.compute_similarity(query_action, rows)
        else:
            relevances = [0.0] * len(rows)

        scored = []
        for r, rel in zip(rows, relevances):
            w_now = _local_decay(r["ts"], r["failure_class"])
            boost = fail_boost if r["outcome"] == "failure" else 1.0
            score = w_now * boost * (1 + relevance_weight * rel)

            vstatus = compute_verification_status(
                last_outcome=r["sig_last_outcome"] or r["outcome"],
                last_ts=r["sig_last_ts"] or r["ts"],
                failure_class=r["sig_failure_class"] or r["failure_class"],
                failure_count=r["sig_failure_count"] or (1 if r["outcome"] == "failure" else 0),
                success_count=r["sig_success_count"] or (1 if r["outcome"] == "success" else 0)
            )

            scored.append({
                **r,
                "weight": w_now,
                "relevance": rel,
                "score": score,
                "verification_status": vstatus.status
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def read_causal_path(self, exp_id: str, max_depth: int = 10) -> list[dict]:
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            chain = []
            curr_id = exp_id
            for _ in range(max_depth):
                cursor.execute("SELECT * FROM experiences WHERE id = ?", (curr_id,))
                row = cursor.fetchone()
                if not row:
                    break
                r = dict(row)
                chain.append(r)
                curr_id = r.get("parent_exp_id")
                if not curr_id:
                    break
        finally:
            conn.close()
        # Return chronologically (oldest first)
        chain.reverse()
        return chain

    def read_signature_status(self, agent_id: str, signature: str, task_namespace: str) -> Optional[dict]:
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute("""
            SELECT * FROM action_signatures 
            WHERE signature = ? AND agent_id = ? AND task_namespace = ?
            """, (signature, agent_id, task_namespace))
            row = cursor.fetchone()
            if not row:
                return None
            r = dict(row)
            vstatus = compute_verification_status(
                last_outcome=r["last_outcome"],
                last_ts=r["last_ts"],
                failure_class=r["failure_class"],
                failure_count=r["failure_count"],
                success_count=r["success_count"]
            )
            return {**r, "verification_status": vstatus.status}
        finally:
            conn.close()

    def prune_stale_experiences(self, agent_id: Optional[str] = None, floor: float = 0.02, dry_run: bool = True) -> dict:
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            # Query candidate experiences to check
            if agent_id:
                cursor.execute("""
                SELECT e.id, e.ts, e.failure_class, e.action_signature, e.agent_id, e.task_namespace,
                       sig.last_outcome, sig.last_ts, sig.failure_class AS sig_failure_class,
                       sig.failure_count, sig.success_count
                FROM experiences e
                LEFT JOIN action_signatures sig ON e.action_signature = sig.signature 
                  AND e.agent_id = sig.agent_id AND e.task_namespace = sig.task_namespace
                WHERE e.agent_id = ?
                """, (agent_id,))
            else:
                cursor.execute("""
                SELECT e.id, e.ts, e.failure_class, e.action_signature, e.agent_id, e.task_namespace,
                       sig.last_outcome, sig.last_ts, sig.failure_class AS sig_failure_class,
                       sig.failure_count, sig.success_count
                FROM experiences e
                LEFT JOIN action_signatures sig ON e.action_signature = sig.signature 
                  AND e.agent_id = sig.agent_id AND e.task_namespace = sig.task_namespace
                """)
            rows = [dict(r) for r in cursor.fetchall()]

            to_delete = []
            for r in rows:
                w = _local_decay(r["ts"], r["failure_class"])
                if w >= floor:
                    continue

                # Verify status is not actively tracked
                vstatus = compute_verification_status(
                    last_outcome=r["last_outcome"] or "success",
                    last_ts=r["last_ts"] or r["ts"],
                    failure_class=r["sig_failure_class"] or r["failure_class"],
                    failure_count=r["failure_count"] or 0,
                    success_count=r["success_count"] or 0
                )
                if vstatus.status in ("ACTIVE_FAILURE", "PROBATION", "CONFIRMED_BROKEN"):
                    continue

                to_delete.append(r["id"])

            if not dry_run and to_delete:
                # Batch delete in SQLite
                cursor.execute(f"DELETE FROM experiences WHERE id IN ({','.join(['?']*len(to_delete))})", to_delete)
                conn.commit()
        finally:
            conn.close()

        return {"eligible_count": len(to_delete), "deleted": (not dry_run), "ids": to_delete}

    def is_healthy(self) -> bool:
        try:
            conn = self._get_conn()
            conn.execute("SELECT 1")
            conn.close()
            return True
        except Exception:
            return False

    def close(self) -> None:
        pass


# -- Fallback Manager Factory --------------------------------------------------
def get_storage_provider() -> BaseStorage:
    """
    Initializes and returns the storage provider configured in .env.
    Defaults to SQLite for local development. Handles automatic fallback to
    SQLite if Neo4j is selected but unreachable.
    """
    storage_type = os.getenv("CEMG_STORAGE_TYPE", "sqlite").lower()
    
    if storage_type == "memory":
        return InMemoryStorage()
        
    elif storage_type == "neo4j":
        # Import Neo4j provider dynamically
        from cemg.neo4j_storage import Neo4jStorage
        provider = Neo4jStorage()
        if provider.is_healthy():
            return provider
        else:
            print("[CEMG] WARNING: Neo4j database unreachable. Falling back to local SQLite database.")
            # Fall through to SQLite

    # Default to SQLite
    db_path = os.getenv("CEMG_SQLITE_PATH", "cemg_memory.db")
    return SqliteStorage(db_path)
