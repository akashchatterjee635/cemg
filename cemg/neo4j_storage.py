from __future__ import annotations

import os
import time
import uuid
import math
from typing import Optional, List, Dict, Any

from neo4j import GraphDatabase, Driver

from cemg.storage import BaseStorage, _local_decay
from cemg.classify import classify_failure, compute_verification_status
from cemg.embeddings import EmbeddingProvider, TfidfCosineProvider
from cemg.security import sanitize_external_content, is_external_source


class Neo4jStorage(BaseStorage):
    """
    Neo4j implementation of the BaseStorage provider.
    Handles all Cypher queries, transaction scopes, and indexing.
    """
    def __init__(self, driver: Optional[Driver] = None):
        self._own_driver = False
        if driver:
            self.driver = driver
        else:
            uri  = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
            user = os.getenv("NEO4J_USER",     "neo4j")
            pwd  = os.getenv("NEO4J_PASSWORD", "neo4j")
            self.driver = GraphDatabase.driver(uri, auth=(user, pwd))
            self._own_driver = True
        
        # Idempotent schema boot if connection is active
        if self.is_healthy():
            self._bootstrap_schema()

    def _bootstrap_schema(self) -> None:
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
        with self.driver.session() as s:
            for stmt in stmts:
                try:
                    s.run(stmt)
                except Exception:
                    pass
        print("[CEMG] Neo4j schema ready")

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
        
        # Calculate decay weights using common local helper
        w_at_write = _local_decay(ts, failure_class)

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

        with self.driver.session() as sess:
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
        params = {"agent_id": agent_id, "fetch_window": fetch_window}
        if task_namespace:
            params["task_namespace"] = task_namespace

        with self.driver.session() as sess:
            result = sess.run(cypher, params)
            raw_rows = [dict(r) for r in result]

        provider = embedding_provider or TfidfCosineProvider()
        if query_action and raw_rows:
            relevances = provider.compute_similarity(query_action, raw_rows)
        else:
            relevances = [0.0] * len(raw_rows)

        scored: list[dict] = []
        for r, rel in zip(raw_rows, relevances):
            w_now = _local_decay(r["ts"], r.get("failure_class"))
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

    def read_causal_path(self, exp_id: str, max_depth: int = 10) -> list[dict]:
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
        with self.driver.session() as sess:
            result = sess.run(cypher, {"exp_id": exp_id})
            return [dict(r) for r in result]

    def read_signature_status(self, agent_id: str, signature: str, task_namespace: str) -> Optional[dict]:
        cypher = """
        MATCH (sig:ActionSignature {signature: $signature, agent_id: $agent_id, task_namespace: $task_namespace})
        RETURN sig.failure_count AS failure_count, sig.success_count AS success_count,
               sig.last_outcome AS last_outcome, sig.last_ts AS last_ts,
               sig.failure_class AS failure_class
        """
        with self.driver.session() as sess:
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

    def prune_stale_experiences(self, agent_id: Optional[str] = None, floor: float = 0.02, dry_run: bool = True) -> dict:
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

        with self.driver.session() as sess:
            rows = [dict(r) for r in sess.run(cypher, params)]

        to_delete = []
        for r in rows:
            w = _local_decay(r["ts"], r.get("failure_class"))
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
                continue
            to_delete.append(r["id"])

        if not dry_run and to_delete:
            with self.driver.session() as sess:
                sess.run(
                    "MATCH (e:Experience) WHERE e.id IN $ids DETACH DELETE e",
                    {"ids": to_delete},
                )

        return {"eligible_count": len(to_delete), "deleted": (not dry_run), "ids": to_delete}

    def is_healthy(self, timeout_s: float = 2.0) -> bool:
        try:
            with self.driver.session() as s:
                s.run("RETURN 1", timeout=timeout_s).consume()
            return True
        except Exception:
            return False

    def close(self) -> None:
        if self._own_driver:
            self.driver.close()
