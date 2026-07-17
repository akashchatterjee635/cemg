import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from cemg.storage import InMemoryStorage, SqliteStorage, get_storage_provider


@pytest.fixture
def sqlite_store(tmp_path):
    db_file = tmp_path / "test_cemg.db"
    store = SqliteStorage(str(db_file))
    yield store
    store.close()


@pytest.fixture
def memory_store():
    store = InMemoryStorage()
    yield store
    store.close()


def test_storage_factory_default(monkeypatch):
    monkeypatch.setenv("CEMG_STORAGE_TYPE", "sqlite")
    monkeypatch.setenv("CEMG_SQLITE_PATH", ":memory:")  # In-memory SQLite
    provider = get_storage_provider()
    assert isinstance(provider, SqliteStorage)
    assert provider.is_healthy()
    provider.close()


def test_storage_factory_memory(monkeypatch):
    monkeypatch.setenv("CEMG_STORAGE_TYPE", "memory")
    provider = get_storage_provider()
    assert isinstance(provider, InMemoryStorage)
    assert provider.is_healthy()
    provider.close()


@pytest.mark.parametrize("store_type", ["sqlite", "memory"])
def test_unified_storage_contract(store_type, sqlite_store, memory_store):
    store = sqlite_store if store_type == "sqlite" else memory_store
    
    # 1. Verify healthy on startup
    assert store.is_healthy()
    
    # 2. Write experiences
    res1 = store.write_experience(
        agent_id="test_agent",
        session_id="session_1",
        action="read_file({'path': 'a.txt'})",
        outcome="failure",
        observed_error="FileNotFoundError: file not found",
        context_hint="read_file",
        tool="read_file",
        params={"path": "a.txt"},
        task_namespace="task_A"
    )
    assert "exp_id" in res1
    assert res1["action_signature"] is not None
    assert res1["failure_class"] == "structural"
    
    # Write a parent relationship experience
    res2 = store.write_experience(
        agent_id="test_agent",
        session_id="session_1",
        action="write_file({'path': 'a.txt', 'text': 'content'})",
        outcome="success",
        context_hint="write_file",
        tool="write_file",
        params={"path": "a.txt", "text": "content"},
        task_namespace="task_A",
        parent_exp_id=res1["exp_id"]
    )
    
    # 3. Read relevant candidate query
    relevant = store.read_relevant(
        agent_id="test_agent",
        query_action="read file path",
        task_namespace="task_A",
        include_failures=True
    )
    assert len(relevant) == 2
    # Check that scores and status are computed correctly
    assert any(r["id"] == res1["exp_id"] for r in relevant)
    assert any(r["id"] == res2["exp_id"] for r in relevant)
    
    # Structural failure signature status check
    failure_rec = [r for r in relevant if r["id"] == res1["exp_id"]][0]
    assert failure_rec["verification_status"] == "ACTIVE_FAILURE"
    
    # 4. Read signature status explicitly
    sig_status = store.read_signature_status(
        agent_id="test_agent",
        signature=res1["action_signature"],
        task_namespace="task_A"
    )
    assert sig_status is not None
    assert sig_status["verification_status"] == "ACTIVE_FAILURE"
    assert sig_status["failure_count"] == 1
    
    # 5. Read causal path
    path = store.read_causal_path(res2["exp_id"])
    assert len(path) == 2
    assert path[0]["id"] == res1["exp_id"]
    assert path[1]["id"] == res2["exp_id"]
    
    # 6. Test pruning stale records
    # Set a time in the past to trigger decay-based pruning eligibility
    old_ts = time.time() - (200 * 86400) # 200 days ago
    
    # Write a success experience way in the past (decays below floor)
    res_stale = store.write_experience(
        agent_id="test_agent",
        session_id="session_1",
        action="some_action()",
        outcome="success",
        task_namespace="task_A",
        ts=old_ts
    )
    
    # Check that it gets pruned
    prune_res = store.prune_stale_experiences(agent_id="test_agent", floor=0.02, dry_run=False)
    assert res_stale["exp_id"] in prune_res["ids"]
    assert prune_res["deleted"] is True
    
    # Verify it is gone from database queries
    relevant_after = store.read_relevant(agent_id="test_agent", task_namespace="task_A")
    assert not any(r["id"] == res_stale["exp_id"] for r in relevant_after)
