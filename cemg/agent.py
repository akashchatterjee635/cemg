"""
cemg/agent.py
-------------
A CEMG-augmented agent loop.

FIX LOG v2 (this pass):
  - Memory retrieval is now wrapped in a graceful-degradation guard:
    if Neo4j is unreachable, is_healthy() catches it and the agent
    runs with an empty memory block instead of crashing. Memory is a
    best-effort enhancement, not a hard dependency.
  - The tool executor now returns THREE things per failure, not two:
    a human-readable result, the outcome, and the raw observed_error
    text -- kept separate from whatever the agent later "reasons" about
    why it failed. This is the fix for the confabulation problem: the
    stored observed_error is what actually happened, unedited.
  - External tool results (web_search, read_file) are tagged with a
    context_hint so cemg.security's sanitiser knows to scrub them
    before storage -- the mitigation for stored prompt injection.
  - Cost is tracked (rough token-count proxy) and stored per experience,
    surfaced back to the LLM alongside reliability -- addressing the
    "agent reroutes to a cheaper-but-less-reliable path because it
    can't see the tradeoff" problem.
  - The agent now snapshots verification status BEFORE each action
    executes (self.decision_snapshots), via cemg.memory.peek_signature_status,
    instead of only tracking which signatures were used and checking
    their status after the fact. This fixes a timing bug: checking
    status retroactively after a full run can miss a genuine violation
    if the same signature's state changed later in the same run.
  - task_namespace is a required constructor field (defaults to
    "default") threaded through every store/recall/peek call, both to
    prevent cross-task contamination in raw memory recall AND (fixed
    this pass) in the ActionSignature aggregate that drives the
    AVOID/PROBATION verification status.

Swap the 4 synthetic tools out for your real tools -- the CEMG memory
layer is completely tool-agnostic.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, Callable

from dotenv import load_dotenv
from neo4j import Driver
from cemg.storage import BaseStorage, get_storage_provider

from cemg.graph import get_driver, bootstrap_schema, is_healthy, DEFAULT_NAMESPACE
from cemg.llm  import get_llm, chat, LLMProvider
from cemg.memory import store_experience, build_memory_block, peek_signature_status

load_dotenv()

MAX_STEPS = int(os.getenv("CEMG_MAX_STEPS", "15"))

# Tools whose OUTPUT is untrusted external content and must be sanitised
# before storage -- see cemg.security.is_external_source().
EXTERNAL_TOOLS = {"web_search", "read_file"}


# -- Tool schema ----------------------------------------------------------------
TOOL_SCHEMA = [
    {"name": "web_search",  "description": "Search the web for information.",
     "parameters": {"query": "string"}},
    {"name": "read_file",   "description": "Read a local text file.",
     "parameters": {"path": "string"}},
    {"name": "write_file",  "description": "Write text to a local file.",
     "parameters": {"path": "string", "text": "string"}},
    {"name": "finish",      "description": "Return the final answer and stop.",
     "parameters": {"answer": "string"}},
]
TOOL_SCHEMA_STR = json.dumps(TOOL_SCHEMA, indent=2)


def _estimate_tokens(*texts: str) -> int:
    """Rough token-count proxy (chars/4) -- swap for real usage.total_tokens
    from your LLM provider's response object when available."""
    return sum(len(t or "") for t in texts) // 4


def _execute_tool(name: str, params: dict) -> tuple[str, str, str]:
    """
    Execute a tool. Returns (result_text, outcome, observed_error).

    observed_error is the RAW error text -- empty string on success.
    This is stored separately from any "reasoning" the agent later
    states, so the recorded cause of a failure is never just the
    agent's own (possibly wrong) explanation.
    """
    try:
        if name == "web_search":
            query = params.get("query", "")
            result = (
                f"[Synthetic] Top result for '{query}': "
                f"'A 2025 overview of {query}' -- key points: lorem ipsum."
            )
            return result, "success", ""

        elif name == "read_file":
            path = params.get("path", "")
            if not os.path.exists(path):
                err = f"FileNotFoundError: no such file or directory: '{path}'"
                return f"File not found: {path}", "failure", err
            with open(path) as f:
                return f.read()[:2000], "success", ""

        elif name == "write_file":
            path = params.get("path", "")
            text = params.get("text", "")
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                f.write(text)
            return f"Wrote {len(text)} chars to {path}", "success", ""

        elif name == "finish":
            return params.get("answer", ""), "success", ""

        else:
            return f"Unknown tool: {name}", "failure", f"UnknownToolError: {name}"

    except Exception as e:
        return f"Tool error: {e}", "failure", f"{type(e).__name__}: {e}"


# -- Agent ------------------------------------------------------------------------
@dataclass
class CEMGAgent:
    agent_id:            str
    llm:                 LLMProvider
    driver:              Driver | BaseStorage
    task_namespace:      str                = DEFAULT_NAMESPACE
    tool_executor:       Optional[Callable] = None
    tool_schema:         Optional[str]      = None
    session_id:          str                = field(default_factory=lambda: str(uuid.uuid4())[:8])
    history:             list[dict]         = field(default_factory=list)
    last_exp_id:         Optional[str]      = None
    step:                int                = 0
    
    # -- in-run tracking, all reset per .run() call --
    run_failures:        int                = 0
    run_fail_actions:    list[str]          = field(default_factory=list)
    decision_snapshots:  list[dict]         = field(default_factory=list)
    memory_degraded:     bool               = False

    # -- system prompt builder --------------------------------------------------
    def _build_system(self, task: str) -> str:
        """
        Graceful degradation: if Neo4j is unreachable, fall back to an
        empty memory block instead of raising -- a database outage
        should degrade the agent to "runs without memory," not crash it.
        """
        memory_block = ""
        if is_healthy(self.driver):
            try:
                memory_block = build_memory_block(
                    self.driver, self.agent_id,
                    query_action=task, task_namespace=self.task_namespace,
                )
            except Exception as e:
                self.memory_degraded = True
                print(f"[CEMG] WARNING: memory retrieval failed ({e}) -- continuing without memory")
        else:
            self.memory_degraded = True
            print("[CEMG] WARNING: Neo4j unreachable -- running without memory this session")

        schema_str = self.tool_schema if self.tool_schema else TOOL_SCHEMA_STR

        return f"""You are a reliable long-horizon agent.
Your task: {task}

{memory_block}

You have these tools:
{schema_str}

At every step output ONLY a JSON object with exactly these keys:
{{
  "thought":    "your reasoning (1-2 sentences)",
  "tool":       "tool name",
  "parameters": {{...}},
  "reasoning":  "why you chose this tool and params"
}}

Rules:
- Memory entries marked AVOID are recent or repeated failures -- do not repeat them.
- Memory entries marked UNCERTAIN are past their cooldown window -- they MAY be
  fixed now, but verify cautiously rather than assuming success.
- If you are unsure, search before acting.
- Use finish() when you have a complete answer.
- Never output anything outside the JSON object.
"""

    @staticmethod
    def _parse_step(raw: str) -> Optional[dict]:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:])
        if raw.endswith("```"):
            raw = raw[:-3].strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except Exception:
                    pass
        return None

    def _store(self, action: str, outcome: str, reasoning: str = "",
               observed_error: str = "", tool: str = "", params: Optional[dict] = None,
               cost_tokens: int = 0) -> dict:
        """
        Wraps store_experience with graceful degradation -- if the DB
        write fails (e.g. connection dropped mid-session), log and
        continue rather than crashing the agent loop over a memory
        write failure.
        """
        try:
            result = store_experience(
                driver         = self.driver,
                agent_id       = self.agent_id,
                session_id     = self.session_id,
                action         = action,
                outcome        = outcome,
                reasoning      = reasoning,
                observed_error = observed_error,
                context_hint   = tool,
                tool           = tool,
                params         = params or {},
                task_namespace = self.task_namespace,
                cost_tokens    = cost_tokens,
                parent_exp_id  = self.last_exp_id,
            )
            self.last_exp_id = result["exp_id"]
            return result
        except Exception as e:
            self.memory_degraded = True
            print(f"[CEMG] WARNING: failed to store experience ({e}) -- continuing")
            return {"exp_id": None, "action_signature": None, "failure_class": None}

    # -- main loop ----------------------------------------------------------------
    def run(self, task: str, verbose: bool = True) -> str:
        # reset per-run tracking
        self.run_failures = 0
        self.run_fail_actions = []
        self.decision_snapshots = []
        self.memory_degraded = False

        system   = self._build_system(task)
        messages = [{"role": "user", "content": f"Begin. Task: {task}"}]

        if verbose:
            print(f"\n{'='*60}")
            print(f"CEMG Agent | id={self.agent_id} | session={self.session_id} | ns={self.task_namespace}")
            print(f"Task: {task}")
            if self.memory_degraded:
                print("[CEMG] running in DEGRADED mode -- no memory available this session")
            elif "AVOID" in system:
                print("[CEMG] Memory loaded -- agent has prior experience")
            print(f"{'='*60}")

        for self.step in range(1, MAX_STEPS + 1):
            if verbose:
                print(f"\n--- Step {self.step} ---")

            raw = chat(self.llm, messages, system=system, max_tokens=512)
            if verbose:
                print(f"LLM raw: {raw[:300]}")

            action = self._parse_step(raw)
            if action is None:
                self._store(
                    action=f"[step {self.step}] output was not valid JSON",
                    outcome="failure",
                    observed_error="ParseError: LLM output was not valid JSON",
                    tool="_parse_step",
                )
                self.run_failures += 1
                self.run_fail_actions.append("invalid_json_output")
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": "Your output was not valid JSON. Output ONLY a JSON object."})
                continue

            tool    = action.get("tool", "")
            params  = action.get("parameters", {})
            reason  = action.get("reasoning", "")
            thought = action.get("thought", "")

            # -- FIX: snapshot status BEFORE acting, not after the run --
            # This is what makes the compliance metric reflect what the
            # agent actually knew at the moment it decided to act, rather
            # than the signature's state after the whole run has already
            # finished (which could have changed if the same signature
            # was used again later in this same run).
            try:
                snapshot = peek_signature_status(self.driver, self.agent_id, tool, params, self.task_namespace)
            except Exception as e:
                snapshot = {"action_signature": None, "status_before": "UNKNOWN"}
                self.memory_degraded = True
                print(f"[CEMG] WARNING: pre-decision status check failed ({e})")
            self.decision_snapshots.append({**snapshot, "tool": tool, "params": params, "step": self.step})

            if verbose:
                print(f"Thought:  {thought}")
                print(f"Tool:     {tool}({params})")
                if snapshot["status_before"] in ("ACTIVE_FAILURE", "CONFIRMED_BROKEN"):
                    print(f"[CEMG] NOTE: memory flagged this action as {snapshot['status_before']} before it was attempted")

            if self.tool_executor:
                result, outcome, observed_error = self.tool_executor(tool, params)
            else:
                result, outcome, observed_error = _execute_tool(tool, params)

            if verbose:
                status = "OK" if outcome == "success" else "FAILED"
                print(f"Result:   [{status}] {result[:200]}")

            if outcome == "failure":
                self.run_failures += 1
                self.run_fail_actions.append(f"{tool}({json.dumps(params)})")

            self._store(
                action         = f"{tool}({json.dumps(params)})",
                outcome        = outcome,
                reasoning      = reason,           # agent's self-report -- kept separate
                observed_error = observed_error,   # raw truth -- what classification reads
                tool           = tool,
                params         = params,
                cost_tokens    = _estimate_tokens(raw, result),
            )

            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"Tool result ({outcome}): {result}"})

            if tool == "finish":
                if verbose:
                    print(f"\n{'='*60}\nDone in {self.step} steps\nAnswer: {result}")
                return result

        self._store(
            action=f"Reached step limit ({MAX_STEPS})",
            outcome="failure",
            observed_error=f"StepLimitError: exceeded {MAX_STEPS} steps",
            tool="_run_loop",
        )
        return f"[CEMG] Agent hit step limit ({MAX_STEPS}). Task incomplete."


def make_agent(agent_id: str, task_namespace: str = DEFAULT_NAMESPACE) -> CEMGAgent:
    driver = get_storage_provider()
    bootstrap_schema(driver)
    llm    = get_llm()
    return CEMGAgent(agent_id=agent_id, llm=llm, driver=driver, task_namespace=task_namespace)
