"""
eval/baselines.py
------------------
Three systems compared on the same task: no memory, text-compression
memory, and CEMG.

FIX LOG v3 (this pass -- the most important fix in the whole project):
  Previously, every run generated a brand-new random agent_id for CEMG,
  so CEMG started every comparison run with ZERO memory -- identical
  to the no-memory baseline. The metrics were honest but the experiment
  wasn't testing anything: there was nothing seeded for CEMG to draw on.

  Fixed by seed_prior_failure(): before each run, a genuine prior-failure
  record is written into a fresh CEMG agent_id using the real
  store_experience() call (same mechanism a live agent would use), and
  the SAME narrative is given to TextCompressionAgent as prior_session_text.
  This is what run_demo.py already did correctly; it just hadn't been
  ported into the eval harness.

  Also this pass:
  - check_task_success() unchanged from the prior fix (still verifies
    the real output artifact, not agent chatter).
  - Failure counting still consistent across all three systems (each
    agent's own run_failures / in-loop counter).
  - Paired significance testing now covers CEMG vs NoMemory AND CEMG vs
    TextCompression (previously only the latter), on both steps and
    failure count -- the CEMG-vs-NoMemory comparison is the one that
    actually supports the paper's headline claim and was missing before.
  Also this pass (fixing the two gaps found in the third review):
  - ActionSignature aggregates (and therefore verification status) are
    now correctly namespace-scoped in graph.py -- fixed there, this
    file benefits automatically since it already uses EVAL_NAMESPACE.
  - Compliance is now computed from cemg.agent.CEMGAgent.decision_snapshots
    (captured BEFORE each action executes) via evaluate_compliance(),
    not from a live re-query of signature state after the run completes.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

from cemg.llm    import get_llm, chat, LLMProvider
from cemg.storage import get_storage_provider
from cemg.graph  import get_driver, bootstrap_schema
from cemg.memory import store_experience, evaluate_compliance

load_dotenv()

MAX_STEPS = int(os.getenv("CEMG_MAX_STEPS", "15"))
EVAL_NAMESPACE = "eval_comparison"

TOOL_SCHEMA = [
    {"name": "internal_rag", "parameters": {"query": "string"}},
    {"name": "web_search",  "parameters": {"query": "string"}},
    {"name": "read_file",   "parameters": {"path": "string"}},
    {"name": "write_file",  "parameters": {"path": "string", "text": "string"}},
    {"name": "legacy_save", "parameters": {"path": "string", "content": "string"}},
    {"name": "finish",      "parameters": {"answer": "string"}},
]
TOOL_SCHEMA_STR = json.dumps(TOOL_SCHEMA, indent=2)

PRIOR_FAILURE_NARRATIVE = (
    "In the previous session, the agent tried to use web_search for private vessel manual information, "
    "which returned irrelevant public results (structurally wrong). It also tried to use legacy_save "
    "to write the report, which always fails with a PermissionError. Finally, it tried to call finish "
    "before actually creating the required file."
)


# -- Shared tool executor --------------------------------------------------------
def _run_tool(name: str, params: dict) -> tuple[str, str, str]:
    """Returns (result_text, outcome, observed_error)."""
    try:
        if name == "internal_rag":
            q = params.get("query", "")
            if "framo" in q.lower() or "sd125" in q.lower():
                return f"Private Manual Snippet: Framo SD125 pump faults are usually caused by hydraulic pressure drops. Check the pressure relief valve.", "success", ""
            return f"No private records found for '{q}'.", "success", ""
        if name == "web_search":
            q = params.get("query", "")
            return f"Public web results for '{q}': Irrelevant consumer pump discussions.", "success", ""
        if name == "legacy_save":
            return "Permission denied to legacy storage.", "failure", "PermissionError: legacy storage is read-only"
        if name == "read_file":
            p = params.get("path", "")
            if not os.path.exists(p):
                return f"File not found: {p}", "failure", f"FileNotFoundError: {p}"
            with open(p) as f:
                return f.read()[:2000], "success", ""
        if name == "write_file":
            p = params.get("path", ""); t = params.get("text", "")
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            with open(p, "w") as f: f.write(t)
            return f"Wrote {len(t)} chars to {p}", "success", ""
        if name == "finish":
            # Allowed only if the required file exists
            if not os.path.exists("data/reports/framo_sd125_summary.md"):
                return "Cannot finish: required report file not found.", "failure", "AssertionError: data/reports/framo_sd125_summary.md does not exist"
            return params.get("answer", ""), "success", ""
        return f"Unknown tool: {name}", "failure", f"UnknownToolError: {name}"
    except Exception as e:
        return str(e), "failure", f"{type(e).__name__}: {e}"


def _parse(raw: str) -> Optional[dict]:
    raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try: return json.loads(m.group())
            except Exception: pass
    return None


# -- Task-completion check -----------------------------------------------------
def check_task_success(output_path: str = "data/reports/framo_sd125_summary.md", min_length: int = 50) -> dict:
    """Verifies the real output artifact."""
    if not os.path.exists(output_path):
        return {"success": False, "length": 0, "reason": f"{output_path} does not exist"}
    with open(output_path) as f:
        content = f.read()
    if len(content.strip()) < min_length:
        return {"success": False, "length": len(content.strip()), "reason": "content too short"}
    if "pressure relief valve" not in content.lower():
        return {"success": False, "length": len(content.strip()), "reason": "missing key finding"}
    return {"success": True, "length": len(content.strip()), "reason": "ok"}


def _reset_output(output_path: str = "data/reports/framo_sd125_summary.md") -> None:
    if os.path.exists(output_path):
        os.remove(output_path)


# -- THE KEY FIX: actually seed CEMG's memory before comparing it ------------
def seed_prior_failure(driver, agent_id: str, task_namespace: str = EVAL_NAMESPACE) -> None:
    """
    Write a genuine prior-failure record into agent_id's CEMG memory.
    """
    r1 = store_experience(
        driver=driver, agent_id=agent_id, session_id=f"seed_{uuid.uuid4().hex[:6]}",
        action="web_search(query='framo sd125 pump fault')", outcome="failure",
        reasoning="Tried to search the public web for private vessel manuals",
        observed_error="Found irrelevant public results instead of actual manuals",
        tool="web_search", params={"query": "framo sd125 pump fault"},
        task_namespace=task_namespace,
    )
    r2 = store_experience(
        driver=driver, agent_id=agent_id, session_id=r1.get("exp_id", "seed"),
        action="legacy_save(content='...', path='data/reports/framo_sd125_summary.md')", outcome="failure",
        reasoning="Tried to use legacy storage which is read-only",
        observed_error="PermissionError: legacy storage is read-only",
        tool="legacy_save", params={"path": "data/reports/framo_sd125_summary.md", "content": "..."},
        task_namespace=task_namespace, parent_exp_id=r1.get("exp_id"),
    )
    store_experience(
        driver=driver, agent_id=agent_id, session_id=r2.get("exp_id", "seed"),
        action="finish(answer='I am done')", outcome="failure",
        reasoning="Tried to finish before the report file was actually created",
        observed_error="AssertionError: data/reports/framo_sd125_summary.md does not exist",
        tool="finish", params={"answer": "I am done"},
        task_namespace=task_namespace, parent_exp_id=r2.get("exp_id"),
    )


# -- Baseline 1: No memory -----------------------------------------------------
@dataclass
class NoMemoryAgent:
    llm:        LLMProvider
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def run(self, task: str) -> dict:
        system = (
            f"You are a helpful agent. Task: {task}\n"
            f"Tools: {TOOL_SCHEMA_STR}\n"
            "Output ONLY JSON: {\"thought\":\"...\",\"tool\":\"...\","
            "\"parameters\":{...},\"reasoning\":\"...\"}"
        )
        messages  = [{"role": "user", "content": f"Begin. Task: {task}"}]
        steps, failures, fail_acts = 0, 0, []

        for step in range(1, MAX_STEPS + 1):
            steps = step
            raw    = chat(self.llm, messages, system=system)
            action = _parse(raw)
            if action is None:
                failures += 1; fail_acts.append("invalid_json_output")
                messages += [{"role":"assistant","content":raw},
                             {"role":"user","content":"Not valid JSON. Try again."}]
                continue
            tool, params = action.get("tool", ""), action.get("parameters", {})
            result, outcome, _ = _run_tool(tool, params)
            if outcome == "failure":
                failures += 1; fail_acts.append(f"{tool}({params})")
            messages += [{"role":"assistant","content":raw},
                         {"role":"user","content":f"Result ({outcome}): {result}"}]
            if tool == "finish":
                break

        task_result = check_task_success()
        return {"steps": steps, "failures": failures, "fail_actions": fail_acts,
                "success": task_result["success"], "task_detail": task_result}


# -- Baseline 2: Text compression ----------------------------------------------
@dataclass
class TextCompressionAgent:
    llm:                 LLMProvider
    prior_session_text:  str  = ""
    session_id:          str  = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def run(self, task: str) -> dict:
        memory_section = f"\nPRIOR SESSION SUMMARY:\n{self.prior_session_text}\n" if self.prior_session_text else ""
        system = (
            f"You are a helpful agent.{memory_section}\n"
            f"Task: {task}\n"
            f"Tools: {TOOL_SCHEMA_STR}\n"
            "Output ONLY JSON: {\"thought\":\"...\",\"tool\":\"...\","
            "\"parameters\":{...},\"reasoning\":\"...\"}"
        )
        messages  = [{"role": "user", "content": f"Begin. Task: {task}"}]
        steps, failures, fail_acts = 0, 0, []

        for step in range(1, MAX_STEPS + 1):
            steps = step
            raw    = chat(self.llm, messages, system=system)
            action = _parse(raw)
            if action is None:
                failures += 1; fail_acts.append("invalid_json_output")
                messages += [{"role":"assistant","content":raw},
                             {"role":"user","content":"Not valid JSON. Try again."}]
                continue
            tool, params = action.get("tool", ""), action.get("parameters", {})
            result, outcome, _ = _run_tool(tool, params)
            if outcome == "failure":
                failures += 1; fail_acts.append(f"{tool}({params})")
            messages += [{"role":"assistant","content":raw},
                         {"role":"user","content":f"Result ({outcome}): {result}"}]
            if tool == "finish":
                break

        task_result = check_task_success()
        return {"steps": steps, "failures": failures, "fail_actions": fail_acts,
                "success": task_result["success"], "task_detail": task_result}


# -- Eval runner -----------------------------------------------------------------
def run_comparison(task: str, n_runs: int = 3):
    """
    Run all three systems on the same task, n_runs times each. CEMG and
    TextCompression are BOTH seeded with the same prior-failure narrative
    before every run -- this is the fix that makes the comparison
    actually test whether structured memory (CEMG) helps more than
    unstructured text-summary memory (B2), versus no memory at all (B1).
    """
    from cemg.agent import CEMGAgent

    llm    = get_llm()
    driver = get_storage_provider()
    bootstrap_schema(driver)

    results: dict[str, list[dict]] = {"NoMemory": [], "TextCompression": [], "CEMG": []}
    compliance_reports: list[dict] = []

    for run in range(n_runs):
        print(f"\n=== Run {run+1}/{n_runs} ===")

        _reset_output()
        print("  B1 NoMemory ...")
        results["NoMemory"].append(NoMemoryAgent(llm=llm).run(task))

        _reset_output()
        print("  B2 TextCompression (seeded with prior-failure narrative) ...")
        b2 = TextCompressionAgent(llm=llm, prior_session_text=PRIOR_FAILURE_NARRATIVE)
        results["TextCompression"].append(b2.run(task))

        _reset_output()
        print("  CEMG (seeded with the SAME prior failure, via real store_experience) ...")
        agent_id = f"eval_{uuid.uuid4().hex[:6]}"
        seed_prior_failure(driver, agent_id)   # <-- the key fix
        cemg = CEMGAgent(agent_id=agent_id, llm=llm, driver=driver, task_namespace=EVAL_NAMESPACE, tool_executor=_run_tool, tool_schema=TOOL_SCHEMA_STR)
        answer = cemg.run(task, verbose=True)
        task_result = check_task_success()

        compliance = evaluate_compliance(cemg.decision_snapshots)
        compliance_reports.append(compliance)

        results["CEMG"].append({
            "steps": cemg.step, "failures": cemg.run_failures,
            "fail_actions": cemg.run_fail_actions,
            "success": task_result["success"], "task_detail": task_result,
        })

    _print_summary(results, compliance_reports, n_runs)
    driver.close()
    return results


def _paired_ttest(a: list[float], b: list[float], label: str) -> None:
    try:
        from scipy import stats as scipy_stats
        if len(a) == len(b) and len(a) >= 3:
            t_stat, p_val = scipy_stats.ttest_rel(a, b)
            sig = "significant (p<0.05)" if p_val < 0.05 else "NOT significant"
            print(f"  {label}: t={t_stat:.3f}  p={p_val:.4f}  -> {sig}")
        else:
            print(f"  {label}: skipped -- need equal-length paired runs, n>=3")
    except ImportError:
        print(f"  {label}: skipped -- scipy not installed (pip install scipy)")


def _print_summary(results: dict[str, list[dict]], compliance_reports: list[dict], n_runs: int) -> None:
    print("\n" + "="*72)
    print(f"{'System':<18} {'Steps (mean+-std)':>20} {'Failures (mean+-std)':>22} {'Success':>10}")
    print("-"*72)

    metrics: dict[str, dict[str, list[float]]] = {}
    for sys_name, runs in results.items():
        steps     = [r["steps"]    for r in runs]
        failures  = [r["failures"] for r in runs]
        successes = sum(1 for r in runs if r["success"])
        metrics[sys_name] = {"steps": steps, "failures": failures}

        s_mean = statistics.mean(steps);    s_std = statistics.stdev(steps)    if len(steps)    > 1 else 0.0
        f_mean = statistics.mean(failures); f_std = statistics.stdev(failures) if len(failures) > 1 else 0.0
        succ_pct = successes / len(runs) * 100

        print(f"{sys_name:<18} {s_mean:>8.1f} +- {s_std:<8.1f} "
              f"{f_mean:>10.1f} +- {f_std:<8.1f} {succ_pct:>9.0f}%")

    print("="*72)

    if compliance_reports:
        total_used = sum(c["total_used"] for c in compliance_reports)
        total_viol = sum(c["violations"] for c in compliance_reports)
        rate = (total_viol / total_used) if total_used else 0.0
        print(f"\nCEMG compliance: {total_used - total_viol}/{total_used} actions correctly avoided "
              f"flagged failures ({rate*100:.1f}% violation rate -- lower is better)")

    if n_runs < 3:
        print("\nNOTE: n_runs < 3 -- variance/significance estimates are not reliable. Use n_runs=3+.")
        return

    print("\nSignificance tests (failures, the metric CEMG's whole premise is about):")
    _paired_ttest(metrics["CEMG"]["failures"], metrics["NoMemory"]["failures"],       "CEMG vs NoMemory      ")
    _paired_ttest(metrics["CEMG"]["failures"], metrics["TextCompression"]["failures"],"CEMG vs TextCompression")

    print("\nSignificance tests (steps to completion):")
    _paired_ttest(metrics["CEMG"]["steps"], metrics["NoMemory"]["steps"],        "CEMG vs NoMemory      ")
    _paired_ttest(metrics["CEMG"]["steps"], metrics["TextCompression"]["steps"], "CEMG vs TextCompression")


if __name__ == "__main__":
    TASK = (
        "Create a maintenance summary for the Framo SD125 pump fault and save it "
        "to data/reports/framo_sd125_summary.md."
    )
    run_comparison(TASK, n_runs=1)
