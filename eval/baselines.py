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
from cemg.graph  import get_driver, bootstrap_schema
from cemg.memory import store_experience, evaluate_compliance

load_dotenv()

MAX_STEPS = int(os.getenv("CEMG_MAX_STEPS", "15"))
EVAL_NAMESPACE = "eval_comparison"

TOOL_SCHEMA = [
    {"name": "web_search",  "parameters": {"query": "string"}},
    {"name": "read_file",   "parameters": {"path": "string"}},
    {"name": "write_file",  "parameters": {"path": "string", "text": "string"}},
    {"name": "finish",      "parameters": {"answer": "string"}},
]
TOOL_SCHEMA_STR = json.dumps(TOOL_SCHEMA, indent=2)

# The scripted prior failure both CEMG and TextCompression are seeded
# with -- identical narrative, different storage mechanism. This is
# what makes the comparison fair: both systems are given the SAME prior
# knowledge, and we measure how well each one's memory representation
# lets an agent actually use it.
PRIOR_FAILURE_NARRATIVE = (
    "In the previous session the agent tried to read a local research "
    "file that did not exist yet (read_file failed: file not found), "
    "then tried to write the summary without first creating the output "
    "directory (write_file failed: directory missing). The task was not "
    "completed."
)


# -- Shared tool executor --------------------------------------------------------
def _run_tool(name: str, params: dict) -> tuple[str, str, str]:
    """Returns (result_text, outcome, observed_error)."""
    try:
        if name == "web_search":
            q = params.get("query", "")
            return (f"[Synthetic] Top result for '{q}': "
                    f"'A 2025 overview of {q}' -- key points: lorem ipsum."), "success", ""
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
def check_task_success(output_path: str = "data/summary.txt", min_bullets: int = 3) -> dict:
    """Verifies the real output artifact. Unchanged from the prior fix pass."""
    if not os.path.exists(output_path):
        return {"success": False, "n_bullets": 0, "reason": f"{output_path} does not exist"}
    with open(output_path) as f:
        content = f.read()
    bullet_pattern = re.compile(r"^\s*[-*\u2022]\s+\S|^\s*\d+[\.\)]\s+\S", re.MULTILINE)
    n_bullets = len(bullet_pattern.findall(content))
    if n_bullets < min_bullets:
        return {"success": False, "n_bullets": n_bullets, "reason": f"only {n_bullets} bullets, need >= {min_bullets}"}
    if len(content.strip()) < 20:
        return {"success": False, "n_bullets": n_bullets, "reason": "content too short"}
    return {"success": True, "n_bullets": n_bullets, "reason": "ok"}


def _reset_output(output_path: str = "data/summary.txt") -> None:
    if os.path.exists(output_path):
        os.remove(output_path)


# -- THE KEY FIX: actually seed CEMG's memory before comparing it ------------
def seed_prior_failure(driver, agent_id: str, task_namespace: str = EVAL_NAMESPACE) -> None:
    """
    Write a genuine prior-failure record into agent_id's CEMG memory,
    using the real store_experience() call -- the same mechanism a live
    agent would use. Without this, CEMG starts every eval run with no
    memory at all, identical to NoMemoryAgent, and the comparison
    doesn't test anything.
    """
    past_ts = time.time() - 7_200  # 2 hours ago -- recent enough to still be an ACTIVE_FAILURE

    r1 = store_experience(
        driver=driver, agent_id=agent_id, session_id=f"seed_{uuid.uuid4().hex[:6]}",
        action="read_file(path='data/kg_research.txt')", outcome="failure",
        reasoning="Assumed a local research file already existed from a previous run",
        observed_error="FileNotFoundError: data/kg_research.txt",
        tool="read_file", params={"path": "data/kg_research.txt"},
        task_namespace=task_namespace,
    )
    store_experience(
        driver=driver, agent_id=agent_id, session_id=r1.get("exp_id", "seed"),
        action="write_file(path='data/summary.txt', text='...')", outcome="failure",
        reasoning="Tried to write without first confirming the directory exists",
        observed_error="FileNotFoundError: directory 'data' does not exist",
        tool="write_file", params={"path": "data/summary.txt"},
        task_namespace=task_namespace, parent_exp_id=r1.get("exp_id"),
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
    driver = get_driver()
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
        cemg = CEMGAgent(agent_id=agent_id, llm=llm, driver=driver, task_namespace=EVAL_NAMESPACE)
        answer = cemg.run(task, verbose=False)
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
        "Research key differences between TKG and static KG for agent memory, "
        "then write a 3-bullet summary to data/summary.txt"
    )
    run_comparison(TASK, n_runs=3)
