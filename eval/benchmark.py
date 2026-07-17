import argparse
import json
import math
import os
import random
import sys
import time
from typing import List, Dict, Any

# Ensure parent directory is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cemg.classify import classify_failure, compute_verification_status, cooldown_days


# -- Statistical T-Test (Pure Python fallback if scipy is missing) --------------
def calculate_paired_t_test(a: List[float], b: List[float]) -> tuple[float, float]:
    """
    Computes a paired t-test for two lists of equal length.
    Returns (t_statistic, p_value).
    Uses math library to avoid external scipy dependency while keeping metrics robust.
    """
    n = len(a)
    if n < 2 or len(b) != n:
        return 0.0, 1.0
    
    differences = [x - y for x, y in zip(a, b)]
    mean_diff = sum(differences) / n
    
    variance = sum((d - mean_diff) ** 2 for d in differences) / (n - 1)
    if variance == 0:
        return 0.0, 1.0
        
    std_err = math.sqrt(variance / n)
    t_stat = mean_diff / std_err
    
    # Compute two-tailed p-value using a simple t-distribution approximation
    # Degrees of freedom df = n - 1
    df = n - 1
    # Simple polynomial approximation for cumulative t-distribution
    # suitable for basic statistics reporting.
    x = abs(t_stat)
    # Approximation of normal CDF (as df gets larger, t-dist approaches normal)
    # Good enough for corporate reporting/dashboard purposes.
    fact = 1.0 / (1.0 + 0.2316419 * x)
    d = 0.3989423 * math.exp(-x * x / 2.0)
    prob = 1.0 - d * (0.3193815 * fact - 0.3565638 * (fact**2) + 1.7814779 * (fact**3) - 1.821256 * (fact**4) + 1.330274 * (fact**5))
    p_val = 2.0 * (1.0 - prob)
    return t_stat, min(max(p_val, 0.0), 1.0)


# -- Mock Agent Simulation for Benchmarking ------------------------------------
class MockAgentSimulator:
    """
    Simulates the agent loops deterministically under different memory strategies
    over N runs. Allows generating statistical benchmarks without incurring LLM API
    costs or requiring live database instances.
    """
    def __init__(self, failure_probability: float = 0.15):
        # A = Structural failure (cost 5)
        # B = Transient timeout (cost 1), recovers on day 15
        # C = Successful fallback (cost 10)
        self.approaches = {
            "A": {"cost": 5, "observed_error": "TypeError: invalid parameter 'region'"},
            "B": {"cost": 1, "observed_error": "ConnectionTimeoutError: upstream timed out"},
            "C": {"cost": 10, "observed_error": None}
        }
        self.fail_prob = failure_probability

    def run_session(self, strategy: str, history: List[Dict], day: int) -> Dict[str, Any]:
        """
        Runs one agent session under a specific strategy.
        """
        steps = 0
        failures = 0
        total_cost = 0
        violation_count = 0
        total_decisions = 0
        success = False
        
        # Determine the order in which the agent tries approaches
        if strategy == "NoMemory":
            # Baseline: Always tries cheapest first (B, A, C) without learning
            order = ["B", "A", "C"]
        elif strategy == "StaticBlacklist":
            # Naive memory: Avoids any approach that failed in history forever
            failed_set = {h["approach"] for h in history if h["outcome"] == "failure"}
            order = [a for a in ["B", "A", "C"] if a not in failed_set]
            # Fallback if everything is blacklisted
            if not order:
                order = ["B", "A", "C"]
        elif strategy == "CEMG":
            # CEMG State Machine:
            # Reconstruct status of each approach based on history
            status_map = {}
            for a in ["B", "A", "C"]:
                rec = [h for h in history if h["approach"] == a]
                if not rec:
                    status_map[a] = "CLEAN"
                    continue
                # Aggregate facts
                failure_count = sum(1 for r in rec if r["outcome"] == "failure")
                success_count = sum(1 for r in rec if r["outcome"] == "success")
                last = rec[-1]
                v = compute_verification_status(
                    last_outcome=last["outcome"],
                    last_ts=last["day"] * 86400.0,
                    failure_class=last["failure_class"],
                    failure_count=failure_count,
                    success_count=success_count,
                    now=day * 86400.0
                )
                status_map[a] = v.status

            # Preferred are CLEAN, RESOLVED, PROBATION (cheapest first)
            # Deferred are ACTIVE_FAILURE, CONFIRMED_BROKEN
            candidates = sorted(["B", "A", "C"], key=lambda x: self.approaches[x]["cost"])
            preferred = [a for a in candidates if status_map[a] not in ("ACTIVE_FAILURE", "CONFIRMED_BROKEN")]
            deferred = [a for a in candidates if status_map[a] in ("ACTIVE_FAILURE", "CONFIRMED_BROKEN")]
            order = preferred + deferred
            
            # Compliance tracking snapshot check
            for a in order[:1]: # Check the first decision choice
                total_decisions += 1
                if status_map[a] in ("ACTIVE_FAILURE", "CONFIRMED_BROKEN"):
                    violation_count += 1
        
        # Execute the decisions
        tried = []
        for approach in order:
            steps += 1
            total_cost += self.approaches[approach]["cost"]
            
            # Determine outcome
            if approach == "A":
                outcome = "failure"
            elif approach == "B":
                outcome = "success" if day >= 15 else "failure"
            else: # C
                outcome = "success"
                
            tried.append({
                "approach": approach,
                "outcome": outcome,
                "observed_error": self.approaches[approach]["observed_error"]
            })
            
            if outcome == "failure":
                failures += 1
            else:
                success = True
                break # Reached the goal
                
        return {
            "steps": steps,
            "failures": failures,
            "cost": total_cost,
            "success": success,
            "violations": violation_count,
            "decisions": total_decisions,
            "tried": tried
        }


# -- Runner and Report Generator -----------------------------------------------
def run_benchmark(n_runs: int = 30):
    sim = MockAgentSimulator()
    strategies = ["NoMemory", "StaticBlacklist", "CEMG"]
    
    # Store history of writes: strategy -> list of experience records
    # Experience record: {approach, outcome, failure_class, day}
    histories = {s: [] for s in strategies}
    
    # Aggregate data per run: strategy -> list of results
    raw_data = {s: {"steps": [], "failures": [], "cost": [], "success": []} for s in strategies}
    compliance_violations = 0
    compliance_decisions = 0
    
    for day in range(n_runs):
        for strategy in strategies:
            res = sim.run_session(strategy, histories[strategy], day)
            
            # Record metrics
            raw_data[strategy]["steps"].append(res["steps"])
            raw_data[strategy]["failures"].append(res["failures"])
            raw_data[strategy]["cost"].append(res["cost"])
            raw_data[strategy]["success"].append(1.0 if res["success"] else 0.0)
            
            # Add to history for subsequent runs
            for attempt in res["tried"]:
                fc = classify_failure(attempt["observed_error"]) if attempt["outcome"] == "failure" else None
                histories[strategy].append({
                    "approach": attempt["approach"],
                    "outcome": attempt["outcome"],
                    "failure_class": fc,
                    "day": day
                })
                
            if strategy == "CEMG":
                compliance_violations += res["violations"]
                compliance_decisions += res["decisions"]

    # Calculate summary metrics
    report = {}
    for s in strategies:
        steps_list = raw_data[s]["steps"]
        failures_list = raw_data[s]["failures"]
        cost_list = raw_data[s]["cost"]
        success_list = raw_data[s]["success"]
        
        n = len(steps_list)
        mean_steps = sum(steps_list) / n
        mean_failures = sum(failures_list) / n
        mean_cost = sum(cost_list) / n
        success_rate = (sum(success_list) / n) * 100.0
        
        # Calculate standard deviation
        var_steps = sum((x - mean_steps) ** 2 for x in steps_list) / (n - 1) if n > 1 else 0.0
        var_failures = sum((x - mean_failures) ** 2 for x in failures_list) / (n - 1) if n > 1 else 0.0
        var_cost = sum((x - mean_cost) ** 2 for x in cost_list) / (n - 1) if n > 1 else 0.0
        
        report[s] = {
            "mean_steps": round(mean_steps, 2),
            "std_steps": round(math.sqrt(var_steps), 2),
            "mean_failures": round(mean_failures, 2),
            "std_failures": round(math.sqrt(var_failures), 2),
            "mean_cost": round(mean_cost, 2),
            "std_cost": round(math.sqrt(var_cost), 2),
            "success_rate": round(success_rate, 1),
            "total_accumulated_cost": sum(cost_list)
        }

    # Perform Significance Tests (CEMG vs Baselines)
    t_steps_nm, p_steps_nm = calculate_paired_t_test(raw_data["CEMG"]["steps"], raw_data["NoMemory"]["steps"])
    t_fails_nm, p_fails_nm = calculate_paired_t_test(raw_data["CEMG"]["failures"], raw_data["NoMemory"]["failures"])
    
    t_steps_bl, p_steps_bl = calculate_paired_t_test(raw_data["CEMG"]["steps"], raw_data["StaticBlacklist"]["steps"])
    t_fails_bl, p_fails_bl = calculate_paired_t_test(raw_data["CEMG"]["failures"], raw_data["StaticBlacklist"]["failures"])

    compliance_rate = ((compliance_decisions - compliance_violations) / compliance_decisions * 100.0) if compliance_decisions else 100.0

    # Output JSON Report
    json_output = {
        "summary": report,
        "significance": {
            "vs_no_memory": {
                "steps": {"t_stat": round(t_steps_nm, 3), "p_val": round(p_steps_nm, 5), "significant": p_steps_nm < 0.05},
                "failures": {"t_stat": round(t_fails_nm, 3), "p_val": round(p_fails_nm, 5), "significant": p_fails_nm < 0.05}
            },
            "vs_static_blacklist": {
                "steps": {"t_stat": round(t_steps_bl, 3), "p_val": round(p_steps_bl, 5), "significant": p_steps_bl < 0.05},
                "failures": {"t_stat": round(t_fails_bl, 3), "p_val": round(p_fails_bl, 5), "significant": p_fails_bl < 0.05}
            }
        },
        "compliance": {
            "total_decisions": compliance_decisions,
            "violations": compliance_violations,
            "compliance_rate_pct": round(compliance_rate, 2)
        }
    }
    
    # Save report
    report_path = "eval/benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(json_output, f, indent=2)

    # Print Clean Corporate Dashboard
    print("\n" + "="*80)
    print("                      CEMG TECHNICAL BENCHMARK REPORT                      ")
    print("="*80)
    print(f"Sample Size (Runs/Days): {n_runs} | Environment: Windows (Sandbox)")
    print(f"Metrics Output Saved to: file:///{os.path.abspath(report_path).replace('\\', '/')}")
    print("-"*80)
    print(f"{'Strategy':<18} | {'Avg Steps':<12} | {'Avg Failures':<12} | {'Avg Cost/Run':<12} | {'Success Rate'}")
    print("-"*80)
    for s in ["NoMemory", "StaticBlacklist", "CEMG"]:
        r = report[s]
        steps_str = f"{r['mean_steps']} ±{r['std_steps']}"
        fails_str = f"{r['mean_failures']} ±{r['std_failures']}"
        cost_str = f"${r['mean_cost']} ±{r['std_cost']}"
        print(f"{s:<18} | {steps_str:<12} | {fails_str:<12} | {cost_str:<12} | {r['success_rate']}%")
    print("-"*80)
    
    # Print Savings Overview
    savings_vs_nm = ((report["NoMemory"]["total_accumulated_cost"] - report["CEMG"]["total_accumulated_cost"]) 
                     / report["NoMemory"]["total_accumulated_cost"] * 100.0)
    savings_vs_bl = ((report["StaticBlacklist"]["total_accumulated_cost"] - report["CEMG"]["total_accumulated_cost"]) 
                     / report["StaticBlacklist"]["total_accumulated_cost"] * 100.0)
    
    print("\n[BUSINESS EFFICIENCY METRICS]")
    print(f"  * Cumulative Cost Savings vs. No Memory:       {savings_vs_nm:.2f}%")
    print(f"  * Cumulative Cost Savings vs. Static Blacklist: {savings_vs_bl:.2f}%")
    print(f"  * Action Compliance Rate (Safety Metric):       {compliance_rate:.2f}%")
    
    print("\n[STATISTICAL SIGNIFICANCE (P-VALUES)]")
    def format_sig(p):
        return f"{p:.5f} (CONFIRMED SIGNIFICANT)" if p < 0.05 else f"{p:.5f} (NOT SIGNIFICANT)"
        
    print(f"  * Step reduction vs. No Memory:       p = {format_sig(p_steps_nm)}")
    print(f"  * Failure reduction vs. No Memory:    p = {format_sig(p_fails_nm)}")
    print(f"  * Step reduction vs. Blacklist:       p = {format_sig(p_steps_bl)}")
    print(f"  * Failure reduction vs. Blacklist:    p = {format_sig(p_fails_bl)}")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CEMG Technical Benchmarking Harness")
    parser.add_argument("--runs", type=int, default=30, help="Number of benchmark runs/days (default: 30)")
    args = parser.parse_args()
    run_benchmark(args.runs)
