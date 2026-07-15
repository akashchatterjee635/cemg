"""
eval/simulate_convergence.py
------------------------------
A deterministic, dependency-free simulation comparing three memory
strategies against the SAME synthetic world, using the REAL,
unmodified cemg/classify.py logic (decay math, failure classification,
verification state machine) -- no mocking of that code, no Neo4j
required, no LLM API required.

WHY THIS EXISTS: this sandbox has no live Neo4j instance and no LLM API
credentials, so a true end-to-end benchmark (real agent + real database
+ real LLM) can't be run here. This script is the honest substitute --
it exercises the actual production scoring/classification functions
against a scripted synthetic scenario, so the numbers below are REAL
outputs of REAL code, just running against a synthetic "world" instead
of a live deployment.

THE WORLD:
  Three ways to reach the goal, each with a distinct cost (think: token
  cost or latency) and distinct reliability over time:

    Approach A: cost 5, STRUCTURALLY broken -- never works, ever.
                (e.g. calling an API with a fundamentally wrong parameter)
    Approach B: cost 1 (cheapest), TRANSIENTLY broken for the first 15
                simulated days, then starts working from day 15 onward.
                (e.g. a third-party service outage that gets fixed)
    Approach C: cost 10 (most expensive), always works.
                (e.g. a slow but reliable fallback)

  30 simulated sessions run, one per day. Whichever agent reaches the
  goal in a session pays the cost of every approach it tried that
  session (failed attempts aren't free -- they cost tokens/time too).

THREE AGENTS COMPARED:
  1. NoMemory        -- no memory at all. Every session, tries
                        approaches in the same fixed default order
                        (cheapest-first: B, A, C) since it has no
                        history to reason from.
  2. StaticBlacklist  -- proxy for how most current agent-memory tools
                        behave: once something fails, avoid it FOREVER.
                        No reconsideration, no decay, no re-verification.
  3. CEMG             -- the real classify_failure() / cooldown_days() /
                        compute_verification_status() functions from
                        this repo, unmodified. Classifies each failure,
                        assigns a class-specific cooldown, and re-tests
                        once a failure's cooldown has passed.
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cemg.classify import classify_failure, compute_verification_status, cooldown_days

# -- Synthetic world ----------------------------------------------------------
APPROACHES = {
    "A": {"cost": 5,  "observed_error_on_fail": "TypeError: invalid parameter 'region'"},
    "B": {"cost": 1,  "observed_error_on_fail": "ConnectionTimeoutError: upstream timed out"},
    "C": {"cost": 10, "observed_error_on_fail": None},   # never fails
}
B_FIXED_ON_DAY = 15
N_DAYS = 30
DAY_SECONDS = 86_400


def world_outcome(approach: str, day: int) -> str:
    """Ground truth: does this approach succeed on this simulated day?"""
    if approach == "A":
        return "failure"                                  # never works
    if approach == "B":
        return "success" if day >= B_FIXED_ON_DAY else "failure"
    if approach == "C":
        return "success"                                   # always works
    raise ValueError(approach)


# -- Agent 1: No memory --------------------------------------------------------
def run_no_memory(n_days: int = N_DAYS, order: list = None) -> dict:
    """
    Tries approaches in a FIXED order every single session -- forever,
    because it has no way to remember what happened before.

    order defaults to ["B","A","C"] (cheapest-first) -- but that's a
    generous assumption: it implicitly gives NoMemory free cost-awareness
    it doesn't actually have. A more realistic default for an LLM with
    no memory is whatever order the tools happen to be declared in
    (here modelled as alphabetical: A, B, C) -- run BOTH and compare,
    since which one you pick materially changes how "competitive" the
    no-memory baseline looks.
    """
    order = order or ["B", "A", "C"]
    total_cost = 0
    daily_costs = []
    for day in range(n_days):
        session_cost = 0
        for approach in order:
            session_cost += APPROACHES[approach]["cost"]
            if world_outcome(approach, day) == "success":
                break
        total_cost += session_cost
        daily_costs.append(session_cost)
    return {"name": "NoMemory", "total_cost": total_cost, "daily_costs": daily_costs, "order": order}


# -- Agent 2: Static blacklist (proxy for naive "current" memory tools) -------
def run_static_blacklist(n_days: int = N_DAYS) -> dict:
    """Once an approach fails, it is avoided FOREVER -- no decay, no
    re-verification, ever. This is the behaviour of a simple 'remember
    what failed and never try it again' memory layer, which is roughly
    what a plain vector-store or session-summary memory gives you."""
    blacklist = set()
    total_cost = 0
    daily_costs = []
    for day in range(n_days):
        session_cost = 0
        for approach in ["B", "A", "C"]:
            if approach in blacklist:
                continue
            session_cost += APPROACHES[approach]["cost"]
            if world_outcome(approach, day) == "success":
                break
            else:
                blacklist.add(approach)   # permanent, no matter WHY it failed
        total_cost += session_cost
        daily_costs.append(session_cost)
    return {"name": "StaticBlacklist", "total_cost": total_cost, "daily_costs": daily_costs,
            "final_blacklist": blacklist}


# -- Agent 3: CEMG -- real classify.py logic, unmodified ----------------------
def run_cemg(n_days: int = N_DAYS) -> dict:
    """
    Uses the ACTUAL production functions from cemg/classify.py:
      - classify_failure(observed_error)        -- transient vs structural
      - compute_verification_status(...)        -- the 4-state machine
      - cooldown_days(failure_class)             -- class-specific cooldown

    Memory is a plain Python dict here (standing in for the Neo4j
    ActionSignature aggregate, since no live Neo4j is available in this
    sandbox) -- but the DECISION LOGIC querying that memory is the
    real, unmodified library code.
    """
    # in-memory stand-in for the Neo4j ActionSignature aggregate
    memory: dict[str, dict] = {}   # approach -> {failure_count, success_count, last_outcome, last_ts, failure_class}

    total_cost = 0
    daily_costs = []
    sim_start = time.time() - n_days * DAY_SECONDS   # so "day 0" is n_days ago, "day N" approaches now

    for day in range(n_days):
        now_ts = sim_start + day * DAY_SECONDS
        session_cost = 0
        tried_this_session = []

        # decide order: prefer CLEAN/RESOLVED/PROBATION, defer ACTIVE_FAILURE/CONFIRMED_BROKEN,
        # and among the preferred set, cheapest first (this is the same
        # "show cost alongside reliability" idea from the design discussion)
        def status_for(approach):
            rec = memory.get(approach)
            if rec is None:
                return "CLEAN"
            v = compute_verification_status(
                last_outcome=rec["last_outcome"], last_ts=rec["last_ts"],
                failure_class=rec["failure_class"], failure_count=rec["failure_count"],
                success_count=rec["success_count"], now=now_ts,
            )
            return v.status

        avoid_now   = {"ACTIVE_FAILURE", "CONFIRMED_BROKEN"}
        candidates  = sorted(APPROACHES.keys(), key=lambda a: APPROACHES[a]["cost"])
        preferred   = [a for a in candidates if status_for(a) not in avoid_now]
        deferred    = [a for a in candidates if status_for(a) in avoid_now]
        order = preferred + deferred   # only touch a flagged approach if nothing else is left

        for approach in order:
            session_cost += APPROACHES[approach]["cost"]
            tried_this_session.append(approach)
            outcome = world_outcome(approach, day)

            rec = memory.setdefault(approach, {"failure_count": 0, "success_count": 0,
                                                "last_outcome": None, "last_ts": now_ts,
                                                "failure_class": None})
            if outcome == "failure":
                fc = classify_failure(APPROACHES[approach]["observed_error_on_fail"])
                rec["failure_count"] += 1
                rec["last_outcome"] = "failure"
                rec["last_ts"] = now_ts
                rec["failure_class"] = fc
            else:
                rec["success_count"] += 1
                rec["last_outcome"] = "success"
                rec["last_ts"] = now_ts
                break   # success -- stop trying more approaches this session

        total_cost += session_cost
        daily_costs.append(session_cost)

    return {"name": "CEMG", "total_cost": total_cost, "daily_costs": daily_costs, "final_memory": memory}


# -- Report ---------------------------------------------------------------------
def _day_of_convergence(daily_costs: list, optimal_cost: int, after_day: int) -> str:
    """First day, after `after_day`, on which this system's daily cost
    reached the true optimum -- and stayed there for the rest of the run."""
    for day in range(after_day, len(daily_costs)):
        if all(c == optimal_cost for c in daily_costs[day:]):
            return str(day)
    return "never (within simulation window)"


def main():
    no_mem_generous     = run_no_memory(order=["B", "A", "C"])   # cheapest-first (flattering)
    no_mem_realistic    = run_no_memory(order=["A", "B", "C"])   # alphabetical / schema-order (realistic)
    static              = run_static_blacklist()
    cemg                = run_cemg()

    print("="*78)
    print(f"SIMULATION: {N_DAYS} sessions, B fixed on day {B_FIXED_ON_DAY}")
    print(f"Costs: A=5 (never works) | B=1 (broken until day {B_FIXED_ON_DAY}) | C=10 (always works)")
    print("="*78)
    print(f"{'System':<28}{'Total Cost':>14}{'Avg Cost/Day':>16}{'Cost After Day '+str(B_FIXED_ON_DAY):>20}")
    print("-"*78)

    for result in [no_mem_generous, no_mem_realistic, static, cemg]:
        total = result["total_cost"]
        avg   = total / N_DAYS
        after = sum(result["daily_costs"][B_FIXED_ON_DAY:]) / (N_DAYS - B_FIXED_ON_DAY)
        label = result["name"]
        if "order" in result:
            label += f" (order={''.join(result['order'])})"
        print(f"{label:<28}{total:>14}{avg:>16.2f}{after:>20.2f}")

    print("="*78)
    print()
    print(f"Day the system converges to the TRUE OPTIMUM (cost=1, i.e. found that B is fixed) "
          f"and never regresses, starting the search from day {B_FIXED_ON_DAY}:")
    print(f"  NoMemory (cheapest-first, order=BAC):  {_day_of_convergence(no_mem_generous['daily_costs'], 1, B_FIXED_ON_DAY)}"
          f"   <- only converges instantly because its SCRIPTED default order happens to match optimal")
    print(f"  NoMemory (alphabetical, order=ABC):     {_day_of_convergence(no_mem_realistic['daily_costs'], 1, B_FIXED_ON_DAY)}"
          f"   <- never finds the cheap path; always pays for A's guaranteed failure first")
    print(f"  StaticBlacklist (today's typical memory): {_day_of_convergence(static['daily_costs'], 1, B_FIXED_ON_DAY)}"
          f"   <- permanently blacklisted B after its FIRST failure, can never recover")
    print(f"  CEMG:                                   {_day_of_convergence(cemg['daily_costs'], 1, B_FIXED_ON_DAY)}"
          f"   <- rediscovers B works via scheduled re-verification, then stays converged")

    print()
    print(f"CEMG final classification memory: {cemg['final_memory']}")
    print(f"StaticBlacklist final blacklist (permanent, forever): {static['final_blacklist']}")
    print()
    savings_vs_static  = static["total_cost"] - cemg["total_cost"]
    savings_vs_realistic_nomem = no_mem_realistic["total_cost"] - cemg["total_cost"]
    print(f"CEMG total savings vs StaticBlacklist (today's typical memory tool): "
          f"{savings_vs_static} cost units ({savings_vs_static/static['total_cost']*100:.1f}% cheaper)")
    print(f"CEMG total savings vs realistic NoMemory (alphabetical order):      "
          f"{savings_vs_realistic_nomem} cost units ({savings_vs_realistic_nomem/no_mem_realistic['total_cost']*100:.1f}% cheaper)")


if __name__ == "__main__":
    main()
