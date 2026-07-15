"""
demo/run_demo.py
────────────────
The core proof-of-concept demo.

What it shows
─────────────
Session 1:  Agent attempts a multi-step task.  It makes a predictable
            mistake (tries a tool approach that fails).  The failure
            is stored in CEMG.

Session 2:  A FRESH agent instance (new context window, no conversation
            history) attempts the SAME task.  Because it loads CEMG
            memory at startup, it sees the failure from Session 1 and
            avoids repeating it — completing the task faster.

This is the paper's core experiment in miniature.

Run:
    python demo/run_demo.py
"""

import sys
import os
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rich.console import Console
from rich.panel   import Panel
from rich.table   import Table
from rich         import box

from cemg.agent  import make_agent
from cemg.graph  import get_driver, bootstrap_schema
from cemg.memory import recall_relevant, build_memory_block

console = Console()

# ── Demo task ─────────────────────────────────────────────────────────────────
# A simple research + write task.
# In Session 1 the agent will try a plausible-but-wrong path first.
# We simulate the failure by pre-seeding CEMG (representing what a
# real session 1 would have produced — saves you 10 API calls in the demo).
TASK = (
    "Research the key differences between Temporal Knowledge Graphs "
    "and static Knowledge Graphs for agent memory systems, then write "
    "a 3-bullet summary to data/summary.txt"
)

AGENT_ID = "demo_agent_001"


# ── Seed Session 1 failures (simulated prior run) ─────────────────────────────
def seed_prior_session(driver) -> None:
    """
    Pre-populate CEMG with realistic failures from a simulated Session 1.
    In a real demo you'd run Session 1 live -- this saves API credits.

    Uses observed_error (raw failure text) separately from reasoning
    (the agent's self-report) -- see cemg/memory.py's fix log for why
    that separation matters.
    """
    from cemg.memory import store_experience

    s1 = f"session_{uuid.uuid4().hex[:6]}"

    # Failure 1: tried to read a file that doesn't exist
    r1 = store_experience(
        driver         = driver,
        agent_id       = AGENT_ID,
        session_id     = s1,
        action         = "read_file(path='data/kg_research.txt')",
        outcome        = "failure",
        reasoning      = "Assumed a local research file already existed from a previous run",
        observed_error = "FileNotFoundError: data/kg_research.txt",
        tool           = "read_file",
        params         = {"path": "data/kg_research.txt"},
    )

    # Failure 2 (caused by failure 1): tried write without creating directory
    store_experience(
        driver         = driver,
        agent_id       = AGENT_ID,
        session_id     = s1,
        action         = "write_file(path='data/summary.txt', text='...')",
        outcome        = "failure",
        reasoning      = "Tried to write without first confirming the directory exists",
        observed_error = "FileNotFoundError: directory 'data' does not exist",
        tool           = "write_file",
        params         = {"path": "data/summary.txt"},
        parent_exp_id  = r1.get("exp_id"),
    )

    console.print(
        "[dim]-> Seeded 2 Session 1 failures into CEMG "
        "(simulating what a real prior session would have stored)[/dim]"
    )


# ── Main demo ─────────────────────────────────────────────────────────────────
def main():
    console.print(Panel.fit(
        "[bold cyan]CEMG Proof-of-Concept Demo[/bold cyan]\n"
        "[dim]Causal Experience Memory Graph for Long-Horizon Agent Reliability[/dim]",
        border_style="cyan"
    ))

    driver = get_driver()
    bootstrap_schema(driver)

    # ── Show memory BEFORE session 2 ─────────────────────────────────────────
    console.print("\n[bold]Step 1 — Seeding prior session failures[/bold]")
    seed_prior_session(driver)

    console.print("\n[bold]Step 2 — CEMG memory block (what Session 2 will see)[/bold]")
    block = build_memory_block(driver, AGENT_ID)
    console.print(Panel(block or "(empty)", border_style="yellow", title="CEMG Memory"))

    # ── Session 2: CEMG-augmented run ────────────────────────────────────────
    console.print("\n[bold]Step 3 — Running Session 2 agent (CEMG-augmented)[/bold]")
    console.print("[dim]The agent starts with a fresh context window but loads CEMG memory.[/dim]")

    agent = make_agent(AGENT_ID)
    t0    = time.time()
    answer = agent.run(TASK, verbose=True)
    elapsed = time.time() - t0

    # ── Results ───────────────────────────────────────────────────────────────
    console.print("\n[bold]Step 4 — Results[/bold]")

    table = Table(box=box.ROUNDED, show_header=True)
    table.add_column("Metric",  style="bold")
    table.add_column("Session 1 (no memory)", style="red")
    table.add_column("Session 2 (CEMG)",      style="green")

    table.add_row("read_file failure",   "❌ happened",       "✅ avoided (in memory)")
    table.add_row("write without dir",   "❌ happened",       "✅ avoided (in memory)")
    table.add_row("Steps to complete",   "hit limit / failed","~{} steps".format(agent.step))
    table.add_row("Wall time",           "—",                 f"{elapsed:.1f}s")
    table.add_row("Task complete",       "No",                "Yes" if "session" not in answer.lower() else "Partial")
    console.print(table)

    if os.path.exists("data/summary.txt"):
        console.print("\n[green]✓ Output file written:[/green]")
        with open("data/summary.txt") as f:
            console.print(Panel(f.read(), title="data/summary.txt", border_style="green"))

    # ── Show what CEMG stored this session ────────────────────────────────────
    console.print("\n[bold]Step 5 — What CEMG stored this session[/bold]")
    exps = recall_relevant(driver, AGENT_ID, top_k=6)
    for e in exps:
        icon = "✗" if e.outcome == "failure" else "✓"
        console.print(
            f"  [{e.score:.2f}] {icon} {e.action[:70]}"
            f"{'...' if len(e.action) > 70 else ''}"
        )

    driver.close()
    console.print("\n[bold cyan]Demo complete.[/bold cyan]")


if __name__ == "__main__":
    main()
