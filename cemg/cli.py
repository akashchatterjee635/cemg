"""
cemg/cli.py
-----------
A single command-line entry point for everything CEMG can do --
inspect memory, check status before acting, run the demo, run the
convergence simulation, prune stale data, and launch the API server.

This is a thin layer over cemg/memory.py and cemg/graph.py -- no new
logic lives here. The point is ergonomics: quickly poking at memory
state during development shouldn't require writing a new Python script
or crafting a curl command every time.

Usage (after `pip install -e .`, see pyproject.toml):
    cemg recall --agent-id demo_agent --query "read the config file"
    cemg peek --agent-id demo_agent --tool read_file --params '{"path":"x.txt"}'
    cemg store --agent-id demo_agent --session-id s1 --action "..." --outcome failure
    cemg prune --dry-run
    cemg health
    cemg demo
    cemg simulate
    cemg serve

Or without installing, from the repo root:
    python -m cemg.cli recall --agent-id demo_agent
"""

from __future__ import annotations

import json
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

app = typer.Typer(
    name="cemg",
    help="Causal Experience Memory Graph -- inspect, debug, and run the memory layer from the command line.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


def _get_driver_or_exit():
    """
    Shared connection helper for every DB-backed command.
    """
    from cemg.storage import get_storage_provider
    from cemg.graph import bootstrap_schema
    driver = get_storage_provider()
    bootstrap_schema(driver)
    return driver


# -- recall --------------------------------------------------------------------
@app.command()
def recall(
    agent_id:       str  = typer.Option(..., "--agent-id", help="Agent identifier"),
    query:          str  = typer.Option("", "--query", help="Current task -- ranks recall by relevance to this"),
    namespace:      Optional[str] = typer.Option(None, "--namespace", help="Filter to one task namespace"),
    top_k:          int  = typer.Option(10, "--top-k"),
    show_block:     bool = typer.Option(False, "--as-prompt", help="Show the formatted LLM-prompt block instead of a table"),
):
    """Show the most relevant memories for an agent right now."""
    from cemg.memory import recall_relevant, build_memory_block

    driver = _get_driver_or_exit()

    if show_block:
        block = build_memory_block(driver, agent_id, query_action=query, task_namespace=namespace, top_k=top_k)
        console.print(Panel(block or "(no memory yet)", title="Memory block (as sent to the LLM)", border_style="cyan"))
        return

    experiences = recall_relevant(driver, agent_id, query_action=query, task_namespace=namespace, top_k=top_k)
    if not experiences:
        console.print("[dim]No memory found for this agent yet.[/dim]")
        return

    table = Table(box=box.ROUNDED, show_header=True, title=f"CEMG recall -- agent_id={agent_id}")
    table.add_column("Status", style="bold")
    table.add_column("Action")
    table.add_column("Cause")
    table.add_column("Weight", justify="right")
    table.add_column("Rel.", justify="right")
    table.add_column("Score", justify="right")

    status_style = {
        "ACTIVE_FAILURE": "red", "CONFIRMED_BROKEN": "red",
        "PROBATION": "yellow", "RESOLVED": "green", "CLEAN": "green",
    }
    for e in experiences:
        style = status_style.get(e.verification_status, "white")
        table.add_row(
            f"[{style}]{e.verification_status}[/{style}]",
            e.action[:50],
            (e.observed_error or e.reasoning or "")[:40],
            f"{e.temporal_weight:.2f}",
            f"{e.relevance:.2f}",
            f"{e.score:.2f}",
        )
    console.print(table)


# -- peek ------------------------------------------------------------------------
@app.command()
def peek(
    agent_id:   str = typer.Option(..., "--agent-id"),
    tool:       str = typer.Option(..., "--tool"),
    params:     str = typer.Option("{}", "--params", help="JSON-encoded params dict"),
    namespace:  str = typer.Option("default", "--namespace"),
):
    """
    Check verification status for a (tool, params) call BEFORE running it.
    Exactly what CEMGAgent calls internally before every tool execution --
    useful to manually check "would this be flagged?" during debugging.
    """
    from cemg.memory import peek_signature_status

    driver = _get_driver_or_exit()
    try:
        parsed_params = json.loads(params)
    except json.JSONDecodeError:
        console.print(f"[red]--params must be valid JSON, got:[/red] {params!r}")
        raise typer.Exit(code=1)

    result = peek_signature_status(driver, agent_id, tool, parsed_params, namespace)
    style = {"ACTIVE_FAILURE": "red", "CONFIRMED_BROKEN": "red", "PROBATION": "yellow"}.get(result["status_before"], "green")
    console.print(f"Signature: [dim]{result['action_signature']}[/dim]")
    console.print(f"Status:    [{style}]{result['status_before']}[/{style}]")


# -- store -----------------------------------------------------------------------
@app.command()
def store(
    agent_id:       str = typer.Option(..., "--agent-id"),
    session_id:     str = typer.Option(..., "--session-id"),
    action:         str = typer.Option(..., "--action"),
    outcome:        str = typer.Option(..., "--outcome", help="success | failure | partial"),
    reasoning:      str = typer.Option("", "--reasoning"),
    observed_error: str = typer.Option("", "--observed-error", help="Raw tool error text, if outcome=failure"),
    tool:           str = typer.Option("", "--tool"),
    params:         str = typer.Option("{}", "--params"),
    namespace:      str = typer.Option("default", "--namespace"),
):
    """Manually store one experience -- mainly for testing/seeding during development."""
    from cemg.memory import store_experience

    driver = _get_driver_or_exit()
    try:
        parsed_params = json.loads(params)
    except json.JSONDecodeError:
        console.print(f"[red]--params must be valid JSON, got:[/red] {params!r}")
        raise typer.Exit(code=1)

    result = store_experience(
        driver=driver, agent_id=agent_id, session_id=session_id, action=action,
        outcome=outcome, reasoning=reasoning, observed_error=observed_error,
        tool=tool, params=parsed_params, task_namespace=namespace,
    )
    console.print(f"[green]Stored.[/green] exp_id={result['exp_id']}  "
                  f"action_signature={result['action_signature']}  failure_class={result['failure_class']}")


# -- prune -----------------------------------------------------------------------
@app.command()
def prune(
    agent_id:  Optional[str] = typer.Option(None, "--agent-id", help="Limit to one agent; omit for all"),
    live:      bool = typer.Option(False, "--live", help="Actually delete. Default is a dry run that only reports what WOULD be deleted."),
):
    """Delete decayed, no-longer-tracked experiences. Always dry-runs by default."""
    from cemg.memory import prune as prune_fn

    driver = _get_driver_or_exit()
    dry_run = not live
    result = prune_fn(driver, agent_id=agent_id, dry_run=dry_run)

    mode = "[yellow]DRY RUN[/yellow]" if dry_run else "[red]LIVE DELETE[/red]"
    console.print(f"{mode} -- eligible: {result['eligible_count']}, deleted: {result['deleted']}")
    if dry_run and result["eligible_count"] > 0:
        console.print("[dim]Re-run with --live to actually delete these.[/dim]")


# -- health ------------------------------------------------------------------------
@app.command()
def health():
    """Check connectivity to storage and LLM provider configuration."""
    from cemg.storage import get_storage_provider
    import os

    try:
        store = get_storage_provider()
        ok = store.is_healthy()
        store_type = type(store).__name__
        console.print(f"Storage ({store_type}):  {'[green]connected[/green]' if ok else '[red]unreachable[/red]'}")
        store.close()
    except Exception as e:
        console.print(f"Storage:  [red]error: {e}[/red]")

    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_openai    = bool(os.getenv("OPENAI_API_KEY"))
    if has_anthropic:
        console.print("LLM:    [green]Claude configured[/green] (ANTHROPIC_API_KEY set)")
    elif has_openai:
        console.print("LLM:    [green]OpenAI-compatible configured[/green] (OPENAI_API_KEY set)")
    else:
        console.print("LLM:    [red]none configured[/red] -- set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env")


# -- demo / simulate / serve -- thin wrappers around existing scripts -----------
@app.command()
def demo():
    """Run the session-1-fails / session-2-avoids-it demo."""
    import subprocess
    subprocess.run([sys.executable, "demo/run_demo.py"])


@app.command()
def simulate():
    """
    Run the dependency-free convergence simulation (no Neo4j/API key
    required) -- compares CEMG's real decay/classification code against
    a no-memory baseline and a static-blacklist baseline.
    """
    import subprocess
    subprocess.run([sys.executable, "eval/simulate_convergence.py"])


@app.command()
def serve(
    port: int = typer.Option(8100, "--port"),
    no_reload: bool = typer.Option(False, "--no-reload", help="Disable auto-reload (recommended outside local dev)"),
):
    """Launch the FastAPI server (same as running uvicorn directly)."""
    import uvicorn
    uvicorn.run("cemg.api:app", host="0.0.0.0", port=port, reload=(not no_reload))


if __name__ == "__main__":
    app()
