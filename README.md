# CEMG -- Causal Experience Memory Graph

> A temporal causal memory API for long-horizon LLM agents.
> No training required. Works with any LLM. Runs on your existing Neo4j.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue)]()
[![Neo4j 5.x](https://img.shields.io/badge/neo4j-5.x-green)]()
[![Tests](https://img.shields.io/badge/tests-48%20passing-brightgreen)]()

---

## The problem in one line

Current agent memory systems store *what the agent knows*.
CEMG stores *what the agent tried, whether it worked, why, and whether
that reason still holds* -- so the agent never blindly repeats the same
mistake, and never gets permanently stuck avoiding something that's
since been fixed.

## What makes this different from a simple "avoid list"

| Capability | Naive avoid-list | CEMG |
|---|:---:|:---:|
| Recent failures ranked above old ones | ✗ (or frozen at write time) | ✓ live-recomputed decay |
| Retrieval filtered by current task relevance | ✗ | ✓ keyword-overlap scoring |
| Transient (server hiccup) vs structural (wrong reasoning) failures | ✗ (one bucket) | ✓ separate decay rates |
| A failure can be re-tested once its cooldown passes | ✗ (permanent block) | ✓ ACTIVE_FAILURE -> PROBATION -> RESOLVED/CONFIRMED_BROKEN |
| Cause is the raw error, not the agent's self-report | ✗ | ✓ `observed_error` kept separate from `reasoning` |
| Cost (tokens/steps) shown alongside reliability | ✗ | ✓ so the LLM sees a real tradeoff, not just avoid/allow |
| Stored-prompt-injection defense on external content | ✗ | ✓ sanitisation before write |
| Cross-task memory isolation | ✗ | ✓ `task_namespace` |
| Graceful degradation if the DB is unreachable | ✗ (crashes) | ✓ falls back to memory-less operation |
| Decay-triggered deletion (PII / unbounded growth) | ✗ | ✓ `/memory/prune` |
| Verification status correctly isolated per task | ✗ | ✓ `task_namespace` in the ActionSignature match key, not just on raw experiences |
| Compliance measured at the moment of decision | ✗ | ✓ pre-action snapshot via `peek_signature_status`, not a post-hoc re-query |

## Quickstart

```bash
git clone https://github.com/akash/cemg
cd cemg
bash setup.sh            # installs deps, starts Neo4j, copies .env
# -> edit .env: add your ANTHROPIC_API_KEY or OPENAI_API_KEY
pytest tests/ -v          # 58 tests, zero setup required (no live DB needed)
python demo/run_demo.py   # the PoC demo
```

## CLI

`pip install -e .` gives you a real `cemg` command (via `pyproject.toml`'s
`console_scripts` entry point), so you don't need to remember file paths
or write throwaway scripts to poke at memory during development:

```bash
pip install -e .

cemg --help                                        # list every command
cemg health                                         # check Neo4j + LLM config
cemg simulate                                       # dependency-free convergence proof (no Neo4j/API key needed)
cemg demo                                           # session-1-fails / session-2-avoids-it demo
cemg recall --agent-id a1 --query "read config"     # inspect memory as a table
cemg recall --agent-id a1 --as-prompt               # see the exact block sent to the LLM
cemg peek --agent-id a1 --tool read_file --params '{"path":"x.txt"}'   # check status BEFORE acting
cemg store --agent-id a1 --session-id s1 --action "..." --outcome failure --observed-error "..."
cemg prune --agent-id a1                            # dry run (default) -- reports what would be deleted
cemg prune --agent-id a1 --live                     # actually deletes
cemg serve --port 8100                              # launch the FastAPI server
```

Every command is a thin wrapper over `cemg/memory.py` -- no logic lives
in the CLI itself, so the library, the API, and the CLI all stay in sync
by construction.

**A note on dependency pinning, kept here rather than hidden:** the
original `typer==0.12.3` / `rich==13.7.1` pins from earlier in this
project were stale enough to be incompatible with the `click` version
pip resolves alongside them today (`Parameter.make_metavar() missing
1 required positional argument: 'ctx'` -- a real error hit and fixed
while building this CLI, not a hypothetical one). Both are now pinned
as ranges (`typer>=0.15,<0.27`, `rich>=13.7,<16`) tested against a
clean install rather than exact versions that silently drift out of
sync with their own dependencies over time.

## Project structure

```
cemg/
├── cemg/
│   ├── graph.py       # Neo4j schema, class-aware decay, verification status, pruning
│   ├── memory.py      # public API: store / recall / causal_path / compliance / prune
│   ├── classify.py    # failure classification (transient/structural) + cooldown state machine
│   ├── security.py    # content sanitisation against stored prompt injection
│   ├── llm.py         # provider-agnostic wrapper (Claude, OpenAI, Mistral...)
│   ├── agent.py       # CEMG-augmented ReAct agent loop, graceful degradation
│   └── api.py         # FastAPI -- REST endpoints
│   └── cli.py         # CLI -- thin wrapper over memory.py for dev/debugging
├── demo/
│   └── run_demo.py    # session 1 (fails) vs session 2 (CEMG avoids repeating it)
├── eval/
│   └── baselines.py   # B1 NoMemory, B2 TextCompression, CEMG -- properly seeded comparison
├── tests/
│   ├── test_core.py       # decay, relevance, task-success checks
│   ├── test_classify.py   # failure classification + verification state machine
│   └── test_security.py   # sanitisation against stored prompt injection
├── setup.sh
├── requirements.txt
└── .env.example
pyproject.toml (root, alongside this file) # packaging + console_scripts entry point for the `cemg` command
```

## The API endpoints

```
POST /memory/store_experience     store one action-outcome, with observed_error kept
                                   separate from the agent's self-reported reasoning
GET  /memory/recall_relevant      ranked retrieval, live decay + relevance + verification status
GET  /memory/causal_path/{id}     reconstruct the decision chain to any outcome
GET  /memory/context_block        formatted memory string ready for an LLM system prompt
POST /memory/check_compliance     did the agent actually avoid what it was told to avoid
POST /memory/prune                decay-triggered deletion (dry_run=True by default)
GET  /health                      reports Neo4j connectivity distinctly from "all fine"
```

Docs auto-generated at `http://localhost:8100/docs` when the server is running.

## Why failures aren't just "avoid forever"

Every stored failure carries a `failure_class` (`transient` or `structural`),
detected from the raw tool error text -- never from the agent's own
explanation, since that can be a confabulated post-hoc story. Each class
decays at its own rate:

- **transient** (timeouts, 5xx, rate limits): half-life ~3 days. A server
  hiccup fades from memory fast, on its own, with no manual cleanup needed.
- **structural** (wrong assumption, wrong tool choice): half-life ~100 days.
  A real mistake stays flagged for a long time, because the reasoning that
  caused it doesn't fix itself just because time passed.

Once a failure's cooldown (`1/lambda` for its class) has passed, its status
moves from `ACTIVE_FAILURE` (hard avoid) to `PROBATION` (worth re-testing,
not a permanent block). If it's retried and succeeds, it becomes `RESOLVED`.
If retried and fails again, it becomes `CONFIRMED_BROKEN` -- stronger
evidence than a single failure.

## Running the comparison eval

```bash
python eval/baselines.py
```

Runs B1 (no memory), B2 (text compression), and CEMG -- with CEMG and B2
**both seeded with the same prior-failure narrative** before every run,
so the comparison actually tests whether structured memory helps more
than an unstructured summary, rather than comparing CEMG-with-no-memory
to a baseline that also has no memory.

Prints mean +/- std for steps and failures, a compliance rate (did the
agent avoid what its own memory flagged), and paired significance tests
(CEMG vs. NoMemory, CEMG vs. TextCompression) when `n_runs >= 3`.

## Key tunable parameters (.env)

| Variable | Default | Effect |
|---|---|---|
| `CEMG_LAMBDA` | `0.03` | Decay speed for successes/unclassified experiences |
| `CEMG_TOP_K` | `10` | Max experiences returned by /recall |
| `CEMG_FAILURE_BOOST` | `2.0` | How much failures are upweighted vs successes |
| `CEMG_RELEVANCE_WEIGHT` | `1.5` | How much task-relevance boosts a memory's score |
| `CEMG_PRUNE_FLOOR` | `0.02` | Decay weight below which an experience is eligible for deletion |
| `CEMG_DEFAULT_NAMESPACE` | `default` | Task namespace used when none is given |
| `CEMG_MAX_STEPS` | `15` | Max steps per agent session |

Failure-class decay constants (`transient=0.30`, `structural=0.01`) live
in `cemg/classify.py` rather than `.env` -- they're a research parameter
worth version-controlling, not an operational knob.

## How to use CEMG in your own agent (3 lines)

```python
from cemg import make_agent

agent = make_agent("my_agent_id", task_namespace="my_project")
answer = agent.run("your task here")   # memory auto-loads + auto-saves, degrades gracefully if Neo4j is down
```

## Known limitations (stated, not hidden)

- **Action signatures are exact-match**, not fuzzy -- `read_file(path=A)`
  and `read_file(path=B)` are tracked as different signatures. Grouping
  structurally similar calls is a real extension, not yet built.
- **Relevance scoring is keyword overlap**, not semantic embedding --
  fast and dependency-free, but "search the web" and "look this up
  online" won't overlap. Swap in an embedding model if you need
  semantic matching; the ranking formula doesn't care how relevance
  is computed.
- **Sanitisation is pattern-based**, not a complete defense against
  stored prompt injection -- it reduces blast radius, it doesn't
  guarantee safety. Structural separation of "data" from "instructions"
  at the model/tooling level is the more complete fix. It also currently
  covers `reasoning`/`observed_error` only; raw tool RESULT content isn't
  persisted to CEMG at all yet, so today's exposure is narrower than it
  will be once richer memory (storing full tool outputs) gets added --
  worth revisiting the sanitiser's scope at that point.
- **Pruning has no scheduler.** `prune_stale_experiences()` / `POST /memory/prune`
  are correct and tested, but nothing calls them automatically -- wire
  up a cron job or background task before relying on this for real
  data retention compliance.

### Fixed in the most recent pass (previously open issues)

- ~~ActionSignature aggregates leaking verification status across
  task_namespace boundaries~~ -- fixed: `task_namespace` is now part of
  the MERGE key for every ActionSignature read and write, not just on
  raw Experience recall.
- ~~Compliance checked after the run against live state~~ -- fixed:
  `CEMGAgent` now calls `peek_signature_status()` immediately before
  each action executes and stores the result in `decision_snapshots`;
  `evaluate_compliance()` is a pure function over those snapshots, with
  a dedicated regression test proving it catches a violation that a
  post-hoc check would have missed.
- Also found and fixed while addressing the above: the original
  `ActionSignature` uniqueness constraint enforced `signature IS UNIQUE`
  *globally*, which would have broken the moment two different agents
  called the same tool with the same parameters. Replaced with a
  composite index; correctness now comes from the MERGE pattern
  matching all three identity properties together.

## Research context

Inspired by the gap identified in MemoryArena (arXiv:2602.16313, Feb 2026):
*"models that score near-perfectly on passive recall benchmarks drop to
40-60% on active, decision-relevant memory tasks."*

CEMG targets this gap with class-aware temporal decay, a live-recomputed
verification state machine, and a causal-chain structure -- without
requiring model training.
