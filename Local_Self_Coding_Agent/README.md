# Self-Coding Agent

A small, fully local multi-agent system that takes a one-line idea and runs it
through a complete software lifecycle — research, plan, scaffold, code, test,
review, fix, innovate, document — looping until the project is complete or it
stops making progress. Inspired by Claude Code, scaled down, Python-only, no
paid APIs: every LLM call goes through a local (or tunnelled) Ollama server.

## Stack

- **Streamlit** — UI to launch projects and watch the loop live.
- **LangGraph** — orchestrates the agents as a state machine with a feedback loop.
- **LangChain (langchain-ollama)** — LLM access through Ollama, JSON-mode calls.
- **LlamaIndex + FAISS** — `memory.py` keeps a persistent vector index of each
  project's prior docs and reports so later iterations recall what was built
  without re-reading everything. Embeddings come from Ollama.
- **Pydantic** — every value passed between agents is a typed model; `ProjectState`
  is the single contract.
- **Ollama** — runs the local models: one reasoning model, one coding model, one
  embedding model.

Generated projects are always **Python** and are tested with **pytest**.

## Layout

```
self_coding_agent/
├── streamlit_app.py            # UI entry point (light imports only)
├── config.json                 # models, host, limits (the main config)
├── requirements.txt            # the agent's own dependencies
├── scripts/run_pipeline.py     # headless CLI runner (no Streamlit)
├── agent_pipeline/
│   ├── config.py               # reads config.json; env vars override
│   ├── models.py               # Pydantic state + event models
│   ├── llm.py                  # Ollama wrappers + JSON-validated calls
│   ├── memory.py               # FAISS vector store over prior iterations
│   ├── events.py               # JSONL event log + state snapshots
│   ├── workspace.py            # file IO, venv, pytest, import analysis, green snapshots
│   ├── graph.py                # LangGraph wiring + loop control
│   ├── runner.py               # threaded entry the UI calls
│   └── agents/                 # one module per agent: run(state, logger)
├── projects/<name>/            # generated source, tests, docs (kept clean)
└── workspace/<name>/           # per-project runtime: venv, events, state, RAG, snapshots
```

## Setup

```bash
python -m venv /path/to/agent-venv && source /path/to/agent-venv/bin/activate
pip install -r requirements.txt

# Ollama (installed separately) — pull the three models:
ollama pull llama3.1:8b          # reasoning
ollama pull qwen2.5-coder:7b     # coding
ollama pull nomic-embed-text     # embeddings
```

## Configuration

All settings live in `config.json` at the repo root:

```json
{
  "ollama_host": "http://127.0.0.1:11434",
  "reasoning_model": "llama3.1:8b",
  "coding_model": "qwen2.5-coder:7b",
  "embedding_model": "nomic-embed-text:latest",
  "max_iterations": 5,
  "max_fix_attempts": 5,
  "research_results": 5,
  "llm_temperature": 0.2,
  "llm_timeout": 180
}
```

Precedence is **environment variable > config.json > built-in default**, so you
can override any single value for a one-off run without editing the file:

| config.json key   | env override            |
| ----------------- | ----------------------- |
| `ollama_host`     | `OLLAMA_HOST`           |
| `reasoning_model` | `SCA_REASONING_MODEL`   |
| `coding_model`    | `SCA_CODING_MODEL`      |
| `embedding_model` | `SCA_EMBED_MODEL`       |
| `max_iterations`  | `SCA_MAX_ITERATIONS`    |
| `max_fix_attempts`| `SCA_MAX_FIX_ATTEMPTS`  |
| `research_results`| `SCA_RESEARCH_RESULTS`  |
| `llm_temperature` | `SCA_TEMPERATURE`       |
| `llm_timeout`     | `SCA_TIMEOUT`           |

A missing or malformed `config.json` falls back to the built-in defaults. The
Streamlit sidebar reads the same file, so the UI always reflects the live config.

## Run

```bash
streamlit run streamlit_app.py
```

Then in the browser:

1. Enter a project name + idea, click **Launch pipeline**.
2. Watch live agent events, generated files, iteration reports, and the final
   `PROJECT_DOC.md`. Use **🔄 Refresh** to pull new events while a run is live.
3. On a finished project, **Continue** runs another iteration; **Force new
   features** makes the innovator propose work even if it thinks it's done.

Headless alternative:

```bash
python scripts/run_pipeline.py --name demo --idea "A CLI todo list with JSON storage"
# optional: --max-iterations 3
```

## The loop

Each iteration walks this graph. Solid arrows are the main path; the bracketed
routes are conditional (chosen by the router functions in `graph.py`):

```
start_iteration
  → research        web search, fetch & clean each page, recall prior RAG context
  → organize        distil a project brief
  → requirements    derive testable requirements (additive across iterations)
  → scaffold        plan files + deps, seed placeholders, build the venv, write conftest
  → coder           implement placeholder files; extend for new features
  → tester          author tests, run pytest
  → test_doctor     if tests fail, fix bugs in the TESTS (never the source)
  → reviewer        structured code review
  → [if failing] bug_finder → bug_fixer → loop:
        ├─ passing            → innovator
        ├─ progress made      → bug_finder (fix more source)
        ├─ stuck once         → test_doctor (maybe the test side is wrong)
        ├─ stuck hard / capped → innovator (give up; finalize reverts to green)
  → innovator       propose the next features (or none)
  → cleaner         delete placeholder files + build/cache dirs
  → reporter        write the per-iteration report
  → doc_writer      (re)write PROJECT_DOC.md
  → finalize        keep the project green, then iterate or stop
```

### Staying green and converging

The hard part of a weak local model is that it writes code (and tests) it can't
always fix. Several mechanisms keep the loop honest and bounded:

- **Green snapshot + revert.** Whenever an iteration ends with all tests passing,
  the project files *and* the requirements/plan are snapshotted
  (`workspace/<name>/.last_green/` + `.green_state.json`). If a later iteration
  ends red, `finalize` restores that snapshot and re-runs the tests — so the
  project on disk is **always left in a passing state**.

- **Abandoned features.** When a feature can't be made to pass, the innovator
  records it and is told not to propose it (or anything similar) again. Reverting
  the requirements alongside the files means the tester won't regenerate the same
  broken tests next iteration — this is what stops the macro-loop of re-adding the
  same unfixable feature forever.

- **No-progress detection.** The bug-fixer compares the set of failing tests
  before and after each patch. If the failures don't change, a no-progress streak
  climbs and the loop bails early instead of burning every attempt.

- **Three independent stop guards** inside the fix loop: `max_fix_attempts`,
  a no-progress streak of 3, and at most 2 `test_doctor` visits. The loop cannot
  run forever.

**The whole run stops when** the innovator proposes no new features,
`max_iterations` is hit, or tests fail for `max_consecutive_failures` iterations
in a row.

**Resumes cleanly.** After the first iteration, `research` pulls the relevant
slices of the prior `PROJECT_DOC.md` and iteration reports from the FAISS index
instead of starting from scratch.

## Generated project vs. workspace

The generated source stays clean and self-describing:

```
projects/<name>/
├── requirements.txt      # rewritten from the plan, deduped by package name
├── conftest.py           # puts the project root on sys.path for pytest
├── <module>.py ...       # implementation
├── tests/                # pytest tests
└── PROJECT_DOC.md        # living doc, read by the next iteration
```

Everything the agent manages lives separately under the workspace:

```
workspace/<name>/
├── .venv/                    # the project's own venv (pip installs land here)
├── events.jsonl             # canonical log the UI tails
├── state.json               # latest pipeline state snapshot
├── iterations/iter_NN/      # per-iteration report.md
├── rag_store/               # persisted FAISS index
├── .last_green/             # last all-green file snapshot (for revert)
├── .green_state.json        # requirements/plan that were green
└── .running                 # liveness marker the UI checks
```

The project's `.venv` is isolated from the agent's own environment: every
pip/pytest subprocess runs with `PYTHONPATH`/`VIRTUAL_ENV` stripped and
`PIP_REQUIRE_VIRTUALENV=1`, so the generated code can never import
`agent_pipeline` or pick up the agent's dependencies.

## Design notes

The pipeline accumulated a set of guard rails to keep a small local model honest:

- **Two specialized fixers, disjoint domains.** `bug_fixer` owns source files and
  is forbidden from touching tests; `test_doctor` owns tests and is forbidden from
  touching source. `test_doctor` runs first to decide whether the *tests* are
  wrong (bad expected values, missing imports, absurd stress inputs) before the
  bug-fixer attacks the source — and it re-enters the loop when the source path
  gets stuck, since a bad test import can only be fixed on the test side.

- **Fixers see the real error, not a digest.** `run_tests` stores pytest's actual
  `FAILURES`/`ERRORS` traceback region (the `ImportError`/`AssertionError` lines),
  and the bug agents receive that plus the failing test code and the source — so
  the model fixes the named cause instead of guessing.

- **Hang handling.** A hanging test or import is the worst case for a weak model.
  pytest runs with a per-test timeout (`pytest-timeout`) and a wall-clock cap;
  on a timeout the result names the test files that never reported a result (the
  prime suspects) and tells the model to bound or remove them.

- **Deterministic imports.** `workspace.project_exports()` AST-walks the source to
  map every public name to its module. The tester/test_doctor get that map in
  their prompt, and `auto_inject_imports()` AST-checks each generated test and
  prepends any missing `from X import Y` — so tests don't fail collection with
  `NameError`.

- **No-regression guard.** In extension mode the coder may only modify a file if
  it's new, an entry point, or named by a pending feature. A working, tested
  module can't be silently rewritten while adding an unrelated feature.

- **Additive iterations.** requirements/scaffold/coder/tester preserve prior work
  and only add what a new feature needs, rather than regenerating the project.

- **Robust LLM JSON.** `chat_json()` retries on schema-echo, bad JSON, and
  validation errors, and coerces common drift (e.g. a list-of-dicts where a list
  of strings was expected).

- **Self-cleaning.** The `cleaner` agent removes empty placeholder files and
  `__pycache__`/`.pytest_cache` so the project tree stays tidy.

- **Fast, non-blocking UI.** Streamlit imports nothing heavy at module load; the
  agent backend is imported lazily inside the Launch/Continue handlers. Source
  scans prune venv/build dirs and the events log is tail-read.

## Adding an agent

Drop a module in `agent_pipeline/agents/` exposing
`run(state, logger) -> ProjectState`, register it in `agents/__init__.py`, and
wire it into `graph.py`. The Pydantic `ProjectState` is the only contract — read
what you need, return what you changed.
```
