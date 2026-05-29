from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ..events import EventLogger
from ..llm import chat, reasoning_llm
from ..models import IterationReport, ProjectState

SYSTEM = (
    "You are the Reporter Agent. Summarize what happened in this iteration in a "
    "natural, human-readable way. Cover: what was researched, what was decided, "
    "what was built, what tests revealed, what bugs were found and fixed, and "
    "what the innovator proposed next. Keep it under ~600 words. Use plain prose "
    "with short headings. Do not invent facts."
)


def run(state: ProjectState, logger: EventLogger) -> ProjectState:
    """Render the iteration summary to markdown and store it on disk."""
    logger.log(state, "reporter", "writing iteration report")

    test = state.last_test_result
    facts = (
        f"Iteration: {state.iteration}\n"
        f"Language: Python\n"
        f"Idea: {state.idea}\n"
        f"Requirements: {len(state.requirements)}\n"
        f"Files written: {len(state.files)}\n"
        f"Tests passed: {test.passed if test else 'n/a'}\n"
        f"Test summary: total={test.total if test else 0}, "
        f"failed={test.failed if test else 0}, errors={test.errors if test else 0}\n"
        f"Bugs found this iteration: {len(state.closed_bugs) + len(state.open_bugs)}\n"
        f"Bugs fixed: {len(state.closed_bugs)}\n"
        f"Bugs still open: {len(state.open_bugs)}\n"
        f"Next feature ideas: {[f.title for f in state.pending_features]}"
    )
    body = chat(reasoning_llm(), SYSTEM, f"# Facts\n{facts}\n\nWrite the iteration report.")

    iter_dir = Path(state.workspace_dir) / "iterations" / f"iter_{state.iteration:02d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    (iter_dir / "report.md").write_text(body, encoding="utf-8")

    report = IterationReport(
        iteration=state.iteration,
        started_at=state.history[-1].started_at if state.history and state.history[-1].iteration == state.iteration else datetime.utcnow(),
        finished_at=datetime.utcnow(),
        requirements=list(state.requirements),
        plan=state.plan,
        test_result=test,
        bugs_found=list(state.closed_bugs) + list(state.open_bugs),
        bugs_fixed=[b.bid for b in state.closed_bugs],
        new_features=list(state.pending_features),
        notes=body,
    )
    if state.history and state.history[-1].iteration == state.iteration:
        state.history[-1] = report
    else:
        state.history.append(report)

    logger.log(state, "reporter", "report written", level="result",
               path=str(iter_dir / "report.md"))
    return state
