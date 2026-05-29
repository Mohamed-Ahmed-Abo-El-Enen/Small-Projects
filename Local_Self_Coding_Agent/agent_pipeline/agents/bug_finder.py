from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from ..events import EventLogger
from ..llm import chat_json, reasoning_llm
from ..models import BugReport, ProjectState
from ..workspace import failing_test_files

SYSTEM = (
    "You are the Bug Finder Agent. You are given the EXACT pytest failure "
    "output, the failing test code, and the source. Each failure names a file, "
    "line, and error. Enumerate the concrete source bugs that cause those exact "
    "failures. Each bug must point at a file/function and propose a concrete "
    "fix tied to a specific failure. Use stable ids B001, B002, ... Do NOT "
    "blame the tests — they are the contract."
)


class _Bugs(BaseModel):
    bugs: list[BugReport] = Field(default_factory=list)


def run(state: ProjectState, logger: EventLogger) -> ProjectState:
    """Ask the LLM to list source bugs implied by the exact test failures."""
    test_output = state.last_test_result.output if state.last_test_result else ""
    if state.last_test_result and state.last_test_result.passed:
        logger.log(state, "bug_finder", "tests pass, no bugs to extract", level="result")
        state.open_bugs = []
        return state

    sources = "\n\n".join(
        f"### {f.path}\n```python\n{f.content[:1500]}\n```" for f in state.files
    )
    failing_tests = failing_test_files(test_output, Path(state.project_dir))
    test_block = "\n\n".join(
        f"### {t.path}\n```python\n{t.content[:1200]}\n```" for t in failing_tests
    ) or "(failing test files not located — read the output above)"

    user = (
        f"# EXACT pytest failure output\n{test_output}\n\n"
        f"# Failing test code (the contract)\n{test_block}\n\n"
        f"# Source files\n{sources}\n\n"
        "Return JSON {\"bugs\": [...]}. Severity in {low, medium, high, critical}. "
        "Set resolved to false."
    )
    parsed = chat_json(reasoning_llm(json_mode=True), SYSTEM, user, _Bugs)
    state.open_bugs = parsed.bugs
    logger.log(state, "bug_finder", f"{len(parsed.bugs)} bugs identified", level="result",
               bugs=[b.model_dump() for b in parsed.bugs])
    return state
