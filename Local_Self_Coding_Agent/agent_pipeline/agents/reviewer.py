from __future__ import annotations

from pydantic import BaseModel

from ..events import EventLogger
from ..llm import chat_json, reasoning_llm
from ..models import ProjectState

SYSTEM = (
    "You are the Code Review Agent. You inspect the produced code and the latest "
    "test output. Comment on correctness, structural quality, missing pieces, and "
    "obvious risks. Be specific and reference file paths. Keep findings short and "
    "actionable."
)


class _Review(BaseModel):
    verdict: str
    strengths: list[str]
    weaknesses: list[str]
    risks: list[str]


def run(state: ProjectState, logger: EventLogger) -> ProjectState:
    """Ask the LLM for a structured review of the current code and tests."""
    logger.log(state, "reviewer", "reviewing implementation and test results")

    sources = "\n\n".join(
        f"### {f.path}\n```python\n{f.content[:1500]}\n```" for f in state.files
    )
    test_output = state.last_test_result.output[-1500:] if state.last_test_result else "(no tests ran)"
    user = (
        f"# Sources\n{sources}\n\n"
        f"# Test output\n{test_output}\n\n"
        "Return a JSON review with verdict, strengths, weaknesses, and risks."
    )
    review = chat_json(reasoning_llm(json_mode=True), SYSTEM, user, _Review)
    logger.log(state, "reviewer", "review complete", level="result", review=review.model_dump())
    return state
