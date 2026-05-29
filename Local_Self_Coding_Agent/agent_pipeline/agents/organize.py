from __future__ import annotations

from pydantic import BaseModel

from ..events import EventLogger
from ..llm import chat_json, reasoning_llm
from ..models import ProjectState, ResearchNote

SYSTEM = (
    "You are the Organizer Agent. Read the project idea and the research notes and "
    "produce a focused brief that defines the scope, the core capabilities, the "
    "non-goals, and the technical building blocks. Be concrete. Keep it usable as "
    "the input for a requirements engineer."
)


class _Brief(BaseModel):
    scope: str
    core_capabilities: list[str]
    non_goals: list[str]
    building_blocks: list[str]


def run(state: ProjectState, logger: EventLogger) -> ProjectState:
    """Turn the idea and research into a structured project brief."""
    logger.log(state, "organize", "distilling research into a project brief")

    research_block = "\n".join(
        f"- {n.title}: {n.summary}" for n in state.research
    ) or "(no research notes available)"

    feature_block = "\n".join(
        f"- {f.title}: {f.motivation}" for f in state.pending_features
    )
    feature_section = (
        f"# Carry-over feature ideas from previous iteration\n{feature_block}\n\n"
        if feature_block else ""
    )

    user = (
        f"# Idea\n{state.idea}\n\n"
        f"# Research notes\n{research_block}\n\n"
        f"{feature_section}"
        "Return a JSON brief that organizes scope, core capabilities, non-goals, and building blocks. "
        "Building blocks should be expressed in terms idiomatic to Python."
    )
    brief = chat_json(reasoning_llm(json_mode=True), SYSTEM, user, _Brief)

    summary = (
        f"Scope: {brief.scope}\n"
        f"Capabilities: {', '.join(brief.core_capabilities)}\n"
        f"Non-goals: {', '.join(brief.non_goals)}\n"
        f"Building blocks: {', '.join(brief.building_blocks)}"
    )
    state.research.append(ResearchNote(source="organizer", title="Project brief", summary=summary))

    logger.log(state, "organize", "brief ready", level="result", brief=brief.model_dump())
    return state
