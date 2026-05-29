from __future__ import annotations

from pydantic import BaseModel, Field

from ..events import EventLogger
from ..llm import chat_json, reasoning_llm
from ..models import ProjectState, Requirement

SYSTEM_INITIAL = (
    "You are the Requirements Agent. Convert the project brief into a list of "
    "requirements. Every requirement must be testable, atomic, and tagged as "
    "functional, non_functional, or constraint. Write 1-3 acceptance checks for "
    "each. Use stable ids of the form R001, R002, ..."
)

SYSTEM_EXTEND = (
    "You are the Requirements Agent operating in EXTENSION mode. The project "
    "already has requirements that MUST be preserved unchanged. Your job is to "
    "produce ONLY the new requirements needed to cover the pending features. "
    "Do NOT restate or modify existing requirements. Each new requirement must "
    "be testable, atomic, and use a fresh id that does not collide with any "
    "existing one."
)


class _Reqs(BaseModel):
    requirements: list[Requirement] = Field(default_factory=list)


def _next_rid(existing: list[Requirement]) -> str:
    """Return the next free Rxxx id given the existing requirements."""
    used = []
    for r in existing:
        if r.rid.startswith("R") and r.rid[1:].isdigit():
            used.append(int(r.rid[1:]))
    n = max(used) + 1 if used else 1
    return f"R{n:03d}"


def run(state: ProjectState, logger: EventLogger) -> ProjectState:
    """Draft the initial requirements list or append new ones on resume."""
    has_prior = bool(state.requirements)

    if has_prior and not state.pending_features:
        logger.log(state, "requirements",
                   f"reusing {len(state.requirements)} existing requirements "
                   "(no new features)", level="result")
        return state

    if has_prior:
        existing_block = "\n".join(
            f"- {r.rid} [{r.kind}]: {r.statement}" for r in state.requirements
        )
        features_block = "\n".join(
            f"- {f.title}: {f.motivation}" for f in state.pending_features
        )
        next_id = _next_rid(state.requirements)
        user = (
            f"# Idea\n{state.idea}\n\n"
            f"# Existing requirements (DO NOT modify or repeat)\n{existing_block}\n\n"
            f"# New features to cover\n{features_block}\n\n"
            f"Start the new requirement ids at {next_id} and increment from there. "
            "Return JSON {\"requirements\": [...]} with ONLY the new requirements."
        )
        parsed = chat_json(reasoning_llm(json_mode=True), SYSTEM_EXTEND, user, _Reqs)
        existing_ids = {r.rid for r in state.requirements}
        added = [r for r in parsed.requirements if r.rid not in existing_ids]
        state.requirements = list(state.requirements) + added
        logger.log(state, "requirements",
                   f"added {len(added)} new requirements (total {len(state.requirements)})",
                   level="result", added=[r.rid for r in added])
        return state

    notes = "\n".join(f"- {n.title}: {n.summary}" for n in state.research)
    open_bugs = "\n".join(f"- {b.bid}: {b.description}" for b in state.open_bugs)
    user = (
        f"# Idea\n{state.idea}\n\n"
        f"# Brief and notes\n{notes}\n\n"
        + (f"# Outstanding bugs to keep in mind\n{open_bugs}\n\n" if open_bugs else "")
        + "Return a JSON object with a `requirements` array."
    )
    reqs = chat_json(reasoning_llm(json_mode=True), SYSTEM_INITIAL, user, _Reqs)
    state.requirements = reqs.requirements
    logger.log(state, "requirements", f"{len(reqs.requirements)} requirements drafted",
               level="result", count=len(reqs.requirements))
    return state
