from __future__ import annotations

import uuid

from pydantic import BaseModel, Field

from ..events import EventLogger
from ..llm import chat_json, reasoning_llm
from ..models import FeatureIdea, ProjectState

SYSTEM = (
    "You are the Innovation Agent. Look at the working project, the requirements "
    "already satisfied, and any prior iteration documentation. Propose a small "
    "set (0 to 4) of high-value next features. Each feature must be concrete, "
    "non-trivial, and a clear extension of the current scope. Return an empty "
    "list if the project is genuinely complete and there is nothing meaningful "
    "left to add."
)


class _Ideas(BaseModel):
    features: list[FeatureIdea] = Field(default_factory=list)


def run(state: ProjectState, logger: EventLogger) -> ProjectState:
    """Return up to four feature ideas (or none if the project should rest)."""
    tests_failing = (
        state.last_test_result is not None and not state.last_test_result.passed
    )

    if tests_failing and not state.force_innovation:
        abandoned = [f.title for f in state.pending_features]
        if abandoned:
            state.abandoned_features = list(
                dict.fromkeys(state.abandoned_features + abandoned)
            )
            logger.log(
                state, "innovator",
                f"tests are failing — abandoning {len(abandoned)} unfixable "
                f"feature(s) so they are not retried: {abandoned}",
                level="warn", abandoned=abandoned,
            )
        else:
            logger.log(
                state, "innovator",
                "tests are failing — skipping innovation so the loop can stop",
                level="warn",
            )
        state.pending_features = []
        return state

    logger.log(state, "innovator", "looking for next features")

    sources = "\n\n".join(f"- {f.path}" for f in state.files)
    reqs = "\n".join(f"- {r.rid}: {r.statement}" for r in state.requirements)
    history = "\n".join(
        f"- iter {h.iteration}: {len(h.requirements)} reqs, "
        f"{len(h.bugs_fixed)} bugs fixed, "
        f"{len(h.new_features)} features proposed"
        for h in state.history
    )

    force_block = ""
    if state.force_innovation:
        force_block = (
            "\n# IMPORTANT\n"
            "The user has explicitly asked for the project to be extended. "
            "You MUST propose at least 1 and at most 4 concrete new features that "
            "build on the existing project — do not return an empty list.\n"
        )

    abandoned_block = ""
    if state.abandoned_features:
        abandoned_block = (
            "\n# DO NOT PROPOSE THESE (already tried and could not be made to "
            "pass tests — avoid them and anything similar)\n"
            + "\n".join(f"- {t}" for t in state.abandoned_features)
            + "\nPrefer simple, self-contained features. Avoid anything needing "
            "multiprocessing, async, networking, external services, or heavy "
            "third-party libraries.\n"
        )

    user = (
        f"# Idea\n{state.idea}\n\n"
        f"# Existing files\n{sources}\n\n"
        f"# Requirements satisfied\n{reqs}\n\n"
        f"# Iteration history\n{history or '(first iteration)'}\n"
        f"{abandoned_block}"
        f"{force_block}\n"
        f"Iteration {state.iteration} of max {state.max_iterations}. "
        "If genuinely complete and not forced, return {\"features\": []}."
    )
    ideas = chat_json(reasoning_llm(json_mode=True), SYSTEM, user, _Ideas)

    for f in ideas.features:
        if not f.fid:
            f.fid = "F" + uuid.uuid4().hex[:6]

    state.pending_features = ideas.features
    state.force_innovation = False
    logger.log(state, "innovator", f"{len(ideas.features)} new features proposed",
               level="result", features=[f.model_dump() for f in ideas.features])
    return state
