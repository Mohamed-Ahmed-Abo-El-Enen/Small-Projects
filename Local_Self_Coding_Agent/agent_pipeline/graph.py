from __future__ import annotations

import json
import time
import traceback
from datetime import datetime
from typing import Any

from langgraph.graph import END, StateGraph

from .agents import (
    bug_finder,
    bug_fixer,
    cleaner,
    coder,
    doc_writer,
    innovator,
    organize,
    reporter,
    requirements,
    research,
    reviewer,
    scaffold,
    test_doctor,
    tester,
)
from .events import EventLogger
from .llm import LLMUnavailable, ping_ollama
from .models import IterationReport, ProjectPlan, ProjectState, Requirement
from .workspace import (
    has_green_snapshot,
    load_green_state,
    restore_green,
    run_tests,
    save_green_state,
    snapshot_green,
)


def _wrap(agent_module, logger: EventLogger):
    """Wrap an agent's run() with timing, logging, and error handling."""
    name = agent_module.__name__.rsplit(".", 1)[-1]

    def node(state: ProjectState) -> ProjectState:
        if state.done:
            return state
        logger.log(state, name, "step start")
        t0 = time.monotonic()
        try:
            new_state = agent_module.run(state, logger)
        except LLMUnavailable as exc:
            state.done = True
            state.stop_reason = f"LLM backend unreachable: {exc}"
            logger.log(state, name, f"aborting iteration: {exc}", level="error")
            logger.snapshot(state)
            return state
        except Exception as exc:
            logger.log(state, name, f"step crashed: {exc}", level="error",
                       trace=traceback.format_exc())
            return state
        elapsed = time.monotonic() - t0
        logger.log(new_state, name, f"step done in {elapsed:.1f}s")
        logger.snapshot(new_state)
        return new_state
    return node


def _start_iteration(state: ProjectState, logger: EventLogger) -> ProjectState:
    """Bump the iteration counter and reset per-iteration scratch state."""
    if state.done:
        return state
    state.iteration += 1
    state.fix_attempts = 0
    state.open_bugs = []
    state.closed_bugs = []
    state.last_test_result = None
    state.no_progress_streak = 0
    state.last_failure_keys = []
    state.test_doctor_visits = 0
    state.history.append(
        IterationReport(iteration=state.iteration, started_at=datetime.utcnow())
    )
    logger.log(state, "pipeline", f"=== iteration {state.iteration} starting ===")
    logger.snapshot(state)
    return state


def _no_tests_collected(state: ProjectState) -> bool:
    """True when pytest ran but found zero test functions."""
    if state.last_test_result is None:
        return False
    return "NO TESTS COLLECTED" in (state.last_test_result.output or "")


def _route_after_tests(state: ProjectState) -> str:
    """Pick the next node after tester (via test_doctor)."""
    if state.done:
        return END
    if state.last_test_result and state.last_test_result.passed:
        return "innovator"
    if _no_tests_collected(state):
        return "innovator"
    return "bug_finder"


def _route_after_fix(state: ProjectState) -> str:
    """Pick the next node after a bug-fix attempt."""
    if state.done:
        return END
    if state.last_test_result and state.last_test_result.passed:
        return "innovator"
    if _no_tests_collected(state):
        return "innovator"
    if state.fix_attempts >= state.max_fix_attempts:
        return "innovator"
    if state.no_progress_streak >= 3:
        return "innovator"
    if state.no_progress_streak >= 1 and state.test_doctor_visits < 2:
        return "test_doctor"
    return "bug_finder"


def _restore_green_state(state: ProjectState) -> None:
    """Roll requirements and plan back to the last green snapshot."""
    blob = load_green_state(state.workspace_dir)
    if not blob:
        return
    try:
        state.requirements = [Requirement(**r) for r in blob.get("requirements", [])]
        plan = blob.get("plan")
        state.plan = ProjectPlan(**plan) if plan else state.plan
    except (TypeError, ValueError):
        return
    state.pending_features = []
    state.open_bugs = []


def _finalize(state: ProjectState, logger: EventLogger) -> ProjectState:
    """Keep the project green at the boundary, then decide stop vs. iterate."""
    if state.done:
        logger.snapshot(state)
        return state

    feature_failed = (
        state.last_test_result is not None and not state.last_test_result.passed
    )
    reverted = False

    if feature_failed and has_green_snapshot(state.workspace_dir):
        if restore_green(state.project_dir, state.workspace_dir):
            restored = run_tests(state.project_dir)
            if restored.passed:
                state.last_test_result = restored
                reverted = True
                _restore_green_state(state)
    elif not feature_failed:
        snapshot_green(state.project_dir, state.workspace_dir)
        save_green_state(state.workspace_dir, json.dumps({
            "requirements": [r.model_dump() for r in state.requirements],
            "plan": state.plan.model_dump() if state.plan else None,
        }))

    if feature_failed:
        state.consecutive_failures += 1
    else:
        state.consecutive_failures = 0

    if state.iteration >= state.max_iterations:
        state.done = True
        state.stop_reason = "max iterations reached"
    elif state.consecutive_failures >= state.max_consecutive_failures:
        state.done = True
        state.stop_reason = (
            f"feature did not converge after {state.consecutive_failures} iterations"
        )
    elif not feature_failed and not state.pending_features:
        state.done = True
        state.stop_reason = "innovator returned no new features"

    if reverted:
        revert_note = "; reverted the failed feature, project is green"
    else:
        revert_note = ""

    if state.done:
        logger.log(state, "pipeline",
                   f"stopping: {state.stop_reason}{revert_note}", level="result")
    elif feature_failed:
        logger.log(state, "pipeline",
                   f"iteration {state.iteration}: feature did not converge"
                   f"{revert_note or ' — retrying'} "
                   f"(consecutive failures: {state.consecutive_failures}/"
                   f"{state.max_consecutive_failures})",
                   level="warn")
    else:
        logger.log(state, "pipeline",
                   f"iteration {state.iteration} COMPLETE — all tests passed, "
                   f"{len(state.pending_features)} features queued",
                   level="result")
    logger.snapshot(state)
    return state


def _route_after_finalize(state: ProjectState) -> str:
    """Pure router: end the run or loop to the next iteration."""
    return END if state.done else "start_iteration"


def build_graph(logger: EventLogger) -> Any:
    """Compile the LangGraph state machine."""
    g: StateGraph = StateGraph(ProjectState)

    g.add_node("start_iteration", lambda s: _start_iteration(s, logger))
    g.add_node("research", _wrap(research, logger))
    g.add_node("organize", _wrap(organize, logger))
    g.add_node("requirements", _wrap(requirements, logger))
    g.add_node("scaffold", _wrap(scaffold, logger))
    g.add_node("coder", _wrap(coder, logger))
    g.add_node("tester", _wrap(tester, logger))
    g.add_node("test_doctor", _wrap(test_doctor, logger))
    g.add_node("reviewer", _wrap(reviewer, logger))
    g.add_node("bug_finder", _wrap(bug_finder, logger))
    g.add_node("bug_fixer", _wrap(bug_fixer, logger))
    g.add_node("innovator", _wrap(innovator, logger))
    g.add_node("cleaner", _wrap(cleaner, logger))
    g.add_node("reporter", _wrap(reporter, logger))
    g.add_node("doc_writer", _wrap(doc_writer, logger))
    g.add_node("finalize", lambda s: _finalize(s, logger))

    g.set_entry_point("start_iteration")
    g.add_edge("start_iteration", "research")
    g.add_edge("research", "organize")
    g.add_edge("organize", "requirements")
    g.add_edge("requirements", "scaffold")
    g.add_edge("scaffold", "coder")
    g.add_edge("coder", "tester")
    g.add_edge("tester", "test_doctor")
    g.add_edge("test_doctor", "reviewer")
    g.add_conditional_edges("reviewer", _route_after_tests,
                            {"innovator": "innovator",
                             "bug_finder": "bug_finder",
                             END: END})
    g.add_edge("bug_finder", "bug_fixer")
    g.add_conditional_edges("bug_fixer", _route_after_fix,
                            {"innovator": "innovator",
                             "bug_finder": "bug_finder",
                             "test_doctor": "test_doctor",
                             END: END})
    g.add_edge("innovator", "cleaner")
    g.add_edge("cleaner", "reporter")
    g.add_edge("reporter", "doc_writer")
    g.add_edge("doc_writer", "finalize")
    g.add_conditional_edges("finalize", _route_after_finalize,
                            {"start_iteration": "start_iteration", END: END})
    return g.compile()


def run_pipeline(state: ProjectState, fresh: bool = True) -> ProjectState:
    """Run the agent loop end-to-end and return the final state."""
    logger = EventLogger(state.workspace_dir)
    if fresh:
        logger.reset()
        logger.log(state, "pipeline", "boot",
                   idea=state.idea, project_name=state.project_name)
    else:
        logger.log(state, "pipeline",
                   f"resume from iteration {state.iteration} "
                   f"(force_innovation={state.force_innovation})")
    logger.snapshot(state)

    ok, msg = ping_ollama()
    logger.log(state, "pipeline", msg, level="result" if ok else "error")
    if not ok:
        state.done = True
        state.stop_reason = msg
        logger.snapshot(state)
        return state

    graph = build_graph(logger)
    final_state = graph.invoke(state, config={"recursion_limit": 200})
    if not isinstance(final_state, ProjectState):
        final_state = ProjectState.model_validate(final_state)

    logger.snapshot(final_state)
    logger.log(final_state, "pipeline",
               f"done: {final_state.stop_reason or 'completed'}", level="result")
    return final_state
