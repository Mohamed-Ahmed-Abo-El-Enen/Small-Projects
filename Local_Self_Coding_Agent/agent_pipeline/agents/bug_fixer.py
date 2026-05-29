from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from ..events import EventLogger
from ..llm import chat_json, coding_llm
from ..models import CodeFile, ProjectState
from ..workspace import (
    failing_test_files,
    format_failure_summary,
    is_test_path,
    read_all_sources,
    run_tests,
    setup_project_env,
    summarize_test_failures,
    write_files,
)

SYSTEM = (
    "You are the Bug Fixer Agent. You are given the EXACT pytest failure output "
    "(assertion messages + tracebacks), the failing test code, the bugs the "
    "finder flagged, and the current source. The test code is the contract: it "
    "is correct and you may NOT change it. Fix the SOURCE so those exact "
    "failing tests pass.\n\n"
    "  - Read each traceback: it names the file, line, and exact error "
    "(ZeroDivisionError, AssertionError: assert None == [1,2,3], etc.). Fix "
    "that specific cause.\n"
    "  - Make minimal, targeted edits. Do not introduce new dependencies.\n"
    "  - A function asserted to return a value MUST return that value (don't "
    "leave it returning None / mutating in place).\n"
    "  - Return only the source files you modified."
)


class _Patches(BaseModel):
    files: list[CodeFile] = Field(default_factory=list)
    notes: str = ""


def run(state: ProjectState, logger: EventLogger) -> ProjectState:
    """Patch the source against the real test failures, then re-run pytest."""
    if not state.open_bugs:
        logger.log(state, "bug_fixer", "no bugs to fix", level="result")
        return state

    project_dir = Path(state.project_dir)
    current = read_all_sources(project_dir)
    src_block = "\n\n".join(
        f"### {f.path}\n```python\n{f.content[:1800]}\n```"
        for f in current if not is_test_path(f.path)
    )

    test_output = state.last_test_result.output if state.last_test_result else ""
    prev_keys = set(summarize_test_failures(test_output))
    failing_tests = failing_test_files(test_output, project_dir)
    test_block = "\n\n".join(
        f"### {t.path}\n```python\n{t.content[:1500]}\n```" for t in failing_tests
    ) or "(failing test files not located — see the output above)"

    bug_block = "\n".join(
        f"- {b.bid} [{b.severity}] {b.location}: {b.description} (suggested: {b.suggested_fix})"
        for b in state.open_bugs
    )
    user = (
        f"# EXACT pytest failure output (fix the SOURCE so these pass)\n"
        f"{test_output}\n\n"
        f"# Failing test code (the contract — DO NOT modify these)\n{test_block}\n\n"
        f"# Bugs the finder flagged\n{bug_block}\n\n"
        f"# Current source files\n{src_block}\n\n"
        "Return JSON {\"files\": [...], \"notes\": \"...\"} with the fixed source "
        "files only. Each fix must address a specific failure above."
    )
    patches = chat_json(coding_llm(json_mode=True), SYSTEM, user, _Patches)

    safe_patches = [p for p in patches.files if not is_test_path(p.path)]
    rejected = [p.path for p in patches.files if is_test_path(p.path)]
    if rejected:
        logger.log(state, "bug_fixer",
                   f"ignored {len(rejected)} test-file patches "
                   "(tests are owned by the tester, not bug_fixer)",
                   level="warn", rejected=rejected)

    if safe_patches:
        write_files(project_dir, safe_patches)
        path_to_new = {p.path: p for p in safe_patches}
        merged: list[CodeFile] = []
        for f in state.files:
            merged.append(path_to_new.pop(f.path, f))
        merged.extend(path_to_new.values())
        state.files = merged
        logger.log(state, "bug_fixer", f"patched {len(safe_patches)} files",
                   files=[f.path for f in safe_patches], notes=patches.notes)
        if any(p.path == "requirements.txt" for p in safe_patches):
            ok, msg, bad = setup_project_env(project_dir)
            logger.log(state, "bug_fixer", f"env refreshed: {msg}",
                       level="info" if ok else "warn", bad_packages=bad)
    else:
        logger.log(state, "bug_fixer", "agent returned no patches", level="warn")

    state.fix_attempts += 1
    logger.log(state, "bug_fixer", "re-running tests after patch")
    result = run_tests(project_dir)
    state.last_test_result = result

    if result.passed:
        for b in state.open_bugs:
            b.resolved = True
        state.closed_bugs.extend(state.open_bugs)
        state.open_bugs = []
        state.no_progress_streak = 0
        state.last_failure_keys = []
        logger.log(state, "bug_fixer", "all tests pass", level="result")
    else:
        out = result.output or ""
        new_keys = set(summarize_test_failures(out))
        if new_keys and new_keys == prev_keys:
            state.no_progress_streak += 1
        else:
            state.no_progress_streak = 0
        state.last_failure_keys = sorted(new_keys)
        digest = format_failure_summary(out) or "no FAILED summary in output"
        stuck = (f"; no progress for {state.no_progress_streak} attempt(s)"
                 if state.no_progress_streak else "")
        logger.log(state, "bug_fixer",
                   f"tests still failing (attempt {state.fix_attempts}/"
                   f"{state.max_fix_attempts}) — {digest}{stuck}",
                   level="warn", attempt=state.fix_attempts,
                   no_progress_streak=state.no_progress_streak,
                   failures=summarize_test_failures(out), output_tail=out[-1500:])

    return state
