from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from ..events import EventLogger
from ..llm import chat_json, coding_llm
from ..models import CodeFile, ProjectState
from ..workspace import (
    auto_inject_imports,
    format_failure_summary,
    format_import_map,
    is_test_path,
    project_exports,
    read_all_sources,
    run_tests,
    summarize_test_failures,
    write_files,
)

SYSTEM = (
    "You are the Test Doctor Agent. The project's tests just failed or timed "
    "out. Decide which is wrong: the tests or the source.\n\n"
    "If the TESTS are wrong, rewrite the affected test files. Only return "
    "test files you actually fix.\n\n"
    "If the SOURCE is wrong, return an empty `files` list and explain in "
    "`notes`. The bug-fixer agent will handle source patches.\n\n"
    "Common test bugs to look for:\n"
    "  - MISSING IMPORTS. The test calls `bubble_sort(...)` but the file "
    "never does `from bubble_sort import bubble_sort`. pytest fails "
    "collection with NameError. Add the missing import as the first line.\n"
    "  - WRONG EXPECTED VALUES. Example: `sort(['banana','apple']) == "
    "['banana','apple']` — the expected list isn't sorted, so the test would "
    "fail even on correct code.\n"
    "  - Stress tests with absurd inputs (>1000 elements for O(n²), or "
    "anything likely to exceed a 10s per-test timeout)\n"
    "  - Expecting a list when the function mutates in place and returns None "
    "(or vice versa)\n"
    "  - Tests calling functions or attributes that don't exist in source\n"
    "  - Tests with infinite loops, unbounded recursion, or blocking I/O\n"
    "  - Tests importing modules that aren't in the project\n\n"
    "WHEN REWRITING TESTS:\n"
    "  - KEEP the original `test_*` function names so the requirement→test "
    "linkage stays intact.\n"
    "  - Use SMALL inputs (3-6 elements) where the expected output is "
    "trivial to hand-verify.\n"
    "  - COMPUTE expected values, do not invent them. For sort: expected MUST "
    "equal `sorted(input)`. For any deterministic operation, expected MUST "
    "equal the actual mathematical result."
)


class _TestPatches(BaseModel):
    files: list[CodeFile] = Field(default_factory=list)
    notes: str = ""


def _format_block(files: list[CodeFile]) -> str:
    """Render a list of files as labelled markdown blocks."""
    return "\n\n".join(
        f"### {f.path}\n```python\n{f.content[:1800]}\n```" for f in files
    )


def run(state: ProjectState, logger: EventLogger) -> ProjectState:
    """Inspect failing tests, fix bugs in the test code, re-run pytest."""
    if state.last_test_result is None or state.last_test_result.passed:
        return state

    project_dir = Path(state.project_dir)
    all_files = read_all_sources(project_dir)
    src = [f for f in all_files if not is_test_path(f.path)]
    tests = [f for f in all_files if is_test_path(f.path)]

    if not tests:
        logger.log(state, "test_doctor", "no test files on disk to diagnose")
        return state

    state.test_doctor_visits += 1
    output = state.last_test_result.output or ""
    exports = project_exports(project_dir)
    import_cheatsheet = format_import_map(exports)
    user = (
        f"# Available imports (USE EXACTLY THESE for any project name you reference)\n"
        f"{import_cheatsheet}\n\n"
        f"# Source files\n{_format_block(src)}\n\n"
        f"# Test files (under scrutiny)\n{_format_block(tests)}\n\n"
        f"# Test runner output\n{output}\n\n"
        "Return JSON {\"files\": [...], \"notes\": \"...\"} with ONLY the "
        "test files you actually fix. Empty array if the bug is in the source."
    )

    try:
        patches = chat_json(coding_llm(json_mode=True), SYSTEM, user, _TestPatches)
    except Exception as exc:
        logger.log(state, "test_doctor", f"diagnosis failed: {exc}", level="warn")
        return state

    test_patches = [f for f in patches.files if is_test_path(f.path)]
    rejected = [f.path for f in patches.files if not is_test_path(f.path)]
    if rejected:
        logger.log(state, "test_doctor",
                   f"ignored {len(rejected)} non-test patches",
                   level="warn", rejected=rejected)

    if not test_patches:
        logger.log(state, "test_doctor", "tests look correct — bug is in source",
                   notes=patches.notes)
        return state

    injected: list[str] = []
    for tf in test_patches:
        original = tf.content
        tf.content = auto_inject_imports(original, exports)
        if tf.content != original:
            injected.append(tf.path)
    if injected:
        logger.log(state, "test_doctor",
                   f"auto-injected imports into {len(injected)} test file(s)",
                   files=injected)

    write_files(project_dir, test_patches)
    state.tests = test_patches + [t for t in state.tests
                                  if t.path not in {p.path for p in test_patches}]
    state.no_progress_streak = 0
    logger.log(state, "test_doctor", f"rewrote {len(test_patches)} test files",
               level="result",
               files=[f.path for f in test_patches], notes=patches.notes)

    logger.log(state, "test_doctor", "re-running pytest after test fixes")
    result = run_tests(project_dir)
    state.last_test_result = result

    out = result.output or ""
    if result.passed:
        logger.log(state, "test_doctor",
                   f"tests now pass after test fixes ({result.total} ok)",
                   level="result")
    elif "NO TESTS COLLECTED" in out:
        logger.log(state, "test_doctor",
                   "still no tests collected after rewrite",
                   level="warn", output_tail=out[-800:])
    else:
        digest = format_failure_summary(out) or "still failing (no summary)"
        logger.log(state, "test_doctor",
                   f"tests still failing after test fixes — {digest}; "
                   "bug-fixer will look at the source next",
                   failures=summarize_test_failures(out), output_tail=out[-1500:])
    return state
