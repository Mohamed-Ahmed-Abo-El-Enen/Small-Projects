from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field

from ..events import EventLogger
from ..llm import chat_json, coding_llm
from ..models import CodeFile, ProjectState, Requirement
from ..workspace import (
    auto_inject_imports,
    format_failure_summary,
    format_import_map,
    iter_source_files,
    project_exports,
    run_tests,
    summarize_test_failures,
    write_files,
)

SYSTEM = (
    "You are the Test Author Agent. Write pytest tests. Tests must be "
    "self-contained and avoid network calls. Put them under a `tests/` "
    "directory.\n\n"
    "MANDATORY IMPORTS:\n"
    "  - EVERY test file MUST start with the imports of the functions / "
    "classes it tests. Look at the project sources block; for each "
    "function/class you reference, write `from <module> import <name>`. "
    "If you call `bubble_sort(...)` in a test, the file MUST contain "
    "`from bubble_sort import bubble_sort` (or the matching module path).\n"
    "  - Never reference a name without importing it. pytest will fail "
    "collection with NameError if you do.\n\n"
    "RULES FOR EXPECTED VALUES (critical):\n"
    "  1. Compute expected values, do NOT invent them. If you write "
    "`assert sort([3,1,2]) == X`, then X MUST equal `[1, 2, 3]`.\n"
    "  2. Prefer inputs small enough that the expected output is trivial to "
    "hand-verify (3-6 elements).\n"
    "  3. For sorting tests, expected MUST equal `sorted(input)` exactly.\n"
    "  4. Avoid stress tests with random data or large `range(N)`.\n"
    "  5. One focused assertion per case."
)

_TEST_FUNC_RE = re.compile(r"^\s*def\s+test_\w+\s*\(", re.MULTILINE)
_TEST_DIR_CANDIDATES = ["tests", "test", "src/test", "Tests"]


class _Tests(BaseModel):
    files: list[CodeFile] = Field(default_factory=list)


def _has_real_tests(content: str) -> bool:
    """Return True if the file contains at least one `def test_*` function."""
    return bool(_TEST_FUNC_RE.search(content or ""))


def _existing_tests(project_dir: Path) -> list[CodeFile]:
    """Read test files that contain at least one real test function."""
    out: list[CodeFile] = []
    seen: set[str] = set()
    for sub in _TEST_DIR_CANDIDATES:
        tdir = project_dir / sub
        if not tdir.exists():
            continue
        for p in iter_source_files(tdir):
            rel = p.relative_to(project_dir).as_posix()
            if rel in seen:
                continue
            seen.add(rel)
            try:
                content = p.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if not _has_real_tests(content):
                continue
            out.append(CodeFile(path=rel, content=content))
    return out


def _select_new_requirements(state: ProjectState,
                             tested_rids: set[str]) -> list[Requirement]:
    """Return the requirements whose rid is not referenced in existing tests."""
    return [r for r in state.requirements if r.rid not in tested_rids]


def _all_rids_in_text(text: str, requirements: list[Requirement]) -> set[str]:
    """Return the set of requirement ids that appear anywhere in `text`."""
    return {r.rid for r in requirements if r.rid in text}


def _ensure_tests_init(project_dir: Path) -> None:
    """Make `tests/__init__.py` exist so pytest can import it."""
    init_path = project_dir / "tests" / "__init__.py"
    init_path.parent.mkdir(parents=True, exist_ok=True)
    if not init_path.exists():
        init_path.write_text("", encoding="utf-8")


def _format_sources(state: ProjectState) -> str:
    """Render the project's source files as markdown blocks for the prompt."""
    return "\n\n".join(
        f"### {f.path}\n```python\n{f.content[:1800]}\n```" for f in state.files
    )


def run(state: ProjectState, logger: EventLogger) -> ProjectState:
    """Author new tests where needed, then run pytest and record the result."""
    project_dir = Path(state.project_dir)
    existing = _existing_tests(project_dir)

    tested_rids = _all_rids_in_text(
        "\n".join(f.content for f in existing), state.requirements
    ) if existing else set()
    targets = _select_new_requirements(state, tested_rids)

    if not existing and state.requirements:
        targets = list(state.requirements)
        logger.log(state, "tester",
                   "no real test functions found on disk — generating full suite "
                   f"for {len(targets)} requirements", level="warn")

    if existing and not targets:
        logger.log(state, "tester",
                   f"reusing {len(existing)} existing test files (no new requirements)")
    else:
        scope = "initial test suite" if not existing else \
                f"adding tests for {len(targets)} new requirements"
        logger.log(state, "tester", f"drafting {scope}")

        reqs_block = "\n".join(
            f"- {r.rid}: {r.statement}\n  acceptance: " + "; ".join(r.acceptance)
            for r in targets
        )
        existing_block = "\n\n".join(
            f"### {t.path}\n```python\n{t.content[:1500]}\n```" for t in existing
        ) if existing else "(no existing tests)"

        exports = project_exports(project_dir)
        import_cheatsheet = format_import_map(exports)
        user = (
            f"# Available imports (USE EXACTLY THESE for any project name you reference)\n"
            f"{import_cheatsheet}\n\n"
            f"# Project sources\n{_format_sources(state)}\n\n"
            f"# Existing tests (DO NOT duplicate or rewrite)\n{existing_block}\n\n"
            f"# Requirements that still need tests\n{reqs_block}\n\n"
            "Return JSON {\"files\": [{path, content}, ...]} with NEW test files "
            "only. Place each file under the project's test directory convention. "
            "Paths must not collide with existing test files."
        )
        parsed = chat_json(coding_llm(json_mode=True), SYSTEM, user, _Tests)

        existing_paths = {t.path for t in existing}
        valid_test_prefixes = ("tests/", "test/", "src/test/", "Tests/")
        new_tests = [
            f for f in parsed.files
            if any(f.path.replace("\\", "/").startswith(p) for p in valid_test_prefixes)
            and f.path not in existing_paths
        ]
        injected: list[str] = []
        for tf in new_tests:
            original = tf.content
            tf.content = auto_inject_imports(original, exports)
            if tf.content != original:
                injected.append(tf.path)
        if injected:
            logger.log(state, "tester",
                       f"auto-injected imports into {len(injected)} test file(s)",
                       files=injected)

        if new_tests:
            _ensure_tests_init(project_dir)
            write_files(project_dir, new_tests)
            logger.log(state, "tester", f"wrote {len(new_tests)} new test files",
                       level="result", files=[t.path for t in new_tests])
        elif targets:
            logger.log(state, "tester",
                       "no new test files produced for new requirements", level="warn")

        state.tests = existing + new_tests

    logger.log(state, "tester", "running tests")
    result = run_tests(project_dir)
    state.last_test_result = result

    out = result.output or ""
    fails = summarize_test_failures(out)
    if "NO TESTS COLLECTED" in out:
        logger.log(state, "tester",
                   "pytest collected zero tests — tester needs real `def test_*` "
                   "functions; skipping bug-fix loop for this iteration",
                   level="error", passed=False, total=0, output_tail=out[-800:])
    elif "pytest timed out" in out:
        digest = format_failure_summary(out) or "no per-test failures recorded"
        logger.log(state, "tester", f"pytest timed out — {digest}",
                   level="error", passed=False, failures=fails, output_tail=out[-1500:])
    elif result.passed:
        logger.log(state, "tester", f"tests passed ({result.total} ok)",
                   level="result", passed=True, total=result.total)
    else:
        digest = format_failure_summary(out) or "no FAILED summary in output"
        logger.log(state, "tester", f"tests failed — {digest}",
                   level="warn",
                   passed=False, total=result.total, failed=result.failed,
                   errors=result.errors, failures=fails, output_tail=out[-1500:])
    return state
