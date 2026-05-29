from __future__ import annotations

import ast
from pathlib import Path

from pydantic import BaseModel, Field

from ..events import EventLogger
from ..llm import chat_json, coding_llm
from ..models import CodeFile, ProjectState
from ..workspace import is_test_path, read_all_sources, write_files

SYSTEM_INITIAL = (
    "You are the Coder Agent. Implement one Python source file at a time, "
    "satisfying the listed requirements and the file's stated purpose. The "
    "code must run as-is. No placeholder TODOs. No fabricated imports. Use "
    "only the listed dependencies plus the Python standard library. Keep "
    "functions short and well-named. One short docstring per public function "
    "in plain English."
)

SYSTEM_EXTEND = (
    "You are the Coder Agent operating in EXTENSION mode. The current source "
    "files are provided and most already work. Extend the project so the new "
    "pending features are implemented.\n"
    "  - NEVER rewrite, refactor, or 'improve' a function that already works. "
    "Do not change its signature or return contract. A function that returns "
    "a sorted list must keep returning a sorted list.\n"
    "  - Touch an existing file ONLY if a pending feature genuinely requires "
    "it. If a feature just adds a new algorithm, create a NEW file for it and "
    "wire it into the entry point — do not edit the other algorithms.\n"
    "  - Return ONLY the files you create or must change.\n"
    "  - No placeholder TODOs. No fabricated imports."
)


class _SingleFile(BaseModel):
    path: str
    content: str


class _ExtensionPatch(BaseModel):
    files: list[CodeFile] = Field(default_factory=list)
    notes: str = ""


def _is_placeholder_source(content: str) -> bool:
    """Return True for a .py file whose body is empty or only a docstring."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return False
    body = tree.body

    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    return len(body) == 0


_AGGREGATOR_NAMES = {
    "main.py", "__main__.py", "__init__.py", "app.py", "cli.py", "run.py",
}


def _feature_mentions(path: str, features) -> bool:
    """True if any pending feature's text references this file or its stem."""
    stem = Path(path).stem.lower()
    name = Path(path).name.lower()
    text = " ".join(
        f"{f.title} {f.motivation} {' '.join(f.proposed_changes)}".lower()
        for f in features
    )
    return stem in text or name in text


def _change_allowed(path: str, existing_paths: set[str], features) -> bool:
    """Allow new files, aggregator files, and files a feature references."""
    if path not in existing_paths:
        return True
    if Path(path).name in _AGGREGATOR_NAMES:
        return True

    return _feature_mentions(path, features)


def _write_one(state: ProjectState, file_path: str, file_purpose: str,
               siblings: list[CodeFile]) -> CodeFile:
    """Ask the LLM to produce the full content of a single new file."""
    sibling_block = "\n\n".join(
        f"### {s.path}\n```python\n{s.content[:1500]}\n```"
        for s in siblings if s.content.strip()
    )
    reqs = "\n".join(f"- {r.rid}: {r.statement}" for r in state.requirements)
    deps = ", ".join(state.plan.dependencies) if state.plan else ""

    user = (
        f"# File to implement\nPath: {file_path}\nPurpose: {file_purpose}\n\n"
        f"# Allowed dependencies\n{deps}\n\n"
        f"# Requirements\n{reqs}\n\n"
        f"# Existing project files (for cross-references)\n{sibling_block or '(empty)'}\n\n"
        "Return JSON {\"path\": ..., \"content\": ...} with the complete file content."
    )
    parsed = chat_json(coding_llm(json_mode=True), SYSTEM_INITIAL, user, _SingleFile)
    return CodeFile(path=file_path, content=parsed.content)


def _initial_pass(state: ProjectState, logger: EventLogger,
                  project_dir: Path) -> list[CodeFile]:
    """Write every non-test file in the plan from scratch."""
    targets = [s for s in state.plan.files if not is_test_path(s.path)]
    logger.log(state, "coder", f"implementing {len(targets)} source files",
               files=[s.path for s in targets])
    produced: list[CodeFile] = []
    for spec in targets:
        logger.log(state, "coder", f"writing {spec.path}")
        try:
            cf = _write_one(state, spec.path, spec.purpose, produced)
        except Exception as exc:
            logger.log(state, "coder", f"failed on {spec.path}: {exc}", level="error")
            continue
        produced.append(cf)
        write_files(project_dir, [cf])
    return produced


def _extension_pass(state: ProjectState, logger: EventLogger,
                    project_dir: Path,
                    existing: list[CodeFile]) -> list[CodeFile]:
    """Write only the files that need to be created or changed this iteration."""
    sources = "\n\n".join(
        f"### {f.path}\n```python\n{f.content[:1800]}\n```" for f in existing
    )
    reqs = "\n".join(f"- {r.rid}: {r.statement}" for r in state.requirements)
    features = "\n".join(
        f"- {f.title}: {f.motivation}\n  changes: {'; '.join(f.proposed_changes)}"
        for f in state.pending_features
    )
    deps = ", ".join(state.plan.dependencies) if state.plan else ""
    plan_block = "\n".join(f"- {f.path}: {f.purpose}" for f in state.plan.files) \
        if state.plan else "(no plan)"

    user = (
        f"# Current project files (PRESERVE unless a feature forces a change)\n{sources}\n\n"
        f"# Plan file roster\n{plan_block}\n\n"
        f"# Allowed dependencies\n{deps}\n\n"
        f"# All requirements (existing + new)\n{reqs}\n\n"
        f"# Pending features driving the extension\n{features or '(none)'}\n\n"
        "Return JSON {\"files\": [{path, content}, ...], \"notes\": \"...\"} "
        "containing ONLY the files you create or modify."
    )
    patch = chat_json(coding_llm(json_mode=True), SYSTEM_EXTEND, user, _ExtensionPatch)

    if not patch.files:
        logger.log(state, "coder", "no files needed changes — extension is a no-op",
                   level="result", notes=patch.notes)
        return existing

    existing_paths = {f.path for f in existing}
    candidates = [f for f in patch.files if not is_test_path(f.path)]
    changed = [f for f in candidates
               if _change_allowed(f.path, existing_paths, state.pending_features)]
    preserved = [f.path for f in candidates
                 if not _change_allowed(f.path, existing_paths, state.pending_features)]

    if preserved:
        logger.log(state, "coder",
                   f"preserved {len(preserved)} working file(s) the extension "
                   "tried to rewrite with no feature reason",
                   level="warn", preserved=preserved)

    if not changed:
        logger.log(state, "coder", "extension is a no-op after preserving working files",
                   level="result", notes=patch.notes)
        return existing

    write_files(project_dir, changed)

    logger.log(state, "coder", f"changed {len(changed)} files",
               level="result", files=[f.path for f in changed], notes=patch.notes)

    by_path = {f.path: f for f in changed}
    merged: list[CodeFile] = []

    for f in existing:
        merged.append(by_path.pop(f.path, f))
    merged.extend(by_path.values())

    return merged


def run(state: ProjectState, logger: EventLogger) -> ProjectState:
    """Implement placeholder files, then extend the project for new features."""
    if not state.plan:
        logger.log(state, "coder", "no plan present, skipping", level="warn")
        return state

    project_dir = Path(state.project_dir)
    non_test = [f for f in read_all_sources(project_dir)
                if not is_test_path(f.path)]

    placeholders = [f for f in non_test if _is_placeholder_source(f.content)]
    real_existing = [f for f in non_test if not _is_placeholder_source(f.content)]

    filled: list[CodeFile] = []

    if placeholders:
        logger.log(state, "coder",
                   f"implementing {len(placeholders)} placeholder source file(s)",
                   files=[p.path for p in placeholders])

        for p in placeholders:
            spec = next(
                (s for s in state.plan.files if s.path == p.path),
                None,
            )
            purpose = spec.purpose if spec else p.content.strip(' "\'\n') or p.path
            logger.log(state, "coder", f"writing {p.path}")

            try:
                cf = _write_one(state, p.path, purpose, real_existing + filled)
            except Exception as exc:
                logger.log(state, "coder", f"failed on {p.path}: {exc}", level="error")
                continue

            filled.append(cf)
            write_files(project_dir, [cf])

    current = real_existing + filled
    if not current:
        produced = _initial_pass(state, logger, project_dir)
    else:
        plan_paths = {s.path for s in state.plan.files if not is_test_path(s.path)}
        existing_paths = {f.path for f in current}
        missing_paths = plan_paths - existing_paths

        if state.pending_features or missing_paths:
            produced = _extension_pass(state, logger, project_dir, current)
        else:
            produced = current
            logger.log(state, "coder",
                       "no extension needed; sources up to date", level="result")

    state.files = produced
    logger.log(state, "coder", f"project now has {len(produced)} source files",
               level="result")

    return state
