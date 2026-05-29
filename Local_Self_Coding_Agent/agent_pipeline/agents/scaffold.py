from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from ..events import EventLogger
from ..llm import chat_json, reasoning_llm
from ..models import CodeFile, FileSpec, ProjectPlan, ProjectState
from ..workspace import dedup_requirements, dep_name, is_test_path, setup_project_env

PROJECT_STYLE = (
    "Use a clean Python package layout. Always include `requirements.txt` and "
    "a `tests/` directory with pytest tests."
)

SYSTEM_INITIAL = (
    "You are the Scaffolding Agent. Given the requirements, produce a Python "
    "project plan describing the file layout, dependencies, and entry point. "
    "Follow the layout hint verbatim. Keep the plan minimal but complete: "
    "every requirement should be served by at least one file.\n\n"
    "Always include:\n"
    "  - `.gitignore` covering venvs, caches, build dirs, and editor files\n"
    "  - `README.md` with a one-paragraph summary and a Quickstart block\n"
    "Include WHEN the project genuinely uses runtime configuration (API keys, "
    "DB URLs, ports, model names, log levels, etc.):\n"
    "  - `.env` with the actual default values used by the code\n"
    "  - `.env.example` with the same keys but placeholder values\n"
    "Do NOT include `.env` for purely-algorithmic projects with no runtime config."
)

SYSTEM_EXTEND = (
    "You are the Scaffolding Agent operating in EXTENSION mode. An existing "
    "project layout and dependency list are provided and MUST be preserved "
    "unchanged. Propose ONLY new file specs and new dependencies needed for "
    "the pending features. Reuse existing modules where possible."
)


class _Extension(BaseModel):
    new_files: list[FileSpec] = Field(default_factory=list)
    new_dependencies: list[str] = Field(default_factory=list)


def _seed_files(plan: ProjectPlan) -> list[CodeFile]:
    """Return docstring placeholder files for Python plan entries only."""
    seeds: list[CodeFile] = []
    seen: set[str] = set()
    for spec in plan.files:
        if spec.path in seen or is_test_path(spec.path):
            continue
        if not spec.path.endswith(".py"):
            continue
        seen.add(spec.path)
        seeds.append(CodeFile(path=spec.path, content=f'"""{spec.purpose}"""\n'))
    return seeds


def _initial_plan(state: ProjectState) -> ProjectPlan:
    """Ask the LLM for a fresh ProjectPlan based on the requirements."""
    reqs = "\n".join(f"- [{r.kind}] {r.rid}: {r.statement}" for r in state.requirements)
    user = (
        f"# Project layout hint (follow this)\n{PROJECT_STYLE}\n\n"
        f"# Project name\n{state.project_name}\n\n"
        f"# Idea\n{state.idea}\n\n"
        f"# Requirements\n{reqs}\n\n"
        "Return a ProjectPlan JSON. "
        "Files should be relative paths inside the project root."
    )
    plan = chat_json(reasoning_llm(json_mode=True), SYSTEM_INITIAL, user, ProjectPlan)
    if not plan.name:
        plan.name = state.project_name
    return plan


def _extend_plan(state: ProjectState, plan: ProjectPlan) -> tuple[list[FileSpec], list[str]]:
    """Ask the LLM for files and deps to add on top of the existing plan."""
    existing_files_block = "\n".join(f"- {f.path}: {f.purpose}" for f in plan.files)
    existing_deps = ", ".join(plan.dependencies) or "(none)"
    reqs_block = "\n".join(f"- {r.rid}: {r.statement}" for r in state.requirements)
    features_block = "\n".join(
        f"- {f.title}: {f.motivation}" for f in state.pending_features
    )
    user = (
        f"# Project name\n{plan.name}\n\n"
        f"# Existing files (DO NOT repeat)\n{existing_files_block}\n\n"
        f"# Existing dependencies\n{existing_deps}\n\n"
        f"# Full requirements list (for context)\n{reqs_block}\n\n"
        f"# Pending features driving the extension\n{features_block}\n\n"
        "Return JSON {\"new_files\": [...], \"new_dependencies\": [...]}. "
        "Empty arrays are valid if no extension is needed."
    )
    ext = chat_json(reasoning_llm(json_mode=True), SYSTEM_EXTEND, user, _Extension)
    existing_paths = {f.path for f in plan.files}
    new_files = [f for f in ext.new_files if f.path not in existing_paths]
    existing_pkg_names = {dep_name(d) for d in plan.dependencies}
    new_deps = [d for d in ext.new_dependencies if dep_name(d) not in existing_pkg_names]
    return new_files, new_deps


def _write_conftest(project_dir: Path) -> None:
    """Ensure pytest can resolve `from <module> import ...` from the project root."""
    conftest = project_dir / "conftest.py"
    if conftest.exists():
        return
    conftest.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, str(Path(__file__).resolve().parent))\n",
        encoding="utf-8",
    )


def run(state: ProjectState, logger: EventLogger) -> ProjectState:
    """Draft or extend the plan, write requirements.txt, set up the venv."""
    project_dir = Path(state.project_dir)

    if state.plan is None:
        logger.log(state, "scaffold", "drafting initial project plan")
        plan = _initial_plan(state)
        state.plan = plan
        for f in _seed_files(plan):
            target = project_dir / f.path
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f.content, encoding="utf-8")
    else:
        plan = state.plan
        logger.log(state, "scaffold",
                   f"extending existing plan ({len(plan.files)} files)")
        new_files, new_deps = _extend_plan(state, plan)
        if new_files or new_deps:
            plan.files.extend(new_files)
            merged_deps, _ = dedup_requirements(plan.dependencies + new_deps)
            plan.dependencies = sorted(merged_deps, key=dep_name)
            for spec in new_files:
                if is_test_path(spec.path) or not spec.path.endswith(".py"):
                    continue
                target = project_dir / spec.path
                if target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(f'"""{spec.purpose}"""\n', encoding="utf-8")
            logger.log(state, "scaffold",
                       f"added {len(new_files)} new files, {len(new_deps)} new deps",
                       level="result",
                       new_files=[f.path for f in new_files],
                       new_deps=new_deps)
        else:
            logger.log(state, "scaffold",
                       "no extension needed — plan unchanged", level="result")

    deduped, _ = dedup_requirements(plan.dependencies)
    deduped = sorted(deduped, key=dep_name)
    (project_dir / "requirements.txt").write_text(
        "\n".join(deduped) + ("\n" if deduped else ""), encoding="utf-8"
    )
    plan.dependencies = deduped
    _write_conftest(project_dir)

    deps_count = len(plan.dependencies)
    logger.log(state, "scaffold",
               f"creating venv + pip-installing {deps_count} dependencies "
               "(this can take a few minutes on first run)")

    def _env_progress(msg: str) -> None:
        logger.log(state, "scaffold", f"env: {msg}")

    ok, msg, bad = setup_project_env(project_dir, on_progress=_env_progress)
    logger.log(state, "scaffold", f"env: {msg}",
               level="result" if ok else "error", ok=ok, bad_packages=bad)

    if not ok and bad:
        bad_lower = {b.lower() for b in bad}
        kept: list[str] = []
        dropped: list[str] = []
        for dep in plan.dependencies:
            if dep_name(dep) in bad_lower:
                dropped.append(dep)
            else:
                kept.append(dep)
        if dropped:
            kept, _ = dedup_requirements(kept)
            kept = sorted(kept, key=dep_name)
            plan.dependencies = kept
            (project_dir / "requirements.txt").write_text(
                "\n".join(kept) + ("\n" if kept else ""), encoding="utf-8"
            )
            logger.log(state, "scaffold",
                       f"dropped unresolvable packages and retrying: {dropped}",
                       level="warn", dropped=dropped, remaining=kept)
            ok, msg, bad = setup_project_env(
                project_dir, on_progress=_env_progress,
            )
            logger.log(state, "scaffold", f"env retry: {msg}",
                       level="result" if ok else "error",
                       ok=ok, bad_packages=bad, removed=dropped)

    if not ok:
        state.done = True
        state.stop_reason = f"environment setup failed: {(msg or '').splitlines()[0][:300]}"
        logger.log(state, "scaffold",
                   f"aborting iteration: {state.stop_reason}", level="error")
    return state
