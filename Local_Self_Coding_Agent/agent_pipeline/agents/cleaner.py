from __future__ import annotations

import ast
import shutil
from pathlib import Path

from ..events import EventLogger
from ..models import ProjectState
from ..workspace import iter_source_files


_TRANSIENT_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def _is_empty_placeholder(content: str) -> bool:
    """Return True for a .py file whose body is empty or a lone docstring."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return False

    body = tree.body
    if (body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]

    return len(body) == 0


def _under_venv(path: Path) -> bool:
    """True if the path is inside the project's .venv (we never touch that)."""
    return any(part in {".venv", "venv"} for part in path.parts)


def run(state: ProjectState, logger: EventLogger) -> ProjectState:
    """Remove placeholder .py files and build/cache directories from the project."""
    project_dir = Path(state.project_dir)

    deleted_files: list[str] = []
    deleted_dirs: list[str] = []

    for d in sorted(project_dir.rglob("*"), key=lambda p: -len(p.parts)):
        if not d.is_dir() or _under_venv(d):
            continue

        if d.name in _TRANSIENT_DIRS:
            try:
                shutil.rmtree(d)
                deleted_dirs.append(d.relative_to(project_dir).as_posix())
            except OSError:
                pass

    for p in iter_source_files(project_dir, [".py"]):
        if p.name == "__init__.py" or p.name == "conftest.py":
            continue

        try:
            content = p.read_text(encoding="utf-8")
        except OSError:
            continue

        if _is_empty_placeholder(content):
            try:
                p.unlink()
                deleted_files.append(p.relative_to(project_dir).as_posix())
            except OSError:
                pass

    for d in sorted(project_dir.rglob("*"), key=lambda p: -len(p.parts)):
        if not d.is_dir() or _under_venv(d):
            continue

        try:
            next(d.iterdir())
        except StopIteration:
            try:
                d.rmdir()
                deleted_dirs.append(d.relative_to(project_dir).as_posix() + "/")
            except OSError:
                pass

        except OSError:
            pass

    msg = (f"removed {len(deleted_files)} empty placeholder files, "
           f"{len(deleted_dirs)} cache/empty dirs")
    level = "result" if (deleted_files or deleted_dirs) else "info"

    logger.log(state, "cleaner", msg, level=level,
               files=deleted_files, dirs=deleted_dirs)

    return state
