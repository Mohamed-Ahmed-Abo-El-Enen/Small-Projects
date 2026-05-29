from __future__ import annotations

import ast
from pathlib import Path

from ..events import EventLogger
from ..llm import chat, reasoning_llm
from ..models import ProjectState
from ..workspace import iter_source_files

SYSTEM = (
    "You are the Documentation Agent. Write the project's living documentation. "
    "Cover: overview, scope, architecture, file-by-file API, how to run, how to "
    "test, known issues, and the iteration log. Be precise about function "
    "behaviour. Future iterations of the pipeline will read this file to know "
    "what already exists."
)


def _python_inventory(project_dir: Path) -> str:
    """Walk the project's .py files and list each public function/class."""
    lines: list[str] = []

    for p in iter_source_files(project_dir, [".py"]):
        if "tests" in p.parts:
            continue
        rel = p.relative_to(project_dir).as_posix()

        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError:
            lines.append(f"### {rel}\n(syntax error — could not parse)\n")
            continue

        lines.append(f"### {rel}")

        for node in tree.body:

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node) or "(no docstring)"
                lines.append(f"- `{node.name}()` — {doc.splitlines()[0]}")

            elif isinstance(node, ast.ClassDef):
                doc = ast.get_docstring(node) or "(no docstring)"
                lines.append(f"- class `{node.name}` — {doc.splitlines()[0]}")
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        sdoc = ast.get_docstring(sub) or "(no docstring)"
                        lines.append(f"  - `{sub.name}()` — {sdoc.splitlines()[0]}")

        lines.append("")

    return "\n".join(lines) or "(no python files yet)"


def run(state: ProjectState, logger: EventLogger) -> ProjectState:
    """Compose PROJECT_DOC.md from the plan, inventory, and history."""
    logger.log(state, "doc_writer", "writing project documentation")

    project_dir = Path(state.project_dir)
    inventory = _python_inventory(project_dir)
    history = "\n".join(
        f"- iter {h.iteration}: {h.notes.splitlines()[0] if h.notes else ''}"
        for h in state.history
    )

    test = state.last_test_result

    facts = (
        f"# Project: {state.project_name}\n"
        f"# Language: Python\n"
        f"# Idea\n{state.idea}\n\n"
        f"# Plan\n{state.plan.model_dump_json(indent=2) if state.plan else '(no plan)'}\n\n"
        f"# Source inventory (auto-extracted)\n{inventory}\n\n"
        f"# Latest test result\npassed={test.passed if test else 'n/a'} "
        f"total={test.total if test else 0} failed={test.failed if test else 0}\n\n"
        f"# Iteration history\n{history or '(first iteration)'}\n\n"
        f"# Pending features\n"
        + "\n".join(f"- {f.title}: {f.motivation}" for f in state.pending_features)
    )
    body = chat(reasoning_llm(), SYSTEM, facts + "\n\nWrite the full project documentation in markdown.")

    doc_path = project_dir / "PROJECT_DOC.md"
    doc_path.write_text(body, encoding="utf-8")
    state.final_doc_path = str(doc_path)

    logger.log(state, "doc_writer", "doc written", level="result", path=str(doc_path))
    return state
