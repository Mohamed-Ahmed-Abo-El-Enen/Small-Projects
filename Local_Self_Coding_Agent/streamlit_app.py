from __future__ import annotations

import json
import os
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", message=r".*allowed_objects.*")

import streamlit as st

_PAGE_RENDER_T0 = time.monotonic()

ROOT_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = ROOT_DIR / "projects"
WORKSPACE_DIR = ROOT_DIR / "workspace"

def _cfg(key, env, default, cast=str):
    """Read a setting with precedence: env var > config.json > default."""
    if env in os.environ:
        return cast(os.environ[env])
    try:
        data = json.loads((ROOT_DIR / "config.json").read_text(encoding="utf-8"))
        if key in data:
            return cast(data[key])
    except (OSError, json.JSONDecodeError):
        pass
    return default


OLLAMA_HOST = _cfg("ollama_host", "OLLAMA_HOST", "http://127.0.0.1:11434")
REASONING_MODEL = _cfg("reasoning_model", "SCA_REASONING_MODEL", "llama3.1:8b")
CODING_MODEL = _cfg("coding_model", "SCA_CODING_MODEL", "qwen2.5-coder:7b")
EMBEDDING_MODEL = _cfg("embedding_model", "SCA_EMBED_MODEL", "nomic-embed-text:latest")
MAX_ITERATIONS_DEFAULT = _cfg("max_iterations", "SCA_MAX_ITERATIONS", 5, int)

SOURCE_EXTENSIONS = (".py",)
SKIP_DIRS = {
    ".venv", "venv", "env",
    "build", "target", "obj", "bin", "dist",
    "__pycache__", ".pytest_cache", ".mypy_cache",
    ".git", "node_modules",
}

RUNNING_MARKER_STALE_SECS = 90
MAX_FILES_RENDERED = 80
MAX_EVENTS_RENDERED = 150
EVENTS_TAIL_BYTES = 256 * 1024
EVENTS_BIG_FILE_BYTES = 10 * 1024 * 1024


st.set_page_config(page_title="Self-Coding Agent", layout="wide")


def project_paths(name: str) -> tuple[Path, Path]:
    """Return the (project_dir, workspace_dir) pair for a named project."""
    return PROJECTS_DIR / name, WORKSPACE_DIR / name


def list_projects() -> list[str]:
    """List existing project workspace folders, sorted by name."""
    if not WORKSPACE_DIR.exists():
        return []
    return sorted(p.name for p in WORKSPACE_DIR.iterdir() if p.is_dir())


def is_running(workspace: Path) -> bool:
    """Return True when the project's running marker is fresh."""
    marker = workspace / ".running"
    if not marker.exists():
        return False
    try:
        age = time.time() - marker.stat().st_mtime
    except OSError:
        return False
    if age > RUNNING_MARKER_STALE_SECS:
        try:
            marker.unlink()
        except OSError:
            pass
        return False
    return True


def load_state(workspace: Path) -> dict:
    """Read state.json into a dict, returning {} when missing or invalid."""
    state_path = workspace / "state.json"
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_events(events_path: Path,
                max_lines: int = MAX_EVENTS_RENDERED,
                tail_bytes: int = EVENTS_TAIL_BYTES) -> tuple[list[dict], int]:
    """Return up to `max_lines` events from the tail of the log, plus file size."""
    if not events_path.exists():
        return [], 0
    try:
        size = events_path.stat().st_size
    except OSError:
        return [], 0
    start = max(0, size - tail_bytes)
    try:
        with events_path.open("rb") as fh:
            fh.seek(start)
            chunk = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return [], size
    lines = chunk.splitlines()
    if start > 0 and lines:
        lines = lines[1:]
    out: list[dict] = []
    for line in lines[-max_lines:]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out, size


def _human_bytes(n: int) -> str:
    """Format a byte count using KB/MB/GB units."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.0f} TB"


def list_source_files(project_dir: Path) -> list[Path]:
    """Walk the project for source files, pruning venv and build directories."""
    out: list[Path] = []
    if not project_dir.exists():
        return out
    for dirpath, dirs, files in os.walk(project_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for name in files:
            if name.endswith(SOURCE_EXTENSIONS):
                out.append(Path(dirpath) / name)
                if len(out) >= MAX_FILES_RENDERED * 4:
                    out.sort()
                    return out
    out.sort()
    return out


@st.cache_data(show_spinner=False)
def read_text_cached(path: str, mtime: float) -> str:
    """Read a file's text, cached by (path, mtime) so reruns are cheap."""
    return Path(path).read_text(encoding="utf-8", errors="replace")


st.sidebar.title("Self-Coding Agent")
st.sidebar.caption("Local LLMs via Ollama. No external APIs.")

projects = ["+ New project"] + list_projects()

pending = st.session_state.pop("active_project", None)
if pending and pending in projects:
    st.session_state["sidebar_project"] = pending

if "sidebar_project" not in st.session_state:
    st.session_state["sidebar_project"] = projects[0]

chosen = st.sidebar.selectbox("Project", projects, key="sidebar_project")

st.sidebar.divider()
st.sidebar.markdown(
    f"**Reasoning:** `{REASONING_MODEL}`  \n"
    f"**Coding:** `{CODING_MODEL}`  \n"
    f"**Embed:** `{EMBEDDING_MODEL}`  \n"
    f"**Ollama:** `{OLLAMA_HOST}`"
)


def launch_new_project(name: str, idea: str, max_iter: int) -> None:
    """Start a fresh pipeline run for a brand-new project."""
    with st.spinner("Loading agent backend (first launch only)..."):
        from agent_pipeline.runner import make_state, start_in_background
    state = make_state(idea, name, max_iterations=max_iter)
    start_in_background(state)
    st.session_state["active_project"] = state.project_name
    st.toast(f"Started {state.project_name}", icon="🚀")
    st.rerun()


def resume_project(name: str, force_innovation: bool) -> None:
    """Resume a saved project, optionally forcing the innovator to add features."""
    with st.spinner("Loading agent backend..."):
        from agent_pipeline.runner import (
            load_state as _load,
            prepare_resume,
            start_in_background,
        )
    saved = _load(name)
    if saved is None:
        st.error(f"Could not load saved state for `{name}`.")
        return
    state = prepare_resume(saved)
    if force_innovation:
        state.force_innovation = True
    start_in_background(state, fresh=False)
    st.session_state["active_project"] = name
    icon = "✨" if force_innovation else "▶️"
    st.toast(
        f"Resumed {name} (iter {state.iteration} / cap {state.max_iterations})",
        icon=icon,
    )
    st.rerun()


if chosen == "+ New project":
    st.header("Start a new project")
    st.caption("Generated projects are in Python. Each gets its own isolated `.venv`.")
    with st.form("new_project"):
        name_input = st.text_input("Project name", placeholder="e.g. invoice-parser")
        idea_input = st.text_area(
            "Initiative idea",
            placeholder="Describe the system you want built...",
            height=180,
        )
        max_iter = st.slider("Max iterations", 1, 10, MAX_ITERATIONS_DEFAULT)
        submitted = st.form_submit_button("Launch pipeline", type="primary")

    if submitted:
        if not name_input.strip() or not idea_input.strip():
            st.error("Project name and idea are both required.")
        else:
            launch_new_project(name_input.strip(), idea_input.strip(), max_iter)

else:
    name = chosen
    project_dir, workspace = project_paths(name)
    state = load_state(workspace)
    running = is_running(workspace)
    recent_events, _ = read_events(workspace / "events.jsonl", max_lines=1, tail_bytes=8192)
    latest_event = recent_events[-1] if recent_events else None
    current_agent = latest_event.get("agent") if latest_event else None
    current_msg = latest_event.get("message", "") if latest_event else ""

    st.header(f"Project: {name}")
    st.caption(
        f"Project dir: `{project_dir}`  ·  Workspace: `{workspace}`"
    )

    head_cols = st.columns([1, 6])
    if head_cols[0].button("🔄 Refresh", key=f"refresh_{name}"):
        st.rerun()
    if state.get("done"):
        head_cols[1].success(f"Pipeline finished — {state.get('stop_reason', '')}")
    elif running and current_agent:
        head_cols[1].info(
            f"Pipeline running — currently in **{current_agent}**: {current_msg[:120]}"
        )
    elif running:
        head_cols[1].info("Pipeline running… click Refresh to update.")
    else:
        head_cols[1].warning("Pipeline not currently running.")

    cols = st.columns(5)
    cols[0].metric(
        "Iteration",
        f"{state.get('iteration', 0)}/{state.get('max_iterations', 0)}",
    )
    cols[1].metric("Requirements", len(state.get("requirements", [])))
    cols[2].metric("Files", len(state.get("files", [])))
    test = state.get("last_test_result")
    if test is None:
        test_metric = "—"
    else:
        passed_n = int(test.get("total", 0) or 0)
        failed_n = int(test.get("failed", 0) or 0)
        errors_n = int(test.get("errors", 0) or 0)
        total_n = passed_n + failed_n + errors_n
        if total_n > 0:
            test_metric = f"{passed_n}/{total_n}"
        elif test.get("passed"):
            test_metric = "0/0"
        else:
            test_metric = "fail"
    cols[3].metric(
        "Tests passed", test_metric,
        help="passed / total tests run. '—' = no test run in this iteration yet.",
    )
    cols[4].metric("Open bugs", len(state.get("open_bugs", [])))

    if state:
        finished = bool(state.get("done"))
        label = "Continue (innovate + iterate)" if finished else "Resume pipeline"
        ctrl_cols = st.columns([1, 1, 2])
        if ctrl_cols[0].button(label, type="primary",
                               disabled=running, key=f"continue_{name}"):
            resume_project(name, force_innovation=False)
        if ctrl_cols[1].button("Force new features",
                               disabled=running, key=f"force_{name}"):
            resume_project(name, force_innovation=True)

    st.divider()

    left, right = st.columns([2, 3])

    with left:
        st.subheader("Live agent events")
        events_path = workspace / "events.jsonl"
        events, events_size = read_events(events_path)

        size_cols = st.columns([2, 1])
        size_cols[0].caption(f"Log size: {_human_bytes(events_size)}")
        if events_size > EVENTS_BIG_FILE_BYTES:
            size_cols[0].warning(
                f"Events log is {_human_bytes(events_size)}; only the latest "
                f"{MAX_EVENTS_RENDERED} entries are shown.",
                icon="⚠️",
            )
        if events_size > 0 and size_cols[1].button(
            "Clear log", key=f"clear_events_{name}",
            help="Truncate events.jsonl to free disk space.",
        ):
            try:
                events_path.write_text("", encoding="utf-8")
                st.toast("Events log cleared.", icon="🧹")
                st.rerun()
            except OSError as exc:
                st.error(f"Could not clear log: {exc}")

        if not events:
            st.caption("Waiting for events... click Refresh once they start.")
        else:
            icon_for = {"info": "·", "warn": "!", "error": "x", "result": "+"}
            for ev in reversed(events):
                level = ev.get("level", "info")
                icon = icon_for.get(level, "·")
                ts_raw = ev.get("ts", "")
                try:
                    ts = datetime.fromisoformat(ts_raw).strftime("%H:%M:%S")
                except (ValueError, TypeError):
                    ts = ts_raw[-8:] if ts_raw else "??:??:??"
                line = (
                    f"`{ts}` `iter {ev.get('iteration', 0)}` "
                    f"**{ev.get('agent', '?')}** {icon} {ev.get('message', '')}"
                )
                if level == "error":
                    st.error(line)
                elif level == "warn":
                    st.warning(line)
                else:
                    st.markdown(line)
                if ev.get("data"):
                    with st.expander("data", expanded=False):
                        st.json(ev["data"])

    with right:
        tabs = st.tabs(["Source files", "Documentation", "Iteration reports"])

        with tabs[0]:
            t0 = time.monotonic()
            files = list_source_files(project_dir)
            walk_ms = int((time.monotonic() - t0) * 1000)
            total = len(files)
            truncated = total > MAX_FILES_RENDERED
            files = files[:MAX_FILES_RENDERED]
            header = f"Generated project files ({total})"
            if truncated:
                header += f" — first {MAX_FILES_RENDERED} shown"
            st.subheader(header)
            st.caption(f"Source scan: {walk_ms} ms")
            if not files:
                st.caption("No source files yet.")
            else:
                for p in files:
                    try:
                        content = read_text_cached(str(p), p.stat().st_mtime)
                    except OSError:
                        continue
                    with st.expander(str(p.relative_to(project_dir)), expanded=False):
                        st.code(content, language="python")

        with tabs[1]:
            doc = project_dir / "PROJECT_DOC.md"
            if doc.exists():
                st.markdown(read_text_cached(str(doc), doc.stat().st_mtime))
            else:
                st.caption("PROJECT_DOC.md not generated yet.")

        with tabs[2]:
            iter_dir = workspace / "iterations"
            reports = sorted(iter_dir.glob("iter_*/report.md")) if iter_dir.exists() else []
            if not reports:
                st.caption("No iteration reports yet.")
            else:
                for r in reports:
                    with st.expander(r.parent.name, expanded=False):
                        st.markdown(read_text_cached(str(r), r.stat().st_mtime))


st.divider()
_render_ms = int((time.monotonic() - _PAGE_RENDER_T0) * 1000)
st.caption(
    f"Rendered {datetime.now().strftime('%H:%M:%S')} in {_render_ms} ms · "
    f"Streamlit {st.__version__}"
)
if _render_ms > 2000:
    st.warning(
        f"Render took {_render_ms} ms — that's unusually slow. "
        "Likely cause: huge events.jsonl. Use the 🧹 Clear log button.",
        icon="🐢",
    )
