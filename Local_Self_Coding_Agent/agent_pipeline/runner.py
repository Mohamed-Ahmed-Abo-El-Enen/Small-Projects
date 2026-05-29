from __future__ import annotations

import json
import threading
import time
import traceback
from pathlib import Path

from . import config
from .events import EventLogger
from .graph import run_pipeline
from .models import ProjectState

RUNNING_MARKER_STALE_SECS = 90
HEARTBEAT_INTERVAL_SECS = 15


def make_state(idea: str, project_name: str,
               max_iterations: int | None = None) -> ProjectState:
    """Build a fresh ProjectState for a new project."""
    project_dir, workspace_dir = config.project_paths(project_name)
    return ProjectState(
        project_name=project_name,
        project_dir=str(project_dir),
        workspace_dir=str(workspace_dir),
        idea=idea.strip(),
        max_iterations=max_iterations or config.MAX_ITERATIONS,
    )


def load_state(project_name: str) -> ProjectState | None:
    """Load the saved state.json for a project, or None if missing."""
    _, workspace_dir = config.project_paths(project_name)
    state_path = Path(workspace_dir) / "state.json"
    if not state_path.exists():
        return None
    try:
        return ProjectState.model_validate(json.loads(
            state_path.read_text(encoding="utf-8")))
    except Exception:
        return None


def prepare_resume(state: ProjectState, extra_iterations: int = 1) -> ProjectState:
    """Clear stop flags and reset retry counters so the loop runs with a fresh budget."""
    if state.done and state.iteration >= state.max_iterations:
        state.max_iterations = state.iteration + max(1, extra_iterations)
    if state.done:
        state.force_innovation = True
    state.done = False
    state.stop_reason = ""
    state.fix_attempts = 0
    state.consecutive_failures = 0
    return state


def _spawn_heartbeat(workspace_dir: Path,
                     stop_event: threading.Event) -> threading.Thread:
    """Start a daemon thread that touches the running marker on a tick."""
    marker = workspace_dir / ".running"

    def _tick():
        while not stop_event.wait(HEARTBEAT_INTERVAL_SECS):
            if marker.exists():
                try:
                    marker.touch()
                except OSError:
                    pass

    t = threading.Thread(target=_tick,
                         name=f"sca-hb-{workspace_dir.name}",
                         daemon=True)
    t.start()
    return t


def start_in_background(state: ProjectState, fresh: bool = True) -> threading.Thread:
    """Run the pipeline in a daemon thread with a heartbeat sibling."""
    workspace = Path(state.workspace_dir)
    marker = workspace / ".running"
    stop_event = threading.Event()

    def _target():
        marker.write_text("running", encoding="utf-8")
        _spawn_heartbeat(workspace, stop_event)
        try:
            run_pipeline(state, fresh=fresh)
        except Exception as exc:
            logger = EventLogger(state.workspace_dir)
            logger.log(state, "pipeline", f"crashed: {exc}", level="error",
                       trace=traceback.format_exc())
        finally:
            stop_event.set()
            try:
                marker.unlink()
            except FileNotFoundError:
                pass

    th = threading.Thread(target=_target, name=f"sca-{state.project_name}", daemon=True)
    th.start()
    return th


def is_running(workspace_dir: str | Path,
               stale_after: int = RUNNING_MARKER_STALE_SECS) -> bool:
    """Return True when the pipeline's running marker is fresh."""
    marker = Path(workspace_dir) / ".running"
    if not marker.exists():
        return False
    try:
        age = time.time() - marker.stat().st_mtime
    except OSError:
        return False
    if age > stale_after:
        try:
            marker.unlink()
        except OSError:
            pass
        return False
    return True
