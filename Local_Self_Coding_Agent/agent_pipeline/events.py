from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from .models import Event, ProjectState

_stream_logger = logging.getLogger("self_coding_agent")
if not _stream_logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"
    ))
    _stream_logger.addHandler(_handler)
    _stream_logger.setLevel(logging.INFO)
    _stream_logger.propagate = False

_LEVEL_MAP = {
    "info": logging.INFO,
    "result": logging.INFO,
    "warn": logging.WARNING,
    "error": logging.ERROR,
}


class EventLogger:
    def __init__(self, workspace_dir: str | Path) -> None:
        self.workspace = Path(workspace_dir)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.events_path = self.workspace / "events.jsonl"
        self.state_path = self.workspace / "state.json"

    def log(self, state: ProjectState, agent: str, message: str,
            level: str = "info", **data: Any) -> None:
        """Append one event and stream it to stderr."""
        event = Event(
            iteration=state.iteration,
            agent=agent,
            level=level,  # type: ignore[arg-type]
            message=message,
            data=data,
        )
        with self.events_path.open("a", encoding="utf-8") as fh:
            fh.write(event.model_dump_json() + "\n")

        marker = self.workspace / ".running"
        if marker.exists():
            try:
                marker.touch()
            except OSError:
                pass

        prefix = f"iter{state.iteration:02d} | {agent:<11}"
        _stream_logger.log(_LEVEL_MAP.get(level, logging.INFO),
                           "%s | %s", prefix, message)

    def snapshot(self, state: ProjectState) -> None:
        """Persist the full state to state.json."""
        self.state_path.write_text(state.model_dump_json_pretty(), encoding="utf-8")

    def reset(self) -> None:
        """Delete the events log and state snapshot."""
        if self.events_path.exists():
            self.events_path.unlink()
        if self.state_path.exists():
            self.state_path.unlink()
