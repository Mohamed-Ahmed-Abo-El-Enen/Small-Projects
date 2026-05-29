from __future__ import annotations

import json
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
PROJECTS_DIR = ROOT_DIR / "projects"
WORKSPACE_DIR = ROOT_DIR / "workspace"
CONFIG_PATH = ROOT_DIR / "config.json"

_DEFAULTS = {
    "ollama_host": "http://127.0.0.1:11434",
    "reasoning_model": "llama3.1:8b",
    "coding_model": "qwen2.5-coder:7b",
    "embedding_model": "nomic-embed-text:latest",
    "max_iterations": 5,
    "max_fix_attempts": 5,
    "research_results": 5,
    "llm_temperature": 0.2,
    "llm_timeout": 180,
}

_ENV_KEYS = {
    "ollama_host": "OLLAMA_HOST",
    "reasoning_model": "SCA_REASONING_MODEL",
    "coding_model": "SCA_CODING_MODEL",
    "embedding_model": "SCA_EMBED_MODEL",
    "max_iterations": "SCA_MAX_ITERATIONS",
    "max_fix_attempts": "SCA_MAX_FIX_ATTEMPTS",
    "research_results": "SCA_RESEARCH_RESULTS",
    "llm_temperature": "SCA_TEMPERATURE",
    "llm_timeout": "SCA_TIMEOUT",
}


def _load_file() -> dict:
    """Read config.json, returning {} if it's missing or invalid."""
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _resolve(key, cast):
    """Resolve one setting with precedence: env var > config.json > default."""
    env = _ENV_KEYS[key]
    if env in os.environ:
        return cast(os.environ[env])
    if key in _FILE:
        return cast(_FILE[key])
    return cast(_DEFAULTS[key])


_FILE = _load_file()

OLLAMA_HOST = _resolve("ollama_host", str)
REASONING_MODEL = _resolve("reasoning_model", str)
CODING_MODEL = _resolve("coding_model", str)
EMBEDDING_MODEL = _resolve("embedding_model", str)
MAX_ITERATIONS = _resolve("max_iterations", int)
MAX_FIX_ATTEMPTS = _resolve("max_fix_attempts", int)
RESEARCH_RESULTS = _resolve("research_results", int)
LLM_TEMPERATURE = _resolve("llm_temperature", float)
LLM_TIMEOUT = _resolve("llm_timeout", float)


def project_paths(name: str) -> tuple[Path, Path]:
    """Return the (project_dir, workspace_dir) pair for a named project."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_") or "project"
    project_dir = PROJECTS_DIR / safe
    workspace_dir = WORKSPACE_DIR / safe
    project_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    return project_dir, workspace_dir
