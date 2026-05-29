import argparse
import subprocess
import sys

from src.server import run
from src.db import init_db, list_sessions
from src.tools import list_models, ollama_is_running
from src.config import OLLAMA_BASE, DEFAULT_MODEL
from src.registry import list_all, seed_defaults, REGISTRY_PATH


# ── Commands ───────────────────────────────────────────────────────────────────
def cmd_server(_args):
    """Start the MCP server over stdio."""
    print("Starting MCP server…", file=sys.stderr)
    run()


def cmd_ui(_args):
    """Launch the Streamlit UI."""
    print("Launching Streamlit UI at http://localhost:8501")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
         "--server.headless", "false"],
        check=True,
    )


def cmd_check(_args):
    """Check Ollama connectivity and list available models."""
    print(f"Ollama base : {OLLAMA_BASE}")
    print(f"Default model: {DEFAULT_MODEL}")

    if ollama_is_running():
        print("Status     :Running")
        models = list_models()
        print(f"Models ({len(models)}):")
        for m in models:
            marker = " ← default" if m == DEFAULT_MODEL else ""
            print(f"  • {m}{marker}")
    else:
        print("Status     : Not reachable")
        print("Fix        : Run `ollama serve` in a separate terminal.")
        sys.exit(1)


def cmd_skills(_args):
    """List all registered custom skills and agents."""
    data = list_all()

    print(f"\nSkills ({len(data['skills'])}):")
    for s in data["skills"]:
        print(f"{s['name']:20s} — {s['description']}")

    print(f"\nAgents ({len(data['agents'])}):")
    for a in data["agents"]:
        print(f"{a['name']:20s} — {a['description']}")


def cmd_sessions(_args):
    """List all saved sessions."""
    init_db()

    sessions = list_sessions()
    if not sessions:
        print("No sessions yet. Start the Streamlit UI to create one.")
        return

    print(f"\n{len(sessions)} session(s):\n")
    for s in sessions:
        print(f"  [{s['id'][:8]}…] {s['name']}")
        print(f"  model={s['model']}  updated={s['updated_at']}")


def cmd_seed(_args):
    """Seed the registry with default skills."""
    if REGISTRY_PATH.exists():
        print(f"Registry already exists at {REGISTRY_PATH}. Delete it first to re-seed.")
    else:
        seed_defaults()
        print("Default skills seeded.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Ollama MCP — local AI assistant with tools, sessions, and web search.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("server",   help="Start the MCP server (stdio)")
    sub.add_parser("ui",       help="Launch the Streamlit web UI")
    sub.add_parser("check",    help="Check Ollama connectivity and list models")
    sub.add_parser("skills",   help="List registered custom skills and agents")
    sub.add_parser("sessions", help="List all saved sessions")
    sub.add_parser("seed",     help="Seed default skills into the registry")

    args = parser.parse_args()

    dispatch = {
        "server":   cmd_server,
        "ui":       cmd_ui,
        "check":    cmd_check,
        "skills":   cmd_skills,
        "sessions": cmd_sessions,
        "seed":     cmd_seed,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()