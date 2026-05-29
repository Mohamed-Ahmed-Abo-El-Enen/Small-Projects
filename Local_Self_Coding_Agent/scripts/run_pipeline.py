from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message=r".*allowed_objects.*")

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_pipeline.graph import run_pipeline
from agent_pipeline.runner import make_state


def main() -> int:
    """Parse args and run the pipeline to completion."""
    parser = argparse.ArgumentParser(description="Run the self-coding agent pipeline.")
    parser.add_argument("--name", required=True, help="Project name (folder-safe)")
    parser.add_argument("--idea", required=True, help="The initiative idea")
    parser.add_argument("--max-iterations", type=int, default=None)
    args = parser.parse_args()

    state = make_state(args.idea, args.name, max_iterations=args.max_iterations)
    final = run_pipeline(state)
    print(f"Done: {final.stop_reason or 'completed'}")
    print(f"Project dir:   {final.project_dir}")
    print(f"Workspace dir: {final.workspace_dir}")
    if final.final_doc_path:
        print(f"Project doc:   {final.final_doc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
