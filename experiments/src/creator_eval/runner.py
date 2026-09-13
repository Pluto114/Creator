"""Independent evaluation entry point; metrics are intentionally not fabricated."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from creator_eval import __version__


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Creator independent evaluation scaffold")
    parser.add_argument("--version", action="version", version=f"creator-eval {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="Show implementation status without reading ground truth")
    run = commands.add_parser("run", help="Reserved EvaluationRequest entry; not implemented")
    run.add_argument("--request", required=True, help="Path to the future EvaluationRequest JSON")
    args = parser.parse_args(argv)
    if args.command == "status":
        print(json.dumps({"component": "evaluation", "stage": "scaffold", "metrics_ready": False}))
        return 0
    print(
        "NOT_IMPLEMENTED: evaluation is not connected; no scores or report were produced.",
        file=sys.stderr,
    )
    return 2
