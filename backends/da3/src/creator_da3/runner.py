"""CLI boundary for an isolated model environment; never simulate predictions."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from creator_da3 import __version__


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Creator DA3 adapter scaffold")
    parser.add_argument("--version", action="version", version=f"creator-da3 {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="Show implemented capabilities without loading a model")
    run = commands.add_parser("run", help="Reserved BackendRequest entry; not implemented")
    run.add_argument("--request", required=True, help="Path to the future BackendRequest JSON")
    args = parser.parse_args(argv)
    if args.command == "status":
        print(json.dumps({"component": "da3", "stage": "scaffold", "inference_ready": False}))
        return 0
    print(
        "NOT_IMPLEMENTED: DA3 inference is not connected; no prediction was produced.",
        file=sys.stderr,
    )
    return 2
