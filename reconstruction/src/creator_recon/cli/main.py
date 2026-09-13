"""Small executable shell for the documented CLI; research actions fail explicitly."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from creator_recon import IMPLEMENTATION_STATUS, __version__
from creator_recon.application.case_builder import create_case_from_request
from creator_recon.application.run_service import RunService
from creator_recon.contracts.control import NeverCancelled, NullProgressSink
from creator_recon.errors import FeatureUnavailableError

MAX_REQUEST_BYTES = 1024 * 1024


def build_parser() -> argparse.ArgumentParser:
    """Expose stable command names without claiming the wire DTOs are implemented."""
    parser = argparse.ArgumentParser(
        prog="creator",
        description="Creator thin-structure reconstruction scaffold; no inference implemented.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="Show implementation capabilities, not a job RunStatus.")
    run = commands.add_parser("run", help="Reserved research-job entry; currently exits 3.")
    run.add_argument("--request", type=Path, required=True, help="UTF-8 JSON request file.")
    case = commands.add_parser("case", help="Case import commands.")
    case_commands = case.add_subparsers(dest="case_command", required=True)
    create = case_commands.add_parser("create", help="Reserved case importer; currently exits 3.")
    create.add_argument("--request", type=Path, required=True, help="UTF-8 JSON request file.")
    return parser


def _reject_constant(value: str) -> Any:
    raise ValueError(f"Non-finite JSON constant is not allowed: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_request(path: Path) -> dict[str, Any]:
    """Check only a small JSON envelope, not the unimplemented v1 request schema."""
    with path.open("rb") as stream:
        raw = stream.read(MAX_REQUEST_BYTES + 1)
    if len(raw) > MAX_REQUEST_BYTES:
        raise ValueError("Request JSON exceeds the 1 MiB CLI envelope limit.")
    request = json.loads(
        raw.decode("utf-8"),
        parse_constant=_reject_constant,
        object_pairs_hook=_unique_object,
    )
    if not isinstance(request, dict):
        raise ValueError("Request must be a JSON object; v1 schema support is not implemented.")
    return request


def main(argv: Sequence[str] | None = None) -> int:
    """Return 0 for introspection, 2 for input errors, and 3 for missing features."""
    args = build_parser().parse_args(argv)
    if args.command == "status":
        print(
            json.dumps(
                {
                    "product": "Creator",
                    "version": __version__,
                    "implementation_status": IMPLEMENTATION_STATUS,
                    "status_kind": "installation_capabilities",
                    "schema_validation": "not_implemented",
                    "case_import": "not_implemented",
                    "reconstruction": "not_implemented",
                    "refinement": "not_implemented",
                    "preview_export": "not_implemented",
                    "evaluation": "not_implemented",
                },
                indent=2,
            )
        )
        return 0
    try:
        request = _read_request(args.request)
        cancel = NeverCancelled()
        progress = NullProgressSink()
        if args.command == "run":
            RunService().execute(request, cancel, progress)
        else:
            create_case_from_request(request, cancel)
    except (OSError, UnicodeError, ValueError, RecursionError) as error:
        print(f"E_INPUT: {error}", file=sys.stderr)
        return 2
    except FeatureUnavailableError as error:
        print(f"{error.code}: {error}", file=sys.stderr)
        return 3
    raise AssertionError("Unimplemented research action unexpectedly returned.")
