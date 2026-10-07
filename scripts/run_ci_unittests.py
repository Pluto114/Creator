"""Run unchanged unittest groups and publish failures as GitHub annotations."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def annotation_escape(value):
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def failure_summary(output):
    """Keep the last failure block and terminal summary, bounded for annotations."""
    lines = output.decode("utf-8", errors="replace").splitlines()
    starts = [i for i, line in enumerate(lines)
              if line.startswith(("ERROR:", "FAIL:", "Traceback (most recent call last):"))]
    start = starts[-1] if starts else max(0, len(lines)-40)
    return "\n".join(lines[start:])[-6000:]


def write_output(stream, data):
    # Preserve original stdout/stderr bytes; decoding is only for the annotation.
    if hasattr(stream, "buffer"):
        stream.buffer.write(data)
    else:
        stream.write(data.decode("utf-8", errors="replace"))
    stream.flush()


def run_patterns(patterns, root=ROOT):
    for pattern in patterns:
        print("CI_UNITTEST_START", pattern, flush=True)
        command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", pattern, "-v"]
        result = subprocess.run(command, cwd=root, capture_output=True, check=False)
        write_output(sys.stdout, result.stdout)
        write_output(sys.stderr, result.stderr)
        if result.returncode:
            detail = failure_summary(result.stdout+b"\n"+result.stderr)
            message = f"unittest pattern={pattern}; exit={result.returncode}\n{detail}"
            print("::error::"+annotation_escape(message), flush=True)
            return result.returncode
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("patterns", nargs="+")
    args = parser.parse_args(argv)
    (ROOT / ".runtime").mkdir(exist_ok=True)
    return run_patterns(args.patterns)


if __name__ == "__main__":
    raise SystemExit(main())
