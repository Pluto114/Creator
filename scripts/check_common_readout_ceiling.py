"""Frozen readout ceiling controls: tubes, noise, gaps and nearby parallel rods.

These use analytic truth only for scoring after extraction. A reader that can
read added zero-width axes but loses plausible base surfaces is not a fair judge.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval.common_readout import readout  # noqa: E402
from creator_eval.line_controls import curve_metrics, gap_coverage  # noqa: E402
from run_rod_identity_blender import digest, write_json  # noqa: E402


def tube(segment, radius, noise, seed):
    first, last = np.asarray(segment)
    direction = last - first
    direction /= np.linalg.norm(direction)
    basis = np.eye(3)[np.argmin(np.abs(direction))]
    a = np.cross(direction, basis)
    a /= np.linalg.norm(a)
    b = np.cross(direction, a)
    theta = np.arange(24) * 2 * np.pi / 24
    center = first + np.linspace(0, 1, 201)[:, None] * (last - first)
    ring = radius * (np.cos(theta)[:, None] * a + np.sin(theta)[:, None] * b)
    points = (center[:, None] + ring).reshape(-1, 3)
    return points + np.random.default_rng(seed).normal(0, noise, points.shape)


def run(output):
    layouts = dict(straight=[[[.013, .017, .011], [.013, .017, 1.011]]],
                   tilted=[[[.013, .017, .011], [.363, .187, 1.011]]],
                   gap=[[[.013, .017, .011], [.013, .017, .411]], [[.013, .017, .611], [.013, .017, 1.011]]],
                   parallel=[[[.013, .017, .011], [.013, .017, 1.011]], [[.173, .017, .011], [.173, .017, 1.011]]])
    rows = []
    for label, truth in layouts.items():
        for radius in (0., .005, .02, .05):
            for noise in (0., .0025):
                points = np.concatenate([tube(segment, radius, noise, 22092026 + i) for i, segment in enumerate(truth)])
                for voxel in (.02, .04):
                    result = readout(points, np.empty((0, 2, 3)), {"voxel_size": voxel})
                    row = dict(layout=label, radius_m=radius, noise_sigma_m=noise, voxel_m=voxel, state=result["state"],
                        input_points=len(points), components=result.get("components"), segments=result["segments"],
                        metrics=curve_metrics(result["segments"], truth, tolerance=.025, spacing=.005))
                    if label == "gap":
                        row["gap"] = gap_coverage(result["segments"], [truth[0][1], truth[1][0]], tolerance=.025, spacing=.005)
                    rows.append(row)
    write_json(output, dict(state="complete", scope="analytic surface/noise ceiling of the fixed reader; not model or method efficacy",
        reader_sha256=digest(ROOT / "experiments/src/creator_eval/common_readout.py"), generator_sha256=digest(Path(__file__)),
        no_parameter_retuning=True, row_count=len(rows), rows=rows))
    print("COMMON_READOUT_CEILING", len(rows), "frozen geometry/noise/scale conditions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/experiments/results/2026-09-22-common-readout-ceiling.json")
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    run(args.output)
