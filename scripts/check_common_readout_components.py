"""Paired analytic comparison, with new tube/noise/grid-phase conditions fixed here."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from check_common_readout_ceiling import tube
from run_rod_identity_blender import digest, write_json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/src"))
from creator_eval.common_readout import readout as local_readout  # noqa: E402
from creator_eval.common_readout_components import readout as component_readout  # noqa: E402
from creator_eval.line_controls import curve_metrics, gap_coverage  # noqa: E402


def run(output):
    rows = []
    for split, origin, vector, radii, noise_values, voxels in (
        ("development", [.013, .017, .011], [.35, .17, 1.], (0., .005, .02, .05), (0., .0025), (.02, .04)),
        ("fresh_analytic", [.028, .033, -.027], [.43, -.25, 1.25], (0., .009, .024, .042), (0., .004), (.018, .036)),
    ):
        a, length = np.asarray(origin), vector[2]
        layouts = dict(straight=[[a, a + [0, 0, length]]], tilted=[[a, a + vector]],
                       gap=[[a, a + [0, 0, .4 * length]], [a + [0, 0, .6 * length], a + [0, 0, length]]],
                       parallel=[[a, a + [0, 0, length]], [a + [.16, 0, 0], a + [.16, 0, length]]])
        for label, truth in layouts.items():
            for radius in radii:
                for noise in noise_values:
                    points = np.concatenate([tube(s, radius, noise, 22092026 + i if split == "development" else 22092027 + i) for i, s in enumerate(truth)])
                    for voxel in voxels:
                        for method, reader in (("local_links", local_readout), ("straight_components", component_readout)):
                            result = reader(points, np.empty((0, 2, 3)), {"voxel_size": voxel})
                            row = dict(split=split, layout=label, radius_m=radius, noise_sigma_m=noise, voxel_m=voxel, method=method,
                                state=result["state"], components=result.get("components"), segments=result["segments"],
                                metrics=curve_metrics(result["segments"], truth, tolerance=.025, spacing=.005))
                            if label == "gap":
                                row["gap"] = gap_coverage(result["segments"], [truth[0][1], truth[1][0]], tolerance=.025, spacing=.005)
                            rows.append(row)
    write_json(output, dict(state="complete", scope="component design informed by failed old ceiling; fresh analytic parameters frozen in this script before scoring; not independent physical assets",
        source_sha256={p: digest(ROOT / p) for p in ("experiments/src/creator_eval/common_readout.py", "experiments/src/creator_eval/common_readout_components.py", "scripts/check_common_readout_ceiling.py", "scripts/check_common_readout_components.py")}, row_count=len(rows), rows=rows))
    print("COMPONENT_READOUT_CONTROLS", len(rows), "paired rows", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/experiments/results/2026-09-22-common-readout-components.json")
    args = parser.parse_args()
    from environment_paths import require_project_environment
    require_project_environment(ROOT)
    run(args.output)
