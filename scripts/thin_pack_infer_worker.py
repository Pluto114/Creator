"""RGB-only inference process. Evaluation files are deliberately absent from the request."""

import argparse
from pathlib import Path

from smoke_da3 import run
from thin_pack_gt import read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = read_json(args.request)
    root = Path(__file__).resolve().parents[1]
    images = [root / "data/inputs" / request["bundle_id"] / f["rgb"] for f in request["frames"]]
    input_root = (root / "data/inputs" / request["bundle_id"]).resolve()
    for path, frame in zip(images, request["frames"]):
        if not path.resolve().is_relative_to(input_root) or sha256(path) != frame["sha256"]:
            raise ValueError("RGB input identity mismatch")
    destination = args.request.parent
    write_json(destination / "report.json", {"ok": False, "state": "running"})
    try:
        inference_args = argparse.Namespace(
            images=images,
            model=request["model"],
            process_res=request["process_res"],
            use_ray_pose=False,
        )
        report = run(inference_args, root, destination)
        report.update(
            state="succeeded",
            scope="paired v2 native baseline; estimated cameras; no GT input",
            prediction_sha256=sha256(destination / "prediction.npz"),
        )
        write_json(destination / "report.json", report)
    except Exception as error:
        write_json(
            destination / "report.json", {"ok": False, "state": "failed", "error": str(error)}
        )
        raise


if __name__ == "__main__":
    main()
