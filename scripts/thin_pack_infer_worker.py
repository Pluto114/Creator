"""RGB-only inference process. Evaluation files are deliberately absent from the request."""

import argparse
import re
from pathlib import Path, PureWindowsPath

from smoke_da3 import run
from thin_pack_gt import read_json, sha256, write_json


def safe_component(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_+\-]{0,95}", value):
        raise ValueError("Invalid " + label)
    return value


def input_bundle_path(root, bundle_id):
    safe_component(bundle_id, "input bundle ID")
    base = (root / "data/inputs").resolve()
    bundle = (base / bundle_id).resolve()
    # 光检查图片在bundle里还不够：bundle自己也不能溜到GT目录去。
    if not bundle.is_relative_to(base) or bundle == base:
        raise ValueError("RGB bundle escapes data/inputs")
    return bundle


def rgb_input_path(bundle, relative):
    if not isinstance(relative, str) or not relative:
        raise ValueError("Invalid RGB relative path")
    normalized = relative.replace("\\", "/")
    if (
        PureWindowsPath(relative).drive
        or Path(normalized).is_absolute()
        or ".." in normalized.split("/")
    ):
        raise ValueError("RGB path must stay inside its input bundle")
    path = (bundle / normalized).resolve()
    if not path.is_relative_to(bundle.resolve()) or path == bundle.resolve():
        raise ValueError("RGB path escapes its input bundle")
    return path


def validate_frame_order(group):
    frames = group["frames"]
    ids = [safe_component(frame["frame_id"], "frame ID") for frame in frames]
    if not ids or len(ids) != len(set(ids)) or ids != group["frame_order"]:
        raise ValueError("Frame set/order mismatch")
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = read_json(args.request)
    root = Path(__file__).resolve().parents[1]
    input_root = input_bundle_path(root, request["bundle_id"])
    validate_frame_order(
        {"frames": request["frames"], "frame_order": [f["frame_id"] for f in request["frames"]]}
    )
    images = [rgb_input_path(input_root, f["rgb"]) for f in request["frames"]]
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
