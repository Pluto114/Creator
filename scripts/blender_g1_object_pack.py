"""Render two development assemblies, never target-projected search guides."""

from __future__ import annotations

import json
import sys
from pathlib import Path

FIXED_GUIDE = [[319.5, 20], [319.5, 459]]
GUIDE_SOURCE = "fixed_screen_coordinates_independent_of_target_geometry"


def validate_request(request):
    generator = request["config"]["generator"]
    if generator.get("guide_xyxy") != FIXED_GUIDE or generator.get("guide_source") != GUIDE_SOURCE:
        raise ValueError("Only the predeclared fixed screen guide is permitted")
    cases = request["config"]["cases"]
    ids = [case["case_id"] for case in cases]
    if not ids or len(ids) != len(set(ids)) or not set(ids) <= {"chair01", "aframe01"}:
        raise ValueError("Unknown or duplicate assembly identity")
    if any("guide_endpoints" in case for case in cases):
        raise ValueError("World-space target guides must never reach the renderer")


def main():
    request = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text(encoding="utf-8"))
    validate_request(request)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import blender_rod_fixture_pack as fixture

    fixture.CAD = request["declared_cad"]
    fixture.ASSETS = Path(request["marker_assets"])
    prior = fixture.height.reference.prior
    prior.setup_static_scene = fixture.setup
    prior.position_camera = fixture.height.position
    prior.main()


if __name__ == "__main__":
    main()
