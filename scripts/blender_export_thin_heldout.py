"""Six evaluation-only views from the verified asset; never saves the .blend."""

import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blender_export_thin_pack import camera_record, export_mesh, probes, save_json


def centerline_probes(scene, camera, geometry):
    """Exact subpixel rays supplement the old pixel-center Blender probes."""
    k = np.asarray(camera["K_edge"])
    transform = np.asarray(camera["world_to_camera_cv"])
    inverse = np.linalg.inv(transform)
    w, h = camera["size_wh"]
    by_name = {obj["object"]: obj["surface_id"] for obj in geometry["objects"]}
    result, max_projection = [], 0.0
    targets = [
        (obj["surface_id"], obj["rod_id"], obj["world_centerline_endpoints"], False)
        for obj in geometry["objects"]
        if "world_centerline_endpoints" in obj and obj["duplicate_geometry_of"] is None
    ]
    targets += [(0, gap["rod_id"], gap["world_endpoints"], True) for gap in geometry["gaps"]]
    for sid, rod_id, endpoints, is_gap in targets:
        a, b = np.asarray(endpoints)
        for fraction in np.linspace(0.05, 0.95, 19):
            point = (1 - fraction) * a + fraction * b
            q = transform @ np.r_[point, 1.0]
            projected = k @ q[:3]
            uv = projected[:2] / projected[2]
            blender = world_to_camera_view(scene, scene.camera, Vector(point))
            error = np.max(np.abs(uv - [blender.x * w, (1 - blender.y) * h]))
            max_projection = max(max_projection, float(error))
            direction = inverse[:3, :3] @ (np.linalg.inv(k) @ np.r_[uv, 1.0])
            origin = inverse[:3, 3] + direction * camera["clip_start"]
            hit, loc, _, _, obj, _ = scene.ray_cast(
                bpy.context.evaluated_depsgraph_get(),
                Vector(origin),
                Vector(direction).normalized(),
                distance=(camera["clip_end"] - camera["clip_start"]) * np.linalg.norm(direction),
            )
            z = float((transform @ np.r_[loc, 1.0])[2]) if hit else None
            result.append(
                {
                    "world_point": point.tolist(),
                    "uv_edge": uv.tolist(),
                    "target_camera_z": float(q[2]),
                    "target_segment_surface_id": sid,
                    "target_rod_id": rod_id,
                    "intentional_gap": is_gap,
                    "surface_id": by_name[obj.name] if hit else 0,
                    "z": z,
                }
            )
    return result, max_projection


def main():
    request = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text(encoding="utf-8"))
    root, config = Path(request["root"]), request["config"]
    output = root / request["output"]
    reference = json.loads((root / config["verified_geometry"]).read_text(encoding="utf-8"))
    paired = json.loads((root / config["paired_render_manifest"]).read_text(encoding="utf-8"))
    scene = bpy.context.scene
    assert scene.camera.name == "Camera" and scene.camera.parent.name == "Empty.001"
    assert scene.camera.data.type == "PERSP" and scene.render.engine == "BLENDER_EEVEE"
    assert [scene.render.resolution_x, scene.render.resolution_y] == [1920, 1080]
    assert scene.render.resolution_percentage == 100 and not scene.render.use_border
    assert not scene.camera.data.dof.use_dof and not scene.render.use_motion_blur
    assert scene.render.pixel_aspect_x == scene.render.pixel_aspect_y == 1
    assert scene.compositing_node_group is None and scene.unit_settings.scale_length == 1
    assert not any(obj.modifiers for obj in scene.objects if obj.type == "MESH")
    for material in bpy.data.materials:
        for node in material.node_tree.nodes if material.node_tree else []:
            if node.type == "BSDF_PRINCIPLED":
                assert (
                    not node.inputs["Alpha"].is_linked and node.inputs["Alpha"].default_value == 1
                )
                assert not node.inputs["Transmission Weight"].is_linked
                assert node.inputs["Transmission Weight"].default_value == 0
    render_settings = {
        "engine": scene.render.engine,
        "view_transform": scene.view_settings.view_transform,
        "look": scene.view_settings.look,
        "filter_size": scene.render.filter_size,
        "eevee_taa_render_samples": scene.eevee.taa_render_samples,
    }
    assert render_settings == paired["render"], "Verified render settings changed"
    rods = [obj for obj in scene.objects if obj.type == "MESH" and obj.name.startswith("Cylinder")]
    rig = bpy.data.objects["Empty.001"]
    result = {
        "blender_version": bpy.app.version_string,
        "render": render_settings,
        "scope": config["scope"],
        "eval_only": True,
        "cases": [],
    }
    for case in config["cases"]:
        for obj in rods:
            obj.scale.x = obj.scale.y = case["xy_scale"]
        # 保留原材质默认值。断开砖纹链接不等于另造一套黑白材质。
        for name in ("Material.002", "Material.003"):
            tree = bpy.data.materials[name].node_tree
            socket = next(n for n in tree.nodes if n.type == "BSDF_PRINCIPLED").inputs["Base Color"]
            for link in list(socket.links):
                tree.links.remove(link)
            if case["brick"]:
                tree.links.new(
                    next(n for n in tree.nodes if n.type == "TEX_BRICK").outputs["Color"], socket
                )
        bpy.context.view_layer.update()
        folder = output / case["id"]
        folder.mkdir()
        geometry = export_mesh(scene, folder, rods)
        old = next(c for c in reference["cases"] if c["case_id"] == case["id"])
        original_case = next(c for c in paired["cases"] if c["id"] == case["id"])
        assert geometry["mesh_content_sha256"] == original_case["mesh_content_sha256"]
        # 先把已知五角度逐个对上，再用同一rig的新角度。新角度无法受控就停，不猜相机。
        camera_checks = []
        for prior in old["views"]:
            rig.rotation_euler.z = math.radians(prior["angle_degrees"])
            bpy.context.view_layer.update()
            camera = camera_record(scene)
            k_error = float(
                np.max(np.abs(np.asarray(camera["K_edge"]) - prior["K_pixel_edge_origin"]))
            )
            pose_error = float(
                np.max(
                    np.abs(np.asarray(camera["world_to_camera_cv"]) - prior["world_to_camera_cv"])
                )
            )
            assert k_error < 0.001 and pose_error < 1e-6
            camera_checks.append(
                {
                    "angle_degrees": prior["angle_degrees"],
                    "K_max_abs_error": k_error,
                    "pose_max_abs_error": pose_error,
                }
            )
        save_json(folder / "geometry.json", geometry)
        entry = {
            **case,
            "mesh_content_sha256": geometry["mesh_content_sha256"],
            "original_camera_regression": camera_checks,
            "views": [],
        }
        for angle in config["heldout_angles_degrees"]:
            rig.rotation_euler.z = math.radians(angle)
            bpy.context.view_layer.update()
            assert abs(math.degrees(rig.rotation_euler.z) - angle) < 1e-5
            frame_id = f"view_{angle:+.1f}".replace(".", "p")
            view_folder = folder / frame_id
            view_folder.mkdir()
            camera = camera_record(scene)
            grid, projection_error = probes(scene, camera, geometry)
            lines, line_projection_error = centerline_probes(scene, camera, geometry)
            assert max(projection_error, line_projection_error) < config["projection_tolerance_px"]
            scene.render.filepath = str(view_folder / "rgb.png")
            scene.render.image_settings.file_format = "PNG"
            scene.render.image_settings.color_mode = "RGB"
            scene.render.image_settings.color_depth = "8"
            bpy.ops.render.render(write_still=True)
            save_json(view_folder / "camera.json", camera)
            save_json(view_folder / "blender_ray_probes.json", grid)
            save_json(view_folder / "blender_centerline_probes.json", lines)
            entry["views"].append(
                {
                    "frame_id": frame_id,
                    "angle_degrees": angle,
                    "projection_check_max_px": max(projection_error, line_projection_error),
                }
            )
            print(f"HELDOUT_RENDER {case['id']} {angle}", flush=True)
        result["cases"].append(entry)
        save_json(output / "render_manifest.json", result)
    print("BLENDER_HELDOUT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
