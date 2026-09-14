"""Blender 5.2 worker. Never saves or modifies the source .blend on disk."""

import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def camera_record(scene):
    cam = scene.camera
    width, height = scene.render.resolution_x, scene.render.resolution_y
    p = cam.calc_matrix_camera(
        bpy.context.evaluated_depsgraph_get(), x=width, y=height, scale_x=1, scale_y=1
    )
    t = Matrix.Diagonal((1.0, -1.0, -1.0, 1.0)) @ cam.matrix_world.normalized().inverted()
    k = np.array(
        [
            [width * p[0][0] / 2, 0, width * (1 - p[0][2]) / 2],
            [0, height * p[1][1] / 2, height * (1 + p[1][2]) / 2],
            [0, 0, 1],
        ]
    )
    ki = k.copy()
    ki[:2, 2] -= 0.5
    return {
        "size_wh": [width, height],
        "K_edge": k.tolist(),
        "K_index": ki.tolist(),
        "world_to_camera_cv": [list(row) for row in t],
        "clip_start": cam.data.clip_start,
        "clip_end": cam.data.clip_end,
    }


def export_mesh(scene, folder, rods=None):
    """Evaluated render meshes, with identical duplicate geometry mapped to one segment."""
    deps = bpy.context.evaluated_depsgraph_get()
    objects = []
    vertices, triangles, labels = [], [], []
    signatures = {}
    offset = 0
    rod_names = [
        "Cylinder",
        "Cylinder.001",
        "Cylinder.002",
        "Cylinder.004",
        "Cylinder.005",
        "Cylinder.006",
        "Cylinder.008",
    ]
    for obj in sorted(
        (o for o in scene.objects if o.type == "MESH" and not o.hide_render), key=lambda o: o.name
    ):
        evaluated = obj.evaluated_get(deps)
        mesh = evaluated.to_mesh()
        mesh.calc_loop_triangles()
        verts = np.array([list(evaluated.matrix_world @ v.co) for v in mesh.vertices], np.float32)
        faces = np.array([list(f.vertices) for f in mesh.loop_triangles], np.uint32)
        signature = hashlib.sha256(
            json.dumps(
                sorted(tuple(round(float(v), 7) for v in xyz) for xyz in verts),
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        duplicate = signatures.get(signature)
        if rods is not None:
            canonical = "Cylinder.002" if obj.name == "Cylinder.003" else obj.name
            sid = (
                rod_names.index(canonical) + 1
                if canonical in rod_names
                else {"Plane": 100, "Plane.001": 101}[canonical]
            )
        else:
            sid = {"CalibrationPlane": 100, "CalibrationOccluder": 1}[obj.name]
        record = {
            "object": obj.name,
            "surface_id": sid,
            "rod_id": min(sid, 6) if sid < 100 else 0,
            "duplicate_geometry_of": duplicate,
            "world_vertex_sha256": signature,
        }
        if rods is not None and obj.name.startswith("Cylinder"):
            z = [v.co.z for v in mesh.vertices]
            record.update(
                world_centerline_endpoints=[
                    list(obj.matrix_world @ Vector((0, 0, zz))) for zz in (min(z), max(z))
                ],
                dimensions=list(obj.dimensions),
            )
        if duplicate is None:
            signatures[signature] = obj.name
            vertices.append(verts)
            triangles.append(faces + offset)
            labels.append(np.full(len(faces), sid, np.uint32))
            offset += len(verts)
        objects.append(record)
        evaluated.to_mesh_clear()
    arrays = {
        "vertices": np.concatenate(vertices),
        "triangles": np.concatenate(triangles),
        "surface_ids": np.concatenate(labels),
    }
    np.savez_compressed(folder / "mesh.npz", **arrays)
    mesh_hash = hashlib.sha256(b"".join(a.tobytes() for a in arrays.values())).hexdigest()
    gaps = []
    if rods is not None:
        lower = next(o for o in objects if o["object"] == "Cylinder.006")
        upper = next(o for o in objects if o["object"] == "Cylinder.008")
        endpoints = [lower["world_centerline_endpoints"][1], upper["world_centerline_endpoints"][0]]
        gaps.append(
            {
                "rod_id": 6,
                "segment_surface_ids": [6, 7],
                "label": "intentional_gap_do_not_connect",
                "world_endpoints": endpoints,
                "length_m": float(np.linalg.norm(np.subtract(*endpoints))),
            }
        )
    return {"objects": objects, "gaps": gaps, "mesh_content_sha256": mesh_hash}


def depth_compositor(scene):
    bpy.context.view_layer.use_pass_z = True
    group = bpy.data.node_groups.new("CreatorPairedExport", "CompositorNodeTree")
    scene.compositing_node_group = group
    group.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    render = group.nodes.new("CompositorNodeRLayers")
    output = group.nodes.new("NodeGroupOutput")
    group.links.new(render.outputs["Image"], output.inputs["Image"])
    file = group.nodes.new("CompositorNodeOutputFile")
    file.format.media_type = "IMAGE"
    file.format.file_format = "OPEN_EXR"
    file.format.color_depth = "32"
    file.format.color_mode = "RGB"
    file.format.exr_codec = "ZIP"
    file.save_as_render = False
    file.file_name = ""
    file.file_output_items.new("RGBA", "blender_depth_raw")
    group.links.new(render.outputs["Depth"], file.inputs["blender_depth_raw"])
    return file


def probes(scene, cam, geometry):
    w, h = cam["size_wh"]
    k, t = np.array(cam["K_edge"]), np.array(cam["world_to_camera_cv"])
    inv = np.linalg.inv(t)
    pixels = [
        (float(x) + 0.5, float(y) + 0.5)
        for y in range(7, h, max(1, h // 25))
        for x in range(7, w, max(1, w // 40))
    ]
    projection_error = 0.0
    for obj in geometry["objects"]:
        if "world_centerline_endpoints" not in obj:
            continue
        a, b = np.array(obj["world_centerline_endpoints"])
        for frac in np.linspace(0.05, 0.95, 20):
            p = a * (1 - frac) + b * frac
            q = k @ (t @ np.r_[p, 1])[:3]
            uv = q[:2] / q[2]
            blender_uv = world_to_camera_view(scene, scene.camera, Vector(p))
            projection_error = max(
                projection_error, abs(uv[0] - blender_uv.x * w), abs(uv[1] - (1 - blender_uv.y) * h)
            )
            if 0 <= uv[0] < w and 0 <= uv[1] < h:
                pixels.append(tuple(np.floor(uv) + 0.5))
    by_name = {o["object"]: o["surface_id"] for o in geometry["objects"]}
    output = []
    for u, v in pixels:
        d = inv[:3, :3] @ (np.linalg.inv(k) @ [u, v, 1])
        origin = inv[:3, 3]
        hit, loc, normal, index, obj, matrix = scene.ray_cast(
            bpy.context.evaluated_depsgraph_get(),
            Vector(origin),
            Vector(d).normalized(),
            distance=cam["clip_end"] * float(np.linalg.norm(d)),
        )
        z = float((t @ np.r_[loc, 1])[2]) if hit else None
        if hit and not cam["clip_start"] <= z <= cam["clip_end"]:
            hit = False
        output.append(
            {
                "uv_edge": [u, v],
                "surface_id": by_name[obj.name] if hit else 0,
                "z": z if hit else None,
                "range": float(np.linalg.norm(np.array(loc) - origin)) if hit else None,
            }
        )
    return output, projection_error


def render_view(scene, depth_node, folder, rgb, geometry):
    folder.mkdir(parents=True, exist_ok=False)
    rgb.parent.mkdir(parents=True, exist_ok=True)
    cam = camera_record(scene)
    checks, err = probes(scene, cam, geometry)
    if err > 0.01:
        raise RuntimeError(f"Blender projection mismatch: {err}")
    depth_node.directory = str(folder)
    scene.render.filepath = str(rgb)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    bpy.ops.render.render(write_still=True)
    if not (folder / "blender_depth_raw.exr").is_file():
        raise RuntimeError("Missing synchronized depth pass")
    save_json(folder / "camera.json", cam)
    save_json(folder / "blender_ray_probes.json", checks)
    return {"camera": cam, "projection_check_max_px": err}


def main():
    request = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text(encoding="utf-8"))
    root = Path(request["root"])
    config = request["config"]
    gt, inputs = root / request["gt"], root / request["inputs"]
    scene = bpy.context.scene
    assert scene.camera.name == "Camera" and scene.render.engine == "BLENDER_EEVEE"
    assert scene.render.resolution_percentage == 100 and not scene.render.use_border
    assert not scene.camera.data.dof.use_dof and not scene.render.use_motion_blur
    assert scene.render.pixel_aspect_x == scene.render.pixel_aspect_y == 1
    assert scene.compositing_node_group is None
    rods = [o for o in scene.objects if o.type == "MESH" and o.name.startswith("Cylinder")]
    assert not any(o.modifiers for o in scene.objects if o.type == "MESH")
    assert scene.unit_settings.scale_length == 1
    # This exporter is deliberately scoped to the verified opaque, static asset.
    for mat in bpy.data.materials:
        for node in mat.node_tree.nodes if mat.node_tree else []:
            if node.type == "BSDF_PRINCIPLED":
                assert (
                    not node.inputs["Alpha"].is_linked and node.inputs["Alpha"].default_value == 1
                )
                assert not node.inputs["Transmission Weight"].is_linked
                assert node.inputs["Transmission Weight"].default_value == 0
    result = {
        "blender_version": bpy.app.version_string,
        "render": {
            "engine": scene.render.engine,
            "view_transform": scene.view_settings.view_transform,
            "look": scene.view_settings.look,
            "filter_size": scene.render.filter_size,
            "eevee_taa_render_samples": scene.eevee.taa_render_samples,
        },
        "cases": [],
    }
    node = depth_compositor(scene)
    reference = json.loads((root / config["verified_geometry"]).read_text(encoding="utf-8"))
    for case in config["cases"]:
        for obj in rods:
            obj.scale.x = obj.scale.y = case["xy_scale"]
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
        folder = gt / case["id"]
        folder.mkdir()
        geometry = export_mesh(scene, folder, rods)
        old = next(c for c in reference["cases"] if c["case_id"] == case["id"])
        for obj in geometry["objects"]:
            if "world_centerline_endpoints" in obj:
                prior = next(o for o in old["rods"] if o["object"] == obj["object"])
                assert obj["duplicate_geometry_of"] == prior["duplicate_geometry_of"]
                assert np.allclose(
                    obj["world_centerline_endpoints"],
                    prior["world_centerline_endpoints"],
                    atol=1e-6,
                    rtol=0,
                )
        save_json(folder / "geometry.json", geometry)
        entry = {**case, **geometry, "views": []}
        for angle in config["angles"]:
            bpy.data.objects["Empty.001"].rotation_euler.z = math.radians(angle)
            bpy.context.view_layer.update()
            frame = f"view_{angle:+03d}"
            view = render_view(
                scene, node, folder / frame, inputs / case["id"] / f"{frame}.png", geometry
            )
            prior = next(v for v in old["views"] if v["angle_degrees"] == angle)
            assert np.allclose(
                view["camera"]["K_edge"], prior["K_pixel_edge_origin"], atol=0.001, rtol=0
            )
            assert np.allclose(
                view["camera"]["world_to_camera_cv"], prior["world_to_camera_cv"], atol=1e-6, rtol=0
            )
            entry["views"].append({"frame_id": frame, "angle_degrees": angle, **view})
            print(f"PAIRED_RENDER {case['id']} {angle}", flush=True)
        result["cases"].append(entry)
        save_json(gt / "render_manifest.json", result)
    # Independent small calibration with fronto-parallel surfaces and a clear world background.
    calibration = bpy.data.scenes.new("CreatorDepthCalibration")
    bpy.context.window.scene = calibration
    calibration.render.engine = "BLENDER_EEVEE"
    calibration.render.resolution_x, calibration.render.resolution_y = 320, 180
    calibration.render.resolution_percentage = 100
    c = bpy.data.objects.new("CalibrationCamera", bpy.data.cameras.new("CalibrationCamera"))
    calibration.collection.objects.link(c)
    c.location = (0, 0, 5)
    c.data.lens = 35
    c.data.clip_end = 100
    calibration.camera = c
    for name, x, y, z, sx, sy in [
        ("CalibrationPlane", 0, 0, 0, 1.7, 0.9),
        ("CalibrationOccluder", -0.5, 0, 1, 0.15, 0.5),
    ]:
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(
            [(x - sx, y - sy, z), (x + sx, y - sy, z), (x + sx, y + sy, z), (x - sx, y + sy, z)],
            [],
            [(0, 1, 2, 3)],
        )
        obj = bpy.data.objects.new(name, mesh)
        calibration.collection.objects.link(obj)
    bpy.context.view_layer.update()
    folder = gt / "calibration"
    folder.mkdir()
    geometry = export_mesh(calibration, folder)
    save_json(folder / "geometry.json", geometry)
    render_view(
        calibration, depth_compositor(calibration), folder / "view", folder / "rgb.png", geometry
    )
    print("BLENDER_EXPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
