"""Render a matched material intervention without saving the source scene."""
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blender_export_thin_pack import camera_record, export_mesh, save_json


def main():
    request = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text(encoding="utf-8"))
    root, config = Path(request["root"]), request["config"]
    inputs, truth = root / request["inputs"], root / request["truth"]
    reference = root / "data/eval_gt" / config["reference_bundle"]
    reference_manifest = json.loads((reference / "render_manifest.json").read_text(encoding="utf-8"))
    case = next(c for c in reference_manifest["cases"] if c["id"] == config["reference_case"])
    scene = bpy.context.scene
    assert scene.camera.name == "Camera" and scene.camera.parent.name == "Empty.001"
    settings = {"engine": scene.render.engine, "view_transform": scene.view_settings.view_transform,
                "look": scene.view_settings.look, "filter_size": scene.render.filter_size,
                "eevee_taa_render_samples": scene.eevee.taa_render_samples}
    assert settings == reference_manifest["render"]
    assert scene.compositing_node_group is None and not scene.render.use_motion_blur
    assert [scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage] == [1920, 1080, 100]
    rods = [o for o in scene.objects if o.type == "MESH" and o.name.startswith("Cylinder")]
    for obj in rods:
        obj.scale.x = obj.scale.y = case["xy_scale"]
    for name in ("Material.002", "Material.003"):
        tree = bpy.data.materials[name].node_tree
        socket = next(n for n in tree.nodes if n.type == "BSDF_PRINCIPLED").inputs["Base Color"]
        for link in list(socket.links):
            tree.links.remove(link)
        if case["brick"]:
            tree.links.new(next(n for n in tree.nodes if n.type == "TEX_BRICK").outputs["Color"], socket)
    bpy.context.view_layer.update()
    rig = bpy.data.objects["Empty.001"]
    result = {"scope": config["scope"], "render": settings, "blender_version": bpy.app.version_string, "groups": []}
    for condition in config["conditions"]:
        if condition == "uniform_emission":
            # 只改这三段目标的材质引用，不改共享旧材质，免得背景也被一起涂白。
            material = bpy.data.materials.new("CreatorUniformAppearanceDiagnostic")
            material.use_nodes = True
            tree = material.node_tree
            tree.nodes.clear()
            emit, output = tree.nodes.new("ShaderNodeEmission"), tree.nodes.new("ShaderNodeOutputMaterial")
            emit.inputs["Color"].default_value = [*config["emission_linear_rgb"], 1.0]
            emit.inputs["Strength"].default_value = config["emission_strength"]
            tree.links.new(emit.outputs[0], output.inputs["Surface"])
            for name in config["intervention_objects_generation_only"]:
                obj = bpy.data.objects[name]
                obj.data = obj.data.copy()
                obj.data.materials.clear()
                obj.data.materials.append(material)
                for polygon in obj.data.polygons:
                    polygon.material_index = 0
        elif condition != "original_replica":
            raise ValueError("Unknown appearance condition")
        bpy.context.view_layer.update()
        truth_folder = truth / condition
        truth_folder.mkdir()
        geometry = export_mesh(scene, truth_folder, rods)
        assert geometry["mesh_content_sha256"] == case["mesh_content_sha256"], "Material intervention changed geometry"
        save_json(truth_folder / "geometry.json", geometry)
        group = {"condition": condition, "mesh_content_sha256": geometry["mesh_content_sha256"], "frames": []}
        for angle in config["angles_degrees"]:
            frame_id = f"view_{angle:+03d}"
            rig.rotation_euler.z = math.radians(angle)
            bpy.context.view_layer.update()
            camera = camera_record(scene)
            expected = json.loads((reference / config["reference_case"] / frame_id / "camera.json").read_text(encoding="utf-8"))
            for field in ("K_index", "K_edge", "world_to_camera_cv"):
                assert np.max(np.abs(np.array(camera[field]) - expected[field])) < 1e-6
            image = inputs / condition / (frame_id + ".png")
            image.parent.mkdir(parents=True, exist_ok=True)
            scene.render.filepath = str(image)
            scene.render.image_settings.file_format = "PNG"
            scene.render.image_settings.color_mode = "RGB"
            scene.render.image_settings.color_depth = "8"
            bpy.ops.render.render(write_still=True)
            save_json(truth_folder / (frame_id + "-camera.json"), camera)
            group["frames"].append({"frame_id": frame_id, "angle_degrees": angle, "size_wh": camera["size_wh"]})
            print("APPEARANCE_RENDER", condition, frame_id, flush=True)
        result["groups"].append(group)
    save_json(truth / "render_manifest.json", result)


if __name__ == "__main__":
    main()
