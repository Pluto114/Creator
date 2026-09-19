"""Blender worker for new paired rod-identity layouts; never saves a .blend file."""

import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blender_export_thin_pack import camera_record, probes, save_json


def reset_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.meshes, bpy.data.curves, bpy.data.cameras, bpy.data.lights):
        for item in list(datablocks):
            if item.users == 0:
                datablocks.remove(item)


def principled_material(name, color, roughness, metallic=0.0):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = next(node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = [*color, 1.0]
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    return material


def stripe_material(name):
    material = principled_material(name, (0.12, 0.12, 0.12), 0.38)
    tree = material.node_tree
    bsdf = next(node for node in tree.nodes if node.type == "BSDF_PRINCIPLED")
    texcoord = tree.nodes.new("ShaderNodeTexCoord")
    separate = tree.nodes.new("ShaderNodeSeparateXYZ")
    ramp = tree.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "CONSTANT"
    elements = ramp.color_ramp.elements
    elements[0].position, elements[0].color = 0.0, (0.06, 0.06, 0.06, 1)
    elements[1].position, elements[1].color = 0.57, (0.06, 0.06, 0.06, 1)
    for position, color in (
        (0.60, (0.9, 0.9, 0.9, 1)),
        (0.72, (0.9, 0.9, 0.9, 1)),
        (0.75, (0.06, 0.06, 0.06, 1)),
    ):
        element = elements.new(position)
        element.color = color
    tree.links.new(texcoord.outputs["Generated"], separate.inputs["Vector"])
    tree.links.new(separate.outputs["X"], ramp.inputs["Fac"])
    tree.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    return material


def brick_material(name):
    material = principled_material(name, (0.18, 0.13, 0.09), 0.75)
    tree = material.node_tree
    bsdf = next(node for node in tree.nodes if node.type == "BSDF_PRINCIPLED")
    texcoord = tree.nodes.new("ShaderNodeTexCoord")
    brick = tree.nodes.new("ShaderNodeTexBrick")
    brick.inputs["Color1"].default_value = (0.24, 0.10, 0.045, 1)
    brick.inputs["Color2"].default_value = (0.075, 0.028, 0.012, 1)
    brick.inputs["Mortar"].default_value = (0.035, 0.035, 0.035, 1)
    brick.inputs["Scale"].default_value = 7.0
    brick.inputs["Mortar Size"].default_value = 0.035
    tree.links.new(texcoord.outputs["Generated"], brick.inputs["Vector"])
    tree.links.new(brick.outputs["Color"], bsdf.inputs["Base Color"])
    return material


def tag(obj, surface_id, role, centerline=None):
    obj["creator_surface_id"] = int(surface_id)
    obj["creator_role"] = role
    if centerline is not None:
        obj["creator_centerline_a"] = list(centerline[0])
        obj["creator_centerline_b"] = list(centerline[1])


def add_cylinder(name, endpoints, radius, material, surface_id, role):
    first, last = Vector(endpoints[0]), Vector(endpoints[1])
    direction = last - first
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=32,
        radius=radius,
        depth=direction.length,
        end_fill_type="NGON",
        location=(first + last) / 2,
    )
    obj = bpy.context.object
    obj.name = name
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = direction.to_track_quat("Z", "Y")
    obj.data.materials.append(material)
    tag(obj, surface_id, role, endpoints)
    return obj


def add_cube(name, location, scale, material, surface_id, role):
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    obj.data.materials.append(material)
    tag(obj, surface_id, role)
    return obj


def configure_scene(generator):
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x, scene.render.resolution_y = generator["size_wh"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    scene.render.use_motion_blur = False
    scene.render.filter_size = 1.5
    scene.eevee.taa_render_samples = generator["render_samples"]
    scene.view_settings.view_transform = "Standard"
    # Blender 5.2 exposes only the neutral look for Standard. Keep the render
    # boring and explicit; surprise colour management is a lousy experiment.
    scene.view_settings.look = "None"
    scene.view_settings.exposure = generator.get("exposure", 0.0)
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene.world.color = (0.025, 0.025, 0.025)
    return scene


def setup_static_scene(generator):
    scene = configure_scene(generator)
    camera_data = bpy.data.cameras.new("IdentityCamera")
    camera_data.lens = generator["camera_lens_mm"]
    camera_data.sensor_width = 36.0
    camera_data.clip_start = 0.05
    camera_data.clip_end = 50.0
    camera = bpy.data.objects.new("IdentityCamera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    wall = add_cube(
        "BrickWall", (0, 1.35, 0), (3.2, 0.06, 2.4),
        brick_material("IdentityBrick"), 100, "background_wall",
    )
    if generator.get("brick_axes") == "XZ":
        # The wall faces XZ. Feeding its constant thickness coordinate into
        # Brick.Y paints one mortar row over the whole wall. Oops.
        tree = wall.data.materials[0].node_tree
        brick = next(node for node in tree.nodes if node.type == "TEX_BRICK")
        tex = next(node for node in tree.nodes if node.type == "TEX_COORD")
        separate = tree.nodes.new("ShaderNodeSeparateXYZ")
        combine = tree.nodes.new("ShaderNodeCombineXYZ")
        tree.links.new(tex.outputs["Generated"], separate.inputs["Vector"])
        tree.links.new(separate.outputs["X"], combine.inputs["X"])
        tree.links.new(separate.outputs["Z"], combine.inputs["Y"])
        tree.links.new(combine.outputs["Vector"], brick.inputs["Vector"])
    floor = add_cube(
        "Floor", (0, 0.1, -1.5), (3.2, 2.0, 0.06),
        principled_material("IdentityFloor", (0.12, 0.13, 0.14), 0.8),
        101, "background_floor",
    )
    wall.visible_shadow, floor.visible_shadow = True, True
    for name, location, energy, size in (
        ("Key", (-2.8, -3.5, 3.8), 760.0, 3.0),
        ("Fill", (3.2, -2.2, 1.0), 420.0, 2.5),
        ("Rim", (0.5, 1.0, 3.0), 520.0, 1.8),
    ):
        light_data = bpy.data.lights.new(name, "AREA")
        light_data.energy, light_data.shape, light_data.size = energy, "DISK", size
        light = bpy.data.objects.new(name, light_data)
        scene.collection.objects.link(light)
        light.location = location
        light.rotation_euler = (0, 0, 0)
        light.rotation_mode = "QUATERNION"
        light.rotation_quaternion = (-Vector(location)).to_track_quat("-Z", "Y")
    return scene, camera


def setup_case(case, materials):
    if case["target"]["present"]:
        target = case["target"]
        for index, segment in enumerate(target.get("segments", [target["endpoints"]])):
            add_cylinder(
                f"TargetRod_{index}", segment, target.get("radius", 0.055),
                materials[target["material"]], 1, "target",
            )
    for index, distractor in enumerate(case["distractors"]):
        add_cylinder(
            f"Distractor_{index}", distractor["endpoints"], distractor["radius"],
            materials["distractor"], 2 + index, distractor["role"],
        )
    if case["occluder"]:
        occluder = case["occluder"] if isinstance(case["occluder"], dict) else {}
        add_cube(
            "Occluder", occluder.get("location", (0.0, -0.28, 0.0)),
            occluder.get("scale", (0.24, 0.16, 0.38)),
            materials["occluder"], 50, "occluder",
        )


def clear_case_objects():
    for obj in list(bpy.context.scene.objects):
        if "creator_surface_id" in obj and int(obj["creator_surface_id"]) < 100:
            bpy.data.objects.remove(obj, do_unlink=True)


def export_mesh(scene, folder):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    vertices, triangles, labels, objects = [], [], [], []
    offset = 0
    for obj in sorted(
        (item for item in scene.objects if item.type == "MESH" and not item.hide_render),
        key=lambda item: item.name,
    ):
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        mesh.calc_loop_triangles()
        verts = np.array([list(evaluated.matrix_world @ vertex.co) for vertex in mesh.vertices], np.float32)
        faces = np.array([list(face.vertices) for face in mesh.loop_triangles], np.uint32)
        surface_id = int(obj["creator_surface_id"])
        record = {
            "object": obj.name,
            "surface_id": surface_id,
            "role": str(obj["creator_role"]),
            "vertex_count": len(verts),
            "triangle_count": len(faces),
        }
        if "creator_centerline_a" in obj:
            record["world_centerline_endpoints"] = [
                list(obj["creator_centerline_a"]), list(obj["creator_centerline_b"])
            ]
        objects.append(record)
        vertices.append(verts)
        triangles.append(faces + offset)
        labels.append(np.full(len(faces), surface_id, np.uint32))
        offset += len(verts)
        evaluated.to_mesh_clear()
    arrays = {
        "vertices": np.concatenate(vertices),
        "triangles": np.concatenate(triangles),
        "surface_ids": np.concatenate(labels),
    }
    np.savez_compressed(folder / "mesh.npz", **arrays)
    mesh_hash = hashlib.sha256(b"".join(array.tobytes() for array in arrays.values())).hexdigest()
    return {"objects": objects, "mesh_content_sha256": mesh_hash}


def position_camera(camera, generator, angle_degrees):
    angle = math.radians(angle_degrees)
    radius = generator["camera_radius_m"]
    camera.location = (
        radius * math.sin(angle),
        -radius * math.cos(angle),
        generator["camera_height_m"],
    )
    direction = Vector((0, 0, 0)) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def project_guide(camera, endpoints):
    camera_data = camera_record(bpy.context.scene)
    world_to_camera = np.array(camera_data["world_to_camera_cv"])
    projection = np.array(camera_data["K_index"]) @ world_to_camera[:3, :]
    points = np.c_[np.asarray(endpoints, float), np.ones(2)]
    values = points @ projection.T
    guide = values[:, :2] / values[:, 2, None]
    order = np.argsort(guide[:, 1])
    return camera_data, guide[order].tolist()


def main():
    request = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text(encoding="utf-8"))
    root = Path(request["root"])
    config = request["config"]
    inputs, truth = root / request["inputs"], root / request["truth"]
    reset_scene()
    scene, camera = setup_static_scene(config["generator"])
    materials = {
        "matte": principled_material("TargetMatte", (0.62, 0.64, 0.68), 0.48),
        "glossy": principled_material("TargetGlossy", (0.42, 0.46, 0.52), 0.08, 0.18),
        "stripe": stripe_material("TargetStripe"),
        "distractor": principled_material("Distractor", (0.70, 0.72, 0.74), 0.34),
        "occluder": principled_material("Occluder", (0.18, 0.20, 0.14), 0.9),
    }
    result = {
        "blender_version": bpy.app.version_string,
        "scope": config["scope"],
        "render": {
            "engine": scene.render.engine,
            "size_wh": config["generator"]["size_wh"],
            "samples": scene.eevee.taa_render_samples,
            "view_transform": scene.view_settings.view_transform,
            "look": scene.view_settings.look,
        },
        "cases": [],
    }
    for case in config["cases"]:
        clear_case_objects()
        setup_case(case, materials)
        bpy.context.view_layer.update()
        case_truth = truth / case["case_id"]
        case_truth.mkdir()
        geometry = export_mesh(scene, case_truth)
        save_json(case_truth / "geometry.json", geometry)
        frames = []
        for angle in config["generator"]["angles_degrees"]:
            position_camera(camera, config["generator"], angle)
            bpy.context.view_layer.update()
            if "guide_xyxy" in config["generator"]:
                camera_data = camera_record(scene)
                guide = config["generator"]["guide_xyxy"]
                guide_source = "fixed_screen_coordinates_independent_of_target_geometry"
            else:
                camera_data, guide = project_guide(camera, case["guide_endpoints"])
                guide_source = "synthetic_world_guide_projection"
            checks, projection_error = (None, None)
            if config["generator"].get("independent_probes", False):
                checks, projection_error = probes(scene, camera_data, geometry)
                if projection_error > 0.01:
                    raise RuntimeError(f"Independent Blender projection error: {projection_error}")
            frame_id = f"view_{angle:+03d}"
            rgb = inputs / case["case_id"] / f"{frame_id}.png"
            rgb.parent.mkdir(parents=True, exist_ok=True)
            scene.render.filepath = str(rgb)
            bpy.ops.render.render(write_still=True)
            save_json(case_truth / f"{frame_id}-camera.json", camera_data)
            frames.append(
                {"frame_id": frame_id, "angle_degrees": angle, "camera": camera_data,
                 "guide_xyxy": guide, "guide_source": guide_source,
                 "blender_ray_probes": checks, "projection_check_max_px": projection_error,
                 "rgb": rgb.relative_to(inputs).as_posix()}
            )
            print("IDENTITY_RENDER", case["case_id"], frame_id, flush=True)
        result["cases"].append(
            {"case_id": case["case_id"], "geometry": geometry, "frames": frames}
        )
    save_json(truth / request.get("render_manifest_name", "render_manifest.json"), result)


if __name__ == "__main__":
    main()
