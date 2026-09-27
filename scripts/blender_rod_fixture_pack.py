"""Render declared fixture artwork; never export its projected corners to inputs."""
import json
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blender_rod_height_pack as height

CAD = None
ASSETS = None


def emission(name, image_path=None):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    shader = tree.nodes.new("ShaderNodeEmission")
    shader.inputs["Color"].default_value = (1, 1, 1, 1)
    shader.inputs["Strength"].default_value = 1
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(shader.outputs[0], output.inputs["Surface"])
    if image_path:
        texture = tree.nodes.new("ShaderNodeTexImage")
        texture.image = bpy.data.images.load(str(image_path), check_existing=True)
        texture.image.colorspace_settings.name = "Non-Color"
        texture.interpolation = "Closest"
        uv = tree.nodes.new("ShaderNodeTexCoord")
        tree.links.new(uv.outputs["UV"], texture.inputs["Vector"])
        tree.links.new(texture.outputs["Color"], shader.inputs["Color"])
    return material


def setup(generator):
    scene, camera = height.reference.setup(generator)
    for name in ("ReferenceBoard0", "ReferenceBoard1"):
        bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
    prior = height.reference.prior
    for index, panel in enumerate(CAD["panels"]):
        center = np.array(panel["center_world"])
        center[1] += .015
        prior.add_cube("FixturePanel" + str(index), center, (.72, .014, 1.12),
            emission("WhiteFixture" + str(index)), 110 + index, "declared_fixture")
    for marker in CAD["markers"]:
        # The black marker boundary is CAD geometry. Artwork has a white margin,
        # so its quad is larger; confusing these would bias the whole scale.
        corners = np.array(marker["corners_world"])
        centre = corners.mean(axis=0)
        vertices = centre + 1.25 * (corners - centre)
        mesh = bpy.data.meshes.new("MarkerMesh" + str(marker["marker_id"]))
        mesh.from_pydata(vertices.tolist(), [], [(0, 1, 2, 3)])
        mesh.update()
        uv = mesh.uv_layers.new(name="UVMap")
        for loop, xy in zip(mesh.polygons[0].loop_indices, ((0, 1), (1, 1), (1, 0), (0, 0))):
            uv.data[loop].uv = xy
        obj = bpy.data.objects.new("FixtureMarker" + str(marker["marker_id"]), mesh)
        scene.collection.objects.link(obj)
        obj.data.materials.append(emission("MarkerMaterial" + str(marker["marker_id"]),
            ASSETS / (str(marker["marker_id"]) + ".png")))
        prior.tag(obj, 200 + marker["marker_id"], "declared_fixture_marker")
    return scene, camera


if __name__ == "__main__":
    request = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text(encoding="utf-8"))
    CAD = request["declared_cad"]
    ASSETS = Path(request["marker_assets"])
    height.reference.prior.setup_static_scene = setup
    height.reference.prior.position_camera = height.position
    height.reference.prior.main()
