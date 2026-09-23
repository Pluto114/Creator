"""Render new rod layouts with explicit reference panels and a zoom counterexample."""

import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blender_rod_identity_pack as prior

_setup = prior.setup_static_scene
_position = prior.position_camera


def reference_material(name, seed):
    rng = np.random.default_rng(seed)
    # These are deliberately helpful reference boards. They change acquisition
    # conditions; any success here must not be sold as fixing arbitrary photos.
    grey = rng.uniform(.025, .38, (64, 32))
    rgba = np.ones((512, 256, 4), np.float32)
    rgba[:, :, :3] = np.repeat(np.repeat(grey, 8, axis=0), 8, axis=1)[:, :, None]
    image = bpy.data.images.new(name, width=256, height=512, alpha=True)
    image.pixels.foreach_set(rgba.ravel())
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    coord = tree.nodes.new('ShaderNodeTexCoord')
    split = tree.nodes.new('ShaderNodeSeparateXYZ')
    combine = tree.nodes.new('ShaderNodeCombineXYZ')
    tex = tree.nodes.new('ShaderNodeTexImage')
    tex.image, tex.interpolation = image, 'Linear'
    emission = tree.nodes.new('ShaderNodeEmission')
    output = tree.nodes.new('ShaderNodeOutputMaterial')
    tree.links.new(coord.outputs['Generated'], split.inputs['Vector'])
    tree.links.new(split.outputs['X'], combine.inputs['X'])
    tree.links.new(split.outputs['Z'], combine.inputs['Y'])
    tree.links.new(combine.outputs['Vector'], tex.inputs['Vector'])
    tree.links.new(tex.outputs['Color'], emission.inputs['Color'])
    tree.links.new(emission.outputs['Emission'], output.inputs['Surface'])
    return material


def setup(generator):
    scene, camera = _setup(generator)
    wall = bpy.data.objects['BrickWall']
    wall.data.materials.clear()
    wall.data.materials.append(prior.principled_material('PlainBackdrop', (.025, .028, .03), .85))
    for i, (x, y) in enumerate(((-1.95, 1.20), (1.95, .20))):
        prior.add_cube(f'ReferenceBoard{i}', (x, y, 0), (1.25, .025, 2.4),
                       reference_material(f'ReferenceTexture{i}', generator['reference_seed'] + 37 * i),
                       110 + i, 'reference_board')
    return scene, camera


def position(camera, generator, angle):
    _position(camera, generator, angle)
    camera.data.lens = generator['lens_by_view_mm'][generator['angles_degrees'].index(angle)]


if __name__ == '__main__':
    prior.setup_static_scene = setup
    prior.position_camera = position
    prior.main()
