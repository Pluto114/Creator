"""Same reference boards and rods; change only the camera-height sequence."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blender_rod_reference_pack as reference


def position(camera, generator, angle):
    index = generator['angles_degrees'].index(angle)
    # Keep radius, azimuth, look-at point and lens untouched. The first view
    # stays at the old height, giving us a byte-level render regression too.
    reference.position(camera, {**generator, 'camera_height_m': generator['camera_heights_m'][index]}, angle)


if __name__ == '__main__':
    reference.prior.setup_static_scene = reference.setup
    reference.prior.position_camera = position
    reference.prior.main()
