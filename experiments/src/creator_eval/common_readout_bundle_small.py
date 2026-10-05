"""Same bundle guard, six/twelve-neighbor proposal ablation only."""

from . import common_readout_bundle as bundle

DEFAULTS = {**bundle.DEFAULTS, "local_proposal_small_neighbors": 6}


def policy(config):
    return bundle.policy({**DEFAULTS, **config})


def readout(points, segments, config, region=None):
    return bundle.readout(points, segments, policy(config), region)
