"""Fixed top/bottom image bands avoid dense-track component percolation."""
import numpy as np


def band_split(tracks, features, config):
    cell = config["cell_size_px"]
    upper, lower = config["validation_max_cell_y"], config["train_min_cell_y"]
    if cell <= 0 or upper < 0 or lower <= upper + 1:
        raise ValueError("Need at least one unused cell row between train and validation")
    labels, cells = [], []
    for track in tracks:
        occupied = []
        for view, feature in track:
            xy = np.asarray(features[view]["xy"][feature], float)
            if xy.shape != (2,) or not np.isfinite(xy).all() or (xy < 0).any():
                raise ValueError("Finite nonnegative image coordinates required")
            x, y = np.floor(xy / cell).astype(int)
            occupied.append((view, int(x), int(y)))
        ys = [c[2] for c in occupied]
        label = "validation" if ys and max(ys) <= upper else ("train" if ys and min(ys) >= lower else "excluded")
        labels.append(label)
        cells.append(set(occupied))
    train = set().union(*(c for c, label in zip(cells, labels) if label == "train"))
    validation = set().union(*(c for c, label in zip(cells, labels) if label == "validation"))
    assert not train & validation
    return np.array(labels, dtype="U10"), dict(policy="all observations must stay in the same fixed horizontal band; no balancing or residual selection",
        cell_size_px=cell, validation_max_cell_y=upper, train_min_cell_y=lower,
        train_tracks=labels.count("train"), validation_tracks=labels.count("validation"), excluded_tracks=labels.count("excluded"),
        shared_cells=len(train & validation), train_cells=sorted(train), validation_cells=sorted(validation))
