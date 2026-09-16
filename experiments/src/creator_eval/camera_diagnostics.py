"""Background match consistency is a diagnostic, not an absolute camera-accuracy certificate."""
import numpy as np

from creator_eval.native_diagnostics import homogeneous


def fundamental_from_cameras(k_first, first, k_second, second):
    relative = homogeneous(second) @ np.linalg.inv(homogeneous(first))
    rotation, translation = relative[:3, :3], relative[:3, 3]
    if np.linalg.norm(translation) < 1e-10:
        raise ValueError("No translation baseline for this epipolar diagnostic")
    x, y, z = translation
    cross = np.array([[0.0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.linalg.inv(k_second).T @ cross @ rotation @ np.linalg.inv(k_first)


def sampson_distances(first_xy, second_xy, fundamental):
    first_xy, second_xy = np.asarray(first_xy, float), np.asarray(second_xy, float)
    if first_xy.shape != second_xy.shape or first_xy.ndim != 2 or first_xy.shape[1] != 2:
        raise ValueError("Expected matching Nx2 image coordinates")
    x = np.c_[first_xy, np.ones(len(first_xy))]
    y = np.c_[second_xy, np.ones(len(second_xy))]
    fx, fty = x @ fundamental.T, y @ fundamental
    denominator = np.sum(fx[:, :2]**2, axis=1) + np.sum(fty[:, :2]**2, axis=1)
    valid = denominator > 1e-20
    result = np.full(len(x), np.nan)
    result[valid] = np.abs(np.sum(y*fx, axis=1)[valid]) / np.sqrt(denominator[valid])
    return result


def background_camera_audit(images, matrices, cameras, guides, guide_y, config):
    import cv2

    orb = cv2.ORB_create(nfeatures=config["maximum_features"], fastThreshold=config["fast_threshold"])
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    features = []
    for image, frame_guides in zip(images, guides):
        h, w = image.shape[:2]
        mask = np.full((h, w), 255, np.uint8)
        start, end = guide_y
        ys = np.arange(max(0, start), min(h, end+1))
        for guide in frame_guides.values():
            centers = guide[0] + (ys-start)/(end-start)*(guide[1]-guide[0])
            for y, center in zip(ys, centers):
                left = max(0, int(np.floor(center-config["target_exclusion_half_width"])))
                right = min(w, int(np.ceil(center+config["target_exclusion_half_width"]+1)))
                mask[y, left:right] = 0
        points, descriptors = orb.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), mask)
        features.append((points, descriptors))
    pairs = []
    for first, second in config["pairs"]:
        kp1, des1 = features[first]
        kp2, des2 = features[second]
        row = {"first_view": first, "second_view": second}
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            pairs.append({**row, "state": "insufficient_features", "match_count": 0})
            continue

        def ratio_matches(a, b):
            return {
                m.queryIdx: m.trainIdx for candidate in matcher.knnMatch(a, b, k=2)
                if len(candidate) == 2
                for m, n in [candidate] if m.distance < config["ratio"]*n.distance
            }

        forward, reverse = ratio_matches(des1, des2), ratio_matches(des2, des1)
        matches = [(i, j) for i, j in forward.items() if reverse.get(j) == i]
        row["match_count"] = len(matches)
        if len(matches) < config["minimum_matches"]:
            pairs.append({**row, "state": "insufficient_matches"})
            continue
        a = np.array([kp1[i].pt for i, _ in matches])
        b = np.array([kp2[j].pt for _, j in matches])
        try:
            f = fundamental_from_cameras(matrices[first], cameras[first], matrices[second], cameras[second])
            error = sampson_distances(a, b, f)
            error = error[np.isfinite(error)]
            if not len(error):
                raise ValueError("No valid epipolar residuals")
            row.update(
                median_sampson_px=float(np.median(error)), p95_sampson_px=float(np.quantile(error, 0.95)),
                fraction_within_2px=float(np.mean(error <= 2)), fraction_within_5px=float(np.mean(error <= 5)),
            )
            row["state"] = "consistent" if (
                row["median_sampson_px"] <= config["median_limit_px"]
                and row["fraction_within_5px"] >= config["minimum_fraction_within_5px"]
            ) else "inconsistent"
        except ValueError as error:
            row.update(state="degenerate", reason=str(error))
        pairs.append(row)
    supported = sum(p["state"] == "consistent" for p in pairs)
    assessed = sum(p["state"] in ("consistent", "inconsistent") for p in pairs)
    return {
        "scope": "Frozen ORB mutual-ratio matches excluding the two target strips; diagnostic only",
        # 砖块重复纹理也可能互相骗过匹配，所以这个分数不能代替相机标定。
        "limitation": "Repeated texture and scene structure can produce wrong matches; consistency does not prove metric accuracy.",
        "target_exclusion_only": "Other scene structure is allowed; no GT segmentation was used.",
        "state": "inconclusive" if assessed < config["minimum_assessed_pairs"] else (
            "consistent" if supported >= config["minimum_consistent_pairs"] else "inconsistent"
        ),
        "accepted_pairs": supported, "assessed_pairs": assessed, "pairs": pairs,
        "used_to_modify_cameras": False,
    }
