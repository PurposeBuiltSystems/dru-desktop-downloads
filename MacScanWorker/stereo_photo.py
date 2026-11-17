#!/usr/bin/env python3
"""Refine photo depths with agreement between independently rectified photo pairs."""
import argparse
import time
from pathlib import Path

import cv2
from scan_image_io import read_image
import numpy as np

from dense_photo import backproject_depth, read_text_model
from reconstruct import write_json


def camera_matrix(camera, size):
    params = camera["params"]
    fx, fy, cx, cy = params[:4] if camera["model"] == "PINHOLE" else (params[0], params[0], params[1], params[2])
    sx, sy = size[0] / camera["width"], size[1] / camera["height"]
    return np.array([[fx * sx, 0, (cx + .5) * sx - .5],
                     [0, fy * sy, (cy + .5) * sy - .5], [0, 0, 1.]])


def select_peers(index, poses):
    centers = poses[:, :3, 3]
    distance = np.linalg.norm(centers - centers[index], axis=1)
    direction = poses[:, :3, 2] @ poses[index, :3, 2]
    eligible = np.flatnonzero((distance > .08) & (distance < .5) & (direction > .9))
    return eligible[np.argsort(np.abs(distance[eligible] - .2), kind="stable")[:4]]


def pair_points(left, right, left_k, right_k, left_pose, right_pose):
    """Return source-camera points after disparity, cycle and photo checks."""
    size = (left.shape[1], left.shape[0])
    rotation = right_pose[:3, :3].T @ left_pose[:3, :3]
    translation = right_pose[:3, :3].T @ (left_pose[:3, 3] - right_pose[:3, 3])
    r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(
        left_k, None, right_k, None, size, rotation, translation,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )
    vertical = abs(p2[1, 3]) > abs(p2[0, 3])
    axis = 1 if vertical else 0
    if not np.isfinite(p1).all() or min(p1[0, 0], p1[1, 1]) <= 0:
        return np.empty((0, 3))
    map_left = cv2.initUndistortRectifyMap(left_k, None, r1, p1, size, cv2.CV_32FC1)
    map_right = cv2.initUndistortRectifyMap(right_k, None, r2, p2, size, cv2.CV_32FC1)
    a = cv2.remap(left, *map_left, cv2.INTER_LINEAR)
    b = cv2.remap(right, *map_right, cv2.INTER_LINEAR)
    visible_left = ((map_left[0] >= 0) & (map_left[0] < size[0] - 1)
                    & (map_left[1] >= 0) & (map_left[1] < size[1] - 1))
    visible_right = ((map_right[0] >= 0) & (map_right[0] < size[0] - 1)
                     & (map_right[1] >= 0) & (map_right[1] < size[1] - 1))
    if vertical:
        a, b = a.T.copy(), b.T.copy()
        visible_left, visible_right = visible_left.T.copy(), visible_right.T.copy()
    minimum = -128 if p2[axis, 3] > 0 else 0
    settings = dict(numDisparities=128, blockSize=5, P1=8 * 25, P2=32 * 25,
                    disp12MaxDiff=1, uniquenessRatio=12, speckleWindowSize=100,
                    speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    # Both match directions lose a search-width border. Padding keeps vertical
    # captures and short rectified images inside the shared valid search area.
    padded_a = cv2.copyMakeBorder(a, 0, 0, 128, 128, cv2.BORDER_CONSTANT)
    padded_b = cv2.copyMakeBorder(b, 0, 0, 128, 128, cv2.BORDER_CONSTANT)
    disparity = cv2.StereoSGBM_create(minDisparity=minimum, **settings).compute(padded_a, padded_b)[:, 128:-128].astype(np.float32) / 16
    reverse = cv2.StereoSGBM_create(minDisparity=-minimum - 128, **settings).compute(padded_b, padded_a)[:, 128:-128].astype(np.float32) / 16
    yy, xx = np.mgrid[:a.shape[0], :a.shape[1]]
    peer_x, peer_y = (xx - disparity).astype(np.float32), yy.astype(np.float32)
    back = cv2.remap(reverse, peer_x, peer_y, cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=1e4)
    matched = cv2.remap(b, peer_x, peer_y, cv2.INTER_LINEAR)
    peer_visible = cv2.remap(visible_right.astype(np.uint8), peer_x, peer_y, cv2.INTER_NEAREST) > 0
    valid = (visible_left & peer_visible & (disparity > minimum) & (np.abs(disparity) > 1)
             & (peer_x > 0) & (peer_x < a.shape[1] - 1)
             & (np.abs(disparity + back) < .75)
             & (np.abs(a.astype(float) - matched) < 20))
    if vertical:
        disparity, valid = disparity.T.copy(), valid.T.copy()
    camera = cv2.reprojectImageTo3D(disparity, q) @ r1
    valid &= np.isfinite(camera).all(axis=2) & (camera[..., 2] > .3) & (camera[..., 2] < 5)
    return camera[valid]


def rasterize_depth(camera, intrinsics, shape):
    """Z-buffer pair measurements into the source depth-map calibration."""
    height, width = shape
    finite = np.isfinite(camera).all(axis=1) & (camera[:, 2] > .3) & (camera[:, 2] < 5)
    camera = camera[finite]
    pixel = camera @ intrinsics.T
    uv = np.rint(pixel[:, :2] / pixel[:, 2:3]).astype(np.intp)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    uv, z = uv[inside], camera[inside, 2]
    depth = np.full(height * width, np.inf)
    np.minimum.at(depth, uv[:, 1] * width + uv[:, 0], z)
    depth[~np.isfinite(depth)] = np.nan
    return depth.reshape(height, width)


def consensus_depth(candidates, peer_centers, depth, mask):
    """Require two separated witnesses and agreement across available pairs."""
    if len(candidates) < 2:
        return depth.copy(), mask.copy(), np.zeros_like(mask)
    values = np.stack(candidates)
    available = np.isfinite(values)
    count = available.sum(axis=0)
    # Fill empty columns for reductions, then restore their missing-data status.
    safe = np.where(available, values, np.inf)
    ordered = np.sort(safe, axis=0)
    lower = np.take_along_axis(ordered, np.maximum((count - 1) // 2, 0)[None], axis=0)[0]
    upper = np.take_along_axis(ordered, (count // 2)[None], axis=0)[0]
    median = (lower + upper) / 2
    spread = np.max(np.where(available, values, -np.inf), axis=0) - ordered[0]
    consistent = available & (np.abs(values - np.where(count > 0, median, 0)) < .02)
    separated = np.zeros_like(mask)
    for a in range(len(peer_centers)):
        for b in range(a):
            if np.linalg.norm(peer_centers[a] - peer_centers[b]) > .04:
                separated |= consistent[a] & consistent[b]
    accepted = ((consistent.sum(axis=0) >= 2) & separated & (spread < .04)
                & np.isfinite(median) & ((np.abs(median - depth) < .5) | ~mask))
    return np.where(accepted, median, depth), mask | accepted, accepted


def refine(root):
    started = time.perf_counter()
    cameras, images = read_text_model(root / "sparse/0_text")
    paths = sorted((root / "refined").glob("*.prediction.npz"))
    by_name = {image["name"]: image for image in images}
    images = [by_name[path.name.removesuffix(".prediction.npz")] for path in paths]
    if not images:
        raise ValueError("Stereo refinement needs existing photo depths")
    first = cameras[images[0]["camera_id"]]
    scale = min(1, 640 / max(first["width"], first["height"]))
    size = (round(first["width"] * scale), round(first["height"] * scale))
    photos, matrices = [], []
    for image in images:
        photo = read_image(root / "images" / image["name"], cv2.IMREAD_GRAYSCALE)
        camera = cameras[image["camera_id"]]
        if photo is None or photo.shape != (camera["height"], camera["width"]):
            raise ValueError(f"Photo dimensions disagree with calibration for {image['name']}")
        photos.append(cv2.resize(photo, size))
        matrices.append(camera_matrix(camera, size))
    poses = np.array([image["c2w"] for image in images])
    output = root / "stereo-refined"
    output.mkdir(exist_ok=True)
    report = {"state": "processing", "notice": "Photo-pair estimates. Physical accuracy and completeness unverified.",
              "requirements": {"matching_witnesses": 2, "witness_separation_m": .04,
                               "maximum_pair_spread_m": .04, "maximum_existing_depth_change_m": .5},
              "views": []}
    write_json(root / "stereo-refinement.json", report)
    for index, (image, path) in enumerate(zip(images, paths)):
        with np.load(path) as packed:
            arrays = {key: packed[key] for key in packed.files}
        pose, intrinsics, mask = arrays["camera_to_world"], arrays["intrinsics"], arrays["mask"]
        depth = (arrays["camera_depth"] if "camera_depth" in arrays else
                 (arrays["xyz"] - pose[:3, 3]) @ pose[:3, :3][:, 2])
        candidates, centers, pair_rows = [], [], []
        for peer in select_peers(index, poses):
            try:
                camera = pair_points(photos[index], photos[peer], matrices[index], matrices[peer], poses[index], poses[peer])
            except cv2.error as error:
                pair_rows.append({"image": images[peer]["name"], "skipped": str(error)})
                continue
            candidates.append(rasterize_depth(camera, intrinsics, depth.shape))
            centers.append(poses[peer, :3, 3])
            pair_rows.append({"image": images[peer]["name"], "measurements": len(camera)})
        corrected, new_mask, accepted = consensus_depth(candidates, centers, depth, mask)
        if "camera_depth" in arrays:
            arrays["camera_depth"] = corrected
        if "xyz" in arrays:
            arrays["xyz"] = arrays["xyz"].copy()
            projected = backproject_depth(corrected, intrinsics, pose).astype(arrays["xyz"].dtype)
            arrays["xyz"][accepted] = projected[accepted]
        arrays["mask"] = new_mask
        temporary = output / (path.name + ".partial.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(output / path.name)
        report["views"].append({"image": image["name"], "pairs": pair_rows,
                                "corrected_pixels": int(accepted.sum()),
                                "newly_supported_pixels": int((accepted & ~mask).sum())})
    expected = {path.name for path in paths}
    for previous in output.glob("*.prediction.npz"):
        if previous.name not in expected:
            previous.unlink()
    report.update(state="done", seconds=round(time.perf_counter() - started, 3))
    write_json(root / "stereo-refinement.json", report)
    print(f"Refined {len(images)} photo views using calibrated stereo pairs", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    refine(parser.parse_args().workspace)
