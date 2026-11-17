#!/usr/bin/env python3
"""Align captured photos with pose-guided pairs and a broad-matching fallback."""
import argparse
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import pycolmap
from photo_stages import photo_support

TEMPORAL_OVERLAP = 3
POSE_NEIGHBORS = 5


def pose_guided_pairs(manifest):
    """Keep local sequence coverage and add nearby, similarly directed cameras."""
    images = sorted(manifest["images"], key=lambda image: image["filename"])
    if manifest.get("trackingInterruptions") or len(images) < 6:
        return None
    try:
        values = np.array([image["cameraToWorld"] for image in images], dtype=float)
        if values.shape != (len(images), 16):
            return None
        poses = values.reshape(-1, 4, 4).transpose(0, 2, 1)
    except (KeyError, ValueError, TypeError):
        return None
    if not np.isfinite(poses).all() or not np.allclose(poses[:, 3, :], [0, 0, 0, 1], atol=.001):
        return None
    rotations = poses[:, :3, :3]
    if not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=.02) or not np.allclose(np.linalg.det(rotations), 1, atol=.02):
        return None
    positions = poses[:, :3, 3]
    if np.linalg.norm(np.ptp(positions, axis=0)) < .01:
        return None
    directions = -poses[:, :3, 2]
    pairs = set()
    for index, image in enumerate(images):
        for neighbor in range(index + 1, min(index + TEMPORAL_OVERLAP + 1, len(images))):
            pairs.add(tuple(sorted((image["filename"], images[neighbor]["filename"]))))
        distances = np.linalg.norm(positions - positions[index], axis=1)
        candidates = np.flatnonzero((directions @ directions[index] > .5) & (np.arange(len(images)) != index))
        nearby = candidates[np.argsort(distances[candidates], kind="stable")[:POSE_NEIGHBORS]]
        for neighbor in nearby:
            pairs.add(tuple(sorted((image["filename"], images[neighbor]["filename"]))))
    return sorted(pairs)


def needs_broader_matching(reconstruction, image_count):
    if reconstruction.num_reg_images() != image_count:
        return True
    error = reconstruction.compute_mean_reprojection_error()
    if not np.isfinite(error) or error > 1.2:
        return True
    # Match the downstream depth-hint requirements before accepting narrower pairs.
    for image in reconstruction.images.values():
        depth, _ = photo_support(reconstruction, image)
        if np.count_nonzero(depth) < 100:
            return True
    return False


def sequential_pairs(database, options):
    """Reuse COLMAP's pair selection while batching descriptor matching."""
    connection = pycolmap.Database.open(database)
    try:
        names = {image.image_id: image.name for image in connection.read_all_images()}
        pairs = pycolmap.SequentialPairGenerator(options, connection).all_pairs()
        return [(names[first], names[second]) for first, second in pairs]
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dru-scripts", type=Path, required=True)
    parser.add_argument("--pose-guided", action="store_true")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("dru_adapter", args.dru_scripts / "dru_recon.py")
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    output = args.out
    output.mkdir(parents=True, exist_ok=True)
    scan = output / "scan"
    started = time.perf_counter()
    adapter.unpack(args.scan, scan)
    manifest = adapter.load_manifest(scan)
    report = {"scan_id": manifest.get("id"), "images": len(manifest["images"]), "createdAt": manifest.get("createdAt"), "coordinateSystem": manifest.get("coordinateSystem"), "steps": {}}
    step = time.perf_counter()
    database = adapter.build_database(output, scan, manifest, .03, 8192)
    report["steps"]["features_s"] = round(time.perf_counter() - step, 3)
    report["capture_quality"] = adapter.capture_quality(database, output / "images")
    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = 8
    pairing.quadratic_overlap = True
    pairing.loop_detection = False
    matching = pycolmap.FeatureMatchingOptions()
    matching.guided_matching = True
    pose_pairs = pose_guided_pairs(manifest) if args.pose_guided else None
    pairs = pose_pairs if pose_pairs is not None else sequential_pairs(database, pairing)
    step = time.perf_counter()
    pair_file = output / "matching-pairs.txt"
    pair_file.write_text("".join(f"{first} {second}\n" for first, second in pairs))
    imported = pycolmap.ImportedPairingOptions()
    imported.match_list_path = str(pair_file)
    pycolmap.match_image_pairs(database, matching_options=matching, pairing_options=imported, device=pycolmap.Device.cpu)
    report["steps"]["matching_s"] = round(time.perf_counter() - step, 3)
    options = pycolmap.IncrementalPipelineOptions()
    options.use_prior_position = True
    options.use_robust_loss_on_prior_position = True
    options.prior_position_loss_scale = 7.815
    options.ba_refine_focal_length = True
    options.ba_refine_principal_point = False
    options.ba_refine_extra_params = False
    options.min_num_matches = 15
    options.ba_global_frames_ratio = 1.6
    options.ba_global_points_ratio = 1.6
    options.ba_global_max_refinements = 2
    options.ba_global_max_num_iterations = 40
    options.ba_local_max_num_iterations = 20
    step = time.perf_counter()
    sparse = output / "sparse"
    sparse.mkdir(exist_ok=True)
    reconstructions = pycolmap.incremental_mapping(database, output / "images", sparse, options)
    mapping_seconds = time.perf_counter() - step
    fallback = False
    reconstruction = max(reconstructions.values(), key=lambda model: model.num_reg_images()) if reconstructions else None
    if pose_pairs is not None and (reconstruction is None or needs_broader_matching(reconstruction, len(manifest["images"]))):
        fallback = True
        retry = time.perf_counter()
        pycolmap.match_sequential(database, matching_options=matching, pairing_options=pairing, device=pycolmap.Device.cpu)
        report["steps"]["matching_s"] += round(time.perf_counter() - retry, 3)
        retry_sparse = sparse / "retry"
        retry_sparse.mkdir(exist_ok=True)
        retry = time.perf_counter()
        reconstructions = pycolmap.incremental_mapping(database, output / "images", retry_sparse, options)
        mapping_seconds += time.perf_counter() - retry
        reconstruction = max(reconstructions.values(), key=lambda model: model.num_reg_images()) if reconstructions else None
    if reconstruction is None:
        raise RuntimeError("Photo alignment registered no images")
    model = sparse / "0"
    model.mkdir(exist_ok=True)
    reconstruction.write(model)
    text = sparse / "0_text"
    text.mkdir(exist_ok=True)
    reconstruction.write_text(text)
    report["steps"]["mapping_s"] = round(mapping_seconds, 3)
    xyz, colors = adapter.sparse_points(reconstruction)
    adapter.write_ply(output / "sfm_sparse.ply", xyz, colors)
    report["sfm"] = adapter.compare_to_arcore(reconstruction, manifest)
    report["alignment_profile"] = {"matching": "pose-guided with sequence coverage" if pose_pairs is not None else "batched sequential with quadratic overlap", "candidate_pairs": len(pairs), "pose_neighbors": POSE_NEIGHBORS, "temporal_overlap": TEMPORAL_OVERLAP, "broader_matching_retry": fallback, "overlap": 8, "global_adjustment_ratio": 1.6, "global_max_refinements": 2, "global_max_iterations": 40}
    report["verdict"] = "EXPERIMENTAL: independent dimensional accuracy unverified"
    report["steps"]["total_s"] = round(time.perf_counter() - started, 3)
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
