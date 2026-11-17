#!/usr/bin/env python3
"""Prepare sparse photo hints, refine inferred depths, and reuse the existing exporter."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import numpy as np
from dense_photo import backproject_depth

NOTICE = "Experimental inferred geometry with photo-supported local planar estimates. Small surface details may be smoothed. Dimensional accuracy and completeness unverified."



def photo_support(reconstruction, image):
    """Separate training depth hints and withheld observations for one photo."""
    camera = reconstruction.cameras[image.camera_id]
    world_to_camera = image.cam_from_world().matrix()
    depth = np.zeros((camera.height, camera.width), np.float32)
    held_out = []
    for pixel in image.points2D:
        if not pixel.has_point3D():
            continue
        point = reconstruction.points3D[pixel.point3D_id]
        if point.track.length() < 4 or point.error > 1.5:
            continue
        z = (world_to_camera[:3, :3] @ point.xyz + world_to_camera[:3, 3])[2]
        if not .3 < z < 5:
            continue
        if pixel.point3D_id % 4 == 0:
            held_out.append({"image": image.name, "point_id": pixel.point3D_id, "xyz": point.xyz.tolist()})
            continue
        x, y = np.rint(pixel.xy).astype(int)
        if 0 <= x < camera.width and 0 <= y < camera.height:
            depth[y, x] = z
    return depth, held_out


def prepare(root):
    import pycolmap
    reconstruction = pycolmap.Reconstruction(root / "sparse/0")
    hints = root / "sfm_depth"
    hints.mkdir(exist_ok=True)
    rows = []
    held_out = []
    for image in reconstruction.images.values():
        depth, reserved = photo_support(reconstruction, image)
        held_out.extend(reserved)
        count = int(np.count_nonzero(depth))
        if count < 100:
            raise ValueError(f"Insufficient triangulated photo support for {image.name}: {count} hints")
        np.save(hints / (image.name + ".npy"), depth)
        rows.append({"image": image.name, "hints": count})
    (root / "photo-hints.json").write_text(json.dumps({"training": rows, "held_out": held_out}, indent=2))
    print(f"Prepared {len(rows)} views and reserved {len(held_out)} feature observations")


def local_depth_correction(depth, mask, u, v, observed):
    """Correct smooth local depth bias using nearby triangulated training features."""
    from scipy.spatial import cKDTree
    residual = observed - depth[v, u]
    supported = np.abs(residual) < .25
    u, v, residual = u[supported], v[supported], residual[supported]
    if len(u) < 30:
        return depth, {"supported_features": len(u), "applied": False}
    height, width = depth.shape
    yy, xx = np.mgrid[:height, :width]
    distances, indices = cKDTree(np.column_stack([u, v])).query(np.column_stack([xx.ravel(), yy.ravel()]), k=16)
    # Limit propagation across depth discontinuities and beyond nearby photo support.
    radius = 20 * width / 392
    depth_delta = np.abs(depth.ravel()[:, None] - depth[v, u][indices])
    weights = np.exp(-.5 * (distances / radius) ** 2 - .5 * (depth_delta / .1) ** 2)
    weights[distances > 3 * radius] = 0
    field = (weights * residual[indices]).sum(axis=1) / (weights.sum(axis=1) + .05)
    field = .85 * np.clip(field, -.15, .15).reshape(height, width)
    field[~mask] = 0
    magnitudes = np.abs(field[mask])
    return depth + field, {"supported_features": len(u), "applied": True, "median_correction_m": float(np.median(magnitudes)), "p95_correction_m": float(np.quantile(magnitudes, .95)), "maximum_correction_m": .1275}


def regional_depth_correction(depth, mask, u, v, observed, intrinsics):
    """Constrain connected smooth regions with well-supported local plane fits."""
    import cv2
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    report = {"applied": False, "regions": [], "adjusted_pixels": 0,
              "assumption": "Connected smooth surfaces may be locally planar. Unobserved detail remains unverified."}
    height, width = depth.shape
    if len(observed) < 20 or min(height, width) < 3:
        return depth, report
    # Repeated projections into one reduced-resolution pixel provide one constraint.
    _, unique = np.unique(np.column_stack([u, v]), axis=0, return_index=True)
    u, v, observed = u[unique], v[unique], observed[unique]
    smooth = cv2.bilateralFilter(depth.astype(np.float32), 5, .03, 3)
    camera = backproject_depth(smooth, intrinsics, np.eye(4))
    normals = np.cross(np.gradient(camera, axis=1), np.gradient(camera, axis=0))
    normals /= np.maximum(np.linalg.norm(normals, axis=2, keepdims=True), 1e-8)
    ids = np.arange(height * width).reshape(height, width)
    horizontal = (
        (np.sum(normals[:, 1:] * normals[:, :-1], axis=2) > .96)
        & (np.abs(smooth[:, 1:] - smooth[:, :-1]) < .04)
        & mask[:, 1:] & mask[:, :-1]
    )
    vertical = (
        (np.sum(normals[1:] * normals[:-1], axis=2) > .96)
        & (np.abs(smooth[1:] - smooth[:-1]) < .04)
        & mask[1:] & mask[:-1]
    )
    start = np.concatenate([ids[:, 1:][horizontal], ids[1:][vertical]])
    end = np.concatenate([ids[:, :-1][horizontal], ids[:-1][vertical]])
    graph = coo_matrix((np.ones(len(start), np.uint8), (start, end)),
                       shape=(height * width, height * width))
    _, labels = connected_components(graph, directed=False)
    labels = labels.reshape(height, width)
    feature_labels = labels[v, u]
    correction = np.zeros_like(depth)
    for region in np.unique(feature_labels):
        features = feature_labels == region
        region_mask = (labels == region) & mask
        if features.sum() < 20 or region_mask.sum() < 200:
            continue
        if min(np.ptp(u[features]), np.ptp(v[features])) < 15:
            continue
        design = np.column_stack([
            (u[features] - width / 2) / width,
            (v[features] - height / 2) / width, np.ones(features.sum()),
        ])
        values = observed[features]
        weights = np.ones(len(values))
        for _ in range(8):
            root_weight = np.sqrt(weights)
            coefficients = np.linalg.lstsq(
                design * root_weight[:, None], root_weight / values, rcond=None,
            )[0]
            predicted = 1 / np.maximum(design @ coefficients, 1e-4)
            error = np.abs(predicted - values)
            weights = np.minimum(1, .02 / np.maximum(error, 1e-8))
        if np.mean(error < .03) < .85:
            continue
        rows, columns = np.nonzero(region_mask)
        target_design = np.column_stack([
            (columns - width / 2) / width,
            (rows - height / 2) / width, np.ones(len(columns)),
        ])
        target = 1 / np.maximum(target_design @ coefficients, 1e-4)
        delta = target - depth[rows, columns]
        plane_normal = np.array([
            coefficients[0] * intrinsics[0, 0] / width,
            coefficients[1] * intrinsics[1, 1] / width,
            coefficients[2]
            + coefficients[0] * (intrinsics[0, 2] - width / 2) / width
            + coefficients[1] * (intrinsics[1, 2] - height / 2) / width,
        ])
        plane_normal /= max(np.linalg.norm(plane_normal), 1e-8)
        # A smooth sequence of normals can connect curved surfaces or rounded corners.
        # Preserve pixels whose orientation disagrees with the measured plane.
        orientation_agreement = np.abs(normals[rows, columns] @ plane_normal)
        accepted = (np.abs(delta) < .4) & (orientation_agreement > .9)
        correction[rows[accepted], columns[accepted]] = .9 * np.clip(delta[accepted], -.3, .3)
        report["regions"].append({"training_pixels": int(features.sum()),
                                  "adjusted_pixels": int(accepted.sum()),
                                  "p95_training_residual_m": float(np.quantile(error, .95))})
    report["adjusted_pixels"] = int(np.count_nonzero(correction))
    report["applied"] = bool(report["adjusted_pixels"])
    report["maximum_correction_m"] = float(np.abs(correction).max())
    return depth + correction, report


def refine_view(path, root, raw_k, output, compact):
    data = np.load(path)
    pose, k, mask = data["camera_to_world"], data["intrinsics"], data["mask"]
    height, width = mask.shape
    depth = data["camera_depth"].astype(float) if "camera_depth" in data else ((data["xyz"] - pose[:3, 3]) @ pose[:3, :3][:, 2]).astype(float)
    hint = np.load(root / "sfm_depth" / path.name.replace(".prediction.npz", ".npy"))
    y, x = np.nonzero(hint > 0)
    u = np.rint((x - raw_k[0, 2]) / raw_k[0, 0] * k[0, 0] + k[0, 2]).astype(int)
    v = np.rint((y - raw_k[1, 2]) / raw_k[1, 1] * k[1, 1] + k[1, 2]).astype(int)
    indices = np.where((u >= 0) & (u < width) & (v >= 0) & (v < height))[0]
    indices = indices[mask[v[indices], u[indices]]]
    observed = hint[y[indices], x[indices]]
    predicted = depth[v[indices], u[indices]]
    fit = np.array([1., 0.])
    method = "unadjusted_insufficient_support"
    if len(predicted) >= 100 and np.std(predicted) > .1:
        method = "robust_affine"
        design = np.column_stack([predicted, np.ones(len(predicted))])
        weights = np.ones(len(predicted))
        for _ in range(15):
            fit = np.linalg.lstsq(design * np.sqrt(weights[:, None]), observed * np.sqrt(weights), rcond=None)[0]
            residual = np.abs(design @ fit - observed)
            weights = np.minimum(1, .025 / np.maximum(residual, 1e-8))
    elif len(predicted) >= 30:
        # A nearly planar view cannot constrain both scale and offset reliably.
        method = "robust_offset"
        fit[1] = np.median(observed - predicted)
    accepted = .5 <= fit[0] <= 1.5 and abs(fit[1]) <= .5
    if not accepted:
        fit = np.array([1., 0.])
    corrected_depth = depth * fit[0] + fit[1]
    corrected_depth, regional_report = regional_depth_correction(
        corrected_depth, mask, u[indices], v[indices], observed, k,
    )
    corrected_depth, local_report = local_depth_correction(corrected_depth, mask, u[indices], v[indices], observed)
    arrays = {"mask": mask & (corrected_depth > .3) & (corrected_depth < 5), "intrinsics": k, "camera_to_world": pose}
    if compact:
        # Preserve float64 corrected depth so deferred projection reproduces XYZ exactly.
        arrays["camera_depth"] = corrected_depth
    else:
        arrays["xyz"] = backproject_depth(corrected_depth, k, pose).astype(np.float32)
    np.savez_compressed(output / path.name, **arrays)
    return {"image": path.name, "fit_applied": bool(accepted and method != "unadjusted_insufficient_support"), "method": method, "scale": float(fit[0]), "offset_m": float(fit[1]), "regional_refinement": regional_report, "local_refinement": local_report}


def refine(root, compact=False, workers=1):
    if workers < 1:
        raise ValueError("Refinement needs at least one worker")
    params = json.loads((root / "report.json").read_text())["sfm"]["refined_camera_params"]
    raw_k = np.array([[params[0], 0, params[2]], [0, params[1], params[3]], [0, 0, 1.]])
    output = root / "refined"
    output.mkdir(exist_ok=True)
    paths = sorted(root.glob("*.prediction.npz"))
    if not paths:
        raise ValueError("No dense predictions were produced")
    if workers == 1:
        rows = [refine_view(path, root, raw_k, output, compact) for path in paths]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(paths))) as executor:
            rows = list(executor.map(lambda path: refine_view(path, root, raw_k, output, compact), paths))
    (root / "depth-refinement.json").write_text(json.dumps(rows, indent=2))
    print(f"Refined {len(rows)} views")


def package(root, scripts):
    spec = importlib.util.spec_from_file_location("scan_package", scripts / "package_deliverable.py")
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    from plyfile import PlyData
    data = PlyData.read(root / "mesh.ply")
    vertices = data["vertex"]
    xyz = np.column_stack([vertices[key] for key in ["x", "y", "z"]])
    rgb = np.column_stack([vertices[key] for key in ["red", "green", "blue"]])
    faces = np.stack(data["face"]["vertex_indices"])
    manifest = json.loads((root / "scan/manifest.json").read_text())
    # Geometry is upright in its own local frame. Export no source-frame measurements.
    report = json.loads((root / "report.json").read_text())
    repaired = bool(report.get("surface_repair"))
    notice = report["surface_repair"]["notice"] if repaired else NOTICE
    report["reconstruction_notice"] = notice
    report["coordinateSystem"] = "Local upright reconstruction frame, metres. No survey reference."
    report["mesh"] = "mesh.ply"
    report["verdict"] = "Experimental guided visualization. Independent accuracy unverified" if repaired else "EXPERIMENTAL: independent accuracy unverified"
    (root / "report.json").write_text(json.dumps(report, indent=2))
    meta, _ = exporter.gather_meta(root)
    meta["detail"] = f"{len(manifest['images'])} aligned photos · local upright estimated geometry"
    meta["quality"] = notice
    dense_selection = root / "dense-view-selection.json"
    if dense_selection.exists():
        dense_info = json.loads(dense_selection.read_text())
        meta["detail"] = f"{len(manifest['images'])} aligned photos · {dense_info['views']} dense views · local upright estimated geometry"
        report["dense_view_selection"] = dense_info
        (root / "report.json").write_text(json.dumps(report, indent=2))
    stereo_report = root / "stereo-refinement.json"
    if stereo_report.exists():
        report["photo_pair_refinement"] = json.loads(stereo_report.read_text())
        (root / "report.json").write_text(json.dumps(report, indent=2))
    if repaired:
        meta["detail"] = f"{len(manifest['images'])} source photos · guided repaired visualization"
    meta["verdict"] = "WEAK: accuracy unverified"
    page = exporter.html_page(xyz, rgb, faces, meta, [])
    original_notice = "Geometry is in feet, reconstructed from the phone's photos and ARCore tracking.\n    Expect a few centimetres of error. The frame is local to the scan, not georeferenced."
    assert original_notice in page, "Review the exporter notice before using the updated template"
    page = page.replace(original_notice, "Viewer units are feet and metres. Output files declare their units. Geometry is in a local upright frame. " + notice + " Gaps and distortion may remain. Independently verify before inspection or engineering decisions.")
    (root / "model.html").write_text(page)
    import zipfile
    with zipfile.ZipFile(root / "model-deliverable.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in ["model.html", "model-metres.ply", "model-metres.obj", "report.json", "model-quality.json"]:
            bundle.write(root / name, name)
        if repaired:
            for name in ["repair-provenance.json", "reconstruction-before-repair.ply"]:
                bundle.write(root / name, name)
        bundle.writestr("README.txt", exporter.brand_for(meta.get("product"))["title"] + "\n\n" + notice + "\nPLY and OBJ coordinates are metres. The HTML viewer displays feet and metres. This is inferred geometry with remaining gaps and distortion.\n")
    print("Created local HTML viewer and model ZIP")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "refine", "package"])
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--dru-scripts", type=Path)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.stage == "package":
        package(args.workspace, args.dru_scripts)
    elif args.stage == "refine":
        refine(args.workspace, compact=args.compact, workers=args.workers)
    else:
        prepare(args.workspace)


if __name__ == "__main__":
    main()
