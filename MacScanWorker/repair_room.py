#!/usr/bin/env python3
"""Create a guided room repair from annotated photos and an existing reconstruction."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import tempfile

import cv2
from scan_image_io import read_image
import numpy as np
from plyfile import PlyData
from scipy.spatial import ConvexHull, cKDTree

from dense_photo import read_text_model
from reconstruct import write_json
from result_storage import restore_source, compact_result

NOTICE = "Guided repaired visualization. Flat walls and floor are assumed. Occluded surfaces are inferred. Dimensions, object details and completeness are unverified."


def validate_annotations(annotations, manifest, complete=True):
    """Validate repair markings shared by the desktop and Android editors."""
    if annotations.get("scan_id") != manifest["id"]:
        raise ValueError("Repair markings belong to a different scan")
    images = {image["filename"]: image for image in manifest["images"]}
    step = annotations.get("grid_step_m", .015)
    if not isinstance(step, (int, float)) or not np.isfinite(step) or step <= 0:
        raise ValueError("Choose a positive repair grid size")

    def check_pixels(name, pixels, minimum):
        if name not in images or not isinstance(pixels, list) or not minimum <= len(pixels) <= 1000:
            raise ValueError("Repair markings need valid photo references and points")
        image = images[name]
        for point in pixels:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError("Each marking needs two pixel coordinates")
            x, y = point
            if not all(isinstance(value, (int, float)) and np.isfinite(value) for value in point):
                raise ValueError("Repair coordinates must be finite numbers")
            if not (0 <= x < image["height"] and 0 <= y < image["width"]):
                raise ValueError("Repair marking lies outside its upright photo")

    walls = annotations.get("walls", [])
    if len(walls) != 2:
        raise ValueError("Mark the floor edges of two intersecting walls")
    for wall in walls:
        contacts = wall.get("contacts", [])
        if complete and (not contacts or sum(len(row.get("pixels", [])) for row in contacts) < 2):
            raise ValueError("Each wall needs at least two separated floor-edge points")
        for row in contacts:
            check_pixels(row.get("image"), row.get("pixels"), 1 if complete else 0)
        color = wall.get("fallback_rgb")
        if not isinstance(color, list) or len(color) != 3 or not all(isinstance(v, int) and 0 <= v <= 255 for v in color):
            raise ValueError("Wall fallback colors need three byte values")
    masks = annotations.get("occluders", {})
    if complete and annotations.get("foreground_reference") not in masks:
        raise ValueError("Outline the foreground objects in the reference photo")
    for name, polygon in masks.items():
        check_pixels(name, polygon, 3 if complete else 0)
    references = annotations.get("color_references", [])
    if (complete and not references) or any(name not in images for name in references):
        raise ValueError("Choose valid photos for repaired surface colors")
    return annotations


def load_cameras(root, orientation):
    rotation = np.array(orientation["world_to_upright_rotation"])
    center = np.array(orientation["center_in_sfm_world"])
    models, views = read_text_model(root / "sparse/0_text")
    cameras = {}
    for view in views:
        model = models[view["camera_id"]]
        if model["model"] != "PINHOLE":
            raise ValueError("Guided repair requires the refined PINHOLE camera model")
        fx, fy, cx, cy = model["params"]
        pose = view["c2w"]
        cameras[view["name"]] = {
            "origin": (pose[:3, 3] - center) @ rotation.T,
            "rotation": pose[:3, :3].T @ rotation.T,
            "intrinsics": np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]]),
            "width": model["width"],
            "height": model["height"],
        }
    return cameras


def project(points, camera):
    local = (points - camera["origin"]) @ camera["rotation"].T
    depth = local[:, 2]
    safe_depth = np.maximum(depth, .01)
    k = camera["intrinsics"]
    u = local[:, 0] / safe_depth * k[0, 0] + k[0, 2]
    v = local[:, 1] / safe_depth * k[1, 1] + k[1, 2]
    visible = (depth > .3) & (u > 2) & (u < camera["width"] - 3) & (v > 2) & (v < camera["height"] - 3)
    return u, v, visible


def upright_ray(camera, pixel):
    x, y = pixel
    if not (0 <= x < camera["height"] and 0 <= y < camera["width"]):
        raise ValueError("Annotation lies outside the upright photograph")
    u, v = y, camera["height"] - 1 - x
    k = camera["intrinsics"]
    return np.array([(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], 1.]) @ camera["rotation"]


def intersect_floor(camera, pixel, height):
    ray = upright_ray(camera, pixel)
    if abs(ray[1]) < 1e-6:
        raise ValueError("Floor annotation is too close to the horizon")
    distance = (height - camera["origin"][1]) / ray[1]
    if not .3 < distance < 10:
        raise ValueError("Floor annotation has an unsupported intersection distance")
    return camera["origin"] + distance * ray


def fit_vertical_plane(points):
    points = np.asarray(points)
    horizontal = points[:, [0, 2]]
    center = horizontal.mean(axis=0)
    _, singular_values, vectors = np.linalg.svd(horizontal - center, full_matrices=False)
    if len(points) < 2 or singular_values[0] < .1:
        raise ValueError("Wall anchors need distinct floor contact points")
    normal = np.array([vectors[-1, 0], 0, vectors[-1, 1]])
    offset = -normal[[0, 2]] @ center
    return normal, float(offset), np.abs(points @ normal + offset)


def training_features(root, orientation):
    rotation = np.array(orientation["world_to_upright_rotation"])
    center = np.array(orientation["center_in_sfm_world"])
    training = []
    for line in (root / "sparse/0_text/points3D.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        if int(fields[0]) % 4 and float(fields[7]) < 1.5 and len(fields[8:]) >= 8:
            training.append([float(value) for value in fields[1:4]])
    if len(training) < 200:
        raise ValueError("Insufficient triangulated training features for the floor")
    return (np.array(training) - center) @ rotation.T


def floor_from_features(root, orientation):
    points = training_features(root, orientation)
    heights = points[:, 1]
    number_of_bins = max(1, int(np.ceil(np.ptp(heights) / .02)))
    edges = heights.min() + np.arange(number_of_bins + 1) * .02
    counts, edges = np.histogram(heights, bins=edges)
    peak = edges[np.argmax(counts)] + .01
    support = heights[np.abs(heights - peak) < .025]
    if len(support) < 200:
        raise ValueError("No dominant horizontal floor support")
    height = float(np.median(support))
    return height, {"training_features": len(support), "height_m": height, "p95_feature_residual_m": float(np.quantile(np.abs(support - height), .95))}


def fit_floor_frame(points, approximate_height):
    """Estimate floor tilt from triangulated training features, preserving scale."""
    candidates = points[np.abs(points[:, 1] - approximate_height) < .12]
    fallback = {"method": "horizontal_height_fallback", "normal_original_upright": [0., 1., 0.],
                "offset_m": -approximate_height, "tilt_degrees": 0., "training_features": 0}
    if len(candidates) < 200 or np.linalg.svd(candidates - candidates.mean(axis=0), compute_uv=False)[1] < .1:
        return np.eye(3), fallback
    generator = np.random.default_rng(51)
    best = np.zeros(len(candidates), bool)
    for _ in range(700):
        a, b, c = candidates[generator.choice(len(candidates), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-8:
            continue
        normal /= length
        if abs(normal[1]) < .95:
            continue
        inside = np.abs(candidates @ normal - normal @ a) < .012
        if inside.sum() > best.sum():
            best = inside
    if best.sum() < max(200, .5 * len(candidates)):
        return np.eye(3), fallback
    for _ in range(3):
        center = candidates[best].mean(axis=0)
        _, _, vectors = np.linalg.svd(candidates[best] - center, full_matrices=False)
        normal = vectors[-1]
        if normal[1] < 0:
            normal = -normal
        offset = -normal @ center
        best = np.abs(candidates @ normal + offset) < .012
        if best.sum() < 200 or normal[1] < .95:
            return np.eye(3), fallback
    axis = np.cross(normal, [0., 1., 0.])
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rotation = np.eye(3) + cross + cross @ cross / (1 + normal[1])
    return rotation, {"method": "robust_training_feature_plane", "normal_original_upright": normal.tolist(),
                      "offset_m": float(offset), "tilt_degrees": float(np.degrees(np.arccos(np.clip(normal[1], -1, 1)))),
                      "training_features": int(best.sum()), "p95_feature_residual_m": float(np.quantile(np.abs(candidates[best] @ normal + offset), .95)),
                      "world_to_floor_frame_rotation": rotation.tolist()}


def silhouette_masks(cameras, annotations):
    masks = {}
    for name, polygon in annotations["occluders"].items():
        camera = cameras[name]
        mask = np.zeros((camera["width"], camera["height"]), np.uint8)
        vertices = np.asarray(polygon, dtype=np.int32)
        if len(vertices) < 3:
            raise ValueError("An object silhouette needs at least three vertices")
        cv2.fillPoly(mask, [vertices], 1)
        masks[name] = cv2.dilate(mask, np.ones((17, 17), np.uint8))
    return masks


def masked_pixels(u, v, camera, mask):
    x = np.rint(camera["height"] - 1 - v).astype(int).clip(0, camera["height"] - 1)
    y = np.rint(u).astype(int).clip(0, camera["width"] - 1)
    return mask[y, x] > 0


def foreground_faces(xyz, faces, cameras, masks, reference):
    import open3d as o3d
    camera = cameras[reference]
    u, v, visible = project(xyz[faces].mean(axis=1), camera)
    keep = visible & masked_pixels(u, v, camera, masks[reference])
    # Background can project inside the object outline in one photograph.
    # Reject triangles contradicted by another annotated view of that location.
    centers = xyz[faces].mean(axis=1)
    for name, mask in masks.items():
        if name == reference:
            continue
        u, v, visible = project(centers, cameras[name])
        keep &= ~visible | masked_pixels(u, v, cameras[name], mask)
    retained = faces[keep]
    if not len(retained):
        raise ValueError("The annotated foreground retained no reconstructed triangles")
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(xyz), o3d.utility.Vector3iVector(retained))
    clusters, counts, _ = mesh.cluster_connected_triangles()
    retained = retained[np.asarray(counts)[np.asarray(clusters)] >= 100]
    if not len(retained):
        raise ValueError("No connected foreground surface survived filtering")
    return retained


def regular_grid(lower, upper, step):
    if step <= 0 or np.any(np.asarray(upper) <= lower):
        raise ValueError("Invalid patch dimensions")
    a = np.arange(lower[0], upper[0] + step, step)
    b = np.arange(lower[1], upper[1] + step, step)
    if len(a) * len(b) > 150000:
        raise ValueError("Patch exceeds the import vertex budget. Increase grid_step_m.")
    u, v = np.meshgrid(a, b)
    indices = np.arange(u.size).reshape(u.shape)
    top_left = indices[:-1, :-1].ravel()
    top_right = indices[:-1, 1:].ravel()
    bottom_left = indices[1:, :-1].ravel()
    bottom_right = indices[1:, 1:].ravel()
    faces = np.concatenate([np.column_stack([top_left, top_right, bottom_left]), np.column_stack([top_right, bottom_right, bottom_left])])
    return np.column_stack([u.ravel(), v.ravel()]), faces


def sample_colors(root, points, normal, cameras, masks, references, fallback):
    colors = np.tile(np.array(fallback) / 255, (len(points), 1))
    best = np.zeros(len(points))
    in_frame = np.zeros(len(points), bool)
    for name, camera in cameras.items():
        u, v, visible = project(points, camera)
        in_frame |= visible
        if name not in references:
            continue
        if name in masks:
            visible &= ~masked_pixels(u, v, camera, masks[name])
        direction = points - camera["origin"]
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-8)
        incidence = np.abs(direction @ normal)
        center = 1 - .3 * ((u - camera["width"] / 2) / (camera["width"] / 2)) ** 2 - .3 * ((v - camera["height"] / 2) / (camera["height"] / 2)) ** 2
        score = incidence * center
        selected = visible & (score > best)
        photo = cv2.cvtColor(read_image(root / "images" / name), cv2.COLOR_BGR2RGB)
        ui = np.rint(u).astype(int).clip(0, camera["width"] - 1)
        vi = np.rint(v).astype(int).clip(0, camera["height"] - 1)
        colors[selected] = photo[vi[selected], ui[selected]] / 255
        best[selected] = score[selected]
    supported = best > 0
    if supported.any():
        colors[~supported] = np.median(colors[supported], axis=0)
    return colors, supported, in_frame


def compact(xyz, colors, faces):
    used, indices = np.unique(faces, return_inverse=True)
    return xyz[used], colors[used], indices.reshape(-1, 3)


def repair(root, annotations):
    quality = json.loads((root / "model-quality.json").read_text())
    if quality.get("surface_repair"):
        raise ValueError("Choose the original reconstruction as the repair source")
    orientation = quality["orientation"]
    cameras = load_cameras(root, orientation)
    masks = silhouette_masks(cameras, annotations)
    floor, floor_report = floor_from_features(root, orientation)
    frame_rotation, plane_report = fit_floor_frame(training_features(root, orientation), floor)
    floor_report["original_upright_height_m"] = floor
    floor = -plane_report["offset_m"]
    floor_report.update(height_m=floor, height_frame="fitted floor frame", plane_fit=plane_report)
    for camera in cameras.values():
        camera["origin"] = frame_rotation @ camera["origin"]
        camera["rotation"] = camera["rotation"] @ frame_rotation.T
    planes = []
    wall_reports = []
    for wall in annotations["walls"]:
        contacts = [intersect_floor(cameras[row["image"]], pixel, floor) for row in wall["contacts"] for pixel in row["pixels"]]
        normal, offset, residuals = fit_vertical_plane(contacts)
        if residuals.max() > .08:
            raise ValueError("Wall annotations disagree by more than 8 cm. Review their floor contacts.")
        planes.append((normal, offset))
        wall_reports.append({"name": wall["name"], "contact_points": len(contacts), "median_anchor_residual_m": float(np.median(residuals)), "maximum_anchor_residual_m": float(residuals.max())})
    if len(planes) != 2:
        raise ValueError("This guided room repair needs two intersecting walls")
    normal_matrix = np.array([normal[[0, 2]] for normal, _ in planes])
    if np.linalg.cond(normal_matrix) > 20:
        raise ValueError("The annotated wall planes are nearly parallel")
    intersection = np.linalg.solve(normal_matrix, -np.array([offset for _, offset in planes]))
    corner = np.array([intersection[0], floor, intersection[1]])
    data = PlyData.read(root / "mesh.ply")
    xyz = np.column_stack([data["vertex"][key] for key in ["x", "y", "z"]]) @ frame_rotation.T
    colors = np.column_stack([data["vertex"][key] for key in ["red", "green", "blue"]]) / 255
    faces = np.stack(data["face"]["vertex_indices"])
    retained = foreground_faces(xyz, faces, cameras, masks, annotations["foreground_reference"])
    vertices = [xyz]
    rgb = [colors]
    triangles = [retained]
    face_ranges = [{"name": "Printer and helmet", "kind": "retained_inferred_reconstruction", "first_face_index": 0, "face_count": len(retained)}]
    vertex_offset, face_offset = len(xyz), len(retained)
    camera_center = np.mean([camera["origin"] for camera in cameras.values()], axis=0)
    step = annotations["grid_step_m"]
    for index, (normal, plane_offset) in enumerate(planes):
        horizontal = np.array([-normal[2], 0, normal[0]])
        candidates = xyz[np.abs(xyz @ normal + plane_offset) < .5]
        distances = (candidates - corner) @ horizontal
        if np.median(distances) < 0:
            horizontal = -horizontal
            distances = -distances
        width = float(np.quantile(distances, .998))
        height = float(xyz[:, 1].max() - floor)
        grid, patch_faces = regular_grid([0, 0], [width, height], step)
        patch = corner + grid[:, 0, None] * horizontal + grid[:, 1, None] * [0, 1, 0]
        patch_colors, observed, visible = sample_colors(root, patch, normal, cameras, masks, annotations["color_references"], annotations["walls"][index]["fallback_rgb"])
        patch_faces = patch_faces[visible[patch_faces].all(axis=1)]
        desired_normal = normal * np.sign((camera_center - corner) @ normal)
        if np.cross(horizontal, [0, 1, 0]) @ desired_normal < 0:
            patch_faces = patch_faces[:, [0, 2, 1]]
        used = np.unique(patch_faces)
        photo_colored = int(observed[used].sum())
        patch, patch_colors, patch_faces = compact(patch, patch_colors, patch_faces)
        triangles.append(patch_faces + vertex_offset)
        vertices.append(patch)
        rgb.append(patch_colors)
        face_ranges.append({"name": annotations["walls"][index]["name"], "kind": "inferred_planar_surface", "first_face_index": face_offset, "face_count": len(patch_faces), "photo_colored_vertices": photo_colored, "estimated_paint_vertices": len(patch) - photo_colored, "maximum_plane_residual_m": float(np.max(np.abs(patch @ normal + plane_offset)))})
        vertex_offset += len(patch)
        face_offset += len(patch_faces)
    floor_points = xyz[np.abs(xyz[:, 1] - floor) < .13]
    inside = np.ones(len(floor_points), bool)
    for normal, offset in planes:
        room_side = np.sign(cameras[annotations["foreground_reference"]]["origin"] @ normal + offset)
        inside &= (floor_points @ normal + offset) * room_side > -.02
    footprint = floor_points[inside][:, [0, 2]]
    hull = ConvexHull(footprint)
    grid, floor_faces = regular_grid(footprint.min(axis=0), footprint.max(axis=0), step)
    within = (grid @ hull.equations[:, :2].T + hull.equations[:, 2]).max(axis=1) < 0
    patch = np.column_stack([grid[:, 0], np.full(len(grid), floor), grid[:, 1]])
    patch_colors, observed, visible = sample_colors(root, patch, np.array([0, 1, 0]), cameras, masks, annotations["color_references"], [150, 139, 122])
    floor_faces = floor_faces[(within & visible)[floor_faces].all(axis=1)][:, [0, 2, 1]]
    used = np.unique(floor_faces)
    photo_colored = int(observed[used].sum())
    patch, patch_colors, floor_faces = compact(patch, patch_colors, floor_faces)
    vertices.append(patch)
    rgb.append(patch_colors)
    triangles.append(floor_faces + vertex_offset)
    face_ranges.append({"name": "Floor", "kind": "inferred_planar_surface", "first_face_index": face_offset, "face_count": len(floor_faces), "photo_colored_vertices": photo_colored, "estimated_color_vertices": len(patch) - photo_colored, "maximum_plane_residual_m": float(np.max(np.abs(patch[:, 1] - floor)))})
    xyz, rgb, faces = compact(np.concatenate(vertices), np.concatenate(rgb), np.concatenate(triangles))
    if len(xyz) > 150000 or len(faces) > 200000:
        raise ValueError("Repair exceeds BridgeCheck import limits. Increase grid_step_m.")
    provenance = {"notice": NOTICE, "method": "User-reviewed floor-contact and foreground-silhouette annotations, including adjusted automatic suggestions when available. Foreground triangles must agree with every annotated view in which their centers are visible. Floor tilt is estimated from triangulated training features. Walls meet the fitted floor at the derived corner. A rigid frame change preserves scale. Surface extents use the reconstruction and saved camera fields of view.", "floor_fit": floor_report, "wall_fits": wall_reports, "face_index_base": 0, "face_ranges": face_ranges, "annotations": annotations, "remaining_limitations": "The printer and helmet retain partial inferred geometry. Wall-mounted relief is represented by photo color on flat planes. Hidden paint and floor colors use local median estimates."}
    return xyz @ frame_rotation, np.rint(rgb * 255).clip(0, 255).astype(np.uint8), faces, provenance, quality


def export_repair(source, output, annotations, dru_scripts, source_label, product=None):
    manifest = json.loads((source / "scan/manifest.json").read_text())
    source_status = json.loads((source / "status.json").read_text())
    if product or source_status.get("product"):
        from scan_branding import brand_for
        manifest["product"] = brand_for(product or source_status["product"])["id"]
    override = source / "scan-identity.json"
    if override.exists():
        manifest.update(json.loads(override.read_text()))
    validate_annotations(annotations, manifest)
    if output.exists():
        raise FileExistsError("Choose a new output folder to preserve existing models")
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    xyz, rgb, faces, provenance, quality = repair(source, annotations)
    from fuse_photo import write_models
    from photo_stages import package
    output.mkdir(parents=True)
    (output / "scan").mkdir()
    write_json(output / "scan/manifest.json", manifest)
    shutil.copytree(source / "sparse/0_text", output / "sparse/0_text")
    for name in ["photo-hints.json", "dense-view-selection.json"]:
        shutil.copy2(source / name, output / name)
    shutil.copy2(source / "model-metres.ply", output / "reconstruction-before-repair.ply")
    write_models(output, xyz, rgb, faces, NOTICE)
    rotation = np.array(quality["orientation"]["world_to_upright_rotation"])
    center = np.array(quality["orientation"]["center_in_sfm_world"])
    held = json.loads((source / "photo-hints.json").read_text())["held_out"]
    unique = {feature["point_id"]: feature["xyz"] for feature in held}
    distances, _ = cKDTree(xyz @ rotation + center).query(np.array(list(unique.values())))
    quality.update(notice=NOTICE, vertices=len(xyz), faces=len(faces), surface_repair=provenance, held_out_features={"count": len(unique), "median_nearest_vertex_m": float(np.median(distances)), "p95_nearest_vertex_m": float(np.quantile(distances, .95)), "scope": "Same-capture internal photo feature consistency. Plane assumptions and hidden surfaces require independent verification."})
    quality.pop("direct_photo_colored_vertices", None)
    quality["source_fusion"] = quality.pop("fusion")
    provenance["source_result_directory"] = str(source_label)
    provenance["source_reconstruction_seconds"] = json.loads((source / "timing.json").read_text())["total_seconds"]
    provenance["geometry_and_model_export_seconds"] = round(time.perf_counter() - started, 3)
    write_json(output / "model-quality.json", quality)
    write_json(output / "repair-provenance.json", provenance)
    report = json.loads((source / "report.json").read_text())
    report["surface_repair"] = provenance
    report["reconstruction_notice"] = NOTICE
    report["verdict"] = "EXPERIMENTAL guided repaired visualization with inferred planar surfaces"
    report.pop("timing", None)
    write_json(output / "report.json", report)
    package(output, dru_scripts)
    elapsed = round(time.perf_counter() - started, 3)
    write_json(output / "timing.json", {"state": "done", "total_seconds": elapsed, "reused_scan_outputs": True, "operation": "Guided room repair", "source_reconstruction_seconds": provenance["source_reconstruction_seconds"], "notice": NOTICE})
    status = {"job_id": output.name, "state": "done", "operation": "Guided room repair", "scan_id": manifest["id"], "product": manifest.get("product", "dru"), "started": started_at, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"), "derived_from": str(source_label), "elapsed_seconds_including_validation": elapsed}
    from scan_identity import identity
    status["capture"] = identity(manifest)
    source_status = json.loads((source / "status.json").read_text())
    if source_status.get("archived_source"):
        status.update(archived_source=source_status["archived_source"], source_sha256=source_status["source_sha256"])
    write_json(output / "status.json", status)
    if (source_label / "cleanup.json").exists():
        status["storage"] = compact_result(output, Path(status["archived_source"]), status["source_sha256"])
        status["elapsed_seconds_including_validation"] = round(time.perf_counter() - started, 3)
        write_json(output / "status.json", status)
    print(json.dumps({"result": str(output), "repair_seconds": elapsed, "vertices": len(xyz), "faces": len(faces), "notice": NOTICE}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dru-scripts", type=Path, required=True)
    parser.add_argument("--product", choices=["dru", "bridgecheck"])
    args = parser.parse_args()
    source, output = args.source.resolve(), args.out.resolve()
    annotations = json.loads(args.annotations.read_text())
    identity_path = source / "scan/manifest.json"
    identity = json.loads(identity_path.read_text()) if identity_path.exists() else json.loads((source / "status.json").read_text())
    if identity.get("id", identity.get("scan_id")) != annotations["scan_id"]:
        raise ValueError("Annotations belong to a different saved scan")
    with tempfile.TemporaryDirectory(prefix="purposebuilt-repair-") as directory:
        materialized = restore_source(source, Path(directory))
        export_repair(materialized, output, annotations, args.dru_scripts, source, args.product)


if __name__ == "__main__":
    main()
