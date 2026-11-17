"""Suggest reviewable floor-contact lines from reconstructed vertical surfaces."""
import json
import tempfile
from pathlib import Path
import hashlib

import numpy as np
from plyfile import PlyData

from result_storage import restore_source
from repair_room import load_cameras, project, training_features, floor_from_features, fit_floor_frame, intersect_floor, fit_vertical_plane


def vertical_planes(xyz, floor):
    """Find two substantial nonparallel vertical surfaces using deterministic RANSAC."""
    points = xyz[(xyz[:, 1] > floor + .25) & (xyz[:, 1] < floor + 3.5)]
    if len(points) < 200:
        return []
    rng = np.random.default_rng(86)
    points = points[rng.choice(len(points), min(8000, len(points)), replace=False)]
    planes = []
    for _ in range(2):
        best = None
        for _ in range(400):
            pair = points[rng.choice(len(points), 2, replace=False)][:, [0, 2]]
            tangent = pair[1] - pair[0]
            length = np.linalg.norm(tangent)
            if length < .35:
                continue
            normal = np.array([-tangent[1], tangent[0]]) / length
            if planes and abs(normal @ planes[0][0]) > .85:
                continue
            offset = -normal @ pair[0]
            mask = np.abs(points[:, [0, 2]] @ normal + offset) < .035
            support = points[mask]
            if len(support) < max(100, len(points) * .08):
                continue
            extent = np.quantile(support[:, [0, 2]] @ (tangent / length), [.03, .97])
            height = np.quantile(support[:, 1], .95) - np.quantile(support[:, 1], .05)
            if extent[1] - extent[0] < .6 or height < .5:
                continue
            score = len(support)
            if best is None or score > best[0]:
                best = (score, normal, offset, mask)
        if best is None:
            break
        _, normal, offset, mask = best
        support = points[mask]
        center = support[:, [0, 2]].mean(axis=0)
        _, _, vectors = np.linalg.svd(support[:, [0, 2]] - center, full_matrices=False)
        normal = vectors[-1]
        offset = -normal @ center
        planes.append((normal, offset, support))
        points = points[~mask]
        if len(points) < 200:
            break
    return planes


def propose(root, manifest, manual=None):
    quality = json.loads((root / "model-quality.json").read_text())
    orientation = quality["orientation"]
    cameras = load_cameras(root, orientation)
    floor, _ = floor_from_features(root, orientation)
    rotation, report = fit_floor_frame(training_features(root, orientation), floor)
    floor = -report["offset_m"]
    for camera in cameras.values():
        camera["origin"] = rotation @ camera["origin"]
        camera["rotation"] = camera["rotation"] @ rotation.T
    mesh = PlyData.read(root / "mesh.ply")
    xyz = np.column_stack([mesh["vertex"][axis] for axis in ["x", "y", "z"]]) @ rotation.T
    planes = vertical_planes(xyz, floor)
    if manual and len(planes) == 2:
        contacts = manual.get("walls", [{}])[0].get("contacts", [])
        if sum(len(row["pixels"]) for row in contacts) >= 2:
            points = [intersect_floor(cameras[row["image"]], pixel, floor) for row in contacts for pixel in row["pixels"]]
            normal, offset, _ = fit_vertical_plane(points)
            scores = [abs(normal[[0, 2]] @ plane[0]) for plane in planes]
            if scores[1] > scores[0]:
                planes.reverse()
    walls = []
    for index in range(2):
        contacts = []
        if index < len(planes):
            normal, offset, support = planes[index]
            tangent = np.array([-normal[1], normal[0]])
            extent = np.quantile(support[:, [0, 2]] @ tangent, [.05, .95])
            horizontal = -offset * normal + np.linspace(*extent, 80)[:, None] * tangent
            edge = np.column_stack([horizontal[:, 0], np.full(80, floor), horizontal[:, 1]])
            candidates = []
            for name, camera in cameras.items():
                u, v, visible = project(edge, camera)
                pixels = np.column_stack([camera["height"] - 1 - v[visible], u[visible]])
                if len(pixels) < 10:
                    continue
                span = np.linalg.norm(pixels[-1] - pixels[0]) / camera["height"]
                if span < .15:
                    continue
                candidates.append((span, name, np.rint(pixels[[0, -1]]).astype(int).tolist()))
            for _, name, pixels in sorted(candidates, reverse=True)[:2]:
                contacts.append({"image": name, "pixels": pixels})
        walls.append({"name": ["First wall", "Second wall"][index], "contacts": contacts,
                      "fallback_rgb": [202, 205, 198]})
    occluders = {}
    reference = ""
    return {"scan_id": manifest["id"], "description": "Automatically suggested floor contacts, reviewed and adjusted by the user. Upright photo pixels, clockwise rotation.",
            "grid_step_m": .015, "foreground_reference": reference, "color_references": [reference] if reference else [], "occluders": occluders, "walls": walls}


def suggestions(source, manifest, manual=None):
    """Read a disposable suggestion cache, preserving capture and manual annotations."""
    key = hashlib.sha256(json.dumps(manual, sort_keys=True).encode()).hexdigest()[:12]
    cache = source / f"repair-suggestions-v3-{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    with tempfile.TemporaryDirectory(prefix="dru-suggest-") as directory:
        result = propose(restore_source(source, Path(directory)), manifest, manual)
    # A concurrent reader sees either a complete cache or regenerates suggestions.
    from reconstruct import write_json
    import uuid
    temporary = cache.with_suffix("." + uuid.uuid4().hex + ".tmp")
    write_json(temporary, result)
    temporary.replace(cache)
    return result
