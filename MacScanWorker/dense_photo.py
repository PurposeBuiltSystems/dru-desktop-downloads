#!/usr/bin/env python3
"""Dense reconstruction with MapAnything from a COLMAP text model. Runs as its own process on purpose:
pycolmap and PyTorch each ship an OpenMP runtime and abort when loaded together.

  python3 dense_photo.py <workspace> [--no-depth-hint]

Reads <workspace>/sparse/0_text/{cameras,images}.txt, <workspace>/images/*.jpg, <workspace>/scan/manifest.json
(+ surface_XXXX.bin for depth hints) and writes dense.laz into the workspace (LAS 1.2, 1 mm, RGB, LAZ-compressed).
Per-pixel predictions from overlapping views duplicate the same surface, so points are deduplicated on a
--voxel grid (default 5 mm) before writing.
The worker uses --predictions-only to keep depth maps for later mesh fusion.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # plain HTTPS download; avoids an optional native module
try:  # macOS framework Python ships without root certificates; torch.hub and huggingface_hub need them
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass


def quat_to_rot(qw, qx, qy, qz):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


def read_text_model(model_dir: Path):
    cameras = {}
    for line in (model_dir / "cameras.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        a = line.split()
        cameras[int(a[0])] = {"model": a[1], "width": int(a[2]), "height": int(a[3]), "params": [float(x) for x in a[4:]]}
    images = []
    lines = [l for l in (model_dir / "images.txt").read_text().splitlines() if not l.startswith("#")]
    for i in range(0, len(lines), 2):
        a = lines[i].split()
        if len(a) < 10:
            continue
        qw, qx, qy, qz, tx, ty, tz = (float(x) for x in a[1:8])
        w2c = np.eye(4); w2c[:3, :3] = quat_to_rot(qw, qx, qy, qz); w2c[:3, 3] = [tx, ty, tz]
        images.append({"camera_id": int(a[8]), "name": a[9], "c2w": np.linalg.inv(w2c)})
    images.sort(key=lambda im: im["name"])
    return cameras, images


def read_surface(path: Path):
    raw = path.read_bytes()
    magic, w, h = struct.unpack(">IHH", raw[:8])
    if magic != 0x44525332:
        raise ValueError("bad surface magic")
    rec = np.frombuffer(raw[8:8 + w * h * 11], dtype=np.dtype([("mm", ">i2"), ("c", "u1"), ("u", ">f4"), ("v", ">f4")]))
    return rec["mm"].reshape(h, w).astype(np.float32) / 1000.0, rec["c"].reshape(h, w).astype(np.float32) / 255.0


def write_ply(path: Path, xyz, rgb):
    from plyfile import PlyData, PlyElement
    arr = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(path))


def voxel_dedup(xyz, rgb, voxel):
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    idx.sort()
    return xyz[idx], rgb[idx]


def write_las(path: Path, xyz, rgb):
    import laspy  # LAZ needs the lazrs backend: pip install laspy lazrs
    header = laspy.LasHeader(point_format=2, version="1.2")
    header.scales = np.array([0.001] * 3); header.offsets = np.floor(xyz.min(axis=0))
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.red, las.green, las.blue = (rgb[:, 0].astype(np.uint16) * 257, rgb[:, 1].astype(np.uint16) * 257, rgb[:, 2].astype(np.uint16) * 257)
    las.write(path)


def backproject_depth(depth, intrinsics, camera_to_world):
    """Place each camera-space Z depth on its calibrated image ray."""
    yy, xx = np.mgrid[:depth.shape[0], :depth.shape[1]]
    rays = np.stack([(xx - intrinsics[0, 2]) / intrinsics[0, 0], (yy - intrinsics[1, 2]) / intrinsics[1, 1], np.ones_like(xx)], axis=2)
    return (rays * depth[..., None]) @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]


def resample_sparse_depth(depth, source_intrinsics, target_intrinsics, shape):
    """Project sparse observations into a resized camera, retaining nearest surfaces."""
    depth = np.asarray(depth)
    source = np.asarray(source_intrinsics, dtype=float)
    target = np.asarray(target_intrinsics, dtype=float)
    height, width = shape
    if depth.ndim != 2 or source.shape != (3, 3) or target.shape != (3, 3):
        raise ValueError("Sparse depth needs a depth image and two camera matrices")
    if height < 1 or width < 1 or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Invalid resized camera")
    if min(source[0, 0], source[1, 1], target[0, 0], target[1, 1]) <= 0:
        raise ValueError("Camera focal lengths must be positive")
    y, x = np.nonzero(np.isfinite(depth) & (depth > 0))
    rays = np.linalg.solve(source, np.stack([x, y, np.ones(len(x))]))
    projected = target @ rays
    pixels = np.rint((projected[:2] / projected[2:3]).T)
    inside = (
        np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    )
    u, v = pixels[inside].astype(np.intp).T
    output = np.full(height * width, np.inf, dtype=np.float32)
    np.minimum.at(output, v * width + u, depth[y[inside], x[inside]])
    output[~np.isfinite(output)] = 0
    return output.reshape(height, width)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("workspace", type=Path)
    ap.add_argument("--no-depth-hint", action="store_true")
    ap.add_argument("--model", default="facebook/map-anything-apache")
    ap.add_argument("--quantized-model", type=Path)
    ap.add_argument("--native-int4", action="store_true")
    ap.add_argument("--predictions-only", action="store_true", help="save depth predictions for mesh fusion and omit the intermediate point cloud")
    ap.add_argument("--chunk", type=int, default=0, help="views per inference (0 = all on CUDA, 8 otherwise)")
    ap.add_argument("--voxel", type=float, default=0.005, help="deduplicate points on this grid (m); 0 keeps every pixel")
    ap.add_argument("--long-side", type=int, default=518)
    ap.add_argument("--max-views", type=int, default=0)
    args = ap.parse_args()
    ws = args.workspace.resolve()
    started = time.perf_counter()
    step = started
    try:
        import torch
        from PIL import Image as PILImage
        from mapanything.models import MapAnything
        from mapanything.utils.image import preprocess_inputs
    except ImportError as e:
        print(f"MapAnything not available: {e}", file=sys.stderr)
        return 2
    stage_seconds = {"imports": time.perf_counter() - step}
    cameras, images = read_text_model(ws / "sparse" / "0_text")
    if args.max_views and len(images) > args.max_views:
        indices = np.rint(np.linspace(0, len(images) - 1, args.max_views)).astype(int)
        images = [images[index] for index in indices]
    manifest = json.loads((ws / "scan" / "manifest.json").read_text())
    by_name = {i["filename"]: i for i in manifest["images"]}
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else ("mps" if torch.backends.mps.is_available() else "cpu")
    if args.native_int4 and (not args.quantized_model or device != "mps"):
        raise ValueError("Native four-bit execution needs a compact model and the Mac GPU")
    selection = {"views": len(images), "filenames": [image["name"] for image in images], "long_side": args.long_side, "device": device, "chunk": args.chunk, "weight_storage": "int4" if args.quantized_model else "float32", "expands_weights_to_float32": bool(args.quantized_model), "native_int4": args.native_int4, "intermediate_point_cloud": not args.predictions_only}
    step = time.perf_counter()
    if args.quantized_model:
        from quantized_weights import load_model
        model = load_model(args.quantized_model, MapAnything, native_int4=args.native_int4)
    else:
        model = MapAnything.from_pretrained(args.model)
    model = model.to(device).eval()
    stage_seconds["model_load"] = time.perf_counter() - step
    selection["native_int4_layers"] = getattr(model, "native_int4_layers", 0)
    selection["native_int4_parameters"] = getattr(model, "native_int4_parameters", 0)
    selection["float32_parameters"] = sum(parameter.numel() for parameter in model.parameters())
    selection["float32_expansion_scope"] = "convolution and small layers" if args.native_int4 else "all packed weights"
    (ws / "dense-view-selection.json").write_text(json.dumps(selection, indent=2))
    views = []
    sparse_hints = {}
    hint_retention = []
    step = time.perf_counter()
    for im in images:
        cam = cameras[im["camera_id"]]
        fx, fy, cx, cy = cam["params"][:4] if cam["model"] == "PINHOLE" else (cam["params"][0], cam["params"][0], cam["params"][1], cam["params"][2])
        view = {"img": torch.from_numpy(np.array(PILImage.open(ws / "images" / im["name"]).convert("RGB"))),
                "intrinsics": torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32),
                "camera_poses": torch.tensor(im["c2w"], dtype=torch.float32), "is_metric_scale": torch.tensor([True])}
        meta = by_name.get(im["name"])
        if not args.no_depth_hint and meta and meta.get("surface"):
            depth, conf = read_surface(ws / "scan" / meta["surface"]["filename"])
            depth = np.where(conf >= 0.5, depth, 0.0).astype(np.float32)
            resized = PILImage.fromarray(depth).resize((meta["width"], meta["height"]), PILImage.NEAREST)
            view["depth_z"] = torch.from_numpy(np.array(resized, dtype=np.float32))
        hint_file = ws / 'sfm_depth' / (im['name'] + '.npy')
        if hint_file.exists():
            sparse_hints[im['name']] = np.load(hint_file)
            view['depth_z'] = torch.from_numpy(sparse_hints[im['name']])
        views.append(view)
    stage_seconds["image_loading"] = time.perf_counter() - step
    # Multi-view attention memory grows with the square of the token count. CUDA has flash attention; MPS/CPU
    # materialise the full matrix (28 views at 518 px = 42 GB), so run in chunks of views and merge them in the
    # metric world frame that the supplied poses define.
    chunk = args.chunk if args.chunk else (len(views) if cuda else 8)
    pts, cols = [], []
    step = time.perf_counter()
    for start in range(0, len(views), chunk):
        batch = views[start:start + chunk]
        processed = preprocess_inputs(batch) if args.long_side == 518 else preprocess_inputs(batch, resize_mode="longest_side", size=args.long_side)
        for index, view in enumerate(processed):
            name = images[start + index]["name"]
            if name in sparse_hints:
                hint = sparse_hints[name]
                projected = resample_sparse_depth(
                    hint, batch[index]["intrinsics"].numpy(),
                    view["intrinsics"][0].numpy(), view["img"].shape[-2:],
                )
                hint_retention.append({
                    "image": name,
                    "input": int(np.count_nonzero(hint)),
                    "nearest_resize": int(torch.count_nonzero(view["depth_z"])),
                    "projected": int(np.count_nonzero(projected)),
                })
                view["depth_z"] = torch.from_numpy(projected)[None]
        for v in processed:
            for k, val in list(v.items()):
                if hasattr(val, "to"):
                    v[k] = val.to(device)
        with torch.no_grad():
            preds = model.infer(processed, memory_efficient_inference=True, minibatch_size=None if cuda else 2,
                                use_amp=cuda, apply_mask=True, mask_edges=True)
        batch_points = 0
        for p_index, p in enumerate(preds):
            mask = p["mask"][0, ..., 0].bool().cpu().numpy() if "mask" in p else np.ones(p["pts3d"].shape[1:3], bool)
            # The network also estimates rays and poses. Keep the photo-aligned camera
            # geometry when backprojecting its per-pixel depth into the scan frame.
            depth = p["depth_z"][0, ..., 0].float().cpu().numpy()
            intrinsics = processed[p_index]["intrinsics"][0].float().cpu().numpy()
            mask &= np.isfinite(depth) & (depth > 0)
            pose = batch[p_index]["camera_poses"].double().numpy()
            name = images[start + p_index]["name"]
            arrays = {"camera_depth": depth, "mask": mask, "intrinsics": intrinsics, "camera_to_world": pose}
            if not args.predictions_only:
                full_xyz = backproject_depth(depth, intrinsics, pose)
                arrays["xyz"] = full_xyz.astype(np.float32)
            np.savez_compressed(ws / (name + ".prediction.npz"), **arrays)
            batch_points += int(np.count_nonzero(mask))
            if args.predictions_only:
                continue
            xyz = full_xyz[mask]
            rgb = p["img_no_norm"][0].float().cpu().numpy()[mask]
            if rgb.size and rgb.max() <= 1.0:
                rgb = rgb * 255.0
            pts.append(xyz.astype(np.float32)); cols.append(rgb.clip(0, 255).astype(np.uint8))
        print(f"  views {start + 1}-{start + len(batch)}: {batch_points} points", flush=True)
    stage_seconds["view_processing"] = time.perf_counter() - step
    step = time.perf_counter()
    if not args.predictions_only:
        xyz = np.concatenate(pts); rgb = np.concatenate(cols)
        finite = np.isfinite(xyz).all(axis=1); xyz, rgb = xyz[finite], rgb[finite]
        raw = len(xyz)
        if args.voxel > 0:
            xyz, rgb = voxel_dedup(xyz, rgb, args.voxel)
        write_las(ws / "dense.laz", xyz, rgb)
        print(f"dense (MapAnything on {device}): {raw} points from {len(views)} views -> {len(xyz)} after {args.voxel*1000:.0f} mm dedup")
    stage_seconds["point_cloud_export"] = time.perf_counter() - step
    stage_seconds["total"] = time.perf_counter() - started
    selection["timing_seconds"] = {name: round(seconds, 3) for name, seconds in stage_seconds.items()}
    selection["sparse_hint_retention"] = hint_retention
    (ws / "dense-view-selection.json").write_text(json.dumps(selection, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
