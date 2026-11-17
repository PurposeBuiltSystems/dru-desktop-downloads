#!/usr/bin/env python3
"""Rebuild metric geometry from a DRU Android scan ZIP with open-source tools.

Pipeline (all BSD/Apache/MIT tooling, runs on a laptop; no cloud service):
  1. Unpack the DRU scan ZIP (manifest.json, frame_XXXX.jpg, cloud.ply, surface_XXXX.bin).
  2. Build a COLMAP database: PINHOLE intrinsics from ARCore, SIFT features, sequential matching.
  3. Run COLMAP incremental SfM with the ARCore camera positions as pose priors
     (COLMAP >= 3.11 pose-prior mapper via pycolmap) so the result keeps ARCore's metric scale
     while the image evidence corrects ARCore drift.
  4. Compare the refined trajectory against ARCore (scale factor, residuals) and write a report.
  5. Export: COLMAP sparse model (for OpenMVS, MapAnything, gsplat/nerfstudio), LAS + PLY clouds.
  6. Optional dense step (--dense mapanything | openmvs) followed by Poisson meshing (pymeshlab) -> OBJ.

Usage:
  python3 scripts/scan_recon/dru_recon.py dru-scan-<id>.zip --out /path/to/workspace [--dense none]

Install (Python 3.10+): pip install pycolmap laspy pymeshlab numpy
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import struct
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

try:
    import pycolmap
except ImportError:  # pragma: no cover
    sys.exit("pycolmap is required: pip install pycolmap")

IMAGE_RE = re.compile(r"frame_[0-9]{4}\.jpg")
SURFACE_RE = re.compile(r"surface_[0-9]{4}\.bin")
ALLOWED = {"manifest.json", "cloud.ply", "repair-annotations.json"}
GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenGL camera (y up, -z forward) -> OpenCV (y down, +z forward)


# ----------------------------------------------------------------------------- scan loading
def unpack(src: Path, scan_dir: Path) -> None:
    scan_dir.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        for f in src.iterdir():
            if f.name in ALLOWED or IMAGE_RE.fullmatch(f.name) or SURFACE_RE.fullmatch(f.name):
                shutil.copy2(f, scan_dir / f.name)
        return
    with zipfile.ZipFile(src) as z:
        for info in z.infolist():
            name = info.filename
            if info.is_dir() or "/" in name or "\\" in name:
                raise SystemExit(f"Unexpected ZIP entry: {name}")
            if not (name in ALLOWED or IMAGE_RE.fullmatch(name) or SURFACE_RE.fullmatch(name)):
                raise SystemExit(f"Unsupported ZIP entry: {name}")
            if name == "repair-annotations.json" and info.file_size > 262144:
                raise SystemExit("Repair markings exceed the size limit")
            if info.file_size > 64_000_000:
                raise SystemExit(f"ZIP entry too large: {name}")
            with z.open(info) as fin, open(scan_dir / name, "wb") as fout:
                shutil.copyfileobj(fin, fout)


def load_manifest(scan_dir: Path) -> dict:
    manifest = json.loads((scan_dir / "manifest.json").read_text())
    if manifest.get("schemaVersion") != 3:
        print(f"warning: manifest schemaVersion {manifest.get('schemaVersion')} (expected 3)")
    for img in manifest["images"]:
        for key in ("filename", "cameraToWorld", "width", "height", "fx", "fy", "cx", "cy"):
            if key not in img:
                raise SystemExit(f"manifest image missing {key}")
        if not IMAGE_RE.fullmatch(img["filename"]) or not (scan_dir / img["filename"]).is_file():
            raise SystemExit(f"missing image {img['filename']}")
    return manifest


def camera_to_world_cv(img: dict) -> np.ndarray:
    """ARCore cameraToWorld is a column-major 4x4 (OpenGL camera). Return row-major c2w, OpenCV camera."""
    m = np.array(img["cameraToWorld"], dtype=np.float64).reshape(4, 4).T
    return m @ GL_TO_CV


def read_cloud(path: Path):
    xyz, conf, src = [], [], []
    with open(path) as f:
        for line in f:
            if line.strip() == "end_header":
                break
        for line in f:
            a = line.split()
            if len(a) < 5:
                continue
            xyz.append((float(a[0]), float(a[1]), float(a[2])))
            conf.append(float(a[3]))
            src.append(int(a[4]))
    return np.array(xyz, dtype=np.float64), np.array(conf), np.array(src)


def read_surface(path: Path):
    """surface_XXXX.bin: magic 'DRS2', u16 w, u16 h, then w*h records of (i16 depth mm, u8 confidence, f32 u, f32 v), big-endian."""
    raw = path.read_bytes()
    magic, w, h = struct.unpack(">IHH", raw[:8])
    if magic != 0x44525332:
        raise ValueError("bad surface magic")
    rec = np.frombuffer(raw[8:8 + w * h * 11], dtype=np.dtype([("mm", ">i2"), ("c", "u1"), ("u", ">f4"), ("v", ">f4")]))
    depth = rec["mm"].reshape(h, w).astype(np.float32) / 1000.0
    conf = rec["c"].reshape(h, w).astype(np.float32) / 255.0
    return depth, conf


# ----------------------------------------------------------------------------- COLMAP
def build_database(ws: Path, scan_dir: Path, manifest: dict, prior_sigma_m: float, max_features: int) -> Path:
    image_dir = ws / "images"
    image_dir.mkdir(exist_ok=True)
    for img in manifest["images"]:
        dst = image_dir / img["filename"]
        if not dst.exists():
            shutil.copy2(scan_dir / img["filename"], dst)
    db_path = ws / "database.db"
    if db_path.exists():
        db_path.unlink()
    imgs = manifest["images"]
    fx = float(np.median([i["fx"] for i in imgs])); fy = float(np.median([i["fy"] for i in imgs]))
    cx = float(np.median([i["cx"] for i in imgs])); cy = float(np.median([i["cy"] for i in imgs]))
    widths = {i["width"] for i in imgs}; heights = {i["height"] for i in imgs}
    if len(widths) != 1 or len(heights) != 1:
        raise SystemExit("mixed image sizes are not supported")
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "PINHOLE"
    reader.camera_params = f"{fx},{fy},{cx},{cy}"
    extraction = pycolmap.FeatureExtractionOptions()
    sift = getattr(extraction, "sift", None)
    if sift is not None:
        sift.max_num_features = max_features
        if hasattr(sift, "estimate_affine_shape"):
            sift.estimate_affine_shape = True
        if hasattr(sift, "domain_size_pooling"):
            sift.domain_size_pooling = True
    pycolmap.extract_features(db_path, image_dir, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader, extraction_options=extraction, device=pycolmap.Device.cpu)
    # Pose priors: ARCore camera positions in the session frame (metres).
    by_name = {i["filename"]: i for i in imgs}
    db = pycolmap.Database.open(db_path)
    try:
        cov = np.eye(3) * prior_sigma_m ** 2
        n = 0
        for image in db.read_all_images():
            img = by_name.get(image.name)
            if img is None:
                continue
            prior = pycolmap.PosePrior()
            prior.position = camera_to_world_cv(img)[:3, 3]
            prior.position_covariance = cov
            prior.coordinate_system = pycolmap.PosePriorCoordinateSystem.CARTESIAN
            prior.corr_data_id = image.data_id
            db.write_pose_prior(prior)
            n += 1
    finally:
        db.close()
    print(f"database: {n} images with pose priors (sigma {prior_sigma_m} m), features <= {max_features}/image")
    return db_path


def capture_quality(db_path: Path, image_dir: Path) -> dict:
    """Why a scan reconstructs badly is almost always visible here: few features (blank walls, low light,
    blur) or a scene that moved. Reported so the field crew gets a reason, not just a bad mesh."""
    from PIL import Image as PILImage
    db = pycolmap.Database.open(db_path)
    try:
        images = db.read_all_images()
        keypoints = {im.name: db.num_keypoints_for_image(im.image_id) for im in images}
    finally:
        db.close()
    lum, sharp = {}, {}
    for name in keypoints:
        g = np.asarray(PILImage.open(image_dir / name).convert("L"), dtype=np.float32)
        lum[name] = float(g.mean()); sharp[name] = float(np.var(np.diff(g, axis=1)))
    kp = np.array(list(keypoints.values())); lu = np.array(list(lum.values())); sh = np.array(list(sharp.values()))
    size = PILImage.open(image_dir / next(iter(keypoints))).size
    flags = []
    if np.median(kp) < 1000: flags.append("few features: textureless or dark scene, or blurred frames")
    if np.median(lu) < 60: flags.append("low light")
    if size[0] < 1920: flags.append(f"low resolution keyframes {size[0]}x{size[1]}")
    weak = [n for n, k in keypoints.items() if k < 300]
    if weak: flags.append(f"{len(weak)} frames with <300 features")
    return {"image_size": list(size), "keypoints_per_image": {"median": int(np.median(kp)), "min": int(kp.min())},
            "luminance": {"median": round(float(np.median(lu)), 1), "min": round(float(lu.min()), 1)},
            "sharpness_var_dx": {"median": round(float(np.median(sh)), 1), "min": round(float(sh.min()), 1)},
            "weak_frames": weak, "flags": flags}


def match(db_path: Path, overlap: int) -> None:
    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = overlap
    pairing.quadratic_overlap = True
    pairing.loop_detection = False
    matching = pycolmap.FeatureMatchingOptions()
    if hasattr(matching, "guided_matching"):
        matching.guided_matching = True
    pycolmap.match_sequential(db_path, matching_options=matching, pairing_options=pairing, device=pycolmap.Device.cpu)
    # Exhaustive on top of sequential is cheap for <= 150 images and recovers loop closures (walking back along a trench).
    db = pycolmap.Database.open(db_path)
    try:
        n = db.num_images()
    finally:
        db.close()
    if n <= 150:
        pycolmap.match_exhaustive(db_path, matching_options=matching, device=pycolmap.Device.cpu)


def map_with_priors(db_path: Path, image_dir: Path, sparse_dir: Path, loss_scale: float):
    sparse_dir.mkdir(exist_ok=True)
    opts = pycolmap.IncrementalPipelineOptions()
    opts.use_prior_position = True
    opts.use_robust_loss_on_prior_position = True
    opts.prior_position_loss_scale = loss_scale
    opts.ba_refine_focal_length = True
    opts.ba_refine_principal_point = False
    opts.ba_refine_extra_params = False
    opts.min_num_matches = 15
    recs = pycolmap.incremental_mapping(db_path, image_dir, sparse_dir, opts)
    if not recs:
        raise SystemExit("COLMAP registered no images")
    best_id, best = max(recs.items(), key=lambda kv: kv[1].num_reg_images())
    print(f"mapping: {len(recs)} model(s); best model {best_id} has {best.num_reg_images()} images, {best.num_points3D()} points")
    return best


def umeyama(src: np.ndarray, dst: np.ndarray):
    """Least-squares similarity transform dst ~ s * R @ src + t (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (xs ** 2).sum() / len(src)
    scale = float(np.trace(np.diag(D) @ S) / var_s) if var_s > 0 else 1.0
    t = mu_d - scale * R @ mu_s
    return scale, R, t


def compare_to_arcore(rec, manifest: dict) -> dict:
    by_name = {i["filename"]: i for i in manifest["images"]}
    names, arcore, sfm = [], [], []
    for image in rec.images.values():
        if not image.has_pose:
            continue
        img = by_name.get(image.name)
        if img is None:
            continue
        names.append(image.name)
        arcore.append(camera_to_world_cv(img)[:3, 3])
        sfm.append(np.asarray(image.projection_center()))
    arcore = np.array(arcore); sfm = np.array(sfm)
    out = {"registered": len(names), "total": len(manifest["images"])}
    if len(names) >= 3:
        direct = np.linalg.norm(sfm - arcore, axis=1)
        out["direct_residual_m"] = {"median": float(np.median(direct)), "max": float(direct.max()), "rms": float(np.sqrt(np.mean(direct ** 2)))}
        # Independent similarity fit (Umeyama): if the image evidence disagreed with ARCore's scale,
        # this is where it shows up. 1.000 means the priors and the photos agree on scale.
        scale, R, t = umeyama(sfm, arcore)
        aligned = (scale * (R @ sfm.T)).T + t
        r = np.linalg.norm(aligned - arcore, axis=1)
        out["sim3_scale_sfm_to_arcore"] = float(scale)
        out["sim3_residual_m"] = {"median": float(np.median(r)), "max": float(r.max()), "rms": float(np.sqrt(np.mean(r ** 2)))}
        path = np.sum(np.linalg.norm(np.diff(arcore, axis=0), axis=1))
        out["arcore_path_length_m"] = float(path)
    out["mean_reprojection_px"] = float(rec.compute_mean_reprojection_error())
    out["mean_track_length"] = float(rec.compute_mean_track_length())
    out["points3D"] = int(rec.num_points3D())
    cams = list(rec.cameras.values())
    if cams:
        out["refined_camera_params"] = [float(p) for p in cams[0].params]
    return out


# ----------------------------------------------------------------------------- exports
def write_las(path: Path, xyz: np.ndarray, rgb: np.ndarray | None = None, classification: np.ndarray | None = None) -> None:
    """LAS 1.2 point format 2 at 1 mm; a .laz path is LAZ-compressed (needs the lazrs package, 5-10x smaller)."""
    try:
        import laspy
    except ImportError:
        print("laspy not installed; skipping", path.name)
        return
    if len(xyz) == 0:
        return
    header = laspy.LasHeader(point_format=2, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.floor(xyz.min(axis=0))
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if rgb is not None:
        las.red, las.green, las.blue = (rgb[:, 0].astype(np.uint16) * 257, rgb[:, 1].astype(np.uint16) * 257, rgb[:, 2].astype(np.uint16) * 257)
    if classification is not None:
        las.classification = classification.astype(np.uint8)
    las.write(path)


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray | None = None) -> None:
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\n")
        if rgb is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(len(xyz)):
            if rgb is not None:
                f.write(f"{xyz[i,0]:.4f} {xyz[i,1]:.4f} {xyz[i,2]:.4f} {int(rgb[i,0])} {int(rgb[i,1])} {int(rgb[i,2])}\n")
            else:
                f.write(f"{xyz[i,0]:.4f} {xyz[i,1]:.4f} {xyz[i,2]:.4f}\n")


def sparse_points(rec):
    pts = rec.points3D
    xyz = np.array([p.xyz for p in pts.values()], dtype=np.float64)
    rgb = np.array([p.color for p in pts.values()], dtype=np.uint8) if len(pts) else None
    return xyz, rgb


def poisson_mesh(cloud: Path, out_dir: Path, obj: bool) -> bool:
    """Screened Poisson via pymeshlab in a separate process (OpenMP runtime clash with pycolmap).
    PoissonRecon can abort on some clouds, so a small ladder of voxel/depth settings is tried."""
    import subprocess
    script = Path(__file__).resolve().parent / "mesh_poisson.py"
    for voxel, depth in ((0.02, 8), (0.01, 9), (0.03, 8), (0.02, 7)):
        for stale in ("mesh.obj", "mesh.ply"):
            (out_dir / stale).unlink(missing_ok=True)
        cmd = [sys.executable, "-P", str(script), str(cloud), str(out_dir), "--voxel", str(voxel), "--depth", str(depth)]
        if obj:
            cmd.append("--obj")
        rc = subprocess.run(cmd).returncode
        if rc == 0 and (out_dir / "mesh.ply").exists():
            return True
        print(f"mesh attempt failed (voxel {voxel} m, depth {depth}, exit {rc}); retrying")
    print("mesh step failed")
    return False


# ----------------------------------------------------------------------------- dense
def dense_mapanything(ws: Path, use_depth: bool) -> Path | None:
    """MapAnything (Apache-2.0 weights) in a separate process: pycolmap and PyTorch each bundle an OpenMP
    runtime and abort when loaded into the same process. See dense_mapanything.py."""
    import subprocess
    cmd = [sys.executable, "-P", str(Path(__file__).resolve().parent / "dense_mapanything.py"), str(ws)]
    if not use_depth:
        cmd.append("--no-depth-hint")
    rc = subprocess.run(cmd).returncode
    out = ws / "dense.laz"
    if rc != 0 or not out.exists():
        print(f"dense step failed (exit {rc})")
        return None
    return out


def package_deliverable(ws: Path, budget_mb: float) -> bool:
    """Email-size bundle (see package_deliverable.py); separate process for the same OpenMP reason as the mesh."""
    import subprocess
    script = Path(__file__).resolve().parent / "package_deliverable.py"
    rc = subprocess.run([sys.executable, "-P", str(script), str(ws), "--budget-mb", str(budget_mb)]).returncode
    if rc != 0 or not (ws / "package.json").exists():
        print(f"package step failed (exit {rc})")
        return False
    return True


def dense_openmvs(ws: Path, sparse_dir: Path) -> Path | None:
    import subprocess
    if not shutil.which("DensifyPointCloud") or not shutil.which("InterfaceCOLMAP"):
        print("OpenMVS binaries (InterfaceCOLMAP, DensifyPointCloud, ReconstructMesh, TextureMesh) not on PATH. Skipping dense step.")
        return None
    dense = ws / "dense"
    pycolmap.undistort_images(dense, sparse_dir, ws / "images")
    mvs = ws / "mvs"; mvs.mkdir(exist_ok=True)
    subprocess.run(["InterfaceCOLMAP", "-i", str(dense), "-o", str(mvs / "scene.mvs"), "--image-folder", str(dense / "images")], check=True)
    subprocess.run(["DensifyPointCloud", str(mvs / "scene.mvs"), "-w", str(mvs)], check=True)
    subprocess.run(["ReconstructMesh", str(mvs / "scene_dense.mvs"), "-w", str(mvs)], check=True)
    subprocess.run(["TextureMesh", str(mvs / "scene_dense_mesh.mvs"), "-w", str(mvs), "--export-type", "obj"], check=True)
    return mvs / "scene_dense.ply"


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan", type=Path, help="DRU scan ZIP (dru-scan-<id>.zip) or an unpacked scan folder")
    ap.add_argument("--out", type=Path, required=True, help="workspace directory (created)")
    ap.add_argument("--dense", choices=["none", "mapanything", "openmvs"], default="none")
    ap.add_argument("--prior-sigma", type=float, default=0.03, help="ARCore position prior std-dev in metres (default 0.03)")
    ap.add_argument("--prior-loss-scale", type=float, default=7.815, help="COLMAP prior position loss scale")
    ap.add_argument("--overlap", type=int, default=12, help="sequential matching overlap")
    ap.add_argument("--max-features", type=int, default=8192)
    ap.add_argument("--no-depth-hint", action="store_true", help="do not feed ARCore depth to MapAnything")
    ap.add_argument("--mesh", action="store_true", help="Poisson-mesh the dense cloud to OBJ")
    ap.add_argument("--obj", action="store_true", help="also write mesh.obj (text, ~4x the binary PLY)")
    ap.add_argument("--keep-db", action="store_true", help="keep database.db after a successful run (features/matches, ~20 MB)")
    ap.add_argument("--package-mb", type=float, default=18.0, help="build the email deliverable (measuring HTML, OBJ in feet, LAZ, photos, README) to fit this many MB; 0 disables")
    ap.add_argument("--resume", action="store_true", help="reuse an existing sparse model in the workspace instead of re-running SfM")
    args = ap.parse_args()

    t0 = time.time()
    ws = args.out.resolve(); ws.mkdir(parents=True, exist_ok=True)
    scan_dir = ws / "scan"
    unpack(args.scan.resolve(), scan_dir)
    manifest = load_manifest(scan_dir)
    print(f"scan {manifest.get('id')}: {len(manifest['images'])} images, depthSupported={manifest.get('depthSupported')}, "
          f"{manifest.get('points')} cloud points ({manifest.get('depthPoints')} depth)")

    report = {"scan_id": manifest.get("id"), "images": len(manifest["images"]), "createdAt": manifest.get("createdAt"),
              "coordinateSystem": manifest.get("coordinateSystem"), "steps": {}}

    # ARCore depth cloud as a first product (metric, in the same local frame).
    xyz, conf, src = read_cloud(scan_dir / "cloud.ply")
    write_las(ws / "arcore_cloud.laz", xyz, classification=np.where(src == 1, 1, 0))
    write_ply(ws / "arcore_cloud.ply", xyz)
    report["arcore_cloud_points"] = int(len(xyz))

    sparse_dir = ws / "sparse"; model_dir = sparse_dir / "0"
    db_path = ws / "database.db"
    if args.resume and (model_dir / "images.bin").exists():
        rec = pycolmap.Reconstruction(model_dir)
        print(f"resume: loaded sparse model with {rec.num_reg_images()} images, {rec.num_points3D()} points")
        text_dir = sparse_dir / "0_text"; text_dir.mkdir(exist_ok=True); rec.write_text(text_dir)
        if db_path.exists():
            report["capture_quality"] = capture_quality(db_path, ws / "images")
        elif (ws / "report.json").exists():  # database.db is deleted after an OK run; keep the earlier grading
            report["capture_quality"] = json.loads((ws / "report.json").read_text()).get("capture_quality", {})
    else:
        t = time.time()
        db_path = build_database(ws, scan_dir, manifest, args.prior_sigma, args.max_features)
        report["steps"]["features_s"] = round(time.time() - t, 1)
        report["capture_quality"] = capture_quality(db_path, ws / "images")
        if report["capture_quality"]["flags"]:
            print("capture quality flags:", "; ".join(report["capture_quality"]["flags"]))
        t = time.time(); match(db_path, args.overlap); report["steps"]["matching_s"] = round(time.time() - t, 1)
        t = time.time()
        rec = map_with_priors(db_path, ws / "images", sparse_dir, args.prior_loss_scale)
        report["steps"]["mapping_s"] = round(time.time() - t, 1)
        model_dir.mkdir(exist_ok=True)
        rec.write(model_dir)
        text_dir = sparse_dir / "0_text"; text_dir.mkdir(exist_ok=True)
        rec.write_text(text_dir)
    sxyz, srgb = sparse_points(rec)
    if len(sxyz):
        write_ply(ws / "sfm_sparse.ply", sxyz, srgb)
        write_las(ws / "sfm_sparse.laz", sxyz, srgb)
    report["sfm"] = compare_to_arcore(rec, manifest)
    print("sfm vs ARCore:", json.dumps(report["sfm"], indent=1))

    dense_cloud = None
    if args.dense == "mapanything" and args.resume and (ws / "dense.laz").exists():
        dense_cloud = ws / "dense.laz"; print("resume: reusing dense.laz")
    elif args.dense == "mapanything":
        t = time.time(); dense_cloud = dense_mapanything(ws, not args.no_depth_hint)
        report["steps"]["dense_s"] = round(time.time() - t, 1)
    elif args.dense == "openmvs":
        t = time.time(); dense_cloud = dense_openmvs(ws, model_dir); report["steps"]["dense_s"] = round(time.time() - t, 1)
    report["dense_cloud"] = str(dense_cloud) if dense_cloud else None
    if dense_cloud and args.mesh:
        t = time.time()
        if poisson_mesh(dense_cloud, ws, args.obj):
            report["mesh"] = str(ws / "mesh.ply"); report["steps"]["mesh_s"] = round(time.time() - t, 1)

    sfm = report["sfm"]
    if sfm["registered"] < 0.8 * sfm["total"]:
        verdict = "FAILED: most frames did not register (see capture_quality.flags)"
    elif abs(sfm.get("sim3_scale_sfm_to_arcore", 1.0) - 1.0) > 0.05 or sfm.get("direct_residual_m", {}).get("median", 0) > 0.05:
        verdict = "WEAK: photos and ARCore disagree on scale or path; geometry is held by ARCore priors only"
    else:
        verdict = "OK: photos and ARCore agree; sparse model is metric and ready for dense reconstruction"
    report["verdict"] = verdict
    print(verdict)
    # The COLMAP database (features + matches, ~20 MB) is only needed to rerun matching/mapping. Keep it when
    # something went wrong so the scan can be debugged, drop it otherwise.
    if db_path.exists() and not args.keep_db and verdict.startswith("OK"):
        db_path.unlink()
    report["steps"]["total_s"] = round(time.time() - t0, 1)
    report["outputs"] = sorted(p.name for p in ws.iterdir() if p.is_file())
    (ws / "report.json").write_text(json.dumps(report, indent=2))
    # The email package reads report.json, so it runs after the report is on disk and then updates it.
    if report.get("mesh") and args.package_mb > 0:
        t = time.time()
        if package_deliverable(ws, args.package_mb):
            report["package"] = json.loads((ws / "package.json").read_text())
            report["steps"]["package_s"] = round(time.time() - t, 1)
            report["steps"]["total_s"] = round(time.time() - t0, 1)
            report["outputs"] = sorted(p.name for p in ws.iterdir() if p.is_file())
            (ws / "report.json").write_text(json.dumps(report, indent=2))
    print(f"done in {report['steps']['total_s']} s -> {ws}")


if __name__ == "__main__":
    main()
