#!/usr/bin/env python3
"""Fuse photo-constrained depths and export an upright colored estimated mesh."""
import argparse
import json
from pathlib import Path
import cv2
from scan_image_io import read_image
import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from dense_photo import backproject_depth


def load_views(root, raw_k):
    views = []
    depth_directory = root / "refined"
    stereo_report = root / "stereo-refinement.json"
    if stereo_report.exists():
        stereo = json.loads(stereo_report.read_text())
        if stereo["state"] != "done":
            raise ValueError("Photo-pair refinement is incomplete")
        depth_directory = root / "stereo-refined"
        expected = {view["image"] + ".prediction.npz" for view in stereo["views"]}
        actual = {path.name for path in depth_directory.glob("*.prediction.npz")}
        if actual != expected:
            raise ValueError("Photo-pair depth files disagree with the completion report")
    for path in sorted(depth_directory.glob("*.prediction.npz")):
        data = np.load(path)
        pose, k, mask = data["camera_to_world"], data["intrinsics"], data["mask"]
        xyz = data["xyz"] if "xyz" in data else backproject_depth(data["camera_depth"], k, pose).astype(np.float32)
        depth = ((xyz - pose[:3, 3]) @ pose[:3, :3][:, 2]).astype(np.float32)
        depth[~mask] = 0
        depth = cv2.bilateralFilter(depth, 5, .025, 3)
        depth[~mask] = 0
        yy, xx = np.mgrid[:mask.shape[0], :mask.shape[1]].astype(np.float32)
        u = (xx - k[0, 2]) / k[0, 0] * raw_k[0, 0] + raw_k[0, 2]
        v = (yy - k[1, 2]) / k[1, 1] * raw_k[1, 1] + raw_k[1, 2]
        name = path.name.removesuffix(".prediction.npz")
        photo = cv2.cvtColor(read_image(root / "images" / name), cv2.COLOR_BGR2RGB)
        color = cv2.remap(photo, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR)
        views.append({"name": name, "depth": depth, "mask": mask, "color": color, "k": k, "pose": pose, "xyz": xyz})
    if len(views) < 2:
        raise ValueError("At least two reconstructed photo views are required")
    return views


def fuse(views):
    samples = np.concatenate([view["xyz"][view["mask"]][::100] for view in views])
    if len(samples) < 100:
        raise ValueError("Insufficient depth support")
    lower, upper = np.quantile(samples, [.001, .999], axis=0)
    origin = lower - .15
    length = float((upper - lower).max() + .3)
    resolution = min(384, int(np.ceil(length / .012)))
    volume = o3d.pipelines.integration.UniformTSDFVolume(length=length, resolution=resolution, sdf_trunc=.06, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8, origin=origin)
    for view in views:
        depth, color, k = view["depth"], view["color"], view["k"]
        height, width = depth.shape
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(np.ascontiguousarray(color)), o3d.geometry.Image(np.ascontiguousarray(depth)), depth_scale=1., depth_trunc=5., convert_rgb_to_intensity=False)
        volume.integrate(rgbd, o3d.camera.PinholeCameraIntrinsic(width, height, k[0, 0], k[1, 1], k[0, 2], k[1, 2]), np.linalg.inv(view["pose"]))
    mesh = volume.extract_triangle_mesh()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    if not mesh.has_triangles():
        raise ValueError("Depth fusion produced no surface")
    clusters, counts, _ = mesh.cluster_connected_triangles()
    mesh.remove_triangles_by_mask(np.asarray(counts)[np.asarray(clusters)] < 100)
    mesh.remove_unreferenced_vertices()
    if not mesh.has_triangles():
        raise ValueError("No supported connected surface remained")
    mesh = mesh.filter_smooth_taubin(number_of_iterations=1)
    if len(mesh.triangles) > 200000:
        mesh = mesh.simplify_quadric_decimation(200000)
    if len(mesh.vertices) > 150000:
        mesh = mesh.simplify_quadric_decimation(120000)
    mesh.compute_vertex_normals()
    return mesh, {"voxel_m": length / resolution, "resolution": resolution, "truncation_m": .06}


def photo_colors(root, mesh, views, raw_k):
    xyz, normals = np.asarray(mesh.vertices), np.asarray(mesh.vertex_normals)
    rgb = np.asarray(mesh.vertex_colors).copy() * 255
    best = np.full(len(xyz), -1.)
    for view in views:
        pose, k, mask = view["pose"], view["k"], view["mask"]
        camera = (xyz - pose[:3, 3]) @ pose[:3, :3]
        z = camera[:, 2]
        safe_z = np.maximum(z, .001)
        u = k[0, 0] * camera[:, 0] / safe_z + k[0, 2]
        v = k[1, 1] * camera[:, 1] / safe_z + k[1, 2]
        height, width = mask.shape
        ui, vi = np.rint(u).astype(int).clip(0, width - 1), np.rint(v).astype(int).clip(0, height - 1)
        observed = (view["xyz"] - pose[:3, 3]) @ pose[:3, :3][:, 2]
        residual = np.abs(observed[vi, ui] - z)
        visible = (z > .3) & (u >= 1) & (u < width - 2) & (v >= 1) & (v < height - 2) & mask[vi, ui] & (residual < .09)
        direction = pose[:3, 3] - xyz
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-8)
        incidence = np.abs((normals * direction).sum(axis=1))
        center_weight = 1 - .3 * ((u - width / 2) / (width / 2)) ** 2 - .3 * ((v - height / 2) / (height / 2)) ** 2
        score = incidence * center_weight * np.exp(-residual / .04)
        selected = visible & (score > best)
        photo = cv2.cvtColor(read_image(root / "images" / view["name"]), cv2.COLOR_BGR2RGB)
        raw_u = camera[:, 0] / safe_z * raw_k[0, 0] + raw_k[0, 2]
        raw_v = camera[:, 1] / safe_z * raw_k[1, 1] + raw_k[1, 2]
        # OpenCV remap limits each output dimension to 32766 pixels.
        sampled = np.concatenate([cv2.remap(photo, raw_u[start:start + 16000].astype(np.float32).reshape(-1, 1), raw_v[start:start + 16000].astype(np.float32).reshape(-1, 1), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).reshape(-1, 3) for start in range(0, len(xyz), 16000)])
        rgb[selected] = sampled[selected]
        best[selected] = score[selected]
    return np.rint(rgb).clip(0, 255).astype(np.uint8), int((best >= 0).sum())


def orient(xyz, views, manifest):
    by_name = {image["filename"]: image for image in manifest["images"]}
    first = views[0]
    ar_pose = np.array(by_name[first["name"]]["cameraToWorld"]).reshape(4, 4, order="F")
    up = first["pose"][:3, :3] @ (ar_pose[:3, :3] @ np.diag([1, -1, -1])).T @ np.array([0, 1, 0.])
    up /= np.linalg.norm(up)
    center = (xyz.min(axis=0) + xyz.max(axis=0)) / 2
    back = np.mean([view["pose"][:3, 3] for view in views], axis=0) - center
    back -= up * (back @ up)
    if np.linalg.norm(back) < 1e-6:
        back = -first["pose"][:3, 2]
        back -= up * (back @ up)
    back /= np.linalg.norm(back)
    right = np.cross(up, back)
    right /= np.linalg.norm(right)
    rotation = np.stack([right, up, back])
    return (xyz - center) @ rotation.T, {"center_in_sfm_world": center.tolist(), "world_to_upright_rotation": rotation.tolist()}


def write_models(root, xyz, rgb, faces, notice):
    vertices = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    for index, key in enumerate(["x", "y", "z"]):
        vertices[key] = xyz[:, index]
    for index, key in enumerate(["red", "green", "blue"]):
        vertices[key] = rgb[:, index]
    triangles = np.empty(len(faces), dtype=[("vertex_indices", "i4", (3,))])
    triangles["vertex_indices"] = faces
    PlyData([PlyElement.describe(vertices, "vertex"), PlyElement.describe(triangles, "face")], text=False, comments=[notice]).write(str(root / "mesh.ply"))
    with (root / "model-metres.ply").open("w") as stream:
        stream.write(f"ply\nformat ascii 1.0\ncomment {notice}\nelement vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nelement face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
        for point, color in zip(xyz, rgb):
            stream.write(" ".join([f"{value:.6f}" for value in point] + [str(value) for value in color]) + "\n")
        for face in faces:
            stream.write("3 " + " ".join(str(int(value)) for value in face) + "\n")
    with (root / "model-metres.obj").open("w") as stream:
        stream.write(f"# {notice}\n# Units metres, local upright frame\n")
        for point, color in zip(xyz, rgb):
            stream.write("v " + " ".join(f"{value:.6f}" for value in point) + " " + " ".join(f"{value / 255:.6f}" for value in color) + "\n")
        for face in faces:
            stream.write("f " + " ".join(str(int(value) + 1) for value in face) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    args = parser.parse_args()
    root = args.workspace
    params = json.loads((root / "report.json").read_text())["sfm"]["refined_camera_params"]
    raw_k = np.array([[params[0], 0, params[2]], [0, params[1], params[3]], [0, 0, 1.]])
    views = load_views(root, raw_k)
    mesh, settings = fuse(views)
    world_xyz = np.asarray(mesh.vertices).copy()
    rgb, colored = photo_colors(root, mesh, views, raw_k)
    manifest = json.loads((root / "scan/manifest.json").read_text())
    xyz, orientation = orient(world_xyz, views, manifest)
    notice = "Experimental inferred geometry with photo-supported local planar estimates. Metres. Small surface details may be smoothed. Gaps and distortion may remain. Dimensional accuracy and completeness unverified."
    write_models(root, xyz, rgb, np.asarray(mesh.triangles), notice)
    features = json.loads((root / "photo-hints.json").read_text())["held_out"]
    unique = {feature["point_id"]: feature["xyz"] for feature in features}
    distances, _ = cKDTree(world_xyz).query(np.array(list(unique.values())))
    quality = {"notice": notice, "vertices": len(xyz), "faces": len(mesh.triangles), "direct_photo_colored_vertices": colored, "fusion": settings, "orientation": orientation, "held_out_features": {"count": len(unique), "median_nearest_vertex_m": float(np.median(distances)), "p95_nearest_vertex_m": float(np.quantile(distances, .95)), "scope": "Internal photo feature consistency from the same capture. Independent physical accuracy unverified."}}
    (root / "model-quality.json").write_text(json.dumps(quality, indent=2))
    print(json.dumps(quality, indent=2))


if __name__ == "__main__":
    main()
