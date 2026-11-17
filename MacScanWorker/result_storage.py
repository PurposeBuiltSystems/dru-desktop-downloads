"""Compact successful results while preserving exports and repair context."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

from reconstruct import write_json

ARTIFACTS = frozenset(["model.html", "model-metres.ply", "model-metres.obj", "report.json", "model-quality.json", "repair-provenance.json", "reconstruction-before-repair.ply", "README.txt"])
WORKING_DIRECTORIES = ["scan", "images", "sfm_depth", "refined", "stereo-refined", "sparse"]
WORKING_FILES = ["database.db", "database.db-wal", "database.db-shm", "dense.laz", "sfm_sparse.ply", "mesh.ply", "model.html", "model-metres.ply", "model-metres.obj", "photo-hints.json", "reconstruction-before-repair.ply"]
REQUIRED_EXPORTS = {"model.html", "model-metres.ply", "model-metres.obj", "report.json", "model-quality.json", "README.txt"}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def directory_bytes(root):
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file() and not path.is_symlink())


def validate_bundle(path):
    with zipfile.ZipFile(path) as archive:
        if not REQUIRED_EXPORTS.issubset(archive.namelist()) or archive.testzip() is not None:
            raise ValueError("Export bundle failed validation")


def compact_result(root, source_archive, source_sha256):
    root, source_archive = Path(root), Path(source_archive)
    if digest(source_archive) != source_sha256:
        raise ValueError("Preserved source archive failed its integrity check")
    bundle = root / "model-deliverable.zip"
    validate_bundle(bundle)
    receipt_path = root / "cleanup.json"
    previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    if previous.get("state") == "done" and previous.get("bundle_sha256") == digest(bundle):
        return previous
    for name in WORKING_DIRECTORIES:
        if (root / name).is_symlink():
            raise ValueError("Working folders must belong to the result being compacted")
    before = directory_bytes(root)
    context = [path for path in (root / "sparse/0_text").glob("*") if path.is_file()]
    context += [root / name for name in ["scan/manifest.json", "photo-hints.json", "dense-view-selection.json", "depth-refinement.json", "stereo-refinement.json"] if (root / name).is_file()]
    replacements = {"reconstruction/" + path.relative_to(root).as_posix(): path for path in context}
    replacements.update({name: root / name for name in ["report.json", "model-quality.json", "timing.json", "repair-provenance.json"] if (root / name).is_file()})
    temporary = root / "model-deliverable.zip.partial"
    with zipfile.ZipFile(bundle) as original, zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as packed:
        for member in original.infolist():
            if member.filename not in replacements:
                packed.writestr(member.filename, original.read(member))
        for name, path in replacements.items():
            packed.write(path, name)
    validate_bundle(temporary)
    # Verify the primary geometry and viewer before removing their standalone copies.
    with zipfile.ZipFile(temporary) as packed:
        for name in ["model.html", "model-metres.ply", "model-metres.obj", "reconstruction-before-repair.ply"]:
            path = root / name
            if path.exists() and hashlib.sha256(packed.read(name)).hexdigest() != digest(path):
                raise ValueError("Export differs from its standalone model")
        if "reconstruction/sparse/0_text/cameras.txt" not in packed.namelist():
            raise ValueError("Repair camera context is missing")
    if digest(source_archive) != source_sha256:
        raise ValueError("Preserved source archive changed during compaction")
    temporary.replace(bundle)
    receipt = {"state": "prepared", "bytes_before": previous.get("bytes_before", before), "bundle_sha256": digest(bundle), "source_sha256": source_sha256, "notice": "Original capture and final export retained. Repair camera context is stored inside the export ZIP."}
    write_json(receipt_path, receipt)
    for name in WORKING_DIRECTORIES:
        path = root / name
        if path.exists():
            shutil.rmtree(path)
    for path in [root / name for name in WORKING_FILES] + list(root.glob("*.prediction.npz")):
        if path.is_file() or path.is_symlink():
            path.unlink()
    receipt.update(state="done", retained_result_bytes=directory_bytes(root), retained_source_bytes=source_archive.stat().st_size)
    receipt["retained_total_bytes"] = receipt["retained_result_bytes"] + receipt["retained_source_bytes"]
    write_json(receipt_path, receipt)
    return receipt


def write_member(destination, name, contents):
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe repair context path")
    path = destination.joinpath(*relative.parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)


def restore_source(root, workspace):
    """Materialize a compacted result for the existing guided repair entry point."""
    if (root / "images").is_dir() and (root / "sparse/0_text").is_dir():
        return root
    status = json.loads((root / "status.json").read_text())
    source = Path(status["archived_source"])
    if digest(source) != status["source_sha256"]:
        raise ValueError("Original scan archive failed its integrity check")
    validate_bundle(root / "model-deliverable.zip")
    with zipfile.ZipFile(root / "model-deliverable.zip") as bundle:
        for member in bundle.infolist():
            if member.filename.startswith("reconstruction/"):
                write_member(workspace, member.filename.removeprefix("reconstruction/"), bundle.read(member))
        for name in ["report.json", "model-quality.json", "model-metres.ply"]:
            write_member(workspace, name, bundle.read(name))
        write_member(workspace, "mesh.ply", bundle.read("model-metres.ply"))
    for name in ["status.json", "timing.json"]:
        shutil.copy2(root / name, workspace / name)
    with zipfile.ZipFile(source) as capture:
        manifest = json.loads(capture.read("manifest.json"))
        override = root / "scan-identity.json"
        if override.exists():
            manifest.update(json.loads(override.read_text()))
        write_member(workspace, "scan/manifest.json", json.dumps(manifest).encode())
        for image in manifest["images"]:
            name = image["filename"]
            write_member(workspace, "images/" + name, capture.read(name))
    return workspace
