"""Local guided-repair queue shared by imported scan ZIPs and the results editor."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid
import threading
import zipfile

from reconstruct import write_json
from platform_runtime import active_command

HERE = Path(__file__).resolve().parent
IDENTITY_LOCK = threading.RLock()


def source_job(results, job_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
        raise ValueError("Invalid saved scan")
    source = results / job_id
    status = json.loads((source / "status.json").read_text())
    if status["state"] != "done" or status.get("operation") == "Guided room repair":
        raise ValueError("Choose a completed original reconstruction")
    archive = Path(status["archived_source"])
    with zipfile.ZipFile(archive) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
    if manifest["id"] != status["scan_id"]:
        raise ValueError("Saved capture identity changed")
    return source, status, manifest


def editor_data(results, job_id, token, product=None):
    source, status, manifest = source_job(results, job_id)
    from scan_branding import brand_for, logo_uri, product_query
    brand = brand_for(product or status.get("product") or manifest)
    preset = None
    preset_path = HERE / "office-scan-repair.json"
    if not preset_path.exists():
        preset_path = Path.home() / "Library/Application Support/PurposeBuilt Scan Worker/office-scan-repair.json"
    if preset_path.exists():
        candidate = json.loads(preset_path.read_text())
        if candidate["scan_id"] in [manifest["id"], manifest.get("sourceScanId")]:
            candidate["scan_id"] = manifest["id"]
            preset = candidate
    draft = source / "repair-draft.json"
    if draft.exists():
        from repair_room import validate_annotations
        preset = validate_annotations(json.loads(draft.read_text()), manifest, complete=False)
    suggested = None
    suggestion_note = ""
    try:
        from repair_suggestions import suggestions
        suggested = suggestions(source, manifest, preset)
        count = sum(len(row["pixels"]) for wall in suggested["walls"] for row in wall["contacts"])
        suggestion_note = "Suggested floor edges are ready. Drag their handles to match the photos." if count else "No reliable floor edges were detected. Mark the visible edges in the photos."
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        suggestion_note = "Automatic floor edges are unavailable for this reconstruction. Mark the visible edges manually."
    return {"scan_id": manifest["id"], "brand_title": brand["title"], "brand_logo": logo_uri(brand), "home_url": "/" + product_query(brand["id"]), "suggestions": suggested, "suggestion_note": suggestion_note,
            "draft_url": f"/{job_id}/repair-draft",
            "images": [{key: image[key] for key in ["filename", "width", "height"]} for image in manifest["images"]],
            "preset": preset, "token": token, "photo_url": f"/{job_id}/repair-photo",
            "submit_url": f"/{job_id}/repair-request?product={brand['id']}"}


def editor_page(results, job_id, token, product=None):
    payload = json.dumps(editor_data(results, job_id, token, product)).replace("<", "\\u003c")
    return (HERE / "repair_editor.html").read_text().replace("__REPAIR_DATA__", payload).encode()


def photo_bytes(results, job_id, filename):
    _, status, manifest = source_job(results, job_id)
    if filename not in {image["filename"] for image in manifest["images"]}:
        raise ValueError("Photo is outside this capture")
    with zipfile.ZipFile(status["archived_source"]) as bundle:
        entry = bundle.getinfo(filename)
        if entry.file_size > 20_000_000:
            raise ValueError("Photo exceeds the preview size limit")
        return bundle.read(entry)


def save_identity(results, job_id, details):
    with IDENTITY_LOCK:
        return _save_identity(results, job_id, details)


def _save_identity(results, job_id, details):
    import html
    import re
    from scan_identity import identity
    source, _, manifest = source_job(results, job_id)
    override = source / "scan-identity.json"
    previous = json.loads(override.read_text()) if override.exists() else manifest
    values = {}
    for key in ["scanName", "projectName", "permitNumber"]:
        value = details.get(key, previous.get(key) or "")
        if not isinstance(value, str) or len(value) > 160 or any(ord(char) < 32 for char in value):
            raise ValueError("Use up to 160 printable characters for each scan label")
        values[key] = value.strip()
    if not any(values.values()):
        raise ValueError("Enter a project, permit or scan name")
    capture = identity(dict(manifest, **values))
    def rename_page(page):
        title = html.escape(capture["name"] + " · " + capture["date"])
        page = re.sub(r"<title>.*?</title>", lambda match: "<title>" + title + "</title>", page, count=1, flags=re.S)
        return re.sub(r'(<div class="big">).*?(</div>)', lambda match: match[1] + title + match[2], page, count=1, flags=re.S)
    bundle = source / "model-deliverable.zip"
    temporary = bundle.with_suffix("." + uuid.uuid4().hex + ".tmp")
    try:
        with zipfile.ZipFile(bundle) as original, zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as updated:
            for member in original.infolist():
                if member.filename == "scan-identity.json":
                    continue
                content = original.read(member)
                if member.filename == "model.html":
                    content = rename_page(content.decode()).encode()
                updated.writestr(member, content)
            updated.writestr("scan-identity.json", json.dumps(values, indent=2))
        with zipfile.ZipFile(temporary) as updated:
            if updated.testzip() is not None:
                raise ValueError("Updated scan export failed its integrity check")
        temporary.replace(bundle)
    finally:
        temporary.unlink(missing_ok=True)
    page = source / "model.html"
    if page.exists():
        updated = page.with_suffix("." + uuid.uuid4().hex + ".tmp")
        updated.write_text(rename_page(page.read_text()))
        updated.replace(page)
    write_json(source / "scan-identity.json", values)
    from worker import rebuild_index
    rebuild_index(results)
    return {"ok": True, "name": capture["name"]}


def save_draft(results, job_id, annotations):
    from repair_room import validate_annotations
    source, _, manifest = source_job(results, job_id)
    validate_annotations(annotations, manifest, complete=False)
    # Each request writes a unique temporary file before replacing the complete draft.
    path = source / "repair-draft.json"
    temporary = path.with_suffix("." + uuid.uuid4().hex + ".tmp")
    write_json(temporary, annotations)
    temporary.replace(path)
    return {"ok": True}


def enqueue(results, job_id, annotations, product=None):
    from repair_room import validate_annotations
    _, status, manifest = source_job(results, job_id)
    from scan_branding import brand_for
    brand = brand_for(product or status.get("product") or manifest)
    validate_annotations(annotations, manifest)
    identity = uuid.uuid4().hex
    directory = results.parent / "Repairs" / identity
    directory.mkdir(parents=True)
    write_json(directory / "annotations.json", annotations)
    request = {"id": identity, "source_job": job_id, "state": "queued", "product": brand["id"],
               "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    write_json(directory / "request.json", request)
    return public_status(request)


def public_status(request):
    result = {"state": request["state"], "status_url": f"/_repair-status/{request['id']}"}
    if request.get("result_job"):
        from scan_branding import product_query
        result["model_url"] = f"/{request['result_job']}/model.html" + product_query(request.get("product"))
    if request.get("error"):
        result["error"] = request["error"]
    return result


def read_status(results, identity):
    if not re.fullmatch(r"[a-f0-9]{32}", identity):
        raise ValueError("Invalid repair request")
    request = json.loads((results.parent / "Repairs" / identity / "request.json").read_text())
    return public_status(request)


def process_request(path, config):
    request = json.loads(path.read_text())
    if request["state"] not in ["queued", "processing"]:
        return
    from desktop_access import access_status
    if not access_status(config)["allowed"]:
        return
    root = Path(config["scan_root"])
    previous = request.get("result_job")
    if previous:
        status_path = root / "Results" / previous / "status.json"
        if status_path.exists() and json.loads(status_path.read_text())["state"] == "done":
            request["state"] = "done"
            write_json(path, request)
            return
    job = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-guided-repair-{uuid.uuid4().hex[:8]}"
    request.update(state="processing", result_job=job)
    write_json(path, request)
    output = root / "Results" / job
    log_path = path.parent / "repair.log"
    try:
        source, _, _ = source_job(root / "Results", request["source_job"])
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join([str(HERE), config["dru_scripts"], config.get("runtime_site", ""), config["mapanything_source"]])
        environment["DYLD_LIBRARY_PATH"] = config["native_library_path"]
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        command = [sys.executable, "-P", str(HERE / "repair_room.py"), str(source),
                   "--annotations", str(path.parent / "annotations.json"),
                   "--out", str(output), "--dru-scripts", config["dru_scripts"]]
        if request.get("product"):
            command += ["--product", request["product"]]
        with log_path.open("w") as log:
            subprocess.run(active_command(command), env=environment,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        request["state"] = "done"
    except Exception as error:
        request.update(state="failed", error=f"{error}. The original reconstruction is preserved. See the repair log.")
    write_json(path, request)
