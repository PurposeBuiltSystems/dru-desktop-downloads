#!/usr/bin/env python3
"""Watch a local inbox and reconstruct exported DRU photo scan ZIPs serially."""
import argparse
import base64
from datetime import datetime, timezone
from platform_runtime import lock_file, active_command
import functools
import hashlib
import html
import hmac
import http.server
import io
import json
from pathlib import Path
import shutil
import secrets
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from urllib.parse import parse_qs, unquote, urlsplit

from reconstruct import write_json
from result_storage import ARTIFACTS, compact_result
from scan_branding import brand_for, logo_uri, product_query


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def validate_source(path):
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) > 4096 or sum(entry.file_size for entry in entries) > 2 * 1024**3:
            raise ValueError("Scan ZIP exceeds the local processing size limit")
        manifest = json.loads(archive.read("manifest.json"))
        scan_id = str(uuid.UUID(manifest["id"]))
        if len(manifest.get("images", [])) < 2:
            raise ValueError("A photo reconstruction needs at least two saved images")
        damaged = archive.testzip()
        if damaged:
            raise ValueError(f"Damaged ZIP member: {damaged}")
    return scan_id


def fingerprint(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


INDEX_LOCK = threading.RLock()


def rebuild_index(results):
    with INDEX_LOCK:
        _rebuild_index(results)


def _rebuild_index(results):
    rows = []
    for path in results.glob("*/status.json"):
        status = json.loads(path.read_text())
        timing = path.parent / "timing.json"
        if timing.exists():
            status["timing"] = json.loads(timing.read_text())
        try:
            with zipfile.ZipFile(status["archived_source"]) as capture:
                manifest = json.loads(capture.read("manifest.json"))
            override = path.parent / "scan-identity.json"
            if override.exists():
                manifest.update(json.loads(override.read_text()))
            from scan_identity import identity
            if status.get("operation") != "Guided room repair" or not status.get("capture"):
                status["capture"] = identity(manifest)
        except (KeyError, OSError, ValueError, zipfile.BadZipFile):
            status["capture"] = {"name": "Scan " + status.get("scan_id", "unknown")[:8], "date": "Capture time unavailable"}
        rows.append(status)
    rows.sort(key=lambda row: row["started"], reverse=True)
    write_json(results / "index.json", rows)
    cards = []
    for row in rows:
        name = html.escape(row["job_id"])
        state = html.escape(row["state"])
        seconds = row.get("timing", {}).get("total_seconds")
        minutes, remaining_seconds = divmod(round(seconds or 0), 60)
        duration = f" · {minutes} min {remaining_seconds} sec" if seconds is not None else ""
        scan_label = html.escape(row["capture"]["name"])
        started = html.escape(row["capture"]["date"])
        links = f'<a class="primary" href="{name}/model.html">Open model</a> · <a href="{name}/model-deliverable.zip">Download ZIP</a> · <a href="{name}/timing.json">Timing</a>' if state == "done" else f'<a href="{name}/worker.log">Processing log</a>'
        if state == "done" and row.get("operation") != "Guided room repair":
            links += f' · <a href="{name}/repair.html">Repair room surfaces</a>'
        rename = ""
        if state == "done" and row.get("operation") != "Guided room repair":
            rename = f'<form data-job="{name}"><label>Project or permit <input name="scanName" maxlength="160" value="{scan_label}" required></label><button>Save name</button><span role="status"></span></form>'
        operation = html.escape(row.get("operation", "Scan"))
        storage_note = f'<p>Storage cleanup needs attention. {html.escape(row["cleanup_error"])}</p>' if row.get("cleanup_error") else ""
        cards.append(f'<article><h2>{scan_label}</h2><p>{started}</p><p>{operation} · {state}{duration}</p>{storage_note}<p>{links}</p>{rename}</article>')
    logo_paths = [Path(__file__).parent.parent / "dru-logo.png",
                  Path(__file__).parent / "scan_recon/dru-logo.png",
                  Path(__file__).parent / "Branding/dru-logo.png"]
    logo = next((path for path in logo_paths if path.exists()), None)
    logo_html = '<img alt="" width="52" height="52" src="data:image/png;base64,' + base64.b64encode(logo.read_bytes()).decode("ascii") + '">' if logo else ""
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="15"><title>DRU Scan Tools</title><style>
:root{color-scheme:light dark;--bg:#f2f2f7;--panel:#fff;--text:#121214;--muted:#626267;--blue:#007aff;--border:rgba(18,18,20,.12)}
@media(prefers-color-scheme:dark){:root{--bg:#121214;--panel:#1e1e21;--text:#f5f5f7;--muted:#aeaeb5;--blue:#0a84ff;--border:rgba(255,255,255,.14)}}
*{box-sizing:border-box}body{font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;max-width:1000px;margin:40px auto;padding:0 24px;background:var(--bg);color:var(--text)}
header{display:flex;align-items:center;gap:14px;margin-bottom:28px}header img{border-radius:13px}h1{font-size:28px;line-height:1.2;letter-spacing:-.5px;margin:0}input,button{font:inherit;border:1px solid var(--border);border-radius:10px;padding:9px;background:var(--bg);color:var(--text)}form{display:flex;align-items:center;gap:8px;flex-wrap:wrap}form label{color:var(--muted)}header p{margin:5px 0 0;color:var(--muted)}h2{font-size:18px;overflow-wrap:anywhere;margin:0}article{background:var(--panel);border:1px solid var(--border);padding:22px;margin:16px 0;border-radius:20px}a{color:var(--blue);text-decoration:none;display:inline-block;padding:8px 12px;border-radius:999px}a:hover{text-decoration:underline}a.primary{background:#ff8a3d;color:#121214;font-weight:600;padding:9px 18px}p{line-height:1.5}article p{color:var(--muted);margin:8px 0}.notice{color:var(--muted);font-size:13px;max-width:760px}.instructions{margin-bottom:24px}
</style><header>''' + logo_html + '''<div><h1>DRU Scan Tools</h1><p>Digital ROW Utilities</p></div></header><p class="instructions">Drop an exported DRU photo scan ZIP into the Inbox folder. This Mac processes one scan at a time.</p><p class="notice">Experimental reconstructed geometry. Gaps and distortion may remain. Independently verify dimensions and completeness before inspection or engineering decisions.</p>'''
    if sys.platform == "win32":
        document = document.replace("This Mac", "This PC")
    temporary = results / "index.html.tmp"
    temporary.write_text(document + ("".join(cards) or "<p>Waiting for scans.</p>") + "</html>")
    temporary.replace(results / "index.html")


def archive_source(source, folder, digest):
    target = folder / f"{source.stem}-{digest[:12]}-{uuid.uuid4().hex[:8]}.zip"
    source.rename(target)
    return target


def process(source, config_path, config):
    from desktop_access import require_access
    require_access(config)
    root = Path(config["scan_root"])
    claimed = root / "Processing" / f"{uuid.uuid4()}.zip"
    source.rename(claimed)
    digest = fingerprint(claimed)
    job_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{digest[:12]}-{uuid.uuid4().hex[:6]}"
    output = root / "Results" / job_id
    output.mkdir()
    status = {"job_id": job_id, "state": "processing", "started": timestamp(), "source_name": source.name, "source_sha256": digest, "product": "bridgecheck" if source.name.startswith("bridgecheck-") else "dru"}
    write_json(output / "status.json", status)
    rebuild_index(root / "Results")
    started = time.perf_counter()
    repair_markings = None
    try:
        status["scan_id"] = validate_source(claimed)
        with zipfile.ZipFile(claimed) as archive:
            captured_product = brand_for(json.loads(archive.read("manifest.json")))
            if captured_product["id"] == "bridgecheck":
                status["product"] = "bridgecheck"
        with zipfile.ZipFile(claimed) as archive:
            if "repair-annotations.json" in archive.namelist():
                try:
                    from repair_room import validate_annotations
                    if archive.getinfo("repair-annotations.json").file_size > 262144:
                        raise ValueError("Repair markings exceed the size limit")
                    repair_markings = validate_annotations(json.loads(archive.read("repair-annotations.json")), json.loads(archive.read("manifest.json")))
                except (ValueError, TypeError, KeyError, AttributeError) as error:
                    status["repair_error"] = f"Repair markings could not be used: {error}"
        write_json(output / "status.json", status)
        command = [sys.executable, "-P", str(Path(__file__).with_name("reconstruct.py")), str(claimed), "--out", str(output), "--config", str(config_path)]
        # Prevent idle sleep only while a scan is being processed.
        with (output / "worker.log").open("w") as log:
            subprocess.run(active_command(command), stdout=log, stderr=subprocess.STDOUT, check=True)
        status["state"] = "done"
        status["archived_source"] = str(archive_source(claimed, root / "Archive", digest))
        if config.get("cleanup_after_success"):
            try:
                status["storage"] = compact_result(output, Path(status["archived_source"]), digest)
            except Exception as error:
                status["cleanup_error"] = str(error)
    except Exception as error:
        status["state"] = "failed"
        status["error"] = str(error)
        status["preserved_source"] = str(archive_source(claimed, root / "Failed", digest))
    finally:
        status["finished"] = timestamp()
        status["elapsed_seconds_including_validation"] = round(time.perf_counter() - started, 3)
        write_json(output / "status.json", status)
        rebuild_index(root / "Results")
    if status["state"] == "done" and repair_markings is not None:
        try:
            from repair_requests import enqueue
            status["repair_request"] = enqueue(root / "Results", job_id, repair_markings)
        except Exception as error:
            status["repair_error"] = str(error)
        write_json(output / "status.json", status)
    print(json.dumps(status), flush=True)
    return status["state"] == "done"


class ResultHandler(http.server.SimpleHTTPRequestHandler):
    def local_repair_request(self):
        port = self.server.server_address[1]
        hosts = [f"127.0.0.1:{port}", f"localhost:{port}"]
        return self.headers.get("Host") in hosts

    def bytes_response(self, payload, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self' data:; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'")
        self.end_headers()
        return io.BytesIO(payload)

    def do_POST(self):
        parsed = urlsplit(self.path)
        parts = unquote(parsed.path).strip("/").split("/")
        port = self.server.server_address[1]
        origin = self.headers.get("Origin")
        token = self.headers.get("X-Repair-Token", "")
        if (not self.local_repair_request() or origin not in [None, f"http://127.0.0.1:{port}", f"http://localhost:{port}"]
                or not hmac.compare_digest(token, self.server.repair_token)):
            payload = self.bytes_response(json.dumps({"error": "Reload the local repair page before submitting"}).encode(), "application/json", 403)
            self.copyfile(payload, self.wfile)
            return
        try:
            if len(parts) != 2 or parts[1] not in ["repair-request", "repair-draft", "scan-identity"]:
                raise ValueError("Unknown repair action")
            from desktop_access import require_access
            require_access(getattr(self.server, "configuration", {}))
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 262144 or self.headers.get_content_type() != "application/json":
                raise ValueError("Repair markings need bounded JSON")
            from repair_requests import enqueue, save_draft, save_identity
            action = {"repair-draft": save_draft, "scan-identity": save_identity, "repair-request": enqueue}[parts[1]]
            payload = json.loads(self.rfile.read(length))
            if parts[1] == "repair-request":
                request = enqueue(Path(self.directory), parts[0], payload, parse_qs(parsed.query).get("product", [None])[0])
            else:
                request = action(Path(self.directory), parts[0], payload)
            payload = self.bytes_response(json.dumps(request).encode(), "application/json", 202 if parts[1] == "repair-request" else 200)
        except PermissionError as error:
            payload = self.bytes_response(json.dumps({"error": str(error)}).encode(), "application/json", 402)
        except (ValueError, TypeError, KeyError, AttributeError, OSError, zipfile.BadZipFile) as error:
            payload = self.bytes_response(json.dumps({"error": str(error)}).encode(), "application/json", 400)
        self.copyfile(payload, self.wfile)

    def branded_viewer(self, page, product):
        import importlib.util
        scripts = getattr(self.server, "configuration", {}).get("dru_scripts")
        if scripts is None:
            scripts = Path(__file__).parent.parent.parent / "Digital ROW Utilities/scripts/scan_recon"
        spec = importlib.util.spec_from_file_location("scan_viewer_branding", Path(scripts) / "package_deliverable.py")
        exporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exporter)
        return exporter.brand_viewer(page, product)

    def branded_library(self, page, product):
        brand = brand_for(product)
        query = product_query(brand["id"])
        page = page.replace("DRU Scan Tools", brand["title"]).replace("Digital ROW Utilities", brand["product"])
        page = page.replace("Drop an exported DRU photo scan ZIP", "Drop a compatible photo scan ZIP")
        import re
        page = re.sub(r'(<header><img[^>]*src=")[^"]+("[^>]*>)', lambda match: match[1] + logo_uri(brand) + match[2], page, count=1)
        page = re.sub(r'(href="[^"?]+/(?:model.html|repair.html|model-deliverable.zip))"', lambda match: match[1] + query + '"', page)
        if brand["id"] == "bridgecheck":
            page = page.replace("--blue:#007aff", "--blue:#9e4511").replace("--blue:#0a84ff", "--blue:#ff8a3d")
        return page

    def send_head(self):
        if not self.local_repair_request():
            self.send_error(403, "Use the local companion address")
            return None
        parsed = urlsplit(self.path)
        parts = unquote(parsed.path).strip("/").split("/")
        selected_product = parse_qs(parsed.query).get("product", [None])[0]
        if len(parts) == 2 and (parts[1] in ["repair.html", "repair-photo"] or parts[0] == "_repair-status"):
            try:
                from repair_requests import editor_page, photo_bytes, read_status
                if parts[0] == "_repair-status":
                    return self.bytes_response(json.dumps(read_status(Path(self.directory), parts[1])).encode(), "application/json")
                if parts[1] == "repair.html":
                    return self.bytes_response(editor_page(Path(self.directory), parts[0], self.server.repair_token, selected_product), "text/html; charset=utf-8")
                name = parse_qs(parsed.query).get("name", [""])[0]
                return self.bytes_response(photo_bytes(Path(self.directory), parts[0], name), "image/jpeg")
            except (ValueError, KeyError, OSError, zipfile.BadZipFile) as error:
                self.send_error(400, str(error))
                return None
        if parsed.path in ["/", "/index.html"]:
            page = self.branded_library((Path(self.directory) / "index.html").read_text(), selected_product)
            script = """<script>const token=__TOKEN__;document.querySelectorAll('form[data-job]').forEach(form=>{form.onsubmit=async event=>{event.preventDefault();const status=form.querySelector('[role=status]');try{const response=await fetch('/'+form.dataset.job+'/scan-identity',{method:'POST',headers:{'Content-Type':'application/json','X-Repair-Token':token},body:JSON.stringify({scanName:form.elements.scanName.value})});const result=await response.json();if(!response.ok)throw new Error(result.error);location.reload();}catch(error){status.textContent=error.message;}}});</script>""".replace("__TOKEN__", json.dumps(self.server.repair_token))
            if '<form data-job=' not in page:
                return self.bytes_response(page.encode(), "text/html; charset=utf-8")
            # Editing a name must survive the library's periodic refresh.
            page = page.replace('<meta http-equiv="refresh" content="15">', '')
            return self.bytes_response((page + script).encode(), "text/html; charset=utf-8")
        path = Path(self.translate_path(self.path))
        if len(parts) == 2 and parts[1] in ["model.html", "model-deliverable.zip"] and selected_product is not None:
            try:
                path.resolve().relative_to(Path(self.directory).resolve())
                if parts[1] == "model.html":
                    if path.exists():
                        page = path.read_text()
                    else:
                        with zipfile.ZipFile(path.parent / "model-deliverable.zip") as archive:
                            page = archive.read("model.html").decode()
                    return self.bytes_response(self.branded_viewer(page, selected_product).encode(), "text/html; charset=utf-8")
                packed = io.BytesIO()
                with zipfile.ZipFile(path) as original, zipfile.ZipFile(packed, "w", zipfile.ZIP_DEFLATED) as branded:
                    for member in original.infolist():
                        content = original.read(member)
                        if member.filename == "model.html":
                            content = self.branded_viewer(content.decode(), selected_product).encode()
                        if member.filename != "viewer-brand.json":
                            branded.writestr(member, content)
                    branded.writestr("viewer-brand.json", json.dumps({"product": brand_for(selected_product)["product"]}))
                return self.bytes_response(packed.getvalue(), "application/zip")
            except (ValueError, KeyError, OSError, zipfile.BadZipFile) as error:
                self.send_error(400, str(error))
                return None
        if path.exists():
            return super().send_head()
        try:
            relative = path.resolve().relative_to(Path(self.directory).resolve())
            if len(relative.parts) != 2 or relative.name not in ARTIFACTS:
                return super().send_head()
            with zipfile.ZipFile(path.parent / "model-deliverable.zip") as archive:
                contents = archive.read(relative.name)
        except (ValueError, OSError, KeyError, zipfile.BadZipFile):
            return super().send_head()
        self.send_response(200)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Content-Length", str(len(contents)))
        self.end_headers()
        return io.BytesIO(contents)


def serve(results, port, config=None):
    handler = functools.partial(ResultHandler, directory=str(results))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    server.repair_token = secrets.token_urlsafe(32)
    server.configuration = config or {}
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = Path(config["scan_root"])
    for name in ["Inbox", "Processing", "Results", "Archive", "Failed"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    with (root / ".worker.lock").open("a+") as lock:
        try:
            lock_file(lock)
        except OSError:
            raise SystemExit("The scan worker is already running")
        # A restart preserves interrupted sources and retries into a new result directory.
        for path in (root / "Processing").glob("*.zip"):
            path.rename(root / "Inbox" / f"recovered-{path.name}")
        for path in (root / "Results").glob("*/status.json"):
            status = json.loads(path.read_text())
            if status["state"] == "processing":
                status.update(state="interrupted", finished=timestamp())
                write_json(path, status)
                timing_path = path.parent / "timing.json"
                if timing_path.exists():
                    timing = json.loads(timing_path.read_text())
                    timing["state"] = "interrupted"
                    write_json(timing_path, timing)
        if config.get("cleanup_after_success"):
            for receipt_path in (root / "Results").glob("*/cleanup.json"):
                receipt = json.loads(receipt_path.read_text())
                if receipt.get("state") == "prepared":
                    status_path = receipt_path.parent / "status.json"
                    status = json.loads(status_path.read_text())
                    if status.get("state") == "done":
                        try:
                            status["storage"] = compact_result(receipt_path.parent, Path(status["archived_source"]), status["source_sha256"])
                            status.pop("cleanup_error", None)
                        except Exception as error:
                            status["cleanup_error"] = str(error)
                        write_json(status_path, status)
        rebuild_index(root / "Results")
        server = serve(root / "Results", config["viewer_port"], config)
        print(f"Watching {root / 'Inbox'}; viewer http://127.0.0.1:{config['viewer_port']}/", flush=True)
        failed = False
        while True:
            from desktop_access import access_status
            access = access_status(config)
            write_json(root / "Results" / "access.json", access)
            if not access["allowed"]:
                if args.once:
                    server.shutdown()
                    raise SystemExit(access["message"])
                time.sleep(5)
                continue
            for path in sorted((root / "Inbox").glob("*.zip")):
                try:
                    size = path.stat().st_size
                    time.sleep(2)
                    if path.exists() and size > 0 and path.stat().st_size == size:
                        failed = not process(path, args.config.resolve(), config) or failed
                except OSError as error:
                    print(f"Inbox file unavailable: {error}", flush=True)
            from repair_requests import process_request
            for request in sorted((root / "Repairs").glob("*/request.json")):
                process_request(request, config)
            rebuild_index(root / "Results")
            if args.once:
                server.shutdown()
                raise SystemExit(1 if failed else 0)
            time.sleep(5)


if __name__ == "__main__":
    main()
