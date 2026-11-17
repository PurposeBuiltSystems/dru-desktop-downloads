#!/usr/bin/env python3
"""Run the photo-constrained reconstruction in separate native-library processes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
NOTICE = "Experimental photo reconstruction with photo-supported local planar estimates. Small surface details may be smoothed. Gaps and distortion may remain. Dimensional accuracy and completeness are unverified. Independently verify before engineering or inspection decisions."


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scripts = Path(config["dru_scripts"])
    environment = os.environ.copy()
    environment.update({
        "PYTHONPATH": os.pathsep.join([str(HERE), str(scripts), config["mapanything_source"]] + ([config["runtime_site"]] if config.get("runtime_site") else [])),
        "DYLD_LIBRARY_PATH": config["native_library_path"],
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "4",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    python = [sys.executable, "-P"]
    alignment_command = python + [str(scripts / "dru_recon.py"), str(args.scan), "--out", str(output), "--dense", "none", "--package-mb", "0", "--max-features", "8192"]
    if config.get("fast_alignment", False):
        alignment_command = python + [str(HERE / "fast_alignment.py"), str(args.scan), "--out", str(output), "--dru-scripts", str(scripts)]
        if config.get("pose_guided_matching", False):
            alignment_command += ["--pose-guided"]
    dense_command = python + [str(HERE / "dense_photo.py"), str(output), "--no-depth-hint", "--predictions-only", "--long-side", str(config.get("dense_long_side", 518)), "--chunk", str(config.get("dense_chunk", 8)), "--max-views", str(config.get("dense_max_views", 0))]
    if config.get("quantized_model"):
        dense_command += ["--quantized-model", config["quantized_model"]]
    if config.get("native_int4"):
        dense_command += ["--native-int4"]
    refinement_workers = min(4, os.cpu_count() or 1)
    stages = [
        ("photo_alignment", alignment_command),
        ("photo_depth_hints", python + [str(HERE / "photo_stages.py"), "prepare", str(output)]),
        ("dense_photo_inference", dense_command),
        ("depth_refinement", python + [str(HERE / "photo_stages.py"), "refine", str(output), "--compact", "--workers", str(refinement_workers)]),
        ("photo_pair_refinement", python + [str(HERE / "stereo_photo.py"), str(output)]),
        ("surface_fusion_and_color", python + [str(HERE / "fuse_photo.py"), str(output)]),
        ("viewer_and_export", python + [str(HERE / "photo_stages.py"), "package", str(output), "--dru-scripts", str(scripts)]),
    ]
    started = time.perf_counter()
    timing = {"state": "processing", "notice": NOTICE, "steps_seconds": {}, "reused_scan_outputs": False, "processing_profile": {key: config.get(key) for key in ["dense_long_side", "dense_chunk", "dense_max_views", "fast_alignment", "pose_guided_matching", "quantized_model", "native_int4"]}}
    timing["processing_profile"].update(compact_intermediates=True, refinement_workers=refinement_workers)
    try:
        for name, command in stages:
            timing["stage"] = name
            write_json(output / "timing.json", timing)
            step_started = time.perf_counter()
            print(f"Starting {name}", flush=True)
            with (output / f"{name}.log").open("w") as log:
                subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            timing["steps_seconds"][name] = round(time.perf_counter() - step_started, 3)
            print(f"Finished {name} in {timing['steps_seconds'][name]:.1f} seconds", flush=True)
        timing["state"] = "done"
        report = json.loads((output / "report.json").read_text())
        report["reconstruction_notice"] = NOTICE
        timing["total_seconds"] = round(time.perf_counter() - started, 3)
        report["timing"] = timing
        report["verdict"] = "EXPERIMENTAL: photo-constrained reconstruction; independent dimensional accuracy unverified"
        write_json(output / "report.json", report)
    except Exception as error:
        timing["state"] = "failed"
        timing["error"] = str(error)
        raise
    finally:
        timing["total_seconds"] = round(time.perf_counter() - started, 3)
        write_json(output / "timing.json", timing)
    print(f"Total {timing['total_seconds']:.1f} seconds", flush=True)


if __name__ == "__main__":
    main()
