#!/usr/bin/env python3
"""Email-size deliverable for a reconstructed DRU scan, mirroring the iOS trench-scan bundle.

  python3 package_deliverable.py <workspace> [--budget-mb 18] [--faces 250000] [--voxel 0.01] [--photos 6]

Reads mesh.ply, dense.laz, report.json and scan/manifest.json from a dru_recon.py workspace and writes

  <workspace>/scan-<id8>.html                  self-contained measuring viewer (feet + metres), no network needed
  <workspace>/dru-scan-<id8>-deliverable.zip   README.txt, the viewer, mesh-feet.obj (CAD, feet, Y up),
                                               dense-cloud.laz (metres, Z up), report.json, photos/

The mesh is decimated and the cloud thinned until the ZIP fits the budget. 18 MB is the default because
Gmail's 25 MB limit is measured after base64 encoding (x1.37) and Outlook's default is 20 MB.

Runs as its own process (pymeshlab bundles an OpenMP runtime that clashes with pycolmap's).
"""
from __future__ import annotations
import html as html_escape
from scan_branding import brand_for

import argparse
import base64
import json
import re
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

FT_PER_M = 1.0 / 0.3048
PRODUCT = "Digital ROW Utilities"
VENDOR = "PurposeBuilt Systems"


# ----------------------------------------------------------------------------- inputs
def read_mesh(path: Path):
    from plyfile import PlyData
    ply = PlyData.read(str(path))
    v = ply["vertex"]
    xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    names = v.data.dtype.names
    rgb = np.column_stack([v["red"], v["green"], v["blue"]]).astype(np.uint8) if "red" in names else None
    faces = np.vstack(ply["face"]["vertex_indices"]).astype(np.int64)
    return xyz, rgb, faces


def read_laz(path: Path):
    import laspy
    las = laspy.read(str(path))
    xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
    rgb = (np.column_stack([las.red, las.green, las.blue]) // 257).astype(np.uint8)
    return xyz, rgb


def voxel_thin(xyz, rgb, voxel):
    if voxel <= 0:
        return xyz, rgb
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    idx.sort()
    return xyz[idx], rgb[idx]


# ----------------------------------------------------------------------------- mesh processing
def decimate(xyz, rgb, faces, target_faces):
    """Quadric edge collapse to target_faces; colours re-sampled from the nearest original vertex."""
    if len(faces) <= target_faces:
        return xyz, rgb, faces
    import pymeshlab
    from scipy.spatial import cKDTree
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=xyz, face_matrix=faces))
    ms.apply_filter("meshing_decimation_quadric_edge_collapse", targetfacenum=int(target_faces),
                    preservenormal=True, planarquadric=True)
    m = ms.current_mesh()
    nxyz = m.vertex_matrix().astype(np.float64)
    nfaces = m.face_matrix().astype(np.int64)
    nrgb = rgb[cKDTree(xyz).query(nxyz, k=1)[1]] if rgb is not None else None
    return nxyz, nrgb, nfaces


def vertex_normals(xyz, faces):
    """Area-weighted vertex normals quantised to signed bytes (same scheme as the iOS viewer)."""
    a, b, c = xyz[faces[:, 0]], xyz[faces[:, 1]], xyz[faces[:, 2]]
    fn = np.cross(b - a, c - a)
    acc = np.zeros_like(xyz)
    for k in range(3):
        np.add.at(acc, faces[:, k], fn)
    n = np.linalg.norm(acc, axis=1, keepdims=True)
    acc = np.where(n > 1e-12, acc / np.maximum(n, 1e-12), np.array([0.0, 1.0, 0.0]))
    return np.clip(np.rint(acc * 127), -127, 127).astype(np.int8)


# ----------------------------------------------------------------------------- writers
def write_obj_feet(path: Path, xyz_m, rgb, faces, header: list[str], measurements):
    """Wavefront OBJ in FEET, Y up (the frame the phone tracked in), per-vertex colour as 0..1 floats
    (MeshLab/CloudCompare convention). Recorded measurements are appended as real geometry (two vertices
    and a line) so they survive into any reader, as the iOS export does."""
    ft = xyz_m * FT_PER_M
    lines = [f"# {h}" for h in header]
    lines.append("# UNITS: feet (1 OBJ unit = 1 foot); axes: X right, Y up, -Z forward (local AR frame)")
    if rgb is not None:
        col = rgb.astype(np.float32) / 255.0
        lines += [f"v {x:.4f} {y:.4f} {z:.4f} {r:.3f} {g:.3f} {b:.3f}" for (x, y, z), (r, g, b) in zip(ft, col)]
    else:
        lines += [f"v {x:.4f} {y:.4f} {z:.4f}" for x, y, z in ft]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
    base = len(ft)
    for i, m in enumerate(measurements):
        s, e = np.array(m["start"]) * FT_PER_M, np.array(m["end"]) * FT_PER_M
        lines.append(f"# measurement {i + 1}: {m['label']}")
        lines.append(f"v {s[0]:.4f} {s[1]:.4f} {s[2]:.4f}")
        lines.append(f"v {e[0]:.4f} {e[1]:.4f} {e[2]:.4f}")
        lines.append(f"l {base + 2 * i + 1} {base + 2 * i + 2}")
    path.write_text("\n".join(lines) + "\n")


def write_laz_zup(path: Path, xyz_m, rgb):
    """LAS 1.2 point format 2 at 1 mm, LAZ-compressed, metres, Z up (E = x, N = -z, U = y)."""
    import laspy
    zup = np.column_stack([xyz_m[:, 0], -xyz_m[:, 2], xyz_m[:, 1]])
    header = laspy.LasHeader(point_format=2, version="1.2")
    header.scales = np.array([0.001] * 3)
    header.offsets = np.floor(zup.min(axis=0))
    las = laspy.LasData(header)
    las.x, las.y, las.z = zup[:, 0], zup[:, 1], zup[:, 2]
    las.red, las.green, las.blue = (rgb[:, 0].astype(np.uint16) * 257, rgb[:, 1].astype(np.uint16) * 257, rgb[:, 2].astype(np.uint16) * 257)
    las.write(path)


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def feet_inches(ft: float) -> str:
    whole = int(ft)
    inches = int(round((ft - whole) * 12))
    if inches == 12:
        whole, inches = whole + 1, 0
    return f"{whole}' {inches}\""


# ----------------------------------------------------------------------------- the viewer
HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>@@TITLE@@</title>
<style>
  :root {
    color-scheme: light dark;
    --background:#f2f2f7; --panel:rgba(255,255,255,.94); --text:#121214;
    --muted:#626267; --border:rgba(18,18,20,.12); --blue:#007aff;
    --orange:#ff8a3d; --secondary:rgba(0,122,255,.08); --warning:#805000;
  }
  @media (prefers-color-scheme:dark) {
    :root { --background:#121214; --panel:rgba(30,30,33,.94); --text:#f5f5f7;
      --muted:#aeaeb5; --border:rgba(255,255,255,.14); --blue:#0a84ff;
      --secondary:rgba(10,132,255,.14); --warning:#ffb66e; }
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--background); color:var(--text);
    font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
  canvas { display:block; width:100vw; height:100dvh; touch-action:none; cursor:crosshair; }
  .panel { position:fixed; background:var(--panel); border:1px solid var(--border);
    border-radius:20px; padding:18px; backdrop-filter:blur(16px);
    box-shadow:0 8px 30px rgba(0,0,0,.06); }
  #brand { top:16px; left:16px; right:16px; display:flex; align-items:center;
    gap:12px; padding:14px 18px; z-index:10; }
  #brand .identity { display:flex; align-items:center; gap:12px; color:inherit; text-decoration:none; }
  #brand .identity:focus-visible { outline:3px solid var(--blue); outline-offset:5px; border-radius:12px; }
  #outputs { margin-left:auto; position:relative; }
  #outputs summary { list-style:none; cursor:pointer; padding:10px 18px; border-radius:999px;
    background:var(--orange); color:#121214; font-weight:600; white-space:nowrap; }
  #outputs summary::-webkit-details-marker { display:none; }
  #outputs .files { position:absolute; right:0; top:calc(100% + 12px); width:260px;
    background:var(--panel); border:1px solid var(--border); border-radius:18px;
    padding:10px; box-shadow:0 12px 36px rgba(0,0,0,.16); }
  #outputs button { display:block; width:100%; text-align:left; margin:0; border-radius:10px; background:transparent; }
  #outputs button:hover { background:var(--secondary); }
  #outputs button:disabled { opacity:.5; cursor:wait; }
  #export-status { margin:8px; font-size:11px; color:var(--muted); }
  #brand img { width:44px; height:44px; border-radius:11px; }
  #brand .name { font-size:22px; font-weight:700; letter-spacing:-.5px; line-height:1.2; }
  #brand .by { font-size:12px; color:var(--muted); margin-top:2px; }
  #info { top:108px; left:16px; width:320px; max-width:calc(100vw - 32px);
    max-height:calc(100dvh - 300px); overflow:auto; }
  #readout { bottom:16px; left:16px; width:280px; max-width:calc(100vw - 210px); }
  #orbit { position:fixed; right:16px; bottom:16px; z-index:5; display:grid;
    grid-template-columns:repeat(3,44px); grid-auto-rows:44px; gap:8px; }
  button { background:var(--secondary); color:var(--blue); border:1px solid transparent;
    border-radius:999px; padding:9px 16px; font:inherit; font-size:13px; font-weight:600;
    cursor:pointer; margin-top:10px; }
  button:hover { border-color:var(--blue); }
  button:focus-visible { outline:3px solid var(--blue); outline-offset:3px; }
  #orbit button { margin:0; width:44px; height:44px; padding:0; font-size:18px;
    display:flex; align-items:center; justify-content:center; border-radius:14px;
    background:var(--panel); border:1px solid var(--border); box-shadow:0 4px 16px rgba(0,0,0,.04); }
  #orbit button:hover { border-color:var(--blue); background:var(--secondary); }
  #orbit #reset, #orbit button:active { background:var(--orange); color:#121214; border-color:var(--orange); }
  #orbit .wide { grid-column:1 / -1; width:auto; font-size:13px; border-radius:999px; color:var(--blue); }
  #orbit .spacer { visibility:hidden; }
  #nudge { position:fixed; left:50%; bottom:32px; transform:translateX(-50%);
    background:var(--panel); border:1px solid var(--border); border-radius:999px;
    padding:10px 18px; color:var(--muted); font-size:13px; pointer-events:none; }
  #nudge b { color:var(--text); }
  h1 { font-size:12px; margin:0 0 7px; color:var(--muted); font-weight:600; }
  .big { font-size:19px; font-weight:700; font-variant-numeric:tabular-nums; line-height:1.3; overflow-wrap:anywhere; }
  .recorded { color:var(--text); }
  .yours { color:var(--blue); }
  .muted { color:var(--muted); font-size:12px; line-height:1.45; margin-top:4px; }
  .warn { color:var(--muted); font-size:11px; line-height:1.5; margin-top:12px;
    border-top:1px solid var(--border); padding-top:12px; }
  .q { font-size:12px; line-height:1.5; margin-top:10px; }
  .q.ok { color:var(--text); } .q.weak, .q.failed { color:var(--warning); }
  @media (max-width:1000px) { #nudge { bottom:210px; } }
  @media (max-width:600px) {
    #brand { top:12px; left:12px; right:12px; padding:12px; }
    #brand .name { font-size:18px; }
    #brand .by { display:none; }
    #brand .identity { gap:10px; }
    #outputs summary { padding:9px 12px; font-size:12px; }
    #info { top:100px; left:12px; width:calc(100vw - 24px); max-width:none;
      max-height:27dvh; padding:14px; }
    #readout { left:12px; right:12px; bottom:12px; width:auto; max-width:none; padding:14px; }
    #orbit { right:12px; bottom:162px; }
    #nudge { display:none; }
  }
</style>
</head>
<body>
<canvas id="c"></canvas>

<div class="panel" id="info">
  <h1>Scan</h1>
  <div class="big">@@HEADLINE@@</div>
  <div class="muted">@@DETAIL@@</div>
  <div class="q @@QCLASS@@">@@QUALITY@@</div>
  @@RECORDED@@
  <div class="warn">Geometry is in feet, reconstructed from the phone's photos and ARCore tracking.
    Expect a few centimetres of error. The frame is local to the scan, not georeferenced.</div>
</div>

<header class="panel" id="brand">
  <a class="identity" id="home" href="/" aria-label="DRU Scan Tools main menu" title="Back to main menu">
    <img src="@@DRU_LOGO@@" alt="" width="44" height="44">
    <div>
      <div class="name">DRU Scan Tools</div>
      <div class="by">Digital ROW Utilities · Scan viewer</div>
    </div>
  </a>
  <details id="outputs">
    <summary>Output files</summary>
    <div class="files">
      <button data-export="usdz">USDZ · Quick Look / AR</button>
      <button data-export="obj">OBJ · CAD, feet</button>
      <button data-export="ply">PLY · reconstructed points, metres</button>
      <button data-export="html">HTML · offline viewer</button>
      <button data-export="zip">All files · ZIP</button>
      <p id="export-status" role="status">Estimated local geometry. Independently verify dimensions and completeness.</p>
    </div>
  </details>
</header>

<div id="orbit">
  <button class="spacer"></button>
  <button id="up" title="Tilt up" aria-label="Tilt up">&#9650;</button>
  <button class="spacer"></button>
  <button id="left" title="Turn left" aria-label="Turn left">&#9664;</button>
  <button id="reset" title="Reset view" aria-label="Reset view">&#9673;</button>
  <button id="right" title="Turn right" aria-label="Turn right">&#9654;</button>
  <button id="out" title="Zoom out" aria-label="Zoom out">&minus;</button>
  <button id="down" title="Tilt down" aria-label="Tilt down">&#9660;</button>
  <button id="in" title="Zoom in" aria-label="Zoom in">+</button>
  <button class="wide" id="top" title="Look straight down">Top view</button>
</div>

<div id="nudge"><b>Drag</b> the model to turn it · or use the arrows</div>

<div class="panel" id="readout">
  <h1>Your measurement</h1>
  <div class="big yours" id="dist">click two points</div>
  <div class="muted" id="hint">tap two points on the surface</div>
  <button id="clear">Clear points</button>
</div>

<script>
const POSITIONS = "@@POSITIONS@@";
const INDICES = "@@INDICES@@";
const COLORS = "@@COLORS@@";
const NORMALS = "@@NORMALS@@";
const MARKERS = @@MARKERS@@;   // recorded measurements, feet: [x1,y1,z1,x2,y2,z2]

function bytes(b64) {
  const s = atob(b64), a = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) a[i] = s.charCodeAt(i);
  return a.buffer;
}
const pos = new Float32Array(bytes(POSITIONS));
const idx = new Uint32Array(bytes(INDICES));
const col = new Uint8Array(bytes(COLORS));
const nrm = new Int8Array(bytes(NORMALS));

const canvas = document.getElementById("c");
const gl = canvas.getContext("webgl2") || canvas.getContext("webgl");
if (!gl) { document.body.innerHTML = "<p style='padding:24px'>This browser has no WebGL.</p>"; }

let lo = [Infinity,Infinity,Infinity], hi = [-Infinity,-Infinity,-Infinity];
for (let i = 0; i < pos.length; i += 3)
  for (let k = 0; k < 3; k++) { if (pos[i+k] < lo[k]) lo[k] = pos[i+k]; if (pos[i+k] > hi[k]) hi[k] = pos[i+k]; }
const centre = [(lo[0]+hi[0])/2, (lo[1]+hi[1])/2, (lo[2]+hi[2])/2];
const radius = Math.max(hi[0]-lo[0], hi[1]-lo[1], hi[2]-lo[2]) / 2 || 1;

const cam = { pivot: centre.slice(), dist: radius * 2.6, yaw: 0.6, pitch: 0.35 };
function camPos() {
  return [cam.pivot[0] + Math.cos(cam.pitch)*Math.sin(cam.yaw)*cam.dist,
          cam.pivot[1] + Math.sin(cam.pitch)*cam.dist,
          cam.pivot[2] + Math.cos(cam.pitch)*Math.cos(cam.yaw)*cam.dist];
}
function sub(a,b){return [a[0]-b[0],a[1]-b[1],a[2]-b[2]];}
function cross(a,b){return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];}
function dot(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
function norm(a){const l=Math.hypot(a[0],a[1],a[2])||1;return [a[0]/l,a[1]/l,a[2]/l];}
function lookAt(eye,at,up){
  const z=norm(sub(eye,at)), x=norm(cross(up,z)), y=cross(z,x);
  return [x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0, -dot(x,eye),-dot(y,eye),-dot(z,eye),1];
}
function perspective(fovy,aspect,near,far){
  const f=1/Math.tan(fovy/2);
  return [f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)/(near-far),-1, 0,0,2*far*near/(near-far),0];
}
function mul(a,b){
  const o=new Array(16);
  for(let r=0;r<4;r++)for(let c=0;c<4;c++){ let s=0; for(let k=0;k<4;k++) s+=a[k*4+r]*b[c*4+k]; o[c*4+r]=s; }
  return o;
}

const vs = `attribute vec3 p; attribute vec3 col; attribute vec3 nrm; uniform mat4 mvp;
  varying vec3 vcol; varying vec3 vnrm;
  void main(){ vcol = col; vnrm = nrm; gl_Position = mvp * vec4(p,1.0); }`;
const fs = `precision mediump float; varying vec3 vcol; varying vec3 vnrm;
  void main(){
    float light = 0.45 + 0.55 * abs(dot(normalize(vnrm), normalize(vec3(0.4,0.9,0.3))));
    gl_FragColor = vec4(vcol * light, 1.0);
  }`;
const lineVS = `attribute vec3 p; uniform mat4 mvp;
  void main(){ gl_Position = mvp * vec4(p,1.0); gl_PointSize = 9.0; }`;
const lineFS = `precision mediump float; uniform vec3 col; void main(){ gl_FragColor = vec4(col,1.0); }`;

function shader(type, src){
  const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(s));
  return s;
}
function program(v, f){
  const p = gl.createProgram();
  gl.attachShader(p, shader(gl.VERTEX_SHADER, v)); gl.attachShader(p, shader(gl.FRAGMENT_SHADER, f));
  gl.linkProgram(p); return p;
}
const meshProg = program(vs, fs), lineProg = program(lineVS, lineFS);
if (!gl.getProgramParameter(meshProg, gl.LINK_STATUS)) console.error("mesh program link failed:", gl.getProgramInfoLog(meshProg));

const posBuf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, posBuf); gl.bufferData(gl.ARRAY_BUFFER, pos, gl.STATIC_DRAW);
const colBuf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, colBuf); gl.bufferData(gl.ARRAY_BUFFER, col, gl.STATIC_DRAW);
const nrmBuf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, nrmBuf); gl.bufferData(gl.ARRAY_BUFFER, nrm, gl.STATIC_DRAW);
const idxBuf = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf);
const uintOK = (typeof WebGL2RenderingContext !== "undefined" && gl instanceof WebGL2RenderingContext) || !!gl.getExtension("OES_element_index_uint");
let indexData = idx, indexType = gl.UNSIGNED_INT;
if (!uintOK) {
  if (pos.length / 3 <= 65536) { indexData = new Uint16Array(idx); indexType = gl.UNSIGNED_SHORT; }
  else console.error("This browser needs OES_element_index_uint to draw a mesh this large.");
}
gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indexData, gl.STATIC_DRAW);
const dynBuf = gl.createBuffer();

// ---- picking: a uniform grid over triangles keeps a click fast on a 300k-face mesh
const picks = [];
const GRID = 48;
const cell = [(hi[0]-lo[0])/GRID || 1, (hi[1]-lo[1])/GRID || 1, (hi[2]-lo[2])/GRID || 1];
const grid = new Map();
(function buildGrid(){
  for (let f = 0; f < idx.length; f += 3) {
    let mn = [Infinity,Infinity,Infinity], mx = [-Infinity,-Infinity,-Infinity];
    for (let j = 0; j < 3; j++) { const v = idx[f+j]*3; for (let k = 0; k < 3; k++) { const x = pos[v+k]; if (x < mn[k]) mn[k] = x; if (x > mx[k]) mx[k] = x; } }
    const a = mn.map((x,k) => Math.min(GRID-1, Math.max(0, Math.floor((x-lo[k])/cell[k]))));
    const b = mx.map((x,k) => Math.min(GRID-1, Math.max(0, Math.floor((x-lo[k])/cell[k]))));
    for (let i = a[0]; i <= b[0]; i++) for (let j = a[1]; j <= b[1]; j++) for (let k = a[2]; k <= b[2]; k++) {
      const key = (i*GRID + j)*GRID + k; let arr = grid.get(key); if (!arr) { arr = []; grid.set(key, arr); } arr.push(f);
    }
  }
})();
function rayFromEvent(ev){
  const r = canvas.getBoundingClientRect();
  const x = ((ev.clientX - r.left) / r.width) * 2 - 1, y = 1 - ((ev.clientY - r.top) / r.height) * 2;
  const eye = camPos(), forward = norm(sub(cam.pivot, eye));
  const right = norm(cross(forward, [0,1,0])), up = cross(right, forward);
  const aspect = canvas.width / canvas.height, t = Math.tan(Math.PI/6);
  const dir = norm([forward[0] + right[0]*x*t*aspect + up[0]*y*t, forward[1] + right[1]*x*t*aspect + up[1]*y*t, forward[2] + right[2]*x*t*aspect + up[2]*y*t]);
  return { eye, dir };
}
function hitTri(f, eye, dir){
  const a = idx[f]*3, b = idx[f+1]*3, c = idx[f+2]*3;
  const v0=[pos[a],pos[a+1],pos[a+2]], v1=[pos[b],pos[b+1],pos[b+2]], v2=[pos[c],pos[c+1],pos[c+2]];
  const e1=sub(v1,v0), e2=sub(v2,v0), h=cross(dir,e2), det=dot(e1,h);
  if (Math.abs(det) < 1e-9) return Infinity;
  const inv=1/det, s=sub(eye,v0), u=dot(s,h)*inv; if (u < 0 || u > 1) return Infinity;
  const q=cross(s,e1), v=dot(dir,q)*inv; if (v < 0 || u+v > 1) return Infinity;
  const t=dot(e2,q)*inv; return t > 1e-6 ? t : Infinity;
}
function pick(ev){
  const { eye, dir } = rayFromEvent(ev);
  // 3D-DDA through the grid; stop at the first cell whose best hit lies inside it.
  let tmin = 0, tmax = Infinity;
  for (let k = 0; k < 3; k++) {
    if (Math.abs(dir[k]) < 1e-12) { if (eye[k] < lo[k] || eye[k] > hi[k]) return null; continue; }
    let t0 = (lo[k]-eye[k])/dir[k], t1 = (hi[k]-eye[k])/dir[k]; if (t0 > t1) [t0, t1] = [t1, t0];
    tmin = Math.max(tmin, t0); tmax = Math.min(tmax, t1);
  }
  if (tmax < tmin) return null;
  let p = [eye[0]+dir[0]*tmin, eye[1]+dir[1]*tmin, eye[2]+dir[2]*tmin];
  let c = p.map((x,k) => Math.min(GRID-1, Math.max(0, Math.floor((x-lo[k])/cell[k]))));
  const step = dir.map(d => d > 0 ? 1 : -1);
  const tDelta = dir.map((d,k) => Math.abs(cell[k] / (d || 1e-12)));
  let tNext = dir.map((d,k) => { const edge = lo[k] + (c[k] + (d > 0 ? 1 : 0)) * cell[k]; return Math.abs(d) < 1e-12 ? Infinity : (edge - eye[k]) / d; });
  const seen = new Set(); let best = Infinity;
  for (let n = 0; n < GRID*3; n++) {
    const key = (c[0]*GRID + c[1])*GRID + c[2]; const arr = grid.get(key);
    if (arr) for (const f of arr) { if (seen.has(f)) continue; seen.add(f); const t = hitTri(f, eye, dir); if (t < best) best = t; }
    const k = tNext[0] < tNext[1] ? (tNext[0] < tNext[2] ? 0 : 2) : (tNext[1] < tNext[2] ? 1 : 2);
    if (best <= tNext[k]) break;
    c[k] += step[k]; if (c[k] < 0 || c[k] >= GRID) break; tNext[k] += tDelta[k];
  }
  return isFinite(best) ? [eye[0]+dir[0]*best, eye[1]+dir[1]*best, eye[2]+dir[2]*best] : null;
}
function feetInches(ft){
  const whole = Math.floor(ft), inches = Math.round((ft - whole) * 12);
  return inches === 12 ? (whole+1) + "' 0\"" : whole + "' " + inches + "\"";
}
function metres(ft){ return (ft * 0.3048).toFixed(3) + " m"; }
function updateReadout(){
  const el = document.getElementById("dist");
  if (picks.length < 2) { el.textContent = picks.length === 1 ? "click one more point" : "click two points"; return; }
  const d = Math.hypot(picks[1][0]-picks[0][0], picks[1][1]-picks[0][1], picks[1][2]-picks[0][2]);
  const dy = Math.abs(picks[1][1]-picks[0][1]);
  el.innerHTML = feetInches(d) + ' <span class="muted">point to point · ' + metres(d) + '</span><br>' +
                 '<span style="font-size:18px">' + feetInches(dy) + ' <span class="muted">vertical · ' + metres(dy) + '</span></span>';
}

let dragging = null, last = null, moved = 0;
canvas.addEventListener("pointerdown", e => { dragging = (e.button === 2 || e.shiftKey) ? "pan" : "orbit"; last = [e.clientX, e.clientY]; moved = 0; canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener("pointermove", e => {
  if (!dragging) return;
  const dx = e.clientX - last[0], dy = e.clientY - last[1]; last = [e.clientX, e.clientY]; moved += Math.abs(dx) + Math.abs(dy);
  if (dragging === "orbit") { cam.yaw -= dx * 0.006; cam.pitch = Math.max(-1.5, Math.min(1.5, cam.pitch + dy * 0.006)); }
  else {
    const eye = camPos(), forward = norm(sub(cam.pivot, eye)), right = norm(cross(forward, [0,1,0])), up = cross(right, forward), scale = cam.dist * 0.0022;
    for (let k = 0; k < 3; k++) cam.pivot[k] += (-right[k]*dx + up[k]*dy) * scale;
  }
  draw();
});
canvas.addEventListener("pointerup", e => {
  if (dragging && moved < 5) { const p = pick(e); if (p) { if (picks.length >= 2) picks.length = 0; picks.push(p); updateReadout(); } }
  dragging = null; draw();
});
canvas.addEventListener("contextmenu", e => e.preventDefault());
canvas.addEventListener("wheel", e => { e.preventDefault(); cam.dist = Math.max(0.2, Math.min(radius * 40, cam.dist * (1 + Math.sign(e.deltaY) * 0.12))); draw(); }, { passive: false });
document.getElementById("clear").onclick = () => { picks.length = 0; updateReadout(); draw(); };

function drawPoints(points, colour, mode){
  if (!points.length) return;
  const flat = new Float32Array(points.flat());
  gl.bindBuffer(gl.ARRAY_BUFFER, dynBuf); gl.bufferData(gl.ARRAY_BUFFER, flat, gl.DYNAMIC_DRAW);
  const loc = gl.getAttribLocation(lineProg, "p"); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 0, 0);
  gl.uniform3fv(gl.getUniformLocation(lineProg, "col"), colour);
  gl.drawArrays(mode, 0, points.length);
}
function draw(){
  const dpr = Math.min(devicePixelRatio || 1, 2);
  canvas.width = innerWidth * dpr; canvas.height = innerHeight * dpr;
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.enable(gl.DEPTH_TEST); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  const eye = camPos();
  const mvp = mul(perspective(Math.PI/3, canvas.width/canvas.height, 0.01, radius*100), lookAt(eye, cam.pivot, [0,1,0]));
  gl.useProgram(meshProg);
  gl.uniformMatrix4fv(gl.getUniformLocation(meshProg, "mvp"), false, new Float32Array(mvp));
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf); const mp = gl.getAttribLocation(meshProg, "p"); gl.enableVertexAttribArray(mp); gl.vertexAttribPointer(mp, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, colBuf); const mc = gl.getAttribLocation(meshProg, "col"); gl.enableVertexAttribArray(mc); gl.vertexAttribPointer(mc, 3, gl.UNSIGNED_BYTE, true, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, nrmBuf); const mn = gl.getAttribLocation(meshProg, "nrm"); gl.enableVertexAttribArray(mn); gl.vertexAttribPointer(mn, 3, gl.BYTE, true, 0, 0);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf);
  gl.drawElements(gl.TRIANGLES, idx.length, indexType, 0);
  gl.useProgram(lineProg);
  gl.uniformMatrix4fv(gl.getUniformLocation(lineProg, "mvp"), false, new Float32Array(mvp));
  gl.disable(gl.DEPTH_TEST);
  for (const m of MARKERS) { drawPoints([[m[0],m[1],m[2]],[m[3],m[4],m[5]]], [1,0.24,0.24], gl.LINES); drawPoints([[m[0],m[1],m[2]],[m[3],m[4],m[5]]], [1,0.24,0.24], gl.POINTS); }
  if (picks.length) drawPoints(picks, [0.35,0.65,1], gl.POINTS);
  if (picks.length === 2) drawPoints(picks, [0.35,0.65,1], gl.LINES);
}
addEventListener("resize", draw);

const home = { yaw: cam.yaw, pitch: cam.pitch, dist: cam.dist, pivot: cam.pivot.slice() };
const nudgeEl = document.getElementById("nudge");
function dismissNudge() { if (nudgeEl) { nudgeEl.remove(); } }
function orbit(dYaw, dPitch, scale) {
  cam.yaw += dYaw; cam.pitch = Math.max(-1.5, Math.min(1.5, cam.pitch + dPitch));
  if (scale) cam.dist = Math.max(0.05, cam.dist * scale);
  dismissNudge(); draw();
}
const STEP = 0.32, TILT = 0.22, ZOOM = 1.25;
document.getElementById("left").onclick  = () => orbit(-STEP, 0, 1);
document.getElementById("right").onclick = () => orbit(STEP, 0, 1);
document.getElementById("up").onclick    = () => orbit(0, TILT, 1);
document.getElementById("down").onclick  = () => orbit(0, -TILT, 1);
document.getElementById("in").onclick    = () => orbit(0, 0, 1 / ZOOM);
document.getElementById("out").onclick   = () => orbit(0, 0, ZOOM);
document.getElementById("top").onclick   = () => { cam.pitch = 1.45; cam.yaw = 0; dismissNudge(); draw(); };
document.getElementById("reset").onclick = () => { cam.yaw = home.yaw; cam.pitch = home.pitch; cam.dist = home.dist; cam.pivot = home.pivot.slice(); dismissNudge(); draw(); };
canvas.addEventListener("pointerdown", dismissNudge, { once: true });
canvas.addEventListener("wheel", dismissNudge, { once: true, passive: true });

updateReadout();
draw();
</script>
<script id="dru-export-tools">
@@EXPORT_TOOLS@@
</script>
<script id="dru-export-controls">
const homeLink = document.getElementById("home");
if (location.protocol === "file:" && homeLink.getAttribute("href") === "/") homeLink.href = "http://127.0.0.1:8765/";
const exportNotice = document.querySelector("#info .warn").textContent.trim();
const captureLabel = document.querySelector("#info .big").textContent;
const exportName = "dru-scan-" + (captureLabel.split(" · ")[0].replace(/[^\p{L}\p{N}._-]+/gu, "-").replace(/^[.-]+|[.-]+$/g, "").slice(0, 100) || "scan");
const exportStatus = document.getElementById("export-status");
function viewerHTML() {
  const page = document.documentElement.cloneNode(true);
  page.querySelectorAll("[data-export]").forEach(button => button.removeAttribute("disabled"));
  page.querySelector("#outputs").removeAttribute("open");
  if (location.protocol === "http:" || location.protocol === "https:") page.querySelector("#home").href = new URL("/", location.href).href;
  page.querySelector("#export-status").textContent = "Estimated local geometry. Independently verify dimensions and completeness.";
  return "<!doctype html>\n" + page.outerHTML;
}
function outputFile(format) {
  const builders = {
    usdz: () => ({name: exportName + ".usdz", data: DRUScanExports.usdz(pos, col, idx, MARKERS, exportNotice), type: "model/vnd.usdz+zip"}),
    obj: () => ({name: exportName + "-feet.obj", data: DRUScanExports.obj(pos, col, idx, MARKERS, exportNotice), type: "text/plain"}),
    ply: () => ({name: exportName + "-points-metres.ply", data: DRUScanExports.pointCloud(pos, col, exportNotice), type: "application/octet-stream"}),
    html: () => ({name: exportName + ".html", data: viewerHTML(), type: "text/html"})
  };
  return builders[format]();
}
async function completeBundle() {
  const files = ["usdz", "obj", "ply", "html"].map(outputFile);
  const readme = exportNotice + "\n\nUSDZ and OBJ use feet, Y up. USDZ declares metersPerUnit 0.3048. PLY and F32 use metres, Y up. All geometry is local and has no survey reference.\n\nThe point clouds contain reconstructed model vertices. F32 is packed little-endian xyz float32, compatible with the iOS point-cloud layout. Reconstructed points can include inferred repaired surfaces.\n\nRecorded field-measurement lines are included in USDZ and OBJ when present. Viewer point picks are temporary estimates. Independently verify before inspection or engineering decisions.\n";
  files.push({name: "reconstructed-points-metres.f32", data: DRUScanExports.floatCloud(pos)});
  files.push({name: "README.txt", data: readme});
  files.push({name: "metadata.json", data: JSON.stringify({product: document.querySelector("#brand").dataset.scanProduct, capture_label: captureLabel, notice: exportNotice, vertices: pos.length / 3, triangles: idx.length / 3, coordinate_system: "local Y-up, no survey reference", mesh_units: "feet", point_cloud_units: "metres", recorded_measurements_feet: MARKERS}, null, 2)});
  return {name: exportName + "-all-files.zip", data: await DRUScanExports.compressedZip(files), type: "application/zip"};
}
for (const button of document.querySelectorAll("[data-export]")) {
  button.addEventListener("click", () => {
    const buttons = document.querySelectorAll("[data-export]");
    buttons.forEach(item => item.disabled = true);
    exportStatus.textContent = "Preparing your file…";
    setTimeout(async () => {
      try {
        const file = button.dataset.export === "zip" ? await completeBundle() : outputFile(button.dataset.export);
        const blob = new Blob([file.data], {type: file.type});
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url; link.download = file.name;
        document.body.appendChild(link); link.click(); link.remove();
        setTimeout(() => URL.revokeObjectURL(url), 60000);
        exportStatus.textContent = file.name + " · " + (blob.size / 1048576).toFixed(1) + " MB";
      } catch (error) {
        exportStatus.textContent = "Export failed. " + error.message;
      } finally {
        buttons.forEach(item => item.disabled = false);
      }
    }, 0);
  });
}
</script>
</body>
</html>
"""


def brand_viewer(page: str, product=None) -> str:
    """Apply the selected product presentation while preserving model and measurement data."""
    if '<canvas id="c"></canvas>' not in page or 'const POSITIONS = "' not in page:
        raise ValueError("Unrecognized scan viewer format")
    from scan_branding import brand_for, logo_uri, product_query
    if product is None:
        saved = re.search(r'data-scan-product="([a-z]+)"', page)
        product = saved[1] if saved else None
    brand_info = brand_for(product)
    style = HTML[HTML.index("<style>"):HTML.index("</style>") + len("</style>")]
    brand = HTML[HTML.index('<header class="panel" id="brand">'):HTML.index("</header>") + len("</header>")]
    brand = brand.replace("@@DRU_LOGO@@", logo_uri(brand_info))
    brand = brand.replace('id="brand"', 'id="brand" data-scan-product="' + brand_info["id"] + '"')
    brand = brand.replace("DRU Scan Tools", brand_info["title"]).replace("Digital ROW Utilities", brand_info["product"])
    brand = brand.replace('href="/"', 'href="/' + product_query(brand_info["id"]) + '"')
    if brand_info["id"] == "bridgecheck":
        style = style.replace("--blue:#007aff", "--blue:#9e4511").replace("--blue:#0a84ff", "--blue:#ff8a3d")
    page, styles = re.subn(r"<style>.*?</style>", lambda _: style, page, count=1, flags=re.DOTALL)
    page, brands = re.subn(r'<(?:div|header) class="panel" id="brand"[^>]*>.*?^</(?:div|header)>',
                          lambda _: brand, page, count=1, flags=re.DOTALL | re.MULTILINE)
    if styles != 1 or brands != 1:
        raise ValueError("Scan viewer branding elements were not found")
    headline = re.search(r'<div class="big">(.*?)</div>', page, re.DOTALL)
    if headline:
        title = html_escape.escape(brand_info["title"]) + " · " + headline[1]
        page = re.sub(r"<title>.*?</title>", lambda _: "<title>" + title + "</title>", page, count=1, flags=re.DOTALL)
    page = page.replace("PLY and OBJ units are metres in a local upright frame.", "Output files declare their units. Geometry is in a local upright frame.")
    controls = HTML[HTML.index('<script id="dru-export-tools">'):HTML.rindex("</script>") + len("</script>")]
    tools = Path(__file__).with_name("scan_viewer_exports.js").read_text()
    if brand_info["id"] == "bridgecheck":
        tools = tools.replace('"DRUScan"', '"BridgeCheckScan"')
    controls = controls.replace("@@EXPORT_TOOLS@@", tools)
    controls = controls.replace('homeLink.getAttribute("href") === "/"', 'homeLink.getAttribute("href").startsWith("/")')
    controls = controls.replace('"dru-scan-"', '"' + brand_info["prefix"] + '-"')
    controls = controls.replace('new URL("/", location.href)', 'new URL("/' + product_query(brand_info["id"]) + '", location.href)')
    controls = controls.replace('http://127.0.0.1:8765/', 'http://127.0.0.1:8765/' + product_query(brand_info["id"]))
    page = re.sub(r'<script id="dru-export-tools">.*?</script>\s*<script id="dru-export-controls">.*?</script>\s*', "", page, flags=re.DOTALL)
    return page.replace("</body>", controls + "\n</body>").replace("gl.clearColor(0.051, 0.067, 0.09, 1)", "gl.clearColor(0, 0, 0, 0)")


def html_page(xyz_m, rgb, faces, meta: dict, measurements: list[dict]) -> str:
    meta = {key: html_escape.escape(value) if isinstance(value, str) else value for key, value in meta.items()}
    ft = (xyz_m * FT_PER_M).astype(np.float32)
    normals = vertex_normals(xyz_m, faces)
    colors = rgb if rgb is not None else np.full((len(ft), 3), (140, 184, 224), np.uint8)
    markers = [list(np.round(np.array(m["start"] + m["end"]) * FT_PER_M, 4)) for m in measurements]
    recorded = ""
    if measurements:
        rows = "".join(f"<div class=\"big recorded\">{html_escape.escape(str(m['label']))}</div><div class=\"muted\">{html_escape.escape(str(m['detail']))}</div>" for m in measurements)
        recorded = f"<h1 style=\"margin-top:8px\">Recorded in the field</h1>{rows}"
    v = meta["verdict"]
    qclass = "ok" if v.startswith("OK") else "weak" if v.startswith("WEAK") else "failed"
    return brand_viewer(HTML.replace("@@TITLE@@", f"{meta.get('name', meta['id8'])} — {meta['date']}")
            .replace("@@HEADLINE@@", f"{meta.get('name', meta['id8'])} · {meta['date']}")
            .replace("@@DETAIL@@", meta["detail"])
            .replace("@@QCLASS@@", qclass).replace("@@QUALITY@@", meta["quality"])
            .replace("@@RECORDED@@", recorded)
            .replace("@@PRODUCT@@", PRODUCT).replace("@@VENDOR@@", VENDOR)
            .replace("@@POSITIONS@@", b64(ft)).replace("@@INDICES@@", b64(faces.astype(np.uint32)))
            .replace("@@COLORS@@", b64(colors.astype(np.uint8))).replace("@@NORMALS@@", b64(normals))
            .replace("@@MARKERS@@", json.dumps(markers)), meta.get("product"))


# ----------------------------------------------------------------------------- README
def readme(meta: dict, files: dict, measurements: list[dict], photos: int) -> str:
    q = meta
    lines = [
        f"{brand_for(meta.get('product'))['product']} — 3D SCAN {q['id8']}",
        f"Captured {q['date']} on Android ({q['frames']} photos). Reconstructed off-device with open-source tools",
        "(COLMAP structure-from-motion seeded with the phone's ARCore tracking, MapAnything dense",
        "reconstruction, screened Poisson meshing).",
        "",
        "QUALITY",
        f"  {q['verdict']}",
        f"  {q['quality']}",
        "  The free-scale fit says whether the photos and the phone's own motion tracking agree on",
        "  size; 1.000 means they do. Residuals are how far the refined camera path moved from the",
        "  tracked one. Nothing here is surveyed: expect a few centimetres of error, and treat any",
        "  field measurement of record as authoritative over a distance re-measured on this model.",
        "",
        "COORDINATES",
        "  Local to the scan (the phone's AR session frame). Not georeferenced; no CRS. Tie it to",
        "  the ground with the GNSS shots recorded separately in the app.",
        "",
        "FILES",
        f"  {files['html']}",
        "         Self-contained viewer with a measuring tool. Open it in any browser by double-",
        "         clicking; it needs no internet connection and loads nothing external. Click two",
        "         points for point-to-point and vertical distance in feet-inches and metres.",
        f"  {files['obj']}",
        "         Mesh for CAD and mesh tools (MeshLab, CloudCompare, Blender, ReCap, MicroStation).",
        "         UNITS: FEET. Axes as tracked by the phone: X right, Y up, -Z forward. Per-vertex",
        f"         colour as 0..1 floats. {files['obj_faces']:,} triangles.",
        f"  {files['laz']}",
        "         Dense point cloud, LAS 1.2 (LAZ-compressed), RGB, 1 mm resolution. UNITS: METRES,",
        "         Z UP (x east-ish, y north-ish in the local frame, z = height). Opens directly in",
        f"         CloudCompare, QGIS, Civil 3D, MicroStation/OpenRoads, PDAL. {files['laz_points']:,} points,",
        f"         thinned to one point per {files['voxel_cm']:g} cm.",
        "  report.json",
        "         Processing report: frames registered, reprojection error, path agreement, capture",
        "         quality flags, timings.",
    ]
    if photos:
        lines += [f"  photos/", f"         {photos} of the {q['frames']} capture photos, evenly spaced through the scan."]
    lines += ["", "MEASUREMENTS RECORDED IN THE FIELD"]
    if measurements:
        for i, m in enumerate(measurements, 1):
            lines.append(f"  {i}. {m['label']}  ({m['detail']})")
        lines += ["  Each is drawn in red in the viewer and written into the OBJ as a line between two", "  vertices, so it survives into any reader."]
    else:
        lines.append("  None. The geometry is attached so it can still be measured.")
    lines += [
        "",
        "NOT INCLUDED, TO KEEP THIS UNDER AN EMAIL ATTACHMENT LIMIT",
        f"  The full-resolution mesh ({q['full_faces']:,} triangles), the full dense cloud ({q['full_points']:,}",
        "  points at 5 mm) and the COLMAP sparse model with every photo. They exist on the processing",
        "  machine and can be sent on request. Nothing failed.",
        "",
        f"Built by {VENDOR} · https://purposebuilt.systems/dru",
    ]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- build
def gather_meta(ws: Path) -> tuple[dict, list[dict]]:
    report = json.loads((ws / "report.json").read_text())
    manifest = json.loads((ws / "scan" / "manifest.json").read_text()) if (ws / "scan" / "manifest.json").exists() else {}
    if (ws / "scan-identity.json").exists():
        manifest.update(json.loads((ws / "scan-identity.json").read_text()))
    scan_id = manifest.get("id") or report.get("scan_id") or ws.name
    created = manifest.get("createdAt") or report.get("createdAt") or ""
    from scan_identity import identity
    capture = identity(dict(manifest, createdAt=created, id=scan_id))
    date = capture["date"]
    sfm = report.get("sfm", {})
    cq = report.get("capture_quality", {})
    verdict = report.get("verdict", "")
    quality = (f"{sfm.get('registered', '?')} of {sfm.get('total', report.get('images', '?'))} frames registered, "
               f"reprojection {sfm.get('mean_reprojection_px', 0):.2f} px, free-scale fit {sfm.get('sim3_scale_sfm_to_arcore', 0):.3f}, "
               f"path residual median {sfm.get('direct_residual_m', {}).get('median', 0) * 100:.1f} cm")
    if cq.get("flags"):
        quality += "; " + "; ".join(cq["flags"])
    detail = f"{report.get('images', len(manifest.get('images', [])))} photos · {manifest.get('coordinateSystem', 'local AR frame')}"
    measurements = []
    for m in manifest.get("measurements", []) or []:
        s, e = m.get("start", {}), m.get("end", {})
        try:
            start = [float(s["x"]), float(s["y"]), float(s["z"])]; end = [float(e["x"]), float(e["y"]), float(e["z"])]
        except (KeyError, TypeError, ValueError):
            continue
        d = float(np.linalg.norm(np.array(end) - np.array(start)))
        measurements.append({"start": start, "end": end, "label": f"{feet_inches(d * FT_PER_M)} ({d:.3f} m)",
                             "detail": f"measured {m.get('createdAt', '')[:16].replace('T', ' ')} · {m.get('surfaceMethod', '')}"})
    return {"id": scan_id, "id8": scan_id[:8], "name": capture["name"], "product": (json.loads((ws / "status.json").read_text()).get("product") if (ws / "status.json").exists() else None) or manifest.get("product") or manifest.get("producedBy"), "date": date, "verdict": verdict, "quality": quality, "detail": detail,
            "frames": report.get("images", len(manifest.get("images", [])))}, measurements


def pick_photos(ws: Path, n: int) -> list[Path]:
    src = ws / "scan" if (ws / "scan").exists() else ws / "images"
    frames = sorted(src.glob("frame_*.jpg"))
    if not frames or n <= 0:
        return []
    picks = np.unique(np.linspace(0, len(frames) - 1, min(n, len(frames))).round().astype(int))
    return [frames[i] for i in picks]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workspace", type=Path)
    ap.add_argument("--budget-mb", type=float, default=18.0, help="ZIP size to fit (18 MB clears Gmail's 25 MB post-encoding limit)")
    ap.add_argument("--faces", type=int, default=250000, help="starting triangle count for the viewer and OBJ")
    ap.add_argument("--voxel", type=float, default=0.01, help="starting thinning grid for the LAZ (m)")
    ap.add_argument("--photos", type=int, default=6)
    args = ap.parse_args()
    ws = args.workspace.resolve()
    mesh_path, laz_path = ws / "mesh.ply", ws / "dense.laz"
    if not mesh_path.exists() or not laz_path.exists():
        print("package: needs mesh.ply and dense.laz in the workspace", file=sys.stderr)
        return 2

    meta, measurements = gather_meta(ws)
    xyz, rgb, faces = read_mesh(mesh_path)
    cxyz, crgb = read_laz(laz_path)
    meta["full_faces"], meta["full_points"] = len(faces), len(cxyz)
    photos = pick_photos(ws, args.photos)
    names = {"html": f"scan-{meta['id8']}.html", "obj": "mesh-feet.obj", "laz": "dense-cloud.laz"}
    zip_path = ws / f"dru-scan-{meta['id8']}-deliverable.zip"
    budget = int(args.budget_mb * 1024 * 1024)

    target_faces, voxel = args.faces, args.voxel
    for attempt in range(8):
        tmp = Path(tempfile.mkdtemp(prefix="dru-pkg-", dir=ws))
        try:
            dxyz, drgb, dfaces = decimate(xyz, rgb, faces, target_faces)
            txyz, trgb = voxel_thin(cxyz, crgb, voxel)
            html = html_page(dxyz, drgb, dfaces, meta, measurements)
            (tmp / names["html"]).write_text(html)
            write_obj_feet(tmp / names["obj"], dxyz, drgb, dfaces,
                           [f"{PRODUCT} scan {meta['id']} captured {meta['date']}", meta["verdict"]], measurements)
            write_laz_zup(tmp / names["laz"], txyz, trgb)
            shutil.copy2(ws / "report.json", tmp / "report.json")
            if photos:
                (tmp / "photos").mkdir()
                for p in photos:
                    shutil.copy2(p, tmp / "photos" / p.name)
            files = dict(names, obj_faces=len(dfaces), laz_points=len(txyz), voxel_cm=round(voxel * 100, 2))
            (tmp / "README.txt").write_text(readme(meta, files, measurements, len(photos)))
            if zip_path.exists():
                zip_path.unlink()
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
                z.write(tmp / "README.txt", "README.txt")
                for f in sorted(tmp.rglob("*")):
                    if f.is_file() and f.name != "README.txt":
                        z.write(f, str(f.relative_to(tmp)))
            size = zip_path.stat().st_size
            shutil.copy2(tmp / names["html"], ws / names["html"])
            print(f"package: {len(dfaces)} faces, {len(txyz)} points at {voxel*100:.1f} cm -> {size/1048576:.1f} MB (budget {args.budget_mb:g} MB)", flush=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if size <= budget:
            break
        if target_faces <= 40000 and voxel >= 0.05:
            print("package: cannot get under budget; keeping the smallest attempt", file=sys.stderr)
            break
        target_faces = max(40000, int(target_faces * 0.65))
        voxel = min(0.05, voxel * 1.35)
    summary = {"zip": zip_path.name, "bytes": size, "html": names["html"], "faces": int(len(dfaces)),
               "points": int(len(txyz)), "voxel_m": voxel, "photos": len(photos), "measurements": len(measurements)}
    (ws / "package.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
