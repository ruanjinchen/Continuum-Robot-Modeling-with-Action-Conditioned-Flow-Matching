from __future__ import annotations
import os, sys, time, json, math
import numpy as np
import multiprocessing as mp
from pathlib import Path
from dataclasses import dataclass

from .config import CollectCfg
from .utils import ensure_dir, print_color

# Optional deps
try:
    import mujoco
    from mujoco.renderer import Renderer
except Exception:
    mujoco = None
    Renderer = None
try:
    import open3d as o3d
except Exception:
    o3d = None
# Optional deps for saving images
try:
    from PIL import Image
except Exception:
    Image = None

try:
    import psutil  # optional
except Exception:
    psutil = None


def _detect_actuators(model):
    """
    Generic version: does NOT require actuator names like motor1..
    Returns: act_ids (np.int32[D]), act_names (list[str]), D.
    Order matches the actuator order in the MuJoCo model (as defined in the XML).
    """
    D = int(model.nu)  # num of actuators
    if D <= 0:
        raise RuntimeError("Model has no actuators (cannot control data.ctrl). Please check the XML.")
    ids = np.arange(D, dtype=np.int32)
    names = []
    for i in range(D):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(i))
        if nm is None:
            nm = f"actuator{i + 1}"
        names.append(str(nm))
    return ids, names, D

def _detect_cameras(model):
    """Return all camera ids and names (order matches MuJoCo model order)."""
    ncam = int(model.ncam)
    ids = list(range(ncam))
    names: list[str] = []
    for cid in ids:
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, int(cid))
        if nm is None:
            nm = f"cam{cid}"
        names.append(str(nm))
    return ids, names


def _select_camera_ids(cam_spec: list[str] | None, all_cam_names: list[str]) -> list[int]:
    """
    cam_spec: list of CLI tokens (camera name or id). None/[] means select all.
    Supports:
      - --rgb_cams 0 1 2 3
      - --rgb_cams camera1 camera2
    """
    if not cam_spec:
        return list(range(len(all_cam_names)))

    name_to_id = {n: i for i, n in enumerate(all_cam_names)}
    out: list[int] = []
    for token in cam_spec:
        if token is None:
            continue
        s = str(token).strip()
        if s == "":
            continue

        # Numeric token -> treat as camera id
        if s.lstrip("-").isdigit():
            cid = int(s)
            if cid < 0 or cid >= len(all_cam_names):
                raise ValueError(f"camera id {cid} out of range: model has {len(all_cam_names)} cameras")
            out.append(cid)
        else:
            # Name token -> lookup
            if s not in name_to_id:
                raise ValueError(f"Unknown camera name: {s}. Available cameras: {all_cam_names}")
            out.append(int(name_to_id[s]))

    # Deduplicate and sort (stable output order).
    return sorted(set(out))


def _save_rgb_image(rgb: np.ndarray, path: Path, fmt: str = "png"):
    """Save RGB (uint8, H, W, 3) returned by renderer.render()."""
    ensure_dir(path.parent)
    fmt = (fmt or "png").lower()

    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    # If your images look vertically flipped, uncomment the next line: rgb = rgb[::-1].copy()
    # rgb = rgb[::-1].copy()

    if Image is not None:
        img = Image.fromarray(rgb)
        if fmt in ("jpg", "jpeg"):
            img.save(str(path), quality=95)
        else:
            img.save(str(path))
        return

    # Fallback when Pillow is unavailable (requires imageio).
    import imageio.v2 as imageio
    imageio.imwrite(str(path), rgb)

# ---------- segmentation mask export (bound to RGB) ----------
def _save_mask_image(mask: np.ndarray, path: Path):
    """Save a binary mask: uint8(H, W), values in {0,255}, png."""
    ensure_dir(path.parent)
    if mask.dtype != np.uint8:
        mask = np.clip(mask, 0, 255).astype(np.uint8)

    # If your images look vertically flipped, apply the same flip as RGB here:
    # mask = mask[::-1].copy()

    if Image is not None:
        # mode="L" -> 8-bit grayscale
        Image.fromarray(mask, mode="L").save(str(path))
        return

    import imageio.v2 as imageio
    imageio.imwrite(str(path), mask)


def _render_segmentation(renderer: Renderer, data, cid: int) -> np.ndarray:
    """Render MuJoCo segmentation; typically returns (H, W, 2) int32."""
    renderer.update_scene(data, camera=cid)

    # Ensure the renderer is not left in depth mode
    try:
        renderer.disable_depth_rendering()
    except Exception:
        pass

    if not hasattr(renderer, "enable_segmentation_rendering"):
        raise RuntimeError(
            "Current mujoco.Renderer does not support enable_segmentation_rendering();"
            "Please upgrade the mujoco Python package, or use a depth-based mask instead."
        )

    renderer.enable_segmentation_rendering()
    seg = renderer.render()
    renderer.disable_segmentation_rendering()
    return seg


def _infer_mask_geom_ids(model, prefer_group: int = 1, alpha_thresh: float = 1e-3) -> np.ndarray:
    """
    Infer foreground (robot) geom ids:
    1) If your robot geoms use group=prefer_group in the XML, prefer that group
    2) Otherwise, use alpha_thresh: keep geoms with rgba alpha > thresh (useful when the base is transparent)
    3) Fallback: all geoms
    """
    # 1) group filter (recommended: set TDCR geoms to group=1 in the XML)
    if hasattr(model, "geom_group"):
        gg = np.asarray(model.geom_group)
        ids = np.where(gg == int(prefer_group))[0]
        if ids.size > 0:
            return ids.astype(np.int32)

    # 2) alpha filter (if base geom alpha=0, this excludes it)
    if hasattr(model, "geom_rgba"):
        rgba = np.asarray(model.geom_rgba)
        if rgba.ndim == 2 and rgba.shape[1] == 4:
            ids = np.where(rgba[:, 3] > float(alpha_thresh))[0]
            if ids.size > 0:
                return ids.astype(np.int32)

    # 3) fallback: all geoms
    return np.arange(int(model.ngeom), dtype=np.int32)


def _seg_to_binary_mask(seg: np.ndarray, model, fg_geom_ids: np.ndarray) -> np.ndarray:
    """
    seg -> binary mask (H,W) uint8 {0,255}.
    Compatible with different MuJoCo versions / segmentation channel orders.
    Approach: compute both interpretations and pick the one with more valid geom-labeled pixels.
    """
    seg = np.asarray(seg)
    if seg.ndim == 3 and seg.shape[2] == 2:
        c0 = seg[..., 0].astype(np.int32, copy=False)
        c1 = seg[..., 1].astype(np.int32, copy=False)
    else:
        raise ValueError(f"segmentation output has unexpected shape: {seg.shape}")

    geom_code = int(mujoco.mjtObj.mjOBJ_GEOM)
    ngeom = int(model.ngeom)

    # geom id -> foreground lookup table
    fg = np.zeros(ngeom, dtype=np.bool_)
    fg[np.clip(fg_geom_ids, 0, ngeom - 1)] = True

    def build(objtype, objid):
        valid = (objtype == geom_code) & (objid >= 0) & (objid < ngeom)
        mask = np.zeros(objid.shape, dtype=np.uint8)
        mask[valid] = (fg[objid[valid]].astype(np.uint8) * 255)
        score = int(valid.sum())
        return mask, score

    # Assumption A: (type,id) = (c0,c1)
    mA, sA = build(c0, c1)
    # Assumption B: (type,id) = (c1,c0)
    mB, sB = build(c1, c0)

    return mB if sB > sA else mA


# ---------- math: camera intrinsics & backproject ----------
def _intrinsics(model, res_w, res_h, cid):
    fovy = float(model.cam_fovy[cid])
    fy = res_h / (2.0 * np.tan(np.deg2rad(fovy) / 2.0))
    fx = fy
    cx, cy = (res_w - 1) / 2.0, (res_h - 1) / 2.0
    return np.float32(fx), np.float32(fy), np.float32(cx), np.float32(cy)


def _cam_to_world_fast(rgb, depth, fx, fy, cx, cy, R, p, depth_max, idx_s):
    z = depth[idx_s].astype(np.float32)
    m = (z > 0) & (z < depth_max)
    if not np.any(m):
        return None, None
    u_sel = idx_s[1][m].astype(np.float32)
    v_sel = idx_s[0][m].astype(np.float32)
    z = z[m]
    x = (u_sel - cx) * z / fx
    y = -(v_sel - cy) * z / fy
    zc = -z
    cam = np.stack((x, y, zc), axis=1)
    world = cam @ R.T + p
    col = (rgb[idx_s][m] / 255.0).astype(np.float32)
    return world, col


def _render_rgbd(renderer: Renderer, data, cid: int):
    renderer.update_scene(data, camera=cid)
    renderer.enable_depth_rendering()
    depth = renderer.render()
    renderer.disable_depth_rendering()
    rgb = renderer.render()
    return rgb, depth

def _render_rgb(renderer: Renderer, data, cid: int):
    """Render RGB only (saves one render call vs _render_rgbd)."""
    renderer.update_scene(data, camera=cid)
    # Ensure we are in RGB mode
    try:
        renderer.disable_depth_rendering()
    except Exception:
        pass
    rgb = renderer.render()
    return rgb


# ---------- sampling ----------
def _generate_controls_discrete(cfg: CollectCfg, model, motor_ids):
    lo = model.actuator_ctrlrange[motor_ids, 0].astype(np.float32)
    hi = model.actuator_ctrlrange[motor_ids, 1].astype(np.float32)
    D = len(motor_ids)
    L = int(cfg.levels_per_motor)
    levels = [np.linspace(lo[i], hi[i], L, dtype=np.float32) for i in range(D)]

    # Estimate total combinations
    total = L ** D
    # If total combinations <= 2e6, build the full Cartesian grid; otherwise do random grid-quantized sampling.
    if total <= 2_000_000:
        from itertools import product
        grid = np.array(list(product(*levels)), dtype=np.float32)
        rng = np.random.default_rng(cfg.seed)
        rng.shuffle(grid, axis=0)
        assert cfg.nsample <= len(grid), "Requested nsample exceeds the total number of discrete combinations"
        return grid[:cfg.nsample]

    # Random grid-quantized sampling with strict de-duplication
    rng = np.random.default_rng(cfg.seed)
    span = hi - lo
    acc = []
    seen = set()
    while len(acc) < cfg.nsample:
        m = min(8192, cfg.nsample - len(acc))
        x = lo + span * rng.random((m, D), dtype=np.float32)
        # Quantize each dimension to the nearest grid level
        for i in range(m):
            for d in range(D):
                # levels[d] is a 1D array
                lv = levels[d]
                x[i, d] = lv[np.argmin(np.abs(lv - x[i, d]))]
        # Strict de-duplication
        for i in range(m):
            k = x[i].tobytes()
            if k not in seen:
                seen.add(k)
                acc.append(x[i].copy())
                if len(acc) >= cfg.nsample:
                    break
    return np.stack(acc, axis=0)


def _generate_controls_continuous(cfg: CollectCfg, model, motor_ids):
    rng = np.random.default_rng(cfg.seed)
    lo = model.actuator_ctrlrange[motor_ids, 0].astype(np.float32)
    hi = model.actuator_ctrlrange[motor_ids, 1].astype(np.float32)
    span = hi - lo
    D = len(motor_ids)

    n = cfg.nsample
    block = min(20000, max(1024, n // 10))
    seen = set()
    out = np.zeros((n, D), dtype=np.float32)

    def quantize(v: np.ndarray) -> np.ndarray:
        if cfg.unique_tol is None:
            return v
        q = np.round((v - lo) / cfg.unique_tol).astype(np.int64)
        return lo + q * cfg.unique_tol

    have_min_gap = (cfg.min_gap is not None and cfg.min_gap > 0)
    idx = 0
    tries = 0
    max_tries = 50 * n
    while idx < n and tries < max_tries:
        tries += 1
        m = min(block, n - idx)
        cand = lo + span * rng.random((m, D), dtype=np.float32)
        cand = quantize(cand)

        if have_min_gap:
            keep_mask = np.ones(m, dtype=bool)
            if idx > 0:
                sample_k = min(2048, idx)
                sel = rng.choice(idx, size=sample_k, replace=False)
                base = out[sel]
                chunk = 256
                for s in range(0, m, chunk):
                    e = min(m, s + chunk)
                    cc = cand[s:e][:, None, :]  # (e-s, 1, D)
                    bb = base[None, :, :]  # (1, k, D)
                    d = np.max(np.abs(cc - bb), axis=2)  # (e-s, k)
                    near = (d < cfg.min_gap).any(axis=1)
                    keep_mask[s:e] &= ~near
            cand = cand[keep_mask]
            if len(cand) == 0:
                continue

        kv = [x.tobytes() for x in cand]
        unique_new = [cand[i] for i, k in enumerate(kv) if k not in seen]
        if not unique_new:
            continue
        for x in unique_new:
            seen.add(x.tobytes())
            out[idx] = x
            idx += 1
            if idx >= n:
                break

    if idx < n:
        raise RuntimeError(
            f"Continuous sampling failed to collect {n} unique controls after {tries} attempts, "
            f"please reduce min_gap/unique_tol or decrease nsample."
        )
    return out


# ---------- stability (early-stop) ----------
def _relax_to_stable(model, data, max_steps, vel_eps, qpos_eps, win, zero_vel=True):
    if zero_vel:
        data.qvel[:] = 0
        data.qacc[:] = 0
        data.act[:] = 0
    mujoco.mj_forward(model, data)
    prev = data.qpos.copy()
    ok = 0
    vmax = 0.0;
    dq = 0.0
    for s in range(max_steps):
        mujoco.mj_step(model, data)
        vmax = float(np.max(np.abs(data.qvel)))
        dq = float(np.max(np.abs(data.qpos - prev)))
        prev[:] = data.qpos
        if vmax < vel_eps and dq < qpos_eps:
            ok += 1
            if ok >= win:
                return s + 1, True, vmax, dq
        else:
            ok = 0
    return max_steps, False, vmax, dq


# ---------- GL backend selection ----------
def _pick_gl_backend(prefer: str = "auto") -> str:
    if prefer != "auto":
        return prefer
    if sys.platform.startswith("win"):
        return "glfw"
    headless = ("DISPLAY" not in os.environ and "WAYLAND_DISPLAY" not in os.environ)
    return "egl" if headless else "glfw"


def _make_renderer_robust(model, w, h, prefer="auto", egl_quiet=True):
    if egl_quiet:
        os.environ.setdefault("LIBEGL_DEBUG", "fatal")
        os.environ.setdefault("EGL_LOG_LEVEL", "fatal")
    tried = []

    def _try(b):
        os.environ["MUJOCO_GL"] = b
        try:
            r = Renderer(model, width=w, height=h)
            print_color(f"[collect] Renderer backend: {b}")
            return r
        except Exception as e:
            tried.append((b, str(e)))
            return None

    p = _pick_gl_backend(prefer)
    order = []
    if p == "egl":
        order = ["egl", "osmesa", "glfw"]
    elif p == "glfw":
        order = ["glfw", "egl", "osmesa"]
    else:
        order = [p, "egl", "osmesa", "glfw"]
    for b in order:
        r = _try(b)
        if r is not None:
            return r
    raise RuntimeError("Renderer init failed: " + " | ".join(f"{b}:{m}" for b, m in tried))


def _auto_worker_count(wish, res_w, res_h, ncam):
    cpu = max(1, os.cpu_count() or 4)
    if wish and wish > 0:
        return max(1, min(wish, cpu))
    per_worker_gb = 0.35 + 0.008 * (res_w * res_h / 1e6) * max(1, ncam)
    avail_gb = None
    try:
        if psutil is not None:
            avail_gb = psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        avail_gb = None
    if avail_gb is None:
        return max(1, cpu // 2)
    return max(1, min(cpu, int(0.75 * avail_gb / max(0.1, per_worker_gb))))


# ---------- prewrite motors ----------
def _prewrite_all_motors(ctrls: np.ndarray, out_dir: Path, start: int, act_names: list[str]):
    ensure_dir(out_dir)
    kv = [x.tobytes() for x in ctrls]
    if len(kv) != len(set(kv)):
        raise AssertionError("Duplicate motors found in pre-generated controls; reduce --min_gap or adjust sampling parameters")
    D = ctrls.shape[1]
    for i, ctrl in enumerate(ctrls, start):
        k = f"{i:06d}.json"
        p = out_dir / k
        if not p.exists():
            # New format: explicit control array in XML actuator order; names included for traceability
            obj = {"ctrl": [float(v) for v in ctrl.tolist()], "actuator_names": act_names}
            with open(p, "w") as f:
                json.dump(obj, f, indent=4, ensure_ascii=False)


# ---------- worker ----------
def _collect_worker(args):
    cfg_dict, indices, position = args
    from tqdm import tqdm as _tqdm

    xml = cfg_dict["xml"]
    res_w = cfg_dict["res_w"]
    res_h = cfg_dict["res_h"]
    depth_max = cfg_dict["depth_max"]
    stride = cfg_dict["stride"]
    seed = cfg_dict["seed"]
    out_pcd_dir = Path(cfg_dict["out_pcd_dir"])
    out_json_dir = Path(cfg_dict["out_json_dir"])

    # RGB export options (optional)
    out_rgb_dir = cfg_dict.get("out_rgb_dir", None)
    out_rgb_dir = Path(out_rgb_dir) if out_rgb_dir else None
    rgb_format = str(cfg_dict.get("rgb_format", "png"))
    rgb_only = bool(cfg_dict.get("rgb_only", False))
    rgb_cams = cfg_dict.get("rgb_cams", None)

    export_pcd = not rgb_only
    export_rgb = out_rgb_dir is not None

    backend = cfg_dict["backend"]

    egl_quiet = cfg_dict["egl_quiet"]
    sim_steps = cfg_dict["sim_steps"]
    use_relax = cfg_dict["use_relax"]
    relax_max = cfg_dict["relax_max_steps"]
    vel_eps = cfg_dict["stable_vel_eps"]
    qpos_eps = cfg_dict["stable_qpos_eps"]
    win = cfg_dict["stable_win"]
    zero_vel = cfg_dict["zero_vel_each_ctrl"]
    resume = cfg_dict["resume"]

    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    if export_pcd and o3d is None:
        raise RuntimeError("Point cloud export requires open3d, but it is not installed in this environment.")

    renderer = _make_renderer_robust(model, res_w, res_h, prefer=backend, egl_quiet=egl_quiet)

    motor_ids, motor_names, D = _detect_actuators(model)
    fg_geom_ids = _infer_mask_geom_ids(model) if export_rgb else None

    # Camera selection
    _, all_cam_names = _detect_cameras(model)
    cam_ids = _select_camera_ids(rgb_cams, all_cam_names)

    # Pre-create camera folders (safe in multiprocess)
    if export_rgb:
        for cid in cam_ids:
            ensure_dir(out_rgb_dir / all_cam_names[cid])

    # Only needed for point cloud backprojection
    if export_pcd:
        U_full, V_full = np.meshgrid(np.arange(res_w, dtype=np.float32),
                                     np.arange(res_h, dtype=np.float32))
        U_s = U_full[::stride, ::stride].ravel().astype(np.intp)
        V_s = V_full[::stride, ::stride].ravel().astype(np.intp)
        idx_s = (V_s, U_s)
    else:
        idx_s = None

    rng = np.random.default_rng(seed + position * 1_000_003)
    bar = _tqdm(indices, position=position, ncols=120, desc=f"worker-{position}", leave=True)
    for k in bar:
        # Decide per-sample what to export (support resume with partial outputs)
        pcd_path = (out_pcd_dir / f"{k:06d}.ply") if export_pcd else None
        need_pcd = export_pcd and (not (resume and pcd_path.exists()))

        if export_rgb:
            rgb_paths = [out_rgb_dir / all_cam_names[cid] / f"{k:06d}.{rgb_format}" for cid in cam_ids]
            mask_paths = [out_rgb_dir / all_cam_names[cid] / f"{k:06d}_mask.png" for cid in cam_ids]
            need_rgb = not (resume and all(r.exists() and m.exists() for r, m in zip(rgb_paths, mask_paths)))
        else:
            rgb_paths, mask_paths = [], []
            need_rgb = False

        if (not need_pcd) and (not need_rgb):
            continue

        # --- load ctrl from prewritten json ---
        with open(out_json_dir / f"{k:06d}.json", "r") as f:
            obj = json.load(f)

        if isinstance(obj, (list, tuple)):
            ctrl = np.array(obj, dtype=np.float32).reshape(-1)
        elif isinstance(obj, dict):
            if "ctrl" in obj:
                ctrl = np.array(obj["ctrl"], dtype=np.float32).reshape(-1)
            else:
                # Legacy format compatibility: motor1..motorD
                vals = []
                for j in range(1, D + 1):
                    key = f"motor{j}"
                    if key not in obj:
                        raise KeyError(f"{k:06d}.json is missing key {key}")
                    vals.append(obj[key])
                ctrl = np.array(vals, dtype=np.float32).reshape(-1)
        else:
            raise ValueError("Unsupported JSON motor format")
        assert len(ctrl) == D, f"{k:06d}.json: expected {D} dims, got {len(ctrl)}"

        data.ctrl[motor_ids] = ctrl

        if use_relax:
            _relax_to_stable(model, data, relax_max, vel_eps, qpos_eps, win, zero_vel=zero_vel)
        else:
            for _ in range(sim_steps):
                mujoco.mj_step(model, data)

        pts_all, col_all = [], []
        for cid in cam_ids:
            # per-camera output paths
            if export_rgb:
                out_path = out_rgb_dir / all_cam_names[cid] / f"{k:06d}.{rgb_format}"
                mask_path = out_rgb_dir / all_cam_names[cid] / f"{k:06d}_mask.png"
                need_rgb_img = need_rgb and ((not resume) or (not out_path.exists()))
                need_mask_img = need_rgb and ((not resume) or (not mask_path.exists()))
            else:
                out_path = None
                mask_path = None
                need_rgb_img = False
                need_mask_img = False

            # Skip if we do not need point cloud nor RGB/mask for this sample
            if (not need_pcd) and (not need_rgb_img) and (not need_mask_img):
                continue

            # ---- Render (minimize render calls) ----
            if need_pcd:
                rgb, d = _render_rgbd(renderer, data, cid)
            elif need_rgb_img:
                rgb = _render_rgb(renderer, data, cid)
                d = None
            else:
                rgb = None
                d = None

            # ---- Save RGB ----
            if need_rgb_img and out_path is not None and rgb is not None:
                _save_rgb_image(rgb, out_path, fmt=rgb_format)

            # ---- Save Mask ----
            if need_mask_img and mask_path is not None:
                seg = _render_segmentation(renderer, data, cid)
                mask = _seg_to_binary_mask(seg, model, fg_geom_ids)
                _save_mask_image(mask, mask_path)

            # ---- Point cloud ----
            if need_pcd:
                fx, fy, cx, cy = _intrinsics(model, res_w, res_h, cid)
                R = data.cam_xmat[cid].reshape(3, 3).astype(np.float32)
                p = data.cam_xpos[cid].astype(np.float32)
                pts, col = _cam_to_world_fast(rgb, d, fx, fy, cx, cy, R, p, depth_max, idx_s)
                if pts is not None:
                    pts_all.append(pts)
                    col_all.append(col)

        if need_pcd and pts_all:
            pts = np.concatenate(pts_all, axis=0)
            col = np.concatenate(col_all, axis=0)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(col)
            o3d.io.write_point_cloud(str(pcd_path), pcd, write_ascii=False)

    renderer.close()


# ---------- public entry ----------
def collect_stage(cfg: CollectCfg):
    if mujoco is None or Renderer is None:
        raise RuntimeError("Stage 1 requires mujoco and mujoco.renderer.Renderer.")

    # --- RGB export options are optional; keep backward compatible even if cfg has no fields ---
    out_rgb_dir = getattr(cfg, "out_rgb_dir", None)
    rgb_format = str(getattr(cfg, "rgb_format", "png"))
    rgb_only = bool(getattr(cfg, "rgb_only", False))
    rgb_cams = getattr(cfg, "rgb_cams", None)

    export_pcd = not rgb_only
    export_rgb = out_rgb_dir is not None
    if export_rgb and not isinstance(out_rgb_dir, Path):
        out_rgb_dir = Path(str(out_rgb_dir))

    if export_pcd and o3d is None:
        raise RuntimeError("Point cloud export requires open3d, but it is not installed in this environment.")

    if export_pcd:
        ensure_dir(cfg.out_pcd_dir)
    ensure_dir(cfg.out_json_dir)
    if export_rgb:
        ensure_dir(out_rgb_dir)

    # Probe the model to discover camera count, control ranges, etc.
    model_probe = mujoco.MjModel.from_xml_path(cfg.xml)

    # Camera selection
    _, all_cam_names = _detect_cameras(model_probe)
    cam_ids = _select_camera_ids(rgb_cams, all_cam_names)
    ncam = int(len(cam_ids))

    if export_rgb:
        for cid in cam_ids:
            ensure_dir(out_rgb_dir / all_cam_names[cid])

    print_color(
        f"[collect] export: pcd={'ON' if export_pcd else 'OFF'}, rgb={'ON' if export_rgb else 'OFF'} "
        f"(rgb_only={rgb_only}, cams={len(cam_ids)}/{len(all_cam_names)}, fmt={rgb_format})"
    )


    # === Compute the final worker count nworkers (supports automatic mode when --workers<=0) ===
    if cfg.workers <= 0:
        nworkers = _auto_worker_count(cfg.workers, cfg.res_w, cfg.res_h, ncam)
    else:
        nworkers = int(cfg.workers)
    # Clamp by sample count and ensure at least 1
    nworkers = max(1, min(int(nworkers), int(cfg.nsample)))
    how = "auto" if cfg.workers <= 0 else "explicit"
    print_color(f"[collect] multiprocessing: {'ON' if nworkers > 1 else 'OFF'} "
                f"(workers={nworkers}, mode={how}, ctx={cfg.ctx})")

    # Generate globally-unique motor controls and prewrite JSON to disk for multiprocess safety
    motor_ids_probe, motor_names_probe, D = _detect_actuators(model_probe)
    print_color(f"[collect] detected motors: D={D} (segments={D // 3})")
    if cfg.sampling == "discrete":
        all_ctrls = _generate_controls_discrete(cfg, model_probe, motor_ids_probe)
    else:
        all_ctrls = _generate_controls_continuous(cfg, model_probe, motor_ids_probe)

    if cfg.resume and len(list(Path(cfg.out_pcd_dir).glob("*.ply"))) > 0:
        print_color("[collect] resume is on: skipping samples with existing .ply outputs.")

    _prewrite_all_motors(all_ctrls, cfg.out_json_dir, cfg.start_index, motor_names_probe)

    # ===== Serial path =====
    if nworkers == 1:
        data = mujoco.MjData(model_probe)
        mujoco.mj_forward(model_probe, data)
        renderer = _make_renderer_robust(
            model_probe, cfg.res_w, cfg.res_h,
            prefer=cfg.backend, egl_quiet=cfg.egl_quiet
        )

        motor_ids = motor_ids_probe

        if export_pcd:
            U_full, V_full = np.meshgrid(
                np.arange(cfg.res_w, dtype=np.float32),
                np.arange(cfg.res_h, dtype=np.float32)
            )
            U_s = U_full[::cfg.stride, ::cfg.stride].ravel().astype(np.intp)
            V_s = V_full[::cfg.stride, ::cfg.stride].ravel().astype(np.intp)
            idx_s = (V_s, U_s)
        else:
            idx_s = None

        from tqdm import tqdm as _tqdm
        for i in _tqdm(range(cfg.nsample), ncols=100, desc="collect"):
            k = cfg.start_index + i
            pcd_path = (Path(cfg.out_pcd_dir) / f"{k:06d}.ply") if export_pcd else None
            need_pcd = export_pcd and (not (cfg.resume and pcd_path.exists()))

            if export_rgb:
                rgb_paths = [out_rgb_dir / all_cam_names[cid] / f"{k:06d}.{rgb_format}" for cid in cam_ids]
                mask_paths = [out_rgb_dir / all_cam_names[cid] / f"{k:06d}_mask.png" for cid in cam_ids]
                # Consider the sample complete only if BOTH RGB and mask exist
                need_rgb = not (cfg.resume and all(r.exists() and m.exists() for r, m in zip(rgb_paths, mask_paths)))
            else:
                rgb_paths, mask_paths = [], []
                need_rgb = False

            if (not need_pcd) and (not need_rgb):
                continue

            ctrl = all_ctrls[i]
            data.ctrl[motor_ids] = ctrl

            # Stability early-stop or fixed-step simulation
            if cfg.relax_max_steps and cfg.stable_win:
                _relax_to_stable(
                    model_probe, data, cfg.relax_max_steps,
                    cfg.stable_vel_eps, cfg.stable_qpos_eps, cfg.stable_win,
                    zero_vel=cfg.zero_vel_each_ctrl
                )
            else:
                for _ in range(cfg.sim_steps):
                    mujoco.mj_step(model_probe, data)

            # Render -> merge point cloud -> save
            pts_all, col_all = [], []
            fg_geom_ids = _infer_mask_geom_ids(model_probe) if export_rgb else None
            for cid in cam_ids:
                if export_rgb:
                    out_path = out_rgb_dir / all_cam_names[cid] / f"{k:06d}.{rgb_format}"
                    mask_path = out_rgb_dir / all_cam_names[cid] / f"{k:06d}_mask.png"

                    need_rgb_img = need_rgb and ((not cfg.resume) or (not out_path.exists()))
                    need_mask_img = need_rgb and ((not cfg.resume) or (not mask_path.exists()))
                else:
                    out_path = None
                    mask_path = None
                    need_rgb_img = False
                    need_mask_img = False

                # Skip if we do not need point cloud nor this camera's RGB/mask
                if (not need_pcd) and (not need_rgb_img) and (not need_mask_img):
                    continue

                # ---- Render (minimize render calls) ----
                if need_pcd:
                    rgb, d = _render_rgbd(renderer, data, cid)
                elif need_rgb_img:
                    rgb = _render_rgb(renderer, data, cid)
                    d = None
                else:
                    rgb = None
                    d = None

                # ---- Save RGB ----
                if need_rgb_img and out_path is not None and rgb is not None:
                    _save_rgb_image(rgb, out_path, fmt=rgb_format)

                # ---- Save Mask (bound to RGB export) ----
                if need_mask_img and mask_path is not None:
                    seg = _render_segmentation(renderer, data, cid)
                    mask = _seg_to_binary_mask(seg, model_probe, fg_geom_ids)
                    _save_mask_image(mask, mask_path)

                # Point cloud
                if need_pcd:
                    fx, fy, cx, cy = _intrinsics(model_probe, cfg.res_w, cfg.res_h, cid)
                    R = data.cam_xmat[cid].reshape(3, 3).astype(np.float32)
                    p = data.cam_xpos[cid].astype(np.float32)
                    pts, col = _cam_to_world_fast(rgb, d, fx, fy, cx, cy, R, p, cfg.depth_max, idx_s)
                    if pts is not None:
                        pts_all.append(pts)
                        col_all.append(col)

            if need_pcd and pts_all:
                pts = np.concatenate(pts_all, axis=0)
                col = np.concatenate(col_all, axis=0)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts)
                pcd.colors = o3d.utility.Vector3dVector(col)
                o3d.io.write_point_cloud(str(pcd_path), pcd, write_ascii=False)

        renderer.close()
        print_color("✅ [collect] done.")
        return

    # ===== Parallel path =====
    all_indices = list(range(cfg.start_index, cfg.start_index + cfg.nsample))
    splits = np.array_split(all_indices, nworkers)
    base_cfg = dict(
        xml=cfg.xml, res_w=cfg.res_w, res_h=cfg.res_h, depth_max=cfg.depth_max, stride=cfg.stride,
        seed=cfg.seed, out_pcd_dir=str(cfg.out_pcd_dir), out_json_dir=str(cfg.out_json_dir),
        backend=cfg.backend, egl_quiet=cfg.egl_quiet, sim_steps=cfg.sim_steps,
        use_relax=bool(cfg.relax_max_steps and cfg.stable_win),
        relax_max_steps=cfg.relax_max_steps, stable_vel_eps=cfg.stable_vel_eps,
        stable_qpos_eps=cfg.stable_qpos_eps, stable_win=cfg.stable_win,
        zero_vel_each_ctrl=cfg.zero_vel_each_ctrl, resume=cfg.resume,

        # RGB export options
        out_rgb_dir=str(out_rgb_dir) if export_rgb else None,
        rgb_format=str(rgb_format),
        rgb_only=bool(rgb_only),
        rgb_cams=rgb_cams,
    )

    ctx = mp.get_context(cfg.ctx if cfg.ctx in ("spawn", "fork") else "spawn")
    tasks = [(base_cfg, list(map(int, arr)), rank) for rank, arr in enumerate(splits)]
    with ctx.Pool(processes=nworkers) as pool:
        for _ in pool.imap_unordered(_collect_worker, tasks, chunksize=1):
            pass
    print_color("✅ [collect] done (parallel).")

