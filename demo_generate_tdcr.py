from __future__ import annotations

import os
import json
import glob
import csv
import argparse
import warnings
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import h5py
from tqdm import tqdm

# Your model definitions (must include VelocityNet and HybridMLP)
from models import VelocityNet

try:
    from models import HybridMLP
except Exception:
    HybridMLP = None


# ============================================================
# Chamfer Distance (CD): compiled chamfer_3D only (no fallback)
# ============================================================
_CHAMFER_EXT = None


def _load_chamfer_ext():
    """Lazy-import compiled chamfer_3D extension. No fallback."""
    global _CHAMFER_EXT
    if _CHAMFER_EXT is not None:
        return _CHAMFER_EXT

    import importlib

    try:
        _CHAMFER_EXT = importlib.import_module("chamfer_3D")
    except Exception as e:
        raise ImportError(
            f"[Chamfer] Failed to import compiled extension 'chamfer_3D'. "
            f"Please make sure it is built and importable. Original error: {e}"
        ) from e

    ext_path = getattr(_CHAMFER_EXT, "__file__", "<unknown>")
    print(f"[Chamfer] Using compiled chamfer_3D extension: {ext_path}")
    return _CHAMFER_EXT


@torch.no_grad()
def chamfer_l2(pred_xyz: torch.Tensor, gt_xyz: torch.Tensor) -> torch.Tensor:
    """
    pred_xyz: (B, N, 3)
    gt_xyz:   (B, M, 3)
    return:   (B,) squared-L2 Chamfer = mean_x min_y ||x-y||^2 + mean_y min_x ||y-x||^2

    NOTE: This script requires the compiled CUDA extension (no torch fallback).
    """

    if pred_xyz.ndim != 3 or gt_xyz.ndim != 3:
        raise ValueError(f"Chamfer expects 3D tensors (B,N,3). got pred={pred_xyz.shape} gt={gt_xyz.shape}")
    if pred_xyz.shape[-1] != 3 or gt_xyz.shape[-1] != 3:
        raise ValueError("Chamfer expects xyz only (last dim == 3).")
    if pred_xyz.shape[0] != gt_xyz.shape[0]:
        raise ValueError(f"Batch mismatch: pred B={pred_xyz.shape[0]} gt B={gt_xyz.shape[0]}")
    if (not pred_xyz.is_cuda) or (not gt_xyz.is_cuda):
        raise ValueError("Chamfer in this script requires CUDA tensors (compiled chamfer_3D).")
    if pred_xyz.device != gt_xyz.device:
        raise ValueError(f"Device mismatch: pred={pred_xyz.device} gt={gt_xyz.device}")

    ext = _load_chamfer_ext()

    B, N, _ = pred_xyz.shape
    _, M, _ = gt_xyz.shape

    x = pred_xyz.contiguous().to(dtype=torch.float32)
    y = gt_xyz.contiguous().to(dtype=torch.float32)

    d1 = torch.empty((B, N), device=x.device, dtype=torch.float32)
    d2 = torch.empty((B, M), device=x.device, dtype=torch.float32)
    i1 = torch.empty((B, N), device=x.device, dtype=torch.int32)
    i2 = torch.empty((B, M), device=x.device, dtype=torch.int32)

    _ = ext.forward(x, y, d1, d2, i1, i2)
    return (d1.mean(dim=1) + d2.mean(dim=1)).to(dtype=pred_xyz.dtype)


# ============================================================
# Earth Mover Distance (EMD): compiled emd_ext only (no fallback)
# ============================================================
_EMD_FN = None


def _load_emd_fn():
    """Lazy-import compiled emd_ext and earth_mover_distance. No fallback."""
    global _EMD_FN
    if _EMD_FN is not None:
        return _EMD_FN

    import importlib

    try:
        emd_ext = importlib.import_module("emd_ext")
    except Exception as e:
        raise ImportError(
            f"[EMD] Failed to import compiled extension 'emd_ext'. "
            f"Please make sure it is built and importable. Original error: {e}"
        ) from e

    try:
        # This wrapper will prefer the compiled emd_ext when available.
        # PyTorch>=2.1 will emit FutureWarning for torch.cuda.amp.custom_fwd/custom_bwd
        # used inside third_party/PyTorchEMD/emd.py. We silence ONLY those two warnings
        # to keep logs clean (behavior is unchanged).
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                category=FutureWarning,
                message=r'.*torch\.cuda\.amp\.custom_fwd.*',
            )
            warnings.filterwarnings(
                'ignore',
                category=FutureWarning,
                message=r'.*torch\.cuda\.amp\.custom_bwd.*',
            )
            from third_party.PyTorchEMD.emd import earth_mover_distance
    except Exception as e:
        raise ImportError(
            f"[EMD] Failed to import 'earth_mover_distance' from third_party.PyTorchEMD.emd. "
            f"Original error: {e}"
        ) from e

    ext_path = getattr(emd_ext, "__file__", "<unknown>")
    print(f"[EMD] Using compiled emd_ext extension: {ext_path}")
    _EMD_FN = earth_mover_distance
    return _EMD_FN


@torch.no_grad()
def emd_distance(pred_xyz: torch.Tensor, gt_xyz: torch.Tensor) -> torch.Tensor:
    """
    pred_xyz, gt_xyz: (B, N, 3) with SAME N
    return: (B,) EMD

    NOTE: This script requires compiled emd_ext (no fallback).
    """

    if pred_xyz.ndim != 3 or gt_xyz.ndim != 3:
        raise ValueError(f"EMD expects 3D tensors (B,N,3). got pred={pred_xyz.shape} gt={gt_xyz.shape}")
    if pred_xyz.shape[-1] != 3 or gt_xyz.shape[-1] != 3:
        raise ValueError("EMD expects xyz only (last dim == 3).")
    if pred_xyz.shape[0] != gt_xyz.shape[0]:
        raise ValueError(f"Batch mismatch: pred B={pred_xyz.shape[0]} gt B={gt_xyz.shape[0]}")
    if pred_xyz.shape[1] != gt_xyz.shape[1]:
        raise ValueError(f"EMD requires same number of points. pred N={pred_xyz.shape[1]} gt N={gt_xyz.shape[1]}")
    if (not pred_xyz.is_cuda) or (not gt_xyz.is_cuda):
        raise ValueError("EMD in this script requires CUDA tensors (compiled emd_ext).")
    if pred_xyz.device != gt_xyz.device:
        raise ValueError(f"Device mismatch: pred={pred_xyz.device} gt={gt_xyz.device}")

    fn = _load_emd_fn()

    x = pred_xyz.contiguous().to(dtype=torch.float32)
    y = gt_xyz.contiguous().to(dtype=torch.float32)

    d = fn(x, y, transpose=False)
    if not torch.is_tensor(d):
        raise RuntimeError(f"earth_mover_distance returned non-tensor: {type(d)}")
    if d.ndim == 0:
        d = d.view(1)
    return d.to(dtype=pred_xyz.dtype)


# ============================================================
# Simple PLY writers (ASCII)
# ============================================================
def write_ply_xyz(path: str, points_xyz: np.ndarray):
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"write_ply_xyz expects (N,3). got {points_xyz.shape}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points_xyz.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        np.savetxt(f, points_xyz, fmt="%.6f %.6f %.6f")


def write_ply_xyzrgb(path: str, points_xyzrgb: np.ndarray):
    """points_xyzrgb: (N,6) with rgb in [0,1]"""
    points_xyzrgb = np.asarray(points_xyzrgb, dtype=np.float32)
    if points_xyzrgb.ndim != 2 or points_xyzrgb.shape[1] != 6:
        raise ValueError(f"write_ply_xyzrgb expects (N,6). got {points_xyzrgb.shape}")
    xyz = points_xyzrgb[:, :3]
    rgb = points_xyzrgb[:, 3:6]
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb255 = (rgb * 255.0).clip(0.0, 255.0).astype(np.uint8)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(xyz, rgb255):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


# ============================================================
# Simple PLY reader (ASCII; matches writers above)
# ============================================================
def _read_ply_vertex_count_ascii(path: str) -> int:
    """Fast header-only vertex count reader for ASCII PLY."""
    with open(path, "r") as f:
        first = f.readline().strip()
        if first != "ply":
            raise ValueError(f"Not a PLY file: {path}")
        fmt = f.readline().strip()
        if not fmt.startswith("format ascii"):
            raise ValueError(f"Only ASCII PLY supported in this script. got: '{fmt}' in {path}")
        n_verts: Optional[int] = None
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"PLY header ended unexpectedly: {path}")
            line = line.strip()
            if line.startswith("element vertex"):
                parts = line.split()
                n_verts = int(parts[-1])
            if line == "end_header":
                break
        if n_verts is None:
            raise ValueError(f"Missing 'element vertex' in PLY header: {path}")
        return int(n_verts)


def read_ply_xyz(path: str) -> np.ndarray:
    """Read ASCII PLY (written by this script) and return xyz float32 (N,3)."""
    n_verts = _read_ply_vertex_count_ascii(path)
    pts = np.empty((n_verts, 3), dtype=np.float32)
    with open(path, "r") as f:
        # skip header
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"PLY header ended unexpectedly: {path}")
            if line.strip() == "end_header":
                break
        # read vertices
        for i in range(n_verts):
            line = f.readline()
            if not line:
                raise ValueError(f"PLY ended unexpectedly while reading vertices: {path}")
            parts = line.strip().split()
            if len(parts) < 3:
                raise ValueError(f"Invalid vertex line in {path}: '{line.strip()}'")
            pts[i, 0] = float(parts[0])
            pts[i, 1] = float(parts[1])
            pts[i, 2] = float(parts[2])
    return pts


# ============================================================
# Eval scale / IO helpers
# ============================================================

def load_eval_center_scale(
    eval_norm_json: str | None,
    eval_scale: float | None = None,
    eval_center: List[float] | None = None,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """Resolve (center, scale) for evaluation.

    We assume dataset normalization is:
        norm = (raw - center) / scale
    so the inverse (restore original units) is:
        raw = norm * scale + center

    Priority:
      1) eval_norm_json (tdcr_pipeline/tdcr_to_vsm global_norm_*.json)
      2) eval_scale + eval_center
      3) default: center=[0,0,0], scale=1
    """
    center = np.zeros(3, dtype=np.float32)
    scale = 1.0
    info: Dict[str, Any] = {
        'source': 'default',
        'eval_norm_json': eval_norm_json,
        'eval_scale_arg': eval_scale,
        'eval_center_arg': eval_center,
    }

    if eval_norm_json:
        with open(eval_norm_json, 'r') as f:
            obj = json.load(f)
        if not isinstance(obj, dict) or len(obj) == 0:
            raise ValueError(f"Bad eval_norm_json (expect dict): {eval_norm_json}")
        if 'all' in obj and isinstance(obj['all'], dict):
            c = obj['all'].get('center', None)
            s = obj['all'].get('scale', None)
        else:
            k0 = next(iter(obj.keys()))
            sub = obj[k0]
            if not isinstance(sub, dict):
                raise ValueError(f"Bad eval_norm_json[{k0}] (expect dict): {eval_norm_json}")
            c = sub.get('center', None)
            s = sub.get('scale', None)
        if c is None or s is None:
            raise ValueError(f"eval_norm_json missing center/scale: {eval_norm_json}")
        center = np.asarray(c, dtype=np.float32).reshape(3)
        scale = float(s)
        info['source'] = 'eval_norm_json'
    else:
        if eval_center is not None:
            center = np.asarray(eval_center, dtype=np.float32).reshape(3)
            info['source'] = 'eval_center_arg'
        if eval_scale is not None:
            scale = float(eval_scale)
            info['source'] = 'eval_scale_arg' if info['source'] == 'default' else info['source']

    if not np.isfinite(center).all():
        raise ValueError(f"Invalid eval center: {center}")
    if not (np.isfinite(scale) and scale > 0):
        raise ValueError(f"Invalid eval scale: {scale}")

    info['center'] = center.tolist()
    info['scale'] = float(scale)
    return center.astype(np.float32), float(scale), info


def apply_eval_transform_np(xyz: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    """Restore original units: raw = norm * scale + center."""
    xyz = np.asarray(xyz, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32).reshape(1, 3)
    return xyz * float(scale) + center


def apply_eval_transform_torch(xyz: torch.Tensor, center_t: torch.Tensor, scale: float) -> torch.Tensor:
    """Torch version of restore: raw = norm * scale + center.

    center_t should be broadcastable to xyz (e.g. shape (1,1,3)).
    """
    return xyz * float(scale) + center_t


def resolve_gt_pred_and_out_dirs(args) -> Tuple[str, str, str]:
    """Resolve (gt_dir, pred_dir, metrics_out).

    Recompute mode supports:
      - --gt_dir + --pred_dir (explicit)
      - --pairs_root (expects pairs_root/gt and pairs_root/pred)
      - fallback to --demo_out (expects demo_out/gt and demo_out/pred)

    metrics_out default:
      - if --metrics_out is set -> that
      - else -> pairs_root (recompute) or demo_out (generation)
    """
    # Explicit dirs override everything
    if getattr(args, 'gt_dir', None) or getattr(args, 'pred_dir', None):
        if not (getattr(args, 'gt_dir', None) and getattr(args, 'pred_dir', None)):
            raise ValueError('Must provide both --gt_dir and --pred_dir when using explicit dirs.')
        gt_dir = str(args.gt_dir)
        pred_dir = str(args.pred_dir)
        pair_root = os.path.commonpath([os.path.abspath(gt_dir), os.path.abspath(pred_dir)])
    else:
        pair_root = getattr(args, 'pairs_root', None) or getattr(args, 'demo_out', None)
        if pair_root is None:
            raise ValueError('Missing input folders: set --pairs_root or --demo_out (or --gt_dir/--pred_dir).')
        gt_dir = os.path.join(str(pair_root), 'gt')
        pred_dir = os.path.join(str(pair_root), 'pred')

    if not os.path.isdir(gt_dir) or not os.path.isdir(pred_dir):
        raise FileNotFoundError(
            f"Expect folders: '{gt_dir}' and '{pred_dir}'. "
            "(Each should contain matching .ply filenames.)"
        )

    metrics_out = getattr(args, 'metrics_out', None)
    if metrics_out is None or str(metrics_out).strip() == '':
        # default to input root
        metrics_out = pair_root
    metrics_out = str(metrics_out)
    os.makedirs(metrics_out, exist_ok=True)
    return gt_dir, pred_dir, metrics_out


# ============================================================
# Samplers
# ============================================================
@torch.no_grad()
def euler_sampler(
    net: torch.nn.Module,
    x0: torch.Tensor,
    cond: Optional[torch.Tensor],
    steps: int,
    guidance_scale: float,
    clamp_rgb: bool = True,
) -> torch.Tensor:
    """Euler sampler for dx/dt = v(x,t,cond), integrate t:0->1"""
    dt = 1.0 / float(steps)
    x = x0
    for i in range(steps):
        t = torch.full((x.shape[0],), (i + 0.5) * dt, device=x.device, dtype=x.dtype)
        v = net.guided_velocity(x, t, cond, guidance_scale=guidance_scale)
        x = x + v * dt
        if clamp_rgb and x.shape[-1] == 6:
            x[..., 3:] = x[..., 3:].clamp(0.0, 1.0)
    return x


@torch.no_grad()
def heun_sampler(
    net: torch.nn.Module,
    x0: torch.Tensor,
    cond: Optional[torch.Tensor],
    steps: int,
    guidance_scale: float,
    clamp_rgb: bool = True,
) -> torch.Tensor:
    """Heun / RK2 sampler for dx/dt = v(x,t,cond), integrate t:0->1"""
    dt = 1.0 / float(steps)
    x = x0
    B = x.shape[0]
    for i in range(steps):
        t0 = torch.full((B,), i * dt, device=x.device, dtype=x.dtype)
        v1 = net.guided_velocity(x, t0, cond, guidance_scale=guidance_scale)
        x_euler = x + v1 * dt

        t1 = torch.full((B,), (i + 1) * dt, device=x.device, dtype=x.dtype)
        v2 = net.guided_velocity(x_euler, t1, cond, guidance_scale=guidance_scale)

        x = x + 0.5 * dt * (v1 + v2)

        if clamp_rgb and x.shape[-1] == 6:
            x[..., 3:] = x[..., 3:].clamp(0.0, 1.0)
    return x


# ============================================================
# H5 utils
# ============================================================
def find_h5_files(data_dir: str, split: str) -> List[str]:
    patterns = [
        os.path.join(data_dir, split, "*.h5"),
        os.path.join(data_dir, split, "*.hdf5"),
        os.path.join(data_dir, f"{split}*.h5"),
        os.path.join(data_dir, f"{split}*.hdf5"),
        os.path.join(data_dir, "*.h5"),
        os.path.join(data_dir, "*.hdf5"),
    ]
    files: List[str] = []
    for p in patterns:
        files.extend(glob.glob(p))
    return sorted(set(files))


def _pick_points_key(f: h5py.File, prefer_norm: bool) -> str:
    if prefer_norm and ("data_norm" in f):
        return "data_norm"
    if "data" in f:
        return "data"
    if "data_norm" in f:
        return "data_norm"
    raise KeyError(f"H5 missing points key. Need data_norm/data. keys={list(f.keys())}")


def _pick_cond_key(f: h5py.File) -> str:
    # compat: motors_norm / motor_norm / motors
    if "motors_norm" in f:
        return "motors_norm"
    if "motor_norm" in f:
        return "motor_norm"
    if "motors" in f:
        return "motors"
    raise KeyError(f"H5 missing motor key. Need motors_norm/motor_norm/motors. keys={list(f.keys())}")


def _pick_rgb_key(f: h5py.File, rgb_key: str) -> str:
    if rgb_key in f:
        return rgb_key
    if "rgb" in f:
        return "rgb"
    raise KeyError(f"H5 missing rgb key. want={rgb_key}, keys={list(f.keys())}")


def rgb_to_01(rgb: np.ndarray) -> np.ndarray:
    """
    Accept rgb in:
      - uint8 [0,255]
      - float [0,1]
      - (optional) float [-1,1] (legacy) -> map to [0,1]
    Return float32 in [0,1].
    """

    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.size == 0:
        return rgb
    mx = float(np.max(rgb))
    mn = float(np.min(rgb))
    if mx > 1.0:
        rgb = rgb / 255.0
    elif mn < 0.0:
        # legacy [-1,1] -> [0,1]
        rgb = (rgb + 1.0) * 0.5
    return np.clip(rgb, 0.0, 1.0)


# ============================================================
# Subsampling
# ============================================================
def subsample_np(points: np.ndarray, k: Optional[int], rng: np.random.RandomState) -> np.ndarray:
    if k is None or k <= 0:
        return points
    n = int(points.shape[0])
    if k >= n:
        return points
    idx = rng.choice(n, size=k, replace=False)
    return points[idx]


def subsample_torch_per_example(x: torch.Tensor, k: Optional[int], rng: np.random.RandomState) -> torch.Tensor:
    """x: (B,N,C) -> (B,k,C) using numpy rng for max compatibility"""
    if k is None or k <= 0:
        return x
    B, N, C = x.shape
    if k >= N:
        return x
    out: List[torch.Tensor] = []
    for b in range(B):
        idx_np = rng.choice(N, size=k, replace=False).astype(np.int64)
        idx = torch.from_numpy(idx_np).to(device=x.device, dtype=torch.long)
        out.append(x[b, idx, :])
    return torch.stack(out, dim=0)


# ============================================================
# Utils: batching & CSV
# ============================================================
def batched(items: List[Any], batch_size: int):
    """Yield successive list-batches from a list."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def select_subset_indices(total: int, eval_fraction: float, eval_max_samples: int, eval_seed: int) -> List[int]:
    if total <= 0:
        return []
    if not (0.0 < float(eval_fraction) <= 1.0):
        raise ValueError(f"--eval_fraction must be in (0,1], got {eval_fraction}")
    k = total
    if float(eval_fraction) < 1.0:
        k = max(1, int(round(total * float(eval_fraction))))
    if int(eval_max_samples) > 0:
        k = min(k, int(eval_max_samples))

    if k >= total:
        return list(range(total))

    rng = np.random.RandomState(int(eval_seed))
    sel = rng.choice(total, size=k, replace=False)
    sel = np.sort(sel).tolist()
    return [int(x) for x in sel]


def write_metrics_csv(csv_path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not rows:
        # still write empty file with header?
        with open(csv_path, "w", newline="") as f:
            f.write("name,cd,emd\n")
        return
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# ============================================================
# Model backbone inference & builder
# ============================================================
def infer_backbone(ckpt: Dict[str, Any]) -> str:
    args = ckpt.get("args", {}) or {}
    pf = args.get("pf_backbone", None)
    if pf in ("mlp", "hybrid"):
        return pf

    sd = ckpt.get("model", {}) or {}
    keys = list(sd.keys())
    if any(k.startswith("ctx_net.") for k in keys) or any(k.startswith("head.") for k in keys):
        return "hybrid"
    return "mlp"


def _to_int_list(x, default: List[int]) -> List[int]:
    if x is None:
        return list(default)
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    return [int(x)]


def build_model_from_ckpt_args(backbone: str, ckpt_args: Dict[str, Any]) -> torch.nn.Module:
    cond_dim = int(ckpt_args.get("cond_dim", 0))
    width = int(ckpt_args.get("width", 512))
    depth = int(ckpt_args.get("depth", 6))
    emb_dim = int(ckpt_args.get("emb_dim", 256))
    cfg_drop_p = float(ckpt_args.get("cfg_drop_p", 0.0))

    use_rgb = bool(ckpt_args.get("use_rgb", False))
    point_dim = int(ckpt_args.get("point_dim", 6 if use_rgb else 3))

    if backbone == "mlp":
        return VelocityNet(
            cond_dim=cond_dim,
            point_dim=point_dim,
            width=width,
            depth=depth,
            emb_dim=emb_dim,
            cfg_dropout_p=cfg_drop_p,
        )

    if HybridMLP is None:
        raise ImportError("Checkpoint seems hybrid, but HybridMLP is not available in current project.")

    return HybridMLP(
        cond_dim=cond_dim,
        point_dim=point_dim,
        ctx_dim=int(ckpt_args.get("ctx_dim", 64)),
        ctx_emb_dim=int(ckpt_args.get("ctx_emb_dim", emb_dim)),
        stage_channels=_to_int_list(ckpt_args.get("ctx_stage_channels", None), [128, 256, 256]),
        stage_blocks=_to_int_list(ckpt_args.get("ctx_stage_blocks", None), [2, 2, 2]),
        stage_res=_to_int_list(ckpt_args.get("ctx_stage_res", None), [32, 16, 8]),
        with_se=bool(ckpt_args.get("ctx_with_se", True)),
        norm_type=str(ckpt_args.get("ctx_norm", "group")),
        gn_groups=int(ckpt_args.get("ctx_gn_groups", 32)),
        with_global=bool(ckpt_args.get("ctx_with_global", True)),
        voxel_normalize=bool(ckpt_args.get("ctx_voxel_normalize", True)),
        use_t_gate=True,
        t_gate_k=float(ckpt_args.get("ctx_t_gate_k", 10.0)),
        t_gate_tau=float(ckpt_args.get("ctx_t_gate_tau", 0.8)),
        pf_width=width,
        pf_depth=depth,
        pf_emb_dim=emb_dim,
        cfg_dropout_p=cfg_drop_p,
    )


# ============================================================
# Prior (match training)
# ============================================================
def make_prior_like(
    gt: torch.Tensor,
    prior_std: float,
    color_prior: str = "uniform",
    color_prior_std: float = 1.0,
) -> torch.Tensor:
    """gt: (B,N,3) or (B,N,6)"""
    if gt.shape[-1] == 3:
        return torch.randn_like(gt) * float(prior_std)

    if gt.shape[-1] != 6:
        raise ValueError(f"Unsupported point_dim in prior: {gt.shape[-1]}")
    z = torch.empty_like(gt)
    z[..., :3] = torch.randn_like(gt[..., :3]) * float(prior_std)

    cp = str(color_prior)
    if cp == "uniform":
        z[..., 3:] = torch.rand_like(gt[..., 3:])
    elif cp == "zeros":
        z[..., 3:] = 0.0
    elif cp == "gauss":
        z[..., 3:] = torch.randn_like(gt[..., 3:]) * float(color_prior_std)
    else:
        raise ValueError(f"Unknown color_prior: {cp}")
    return z


# ============================================================
# Metric recompute mode (from demo_out/gt + demo_out/pred)
# ============================================================
@torch.no_grad()

def recompute_metrics_from_folder(args, device: torch.device):
    # Resolve I/O
    gt_dir, pred_dir, metrics_out = resolve_gt_pred_and_out_dirs(args)

    gt_paths = sorted(glob.glob(os.path.join(gt_dir, "*.ply")))
    pred_paths = sorted(glob.glob(os.path.join(pred_dir, "*.ply")))

    gt_map = {os.path.basename(p): p for p in gt_paths}
    pred_map = {os.path.basename(p): p for p in pred_paths}

    common_names = sorted(set(gt_map.keys()) & set(pred_map.keys()))
    if not common_names:
        raise FileNotFoundError(f"[Recompute] No matching .ply pairs found under {gt_dir} and {pred_dir}")

    missing_in_pred = sorted(set(gt_map.keys()) - set(pred_map.keys()))
    missing_in_gt = sorted(set(pred_map.keys()) - set(gt_map.keys()))
    if missing_in_pred:
        raise FileNotFoundError(f"[Recompute] Missing pred files for {len(missing_in_pred)} gt files. e.g. {missing_in_pred[:5]}")
    if missing_in_gt:
        raise FileNotFoundError(f"[Recompute] Missing gt files for {len(missing_in_gt)} pred files. e.g. {missing_in_gt[:5]}")
    assert len(common_names) == len(gt_map) == len(pred_map)

    total = len(common_names)

    eval_seed = int(args.eval_seed) if args.eval_seed is not None else int(args.seed)
    sel = select_subset_indices(
        total=total,
        eval_fraction=float(args.eval_fraction),
        eval_max_samples=int(args.eval_max_samples),
        eval_seed=eval_seed,
    )
    selected_names = [common_names[i] for i in sel]
    print(f"[Recompute] Eval subset: {len(selected_names)}/{total} samples ({100.0 * len(selected_names) / total:.2f}%), seed={eval_seed}")

    # Eval scale (restore original units)
    center_np, scale_f, scale_info = load_eval_center_scale(
        eval_norm_json=getattr(args, 'eval_norm_json', None),
        eval_scale=getattr(args, 'eval_scale', None),
        eval_center=getattr(args, 'eval_center', None),
    )
    print(f"[EvalScale] raw = norm * {scale_f:.8f} + center {center_np.tolist()} (source={scale_info.get('source')})")

    # Export a few restored-scale samples for visual sanity check
    export_k = int(getattr(args, 'export_original_scale_samples', 5) or 0)
    sample_dir = getattr(args, 'original_scale_samples_dir', None)
    if sample_dir is None or str(sample_dir).strip() == '':
        sample_dir = os.path.join(metrics_out, 'OriginalScaleSamples')
    if export_k > 0:
        os.makedirs(sample_dir, exist_ok=True)
        with open(os.path.join(sample_dir, 'transform.json'), 'w') as f:
            json.dump(scale_info, f, ensure_ascii=False, indent=2)

    # Separate RNG streams so enabling/disabling EMD won't change CD results (and vice versa)
    np_rng_cd = np.random.RandomState(int(args.seed) + 1)
    np_rng_emd = np.random.RandomState(int(args.seed) + 2)

    # Resolve metric points
    cd_points: Optional[int] = int(args.cd_points) if int(args.cd_points) > 0 else None

    emd_points: Optional[int] = None
    if args.use_emd:
        if int(args.emd_points) > 0:
            emd_points = int(args.emd_points)
        else:
            emd_points = cd_points  # default: follow cd_points

    # For batching with variable N, group by vertex count if using full points (k is None).
    # If cd_points/emd_points is set (>0), then we'll subsample to fixed k anyway.
    need_fixed_n = (cd_points is None) or (args.use_emd and emd_points is None)
    groups: Dict[int, List[str]] = {}

    if need_fixed_n:
        # read vertex counts to group by N
        for name in tqdm(selected_names, desc="[Recompute] Scanning PLY headers", leave=False, dynamic_ncols=True):
            n_gt = _read_ply_vertex_count_ascii(gt_map[name])
            n_pred = _read_ply_vertex_count_ascii(pred_map[name])
            if n_gt != n_pred:
                raise ValueError(f"[Recompute] Vertex count mismatch for {name}: gt={n_gt} pred={n_pred}")
            groups.setdefault(int(n_gt), []).append(name)
    else:
        # group all into a single bucket
        groups[-1] = selected_names

    all_rows: List[Dict[str, Any]] = []
    cds: List[float] = []
    emds: List[float] = []

    exported = 0

    for group_n, names_in_group in groups.items():
        group_label = f"N={group_n}" if group_n != -1 else "N=fixed_by_subsample"
        bs = int(args.batch_size)
        num_batches = (len(names_in_group) + bs - 1) // bs
        batches = batched(names_in_group, bs)
        for bn, batch_names in enumerate(
            tqdm(
                batches,
                total=num_batches,
                desc=f"[Recompute {group_label}]",
                leave=False,
                dynamic_ncols=True,
                unit='batch',
            )
        ):
            # read points (CPU) - full resolution
            gt_full: List[np.ndarray] = []
            pred_full: List[np.ndarray] = []
            for name in batch_names:
                gt_np = read_ply_xyz(gt_map[name])  # (N,3)
                pred_np = read_ply_xyz(pred_map[name])
                if gt_np.shape[0] != pred_np.shape[0]:
                    raise ValueError(
                        f"[Recompute] Point count mismatch for {name}: gt={gt_np.shape} pred={pred_np.shape}"
                    )
                gt_full.append(gt_np)
                pred_full.append(pred_np)

                # export original-scale samples for the first few pairs
                if export_k > 0 and exported < export_k:
                    gt_os = apply_eval_transform_np(gt_np, center_np, scale_f)
                    pred_os = apply_eval_transform_np(pred_np, center_np, scale_f)
                    write_ply_xyz(os.path.join(sample_dir, f"gt_{name}"), gt_os)
                    write_ply_xyz(os.path.join(sample_dir, f"pred_{name}"), pred_os)
                    exported += 1

            # ----------------------------
            # CD (Chamfer Distance)
            # ----------------------------
            gt_cd_list = gt_full
            pred_cd_list = pred_full
            if cd_points is not None and cd_points > 0:
                gt_cd_list = [subsample_np(a, cd_points, np_rng_cd) for a in gt_full]
                pred_cd_list = [subsample_np(a, cd_points, np_rng_cd) for a in pred_full]

                # Ensure each sample really has cd_points
                for _name, _arr in zip(batch_names, gt_cd_list):
                    if _arr.shape[0] != cd_points:
                        raise ValueError(
                            f"[Recompute] '{_name}' has only {_arr.shape[0]} points (< cd_points={cd_points}). "
                            "Please reduce --cd_points."
                        )
                for _name, _arr in zip(batch_names, pred_cd_list):
                    if _arr.shape[0] != cd_points:
                        raise ValueError(
                            f"[Recompute] '{_name}' pred has only {_arr.shape[0]} points (< cd_points={cd_points}). "
                            "Please reduce --cd_points."
                        )

            # restore original scale BEFORE computing metrics
            gt_cd_list = [apply_eval_transform_np(a, center_np, scale_f) for a in gt_cd_list]
            pred_cd_list = [apply_eval_transform_np(a, center_np, scale_f) for a in pred_cd_list]

            gt_cd_t = torch.from_numpy(np.stack(gt_cd_list, axis=0)).to(device=device, dtype=torch.float32)
            pred_cd_t = torch.from_numpy(np.stack(pred_cd_list, axis=0)).to(device=device, dtype=torch.float32)

            cd_batch = chamfer_l2(pred_cd_t, gt_cd_t)  # (B,)
            cd_vals = cd_batch.detach().cpu().numpy().astype(np.float64).tolist()

            # ----------------------------
            # EMD (optional)
            # ----------------------------
            emd_vals: Optional[List[float]] = None
            if args.use_emd:
                gt_emd_list = gt_full
                pred_emd_list = pred_full

                if emd_points is not None and emd_points > 0:
                    gt_emd_list = [subsample_np(a, emd_points, np_rng_emd) for a in gt_full]
                    pred_emd_list = [subsample_np(a, emd_points, np_rng_emd) for a in pred_full]

                    for _name, _arr in zip(batch_names, gt_emd_list):
                        if _arr.shape[0] != emd_points:
                            raise ValueError(
                                f"[Recompute] '{_name}' has only {_arr.shape[0]} points (< emd_points={emd_points}). "
                                "Please reduce --emd_points."
                            )
                    for _name, _arr in zip(batch_names, pred_emd_list):
                        if _arr.shape[0] != emd_points:
                            raise ValueError(
                                f"[Recompute] '{_name}' pred has only {_arr.shape[0]} points (< emd_points={emd_points}). "
                                "Please reduce --emd_points."
                            )

                # EMD requires same N for pred/gt and also same N across the batch (to stack)
                n0 = int(gt_emd_list[0].shape[0])
                for _b, _name in enumerate(batch_names):
                    if gt_emd_list[_b].shape[0] != pred_emd_list[_b].shape[0]:
                        raise ValueError(
                            f"[Recompute] EMD requires pred/gt same N for '{_name}'. "
                            f"gt={gt_emd_list[_b].shape} pred={pred_emd_list[_b].shape}"
                        )
                    if int(gt_emd_list[_b].shape[0]) != n0:
                        raise ValueError(
                            f"[Recompute] Variable N inside a batch for EMD (got {gt_emd_list[_b].shape[0]} vs {n0}). "
                            "Try setting --emd_points (or reduce --batch_size)."
                        )

                # restore original scale BEFORE EMD
                gt_emd_list = [apply_eval_transform_np(a, center_np, scale_f) for a in gt_emd_list]
                pred_emd_list = [apply_eval_transform_np(a, center_np, scale_f) for a in pred_emd_list]

                gt_emd_t = torch.from_numpy(np.stack(gt_emd_list, axis=0)).to(device=device, dtype=torch.float32)
                pred_emd_t = torch.from_numpy(np.stack(pred_emd_list, axis=0)).to(device=device, dtype=torch.float32)

                emd_batch = emd_distance(pred_emd_t, gt_emd_t)
                emd_vals = emd_batch.detach().cpu().numpy().astype(np.float64).tolist()

            # record
            for j, name in enumerate(batch_names):
                row: Dict[str, Any] = {
                    'name': name,
                    'cd': f"{float(cd_vals[j]):.10f}",
                }
                cds.append(float(cd_vals[j]))
                if args.use_emd and emd_vals is not None:
                    row['emd'] = f"{float(emd_vals[j]):.10f}"
                    emds.append(float(emd_vals[j]))
                all_rows.append(row)

    mean_cd = float(np.mean(cds)) if cds else float('nan')
    std_cd = float(np.std(cds)) if cds else float('nan')
    mean_emd = float(np.mean(emds)) if emds else float('nan')
    std_emd = float(np.std(emds)) if emds else float('nan')

    print(
        f"[Recompute] Done. Samples={len(cds)} mean_CD={mean_cd:.8f} std_CD={std_cd:.8f}" +
        (f" mean_EMD={mean_emd:.8f} std_EMD={std_emd:.8f}" if args.use_emd else '')
    )

    # write per-sample csv
    csv_path = os.path.join(metrics_out, 'metrics_per_sample.csv')
    write_metrics_csv(csv_path, all_rows)

    # write summary
    summary = {
        'mode': 'recompute_metrics',
        'gt_dir': gt_dir,
        'pred_dir': pred_dir,
        'metrics_out': metrics_out,
        'samples': len(cds),
        'mean_cd': mean_cd,
        'std_cd': std_cd,
        'use_emd': bool(args.use_emd),
        'mean_emd': mean_emd if args.use_emd else None,
        'std_emd': std_emd if args.use_emd else None,
        'cd_points': cd_points,
        'emd_points': emd_points,
        'batch_size': int(args.batch_size),
        'eval_fraction': float(args.eval_fraction),
        'eval_max_samples': int(args.eval_max_samples),
        'eval_seed': int(eval_seed),
        'eval_total_samples': int(total),
        'eval_selected_samples': int(len(selected_names)),
        'seed': int(args.seed),
        'metrics_csv': os.path.basename(csv_path),
        'device': str(device),
        'eval_scale_info': scale_info,
        'original_scale_samples_dir': sample_dir if export_k > 0 else None,
        'original_scale_samples_exported': int(exported),
    }
    with open(os.path.join(metrics_out, 'summary.json'), 'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


# ============================================================
# Generation + evaluation mode (from ckpt + dataset)
# ============================================================

def generate_and_evaluate(args, device: torch.device):
    # seed (global)
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    # Separate RNG streams:
    #   - np_rng_data: subsampling for generation/saving
    #   - np_rng_cd:   subsampling for CD evaluation
    #   - np_rng_emd:  subsampling for EMD evaluation
    np_rng_data = np.random.RandomState(int(args.seed) + 0)
    np_rng_cd = np.random.RandomState(int(args.seed) + 1)
    np_rng_emd = np.random.RandomState(int(args.seed) + 2)

    if args.ckpt is None or args.data_dir is None:
        raise ValueError("[Generate] --ckpt and --data_dir are required unless --recompute_metrics is set.")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    ckpt_args = ckpt.get("args", {}) or {}

    backbone = infer_backbone(ckpt)
    print(f"[Demo] Backbone inferred: {backbone}")

    # infer point_dim & rgb_key from ckpt
    use_rgb = bool(ckpt_args.get("use_rgb", False))
    point_dim = int(ckpt_args.get("point_dim", 6 if use_rgb else 3))
    if point_dim not in (3, 6):
        raise ValueError(f"[Demo] Unsupported point_dim={point_dim} (expect 3 or 6).")

    # sampler hyperparams: default from ckpt, allow override
    sample_steps = int(ckpt_args.get("sample_steps", 50))
    if int(args.sample_steps) > 0:
        sample_steps = int(args.sample_steps)

    # prior_std: prefer ckpt prior_std, fallback to point_prior_std (legacy naming)
    prior_std = ckpt_args.get("prior_std", None)
    if prior_std is None:
        prior_std = ckpt_args.get("point_prior_std", 1.0)
    prior_std = float(prior_std)
    if float(args.prior_std) > 0:
        prior_std = float(args.prior_std)

    guidance_scale = float(ckpt_args.get("guidance_scale", 0.0))
    if args.guidance_scale is not None:
        guidance_scale = float(args.guidance_scale)

    # 6D prior knobs
    color_prior = str(ckpt_args.get("color_prior", "uniform"))
    color_prior_std = float(ckpt_args.get("color_prior_std", 1.0))
    if args.color_prior is not None:
        color_prior = str(args.color_prior)
    if args.color_prior_std is not None:
        color_prior_std = float(args.color_prior_std)

    rgb_key = str(ckpt_args.get("rgb_key", "rgb"))
    if args.rgb_key is not None:
        rgb_key = str(args.rgb_key)

    # max_points: default to ckpt te_max_sample_points
    max_points: Optional[int] = None
    if int(args.max_points) > 0:
        max_points = int(args.max_points)
    else:
        te_mp = ckpt_args.get("te_max_sample_points", None)
        if te_mp is not None:
            te_mp = int(te_mp)
            if te_mp > 0:
                max_points = te_mp

    cd_points: Optional[int] = int(args.cd_points) if int(args.cd_points) > 0 else None
    if cd_points is None:
        cd_points = max_points  # if still None -> later use full

    emd_points: Optional[int] = None
    if args.use_emd:
        if int(args.emd_points) > 0:
            emd_points = int(args.emd_points)
        else:
            emd_points = cd_points  # default: follow cd_points

    # build model
    net = build_model_from_ckpt_args(backbone, ckpt_args).to(device)
    net.eval()

    # load weights (prefer EMA unless --no_ema)
    if (not args.no_ema) and (ckpt.get("ema", None) is not None):
        state = ckpt["ema"]
        print("[Demo] Loading EMA weights.")
    else:
        state = ckpt.get("model", None)
        print("[Demo] Loading model weights.")
    if state is None:
        raise KeyError("Checkpoint missing both 'ema' and 'model'.")

    # strict load if possible
    try:
        net.load_state_dict(state, strict=True)
    except Exception as e:
        print(f"[Demo][WARN] strict load failed: {e}. Retrying with strict=False.")
        missing, unexpected = net.load_state_dict(state, strict=False)
        if missing:
            print(f"[Demo][WARN] Missing keys: {missing[:8]}{'...' if len(missing) > 8 else ''}")
        if unexpected:
            print(f"[Demo][WARN] Unexpected keys: {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")

    # output dirs
    out_gt = os.path.join(args.demo_out, "gt")
    out_pred = os.path.join(args.demo_out, "pred")
    if not bool(args.no_save_ply):
        os.makedirs(out_gt, exist_ok=True)
        os.makedirs(out_pred, exist_ok=True)


    # metrics / restored-scale export output
    metrics_out = args.metrics_out if (getattr(args, 'metrics_out', None) not in (None, '')) else args.demo_out
    os.makedirs(metrics_out, exist_ok=True)

    # Eval scale (restore original units for fair metrics)
    center_np, scale_f, scale_info = load_eval_center_scale(
        eval_norm_json=getattr(args, 'eval_norm_json', None),
        eval_scale=getattr(args, 'eval_scale', None),
        eval_center=getattr(args, 'eval_center', None),
    )
    print(f"[EvalScale] raw = norm * {scale_f:.8f} + center {center_np.tolist()} (source={scale_info.get('source')})")

    center_t = torch.from_numpy(center_np).to(device=device, dtype=torch.float32).view(1, 1, 3)

    export_k = int(getattr(args, 'export_original_scale_samples', 5) or 0)
    sample_dir = getattr(args, 'original_scale_samples_dir', None)
    if sample_dir is None or str(sample_dir).strip() == '':
        sample_dir = os.path.join(metrics_out, 'OriginalScaleSamples')
    if export_k > 0:
        os.makedirs(sample_dir, exist_ok=True)
        with open(os.path.join(sample_dir, 'transform.json'), 'w') as f:
            json.dump(scale_info, f, ensure_ascii=False, indent=2)
    exported = 0

    # scan split files
    files = find_h5_files(args.data_dir, args.split)
    if not files:
        raise FileNotFoundError(f"No .h5/.hdf5 found. data_dir={args.data_dir} split={args.split}")

    # choose subset of samples to evaluate (optional)
    eval_seed = int(args.eval_seed) if args.eval_seed is not None else int(args.seed)

    # Count how many examples exist in this split (across all shards),
    # then pick a random subset without replacement.
    import bisect

    cum_counts: List[int] = []
    total = 0
    for _fp in files:
        with h5py.File(_fp, "r") as _f:
            _pts_key = _pick_points_key(_f, prefer_norm=bool(args.use_norm))
            _B = int(_f[_pts_key].shape[0])
        total += _B
        cum_counts.append(total)

    if total <= 0:
        raise RuntimeError(f"No samples found in split='{args.split}' under {args.data_dir}")

    sel = select_subset_indices(
        total=total,
        eval_fraction=float(args.eval_fraction),
        eval_max_samples=int(args.eval_max_samples),
        eval_seed=eval_seed,
    )

    # Map selected global ids -> (file, local_idx) for efficient H5 access
    selected_by_file: Dict[str, List[Tuple[int, int]]] = {}
    for gid in sel:
        fi = bisect.bisect_right(cum_counts, gid)
        start_off = cum_counts[fi - 1] if fi > 0 else 0
        local_i = int(gid - start_off)
        selected_by_file.setdefault(files[fi], []).append((int(gid), local_i))

    # Sort within each file so H5 reads are more sequential
    for _fp, items in selected_by_file.items():
        items.sort(key=lambda x: x[1])

    k = len(sel)
    print(f"[Demo] Eval subset: {k}/{total} samples ({100.0 * k / total:.2f}%), seed={eval_seed}")

    clamp_rgb = not bool(args.no_clamp_rgb)
    sampler_fn = heun_sampler if args.sampler == "heun" else euler_sampler

    print(
        f"[Demo] Found {len(files)} h5 shards. device={device} point_dim={point_dim} steps={sample_steps} "
        f"prior_std={prior_std} max_points={max_points} cd_points={cd_points} emd_points={emd_points} "
        f"guidance_scale={guidance_scale} sampler={args.sampler} clamp_rgb={clamp_rgb} color_prior={color_prior} "
        f"batch_size={int(args.batch_size)} use_emd={bool(args.use_emd)}"
    )

    # metrics
    cds: List[float] = []
    emds: List[float] = []
    rows: List[Dict[str, Any]] = []

    sample_idx = 0

    # Iterate shards, but only process selected indices in each shard
    for fp in files:
        items = selected_by_file.get(fp, None)
        if not items:
            continue

        with h5py.File(fp, "r") as f:
            pts_key = _pick_points_key(f, prefer_norm=bool(args.use_norm))
            pts_ds = f[pts_key]  # (B,N,3) or (B,N,6)

            # cond may not exist for unconditional models
            cond_dim = int(ckpt_args.get("cond_dim", 0))
            cond_ds = None
            if cond_dim > 0:
                cond_key = _pick_cond_key(f)
                cond_ds = f[cond_key]  # (B,D)

            rgb_ds = None
            if point_dim == 6 and pts_ds.shape[-1] == 3:
                rgb_ds = f[_pick_rgb_key(f, rgb_key)]  # (B,N,3)

            bs = int(args.batch_size)
            num_batches = (len(items) + bs - 1) // bs
            batches = batched(items, bs)
            for batch_items in tqdm(
                batches,
                total=num_batches,
                desc=f"[{os.path.basename(fp)}]",
                leave=False,
                dynamic_ncols=True,
                unit='batch',
            ):
                # names & indices
                idxs = [int(i) for (_gid, i) in batch_items]
                gids = [int(_gid) for (_gid, i) in batch_items]

                names: List[str] = []
                for _ in batch_items:
                    sample_idx += 1
                    names.append(f"{sample_idx:06d}.ply")

                # load gt (batch)
                gt_raw = np.asarray(pts_ds[idxs], dtype=np.float32)  # (B,N,3|6)
                if gt_raw.ndim != 3:
                    raise ValueError(f"Unexpected points shape: {gt_raw.shape} in {fp}")

                B = gt_raw.shape[0]
                gt_list: List[np.ndarray] = []

                if point_dim == 6 and gt_raw.shape[-1] == 3:
                    if rgb_ds is None:
                        raise KeyError(f"[Demo] point_dim=6 but rgb not found in H5: {fp}")
                    rgb_raw = np.asarray(rgb_ds[idxs], dtype=np.float32)  # (B,N,3)
                    if rgb_raw.shape[:2] != gt_raw.shape[:2]:
                        raise ValueError(f"xyz/rgb length mismatch: xyz={gt_raw.shape} rgb={rgb_raw.shape} in {fp}")
                    for b in range(B):
                        xyz = gt_raw[b]
                        rgb = rgb_to_01(rgb_raw[b])
                        g = np.concatenate([xyz, rgb.astype(np.float32)], axis=1)
                        if max_points is not None:
                            g = subsample_np(g, max_points, np_rng_data)
                        gt_list.append(g)
                else:
                    # either point_dim==3, or already concatenated 6D
                    for b in range(B):
                        g = gt_raw[b]
                        if point_dim == 6 and g.shape[1] == 6:
                            g = g.copy()
                            g[:, 3:6] = rgb_to_01(g[:, 3:6])
                        if max_points is not None:
                            g = subsample_np(g, max_points, np_rng_data)
                        gt_list.append(g)

                gt_np = np.stack(gt_list, axis=0)  # (B,N',C)
                # save gt
                if not bool(args.no_save_ply):
                    for b, name in enumerate(names):
                        if point_dim == 6:
                            write_ply_xyzrgb(os.path.join(out_gt, name), gt_np[b])
                        else:
                            write_ply_xyz(os.path.join(out_gt, name), gt_np[b, :, :3])

                # cond
                cond_t: Optional[torch.Tensor] = None
                if cond_ds is not None:
                    cond_np = np.asarray(cond_ds[idxs], dtype=np.float32)
                    if cond_np.ndim == 1:
                        cond_np = cond_np.reshape(1, -1)
                    cond_t = torch.from_numpy(cond_np).to(device=device, dtype=torch.float32)

                # to torch
                gt_t = torch.from_numpy(gt_np).to(device=device, dtype=torch.float32)

                # sample
                z = make_prior_like(gt_t, prior_std, color_prior=color_prior, color_prior_std=color_prior_std)
                pred_t = sampler_fn(
                    net,
                    z,
                    cond_t,
                    steps=sample_steps,
                    guidance_scale=guidance_scale,
                    clamp_rgb=clamp_rgb,
                )

                # save pred
                if not bool(args.no_save_ply):
                    pred_np = pred_t.detach().cpu().numpy()
                    for b, name in enumerate(names):
                        if point_dim == 6:
                            pred_np_b = pred_np[b].copy()
                            pred_np_b[:, 3:6] = np.clip(pred_np_b[:, 3:6], 0.0, 1.0)
                            write_ply_xyzrgb(os.path.join(out_pred, name), pred_np_b)
                        else:
                            write_ply_xyz(os.path.join(out_pred, name), pred_np[b, :, :3])

                # Export a few restored-scale samples for visual sanity check
                if export_k > 0 and exported < export_k:
                    pred_np_for_export = pred_t.detach().cpu().numpy()
                    for b, name in enumerate(names):
                        if exported >= export_k:
                            break
                        gt_os = apply_eval_transform_np(gt_np[b, :, :3], center_np, scale_f)
                        pred_os = apply_eval_transform_np(pred_np_for_export[b, :, :3], center_np, scale_f)
                        write_ply_xyz(os.path.join(sample_dir, f"gt_{name}"), gt_os)
                        write_ply_xyz(os.path.join(sample_dir, f"pred_{name}"), pred_os)
                        exported += 1

                # CD (xyz only) -- computed in ORIGINAL units
                pred_cd = pred_t[..., :3]
                gt_cd = gt_t[..., :3]
                if cd_points is not None and cd_points > 0:
                    pred_cd = subsample_torch_per_example(pred_cd, cd_points, np_rng_cd)
                    gt_cd = subsample_torch_per_example(gt_cd, cd_points, np_rng_cd)

                pred_cd_eval = apply_eval_transform_torch(pred_cd, center_t, scale_f)
                gt_cd_eval = apply_eval_transform_torch(gt_cd, center_t, scale_f)

                cd_batch = chamfer_l2(pred_cd_eval, gt_cd_eval)  # (B,)
                cd_vals = cd_batch.detach().cpu().numpy().astype(np.float64).tolist()

                # EMD (optional, xyz only) -- computed in ORIGINAL units
                emd_vals: Optional[List[float]] = None
                if args.use_emd:
                    pred_e = pred_t[..., :3]
                    gt_e = gt_t[..., :3]
                    if emd_points is not None and emd_points > 0:
                        pred_e = subsample_torch_per_example(pred_e, emd_points, np_rng_emd)
                        gt_e = subsample_torch_per_example(gt_e, emd_points, np_rng_emd)
                    if pred_e.shape[1] != gt_e.shape[1]:
                        raise ValueError(f"[Demo] EMD requires same N. pred={pred_e.shape} gt={gt_e.shape}")

                    pred_e_eval = apply_eval_transform_torch(pred_e, center_t, scale_f)
                    gt_e_eval = apply_eval_transform_torch(gt_e, center_t, scale_f)

                    emd_batch = emd_distance(pred_e_eval, gt_e_eval)
                    emd_vals = emd_batch.detach().cpu().numpy().astype(np.float64).tolist()

                # record per-sample
                for b in range(B):
                    row: Dict[str, Any] = {
                        "name": names[b],
                        "gid": int(gids[b]),
                        "shard": os.path.basename(fp),
                        "local_idx": int(idxs[b]),
                        "cd": f"{float(cd_vals[b]):.10f}",
                    }
                    cds.append(float(cd_vals[b]))
                    if args.use_emd and emd_vals is not None:
                        row["emd"] = f"{float(emd_vals[b]):.10f}"
                        emds.append(float(emd_vals[b]))
                    rows.append(row)

    mean_cd = float(np.mean(cds)) if cds else float("nan")
    std_cd = float(np.std(cds)) if cds else float("nan")
    mean_emd = float(np.mean(emds)) if emds else float("nan")
    std_emd = float(np.std(emds)) if emds else float("nan")

    print(
        f"[Demo] Done. Samples={len(cds)} mean_CD={mean_cd:.8f} std_CD={std_cd:.8f}"
        + (f" mean_EMD={mean_emd:.8f} std_EMD={std_emd:.8f}" if args.use_emd else "")
    )

    # write per-sample csv
    os.makedirs(metrics_out, exist_ok=True)
    csv_path = os.path.join(metrics_out, "metrics_per_sample.csv")
    write_metrics_csv(csv_path, rows)

    # write summary
    summary = {        "mode": "generate_and_eval",
        "demo_out": args.demo_out,
        "metrics_out": metrics_out,
        "eval_scale_info": scale_info,
        "original_scale_samples_dir": sample_dir if export_k > 0 else None,
        "original_scale_samples_exported": int(exported),
        "ckpt": args.ckpt,
        "data_dir": args.data_dir,
        "split": args.split,
        "backbone": backbone,
        "point_dim": point_dim,
        "samples": len(cds),
        "mean_cd": mean_cd,
        "std_cd": std_cd,
        "use_emd": bool(args.use_emd),
        "mean_emd": mean_emd if args.use_emd else None,
        "std_emd": std_emd if args.use_emd else None,
        "sample_steps": sample_steps,
        "prior_std": prior_std,
        "guidance_scale": guidance_scale,
        "max_points": max_points,
        "cd_points": cd_points,
        "emd_points": emd_points,
        "sampler": args.sampler,
        "clamp_rgb": clamp_rgb,
        "color_prior": color_prior,
        "color_prior_std": color_prior_std,
        "rgb_key": rgb_key,
        "eval_fraction": float(args.eval_fraction),
        "eval_max_samples": int(args.eval_max_samples),
        "eval_seed": int(eval_seed),
        "eval_total_samples": int(total),
        "eval_selected_samples": int(k),
        "no_save_ply": bool(args.no_save_ply),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "metrics_csv": os.path.basename(csv_path),
        "device": str(device),
    }
    with open(os.path.join(metrics_out, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser("TDCR demo generation + evaluation (supports xyz / xyzrgb)")

    # Mode switch
    ap.add_argument(
        "--recompute_metrics",
        action="store_true",
        default=False,
        help="If set, skip generation and ONLY recompute CD/EMD from an existing demo_out folder (demo_out/gt + demo_out/pred).",
    )

    # Generation inputs (required only when NOT recompute_metrics)
    ap.add_argument("--ckpt", type=str, default=None, help="path to latest.pt / epoch_xxxx.pt (required unless --recompute_metrics)")
    ap.add_argument("--data_dir", type=str, default=None, help="dataset root (required unless --recompute_metrics)")
    ap.add_argument("--split", type=str, default="test", help="dataset split (generation mode only)")

    # Output / input folders
    ap.add_argument(
        "--demo_out",
        type=str,
        default=None,
        help=(
            "Generation output folder. In generation mode, this is where gt/ and pred/ will be written. "
            "In recompute mode, if --pairs_root/--gt_dir/--pred_dir are not set, it falls back to demo_out/gt and demo_out/pred."
        ),
    )

    ap.add_argument(
        "--pairs_root",
        type=str,
        default=None,
        help="[recompute] Root folder that contains 'gt/' and 'pred/' subfolders. Example: TEST/sim_5m_no_base_hybrid_1_3",
    )
    ap.add_argument(
        "--gt_dir",
        type=str,
        default=None,
        help="[recompute] Explicit GT folder containing .ply files. Overrides --pairs_root.",
    )
    ap.add_argument(
        "--pred_dir",
        type=str,
        default=None,
        help="[recompute] Explicit Pred folder containing .ply files. Overrides --pairs_root.",
    )
    ap.add_argument(
        "--metrics_out",
        type=str,
        default=None,
        help=(
            "Where to write metrics_per_sample.csv / summary.json / OriginalScaleSamples. "
            "Default: pairs_root (recompute) or demo_out (generation)."
        ),
    )

    # Performance / batching
    ap.add_argument("--batch_size", type=int, default=1, help="Batch size for generation+evaluation (and recompute). Default=1.")
    ap.add_argument("--device", type=str, default="cuda")

    # Metrics
    ap.add_argument("--use_emd", action="store_true", default=False, help="Enable EMD computation (requires compiled emd_ext).")
    ap.add_argument("--cd_points", type=int, default=0, help="Points for CD computation (<=0: use max_points/full)")
    ap.add_argument("--emd_points", type=int, default=0, help="Points for EMD computation (<=0: follow cd_points)")

    ap.add_argument(
        "--eval_norm_json",
        type=str,
        default=None,
        help=(
            "Global norm json used in dataset normalization (from tdcr_pipeline add-norm / tdcr_to_vsm compute_global_scale). "
            "Metrics are computed after restoring original units: raw = norm * scale + center."
        ),
    )
    ap.add_argument(
        "--eval_scale",
        type=float,
        default=None,
        help="If --eval_norm_json is not provided, use this scale for restoring: raw = norm * scale + center (default: 1).",
    )
    ap.add_argument(
        "--eval_center",
        type=float,
        nargs=3,
        default=None,
        help="If --eval_norm_json is not provided, use this center (x y z) for restoring (default: 0 0 0).",
    )
    ap.add_argument(
        "--export_original_scale_samples",
        type=int,
        default=5,
        help="Export first K GT/Pred pairs (after restoring original scale) into OriginalScaleSamples. 0 disables.",
    )
    ap.add_argument(
        "--original_scale_samples_dir",
        type=str,
        default=None,
        help="Override export folder for OriginalScaleSamples (default: <metrics_out>/OriginalScaleSamples).",
    )
    ap.add_argument(
        "--max_points",
        type=int,
        default=0,
        help="Max points for generation/saving (<=0: use ckpt te_max_sample_points; still 0 -> full)",
    )

    # Sampling params
    ap.add_argument("--sample_steps", type=int, default=0, help="Override ckpt sample_steps (<=0: no override)")
    ap.add_argument(
        "--prior_std",
        type=float,
        default=0.0,
        help="Override ckpt prior_std / point_prior_std (<=0: no override)",
    )
    ap.add_argument("--guidance_scale", type=float, default=None, help="Override ckpt guidance_scale (if set)")

    # 6D knobs
    ap.add_argument("--color_prior", type=str, default=None, choices=["uniform", "zeros", "gauss"], help="Override ckpt color_prior for 6D")
    ap.add_argument("--color_prior_std", type=float, default=None, help="Override ckpt color_prior_std (only for gauss prior)")
    ap.add_argument("--rgb_key", type=str, default=None, help="Override ckpt rgb_key (only used when point_dim==6)")

    # Sampler
    ap.add_argument("--sampler", type=str, default="heun", choices=["heun", "euler"], help="Sampling ODE solver (default: heun/RK2)")
    ap.add_argument("--no_clamp_rgb", action="store_true", default=False, help="Disable clamping rgb to [0,1] during sampling")

    # Eval subset selection (applies to both generation and recompute)
    ap.add_argument("--eval_fraction", type=float, default=1.0, help="Evaluate only this fraction of samples (0<..<=1)")
    ap.add_argument("--eval_max_samples", type=int, default=0, help="Cap evaluated samples after eval_fraction (0: no cap)")
    ap.add_argument("--eval_seed", type=int, default=None, help="Seed used to choose subset samples (default: --seed)")

    # I/O & misc
    ap.add_argument("--no_save_ply", action="store_true", default=False, help="If set, do not write gt/pred ply files (generation mode only)")
    ap.add_argument("--no_ema", action="store_true", default=False, help="Do not use EMA weights (generation mode only)")
    ap.add_argument("--use_norm", action="store_true", default=True, help="prefer data_norm if exists")
    ap.add_argument("--seed", type=int, default=123)

    args = ap.parse_args()

    # -----------------------------
    # Argument sanity checks
    # -----------------------------
    if bool(args.recompute_metrics):
        # Need an input source for GT/PRED
        if (args.gt_dir is not None) or (args.pred_dir is not None):
            if not (args.gt_dir and args.pred_dir):
                raise ValueError("[Recompute] Must provide both --gt_dir and --pred_dir when using explicit dirs.")
        else:
            # pairs_root preferred, fallback to demo_out
            if args.pairs_root is None and args.demo_out is None:
                raise ValueError("[Recompute] Please provide --pairs_root (or --demo_out) as the folder containing gt/ and pred/.")
    else:
        # generation mode
        if args.demo_out is None:
            raise ValueError("[Generate] --demo_out is required in generation mode.")
        if args.ckpt is None or args.data_dir is None:
            raise ValueError("[Generate] --ckpt and --data_dir are required unless --recompute_metrics is set.")

    # Normalize eval_center if provided
    if args.eval_center is not None:
        args.eval_center = list(args.eval_center)

    # device
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[Warn] CUDA not available, falling back to CPU. (CD/EMD compiled ops will then fail.)")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # basic checks
    if int(args.batch_size) <= 0:
        raise ValueError(f"--batch_size must be > 0, got {args.batch_size}")

    # EMD import check early (optional)
    if bool(args.use_emd):
        _ = _load_emd_fn()  # will raise if missing

    # CD import check early (always needed)
    _ = _load_chamfer_ext()  # will raise if missing

    if bool(args.recompute_metrics):
        recompute_metrics_from_folder(args, device=device)
    else:
        generate_and_evaluate(args, device=device)


if __name__ == "__main__":
    main()

"""
Examples:

1) Generation + evaluation (CD only), with batching:
export CUDA_VISIBLE_DEVICES=0
python demo_generate_tdcr.py \
  --ckpt /path/to/ckpts/latest.pt \
  --data_dir datasets/sim/5m_with_base \
  --split test \
  --demo_out demo_out/exp1 \
  --sampler heun \
  --sample_steps 50 \
  --prior_std 0.5 \
  --rgb_key rgb \
  --cd_points 4096 \
  --batch_size 16

2) Generation + evaluation with EMD enabled:
export CUDA_VISIBLE_DEVICES=1
python demo_generate_tdcr.py \
  --ckpt final_results/sim_5m_with_base_hybrid_1_2/ckpts/latest.pt \
  --data_dir datasets/sim/5m_with_base \
  --split test \
  --demo_out final_demo/sim_5m_with_base_hybrid_1_2 \
  --cd_points 4096 \
  --use_emd \
  --emd_points 4096 \
  --sample_steps 100 \
  --batch_size 64

python demo_generate_tdcr.py \
  --ckpt final_results/sim_5m_with_base_mlp_12_27/ckpts/latest.pt \
  --data_dir datasets/sim/5m_with_base \
  --split test \
  --demo_out final_demo/sim_5m_with_base_mlp_12_27 \
  --cd_points 4096 \
  --use_emd \
  --emd_points 4096 \
  --sample_steps 100 \
  --batch_size 64


3) Recompute metrics only (from existing demo_out/gt + demo_out/pred):
export CUDA_VISIBLE_DEVICES=0
python demo_generate_tdcr.py \
  --recompute_metrics \
  --demo_out demo_out/exp1 \
  --cd_points 4096 \
  --use_emd \
  --emd_points 1024 \
  --batch_size 32 \
  --seed 123
"""
