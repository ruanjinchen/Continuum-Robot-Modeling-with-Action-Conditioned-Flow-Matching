from __future__ import annotations
import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from models import VelocityNet

try:
    from models import HybridMLP
except Exception:
    HybridMLP = None


# ============================================================
# PLY writers (ASCII)
# ============================================================
def write_ply_xyz(path: str, points_xyz: np.ndarray) -> None:
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


def write_ply_xyzrgb(path: str, points_xyzrgb: np.ndarray) -> None:
    """points_xyzrgb: (N,6) with rgb in [0,1]"""
    points_xyzrgb = np.asarray(points_xyzrgb, dtype=np.float32)
    if points_xyzrgb.ndim != 2 or points_xyzrgb.shape[1] != 6:
        raise ValueError(f"write_ply_xyzrgb expects (N,6). got {points_xyzrgb.shape}")
    xyz = points_xyzrgb[:, :3]
    rgb = np.clip(points_xyzrgb[:, 3:6], 0.0, 1.0)
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
# Model builder (same logic as demo_generate_tdcr.py)
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
# Eval scale (denormalization)
# ============================================================
def load_eval_center_scale(
    eval_norm_json: Optional[str],
    eval_scale: Optional[float] = None,
    eval_center: Optional[List[float]] = None,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    center = np.zeros(3, dtype=np.float32)
    scale = 1.0
    info: Dict[str, Any] = {
        "source": "default",
        "eval_norm_json": eval_norm_json,
        "eval_scale_arg": eval_scale,
        "eval_center_arg": eval_center,
    }

    if eval_norm_json:
        with open(eval_norm_json, "r") as f:
            obj = json.load(f)
        if not isinstance(obj, dict) or len(obj) == 0:
            raise ValueError(f"Bad eval_norm_json (expect dict): {eval_norm_json}")
        if "all" in obj and isinstance(obj["all"], dict):
            sub = obj["all"]
            info["key_used"] = "all"
        else:
            k0 = next(iter(obj.keys()))
            sub = obj[k0]
            info["key_used"] = k0
        c = sub.get("center", None)
        s = sub.get("scale", None)
        if c is None or s is None:
            raise ValueError(f"eval_norm_json missing center/scale: {eval_norm_json}")
        center = np.asarray(c, dtype=np.float32).reshape(3)
        scale = float(s)
        info["source"] = "eval_norm_json"
    else:
        if eval_center is not None:
            center = np.asarray(eval_center, dtype=np.float32).reshape(3)
            info["source"] = "eval_center_arg"
        if eval_scale is not None:
            scale = float(eval_scale)
            info["source"] = "eval_scale_arg" if info["source"] == "default" else info["source"]

    if not np.isfinite(center).all():
        raise ValueError(f"Invalid eval center: {center}")
    if not (np.isfinite(scale) and scale > 0):
        raise ValueError(f"Invalid eval scale: {scale}")

    info["center"] = center.tolist()
    info["scale"] = float(scale)
    return center.astype(np.float32), float(scale), info


def denorm_xyz_np(xyz: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32)
    return xyz * float(scale) + center.reshape(1, 3).astype(np.float32)


# ============================================================
# Motor sequence loader
# ============================================================
def load_motor_sequence(path: str, delim: str = ",", has_header: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"motor_seq not found: {path}")

    suffix = p.suffix.lower()
    meta: Dict[str, Any] = {"path": str(p.resolve())}

    if suffix == ".json" or suffix == ".jsonl":
        obj = json.loads(p.read_text(encoding="utf-8"))
        hz = 60.0
        if isinstance(obj, dict):
            if "hz" in obj:
                hz = float(obj["hz"])
            if "ctrl_norm" in obj:
                arr = np.asarray(obj["ctrl_norm"], dtype=np.float32)
            elif "ctrl" in obj:
                arr = np.asarray(obj["ctrl"], dtype=np.float32)
                # optional normalization
                if (np.nanmin(arr) < -0.05 or np.nanmax(arr) > 1.05) and isinstance(obj.get("norm", None), dict):
                    n = obj["norm"]
                    if "min" in n and "max" in n:
                        mn = np.asarray(n["min"], dtype=np.float32).reshape(1, -1)
                        mx = np.asarray(n["max"], dtype=np.float32).reshape(1, -1)
                        denom = np.maximum(mx - mn, 1e-8)
                        arr = (arr - mn) / denom
                        meta["normalized_from_raw"] = True
            else:
                raise ValueError("JSON motor_seq must contain ctrl_norm or ctrl")
        elif isinstance(obj, list):
            # list of vectors
            arr = np.asarray(obj, dtype=np.float32)
            hz = 60.0
        else:
            raise ValueError("Unsupported JSON format for motor_seq")

        if arr.ndim != 2:
            raise ValueError(f"motor_seq ctrl_norm must be 2D (T,D). got {arr.shape}")
        meta["hz"] = hz
        meta["T"] = int(arr.shape[0])
        meta["D"] = int(arr.shape[1])
        return arr.astype(np.float32), meta

    # CSV
    rows: List[List[float]] = []
    with open(p, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delim)
        for ri, row in enumerate(reader):
            if ri == 0 and has_header:
                continue
            if not row:
                continue
            rows.append([float(x) for x in row])
    arr = np.asarray(rows, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"CSV motor_seq must be 2D (T,D). got {arr.shape}")
    meta["hz"] = 60.0
    meta["T"] = int(arr.shape[0])
    meta["D"] = int(arr.shape[1])
    return arr.astype(np.float32), meta


# ============================================================
# Sampling
# ============================================================
@torch.no_grad()
def euler_sampler(
    net: torch.nn.Module,
    x0: torch.Tensor,
    cond: Optional[torch.Tensor],
    steps: int,
    guidance_scale: float,
    clamp_rgb: bool,
) -> torch.Tensor:
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
    clamp_rgb: bool,
) -> torch.Tensor:
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


def make_prior(
    B: int,
    N: int,
    C: int,
    device: torch.device,
    prior_std: float,
    color_prior: str,
    color_prior_std: float,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if C == 3:
        return torch.randn((B, N, 3), device=device, dtype=dtype) * float(prior_std)
    if C != 6:
        raise ValueError(f"Unsupported point_dim={C}, expect 3 or 6.")
    z = torch.empty((B, N, 6), device=device, dtype=dtype)
    z[..., :3] = torch.randn((B, N, 3), device=device, dtype=dtype) * float(prior_std)
    cp = str(color_prior)
    if cp == "uniform":
        z[..., 3:] = torch.rand((B, N, 3), device=device, dtype=dtype)
    elif cp == "zeros":
        z[..., 3:] = 0.0
    elif cp == "gauss":
        z[..., 3:] = torch.randn((B, N, 3), device=device, dtype=dtype) * float(color_prior_std)
    else:
        raise ValueError(f"Unknown color_prior={cp}")
    return z


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    ap = argparse.ArgumentParser(
        "TDCR Demo #2: motor-sequence pred only (denorm + timing)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--motor_seq", type=str, required=True)
    ap.add_argument("--demo_out", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--batch_size", type=int, default=1, help="Use 1 if you need per-sample exact timing.")
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--npoints", type=int, default=4096)
    ap.add_argument("--sample_steps", type=int, default=0)
    ap.add_argument("--sampler", type=str, default="heun", choices=["heun", "euler"])
    ap.add_argument("--prior_std", type=float, default=0.0)
    ap.add_argument("--guidance_scale", type=float, default=None)
    ap.add_argument("--no_clamp_rgb", action="store_true", default=False)
    ap.add_argument("--no_ema", action="store_true", default=False)
    ap.add_argument("--color_prior", type=str, default=None, choices=["uniform", "zeros", "gauss"])
    ap.add_argument("--color_prior_std", type=float, default=None)
    ap.add_argument("--seq_fixed_noise", action="store_true", default=True)
    ap.add_argument("--no_seq_fixed_noise", action="store_true", default=False)

    # motor seq csv options
    ap.add_argument("--motor_seq_delim", type=str, default=",", help="CSV delimiter")
    ap.add_argument("--motor_seq_has_header", action="store_true", default=False)

    # denorm
    ap.add_argument("--eval_norm_json", type=str, default=None)
    ap.add_argument("--eval_scale", type=float, default=None)
    ap.add_argument("--eval_center", type=float, nargs=3, default=None)

    args = ap.parse_args()

    if args.no_seq_fixed_noise:
        args.seq_fixed_noise = False

    set_all_seeds(int(args.seed))
    device = torch.device(args.device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    ckpt_args = ckpt.get("args", {}) or {}
    backbone = infer_backbone(ckpt)

    # hyperparams
    sample_steps = int(ckpt_args.get("sample_steps", 50))
    if int(args.sample_steps) > 0:
        sample_steps = int(args.sample_steps)

    prior_std = ckpt_args.get("prior_std", ckpt_args.get("point_prior_std", 1.0))
    prior_std = float(prior_std)
    if float(args.prior_std) > 0:
        prior_std = float(args.prior_std)

    guidance_scale = float(ckpt_args.get("guidance_scale", 0.0))
    if args.guidance_scale is not None:
        guidance_scale = float(args.guidance_scale)

    color_prior = str(ckpt_args.get("color_prior", "uniform"))
    color_prior_std = float(ckpt_args.get("color_prior_std", 1.0))
    if args.color_prior is not None:
        color_prior = str(args.color_prior)
    if args.color_prior_std is not None:
        color_prior_std = float(args.color_prior_std)

    # denorm
    center_np, scale_f, scale_info = load_eval_center_scale(
        eval_norm_json=args.eval_norm_json,
        eval_scale=args.eval_scale,
        eval_center=list(args.eval_center) if args.eval_center is not None else None,
    )
    print(f"[EvalScale] raw = norm * {scale_f:.8f} + center {center_np.tolist()} (source={scale_info.get('source')})")

    # model
    net = build_model_from_ckpt_args(backbone, ckpt_args).to(device)
    net.eval()
    if (not args.no_ema) and (ckpt.get("ema", None) is not None):
        state = ckpt["ema"]
        print("[Demo2] Loading EMA weights.")
    else:
        state = ckpt.get("model", None)
        print("[Demo2] Loading model weights.")
    if state is None:
        raise KeyError("Checkpoint missing both 'ema' and 'model'.")
    try:
        net.load_state_dict(state, strict=True)
    except Exception as e:
        print(f"[Demo2][WARN] strict load failed: {e}. Retrying strict=False.")
        net.load_state_dict(state, strict=False)

    point_dim = int(getattr(net, "point_dim", 3))
    cond_dim = int(getattr(net, "cond_dim", 0))

    # load motor sequence
    ctrl_norm, meta = load_motor_sequence(args.motor_seq, delim=str(args.motor_seq_delim), has_header=bool(args.motor_seq_has_header))
    T, D = ctrl_norm.shape
    if cond_dim > 0 and D != cond_dim:
        raise ValueError(f"motor_seq dim mismatch: got D={D}, expected cond_dim={cond_dim}")

    out_root = Path(args.demo_out)
    out_pred = out_root / "pred"
    out_pred.mkdir(parents=True, exist_ok=True)

    # Keep a copy of the input motor sequence for reproducibility
    try:
        src = Path(args.motor_seq)
        if src.exists():
            (out_root / f"motor_seq_input{src.suffix.lower()}").write_bytes(src.read_bytes())
    except Exception as e:
        print(f"[Demo2][WARN] Failed to copy motor_seq to output dir: {e}")
    (out_root / "meta.json").write_text(
        json.dumps(
            {
                "ckpt": args.ckpt,
                "motor_seq": meta,
                "npoints": int(args.npoints),
                "sample_steps": int(sample_steps),
                "sampler": str(args.sampler),
                "guidance_scale": float(guidance_scale),
                "prior_std": float(prior_std),
                "color_prior": str(color_prior),
                "color_prior_std": float(color_prior_std),
                "point_dim": int(point_dim),
                "cond_dim": int(cond_dim),
                "denorm": {"center": center_np.tolist(), "scale": float(scale_f), "source": scale_info.get("source")},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    clamp_rgb = not bool(args.no_clamp_rgb)
    sampler_fn = heun_sampler if args.sampler == "heun" else euler_sampler

    # fixed noise for all frames (optional)
    fixed_z: Optional[torch.Tensor] = None
    if bool(args.seq_fixed_noise):
        fixed_z = make_prior(
            B=1,
            N=int(args.npoints),
            C=point_dim,
            device=device,
            prior_std=prior_std,
            color_prior=color_prior,
            color_prior_std=color_prior_std,
            dtype=torch.float32,
        )

    # timing output (per sample)
    time_csv = out_root / "timing_compute_ms.csv"
    with open(time_csv, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(
            fcsv,
            fieldnames=["index", "batch_size", "compute_ms_batch", "compute_ms_per_sample_est"],
        )
        writer.writeheader()

        bs = int(args.batch_size)
        num_batches = (T + bs - 1) // bs
        pbar = tqdm(range(num_batches), desc="[Demo2] sampling", dynamic_ncols=True)
        for bi in pbar:
            s = bi * bs
            e = min((bi + 1) * bs, T)
            B = int(e - s)

            cond_np = ctrl_norm[s:e]
            cond_t: Optional[torch.Tensor] = None
            if cond_dim > 0:
                cond_t = torch.from_numpy(cond_np).to(device=device, dtype=torch.float32)

            if fixed_z is not None:
                z = fixed_z.expand(B, -1, -1).contiguous()
            else:
                z = make_prior(
                    B=B,
                    N=int(args.npoints),
                    C=point_dim,
                    device=device,
                    prior_std=prior_std,
                    color_prior=color_prior,
                    color_prior_std=color_prior_std,
                    dtype=torch.float32,
                )

            # --- compute timing (exclude disk) ---
            if device.type == "cuda" and torch.cuda.is_available():
                ev0 = torch.cuda.Event(enable_timing=True)
                ev1 = torch.cuda.Event(enable_timing=True)
                ev0.record()
                pred_t = sampler_fn(
                    net,
                    z,
                    cond_t,
                    steps=sample_steps,
                    guidance_scale=guidance_scale,
                    clamp_rgb=clamp_rgb,
                )
                ev1.record()
                torch.cuda.synchronize()
                ms_batch = float(ev0.elapsed_time(ev1))
            else:
                t0 = time.perf_counter()
                pred_t = sampler_fn(
                    net,
                    z,
                    cond_t,
                    steps=sample_steps,
                    guidance_scale=guidance_scale,
                    clamp_rgb=clamp_rgb,
                )
                ms_batch = float((time.perf_counter() - t0) * 1000.0)

            ms_per = ms_batch / max(B, 1)

            # save preds (denorm)
            pred_np = pred_t.detach().cpu().numpy().astype(np.float32)
            for k in range(B):
                idx = s + k
                arr = pred_np[k].copy()
                arr[:, :3] = denorm_xyz_np(arr[:, :3], center_np, scale_f)
                fn = out_pred / f"{idx:06d}.ply"
                if point_dim == 6:
                    arr[:, 3:6] = np.clip(arr[:, 3:6], 0.0, 1.0)
                    write_ply_xyzrgb(str(fn), arr)
                else:
                    write_ply_xyz(str(fn), arr[:, :3])

                writer.writerow(
                    {
                        "index": int(idx),
                        "batch_size": int(B),
                        "compute_ms_batch": float(ms_batch),
                        "compute_ms_per_sample_est": float(ms_per),
                    }
                )

    print(f"[Demo2] Done. Output: {str(out_root.resolve())}")


if __name__ == "__main__":
    main()