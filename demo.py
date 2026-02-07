from __future__ import annotations

import argparse
import json
import os
import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from models import VelocityNet
try:
    from models import HybridMLP
except Exception:
    HybridMLP = None

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
            f.write(
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )

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


@torch.no_grad()
def euler_sampler(
    net: torch.nn.Module,
    x0: torch.Tensor,
    cond: Optional[torch.Tensor],
    steps: int,
    guidance_scale: float,
    clamp_rgb: bool = True,
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
    clamp_rgb: bool = True,
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
    color_prior: str = "uniform",
    color_prior_std: float = 1.0,
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


# ============================================================
# 反归一化：raw = norm * scale + center
# ============================================================
def load_global_center_scale(global_norm_json: str) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    with open(global_norm_json, "r") as f:
        obj = json.load(f)

    if not isinstance(obj, dict) or len(obj) == 0:
        raise ValueError(f"Bad global_norm_json: {global_norm_json}")

    if "all" in obj and isinstance(obj["all"], dict):
        sub = obj["all"]
        key_used = "all"
    else:
        key_used = next(iter(obj.keys()))
        sub = obj[key_used]
        if not isinstance(sub, dict):
            raise ValueError(f"Bad global_norm_json[{key_used}]: {global_norm_json}")

    center = np.asarray(sub.get("center", [0, 0, 0]), dtype=np.float32).reshape(3)
    scale = float(sub.get("scale", 1.0))
    if not np.isfinite(center).all():
        raise ValueError(f"Invalid center in {global_norm_json}: {center}")
    if not (np.isfinite(scale) and scale > 0):
        raise ValueError(f"Invalid scale in {global_norm_json}: {scale}")

    info = {
        "global_norm_json": global_norm_json,
        "key_used": key_used,
        "center": center.tolist(),
        "scale": scale,
    }
    return center, scale, info


def denorm_xyz(xyz_norm: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    xyz_norm = np.asarray(xyz_norm, dtype=np.float32)
    return xyz_norm * float(scale) + center.reshape(1, 3).astype(np.float32)


# ============================================================
# 从 motor json 推断 global_norm json
# ============================================================
def _try_find_global_norm_json_near(path_like: str) -> Optional[str]:
    if not path_like:
        return None
    p = Path(path_like)
    cand_dirs = []
    if p.exists():
        cand_dirs.append(p.parent)
        cand_dirs.append(p.parent.parent)
    if p.is_dir():
        cand_dirs.insert(0, p)

    for d in cand_dirs:
        if d is None or not d.exists():
            continue
        pri = list(d.glob("global_norm_scope-all_anchor-origin.json"))
        if pri:
            return str(pri[0])
        any_json = sorted(d.glob("global_norm_*.json"))
        if any_json:
            return str(any_json[0])
    return None


def resolve_global_norm_json(
    motor_json_paths: List[str],
    explicit_eval_norm_json: Optional[str],
    test_dir: str,
) -> str:
    if explicit_eval_norm_json:
        if not Path(explicit_eval_norm_json).exists():
            raise FileNotFoundError(f"--eval_norm_json not found: {explicit_eval_norm_json}")
        return explicit_eval_norm_json

    for jp in motor_json_paths[:5]:
        try:
            obj = json.loads(Path(jp).read_text(encoding="utf-8"))
        except Exception:
            continue

        for k in ("data_norm_json", "eval_norm_json", "global_norm_json"):
            v = obj.get(k, None)
            if isinstance(v, str) and v.strip():
                if Path(v).exists():
                    return v
                rel = (Path(jp).parent / v).resolve()
                if rel.exists():
                    return str(rel)

        norm_obj = obj.get("norm", {}) if isinstance(obj.get("norm", {}), dict) else {}
        stats_file = norm_obj.get("stats_file", None)
        if isinstance(stats_file, str) and stats_file.strip():
            found = _try_find_global_norm_json_near(stats_file)
            if found:
                return found

    td = Path(test_dir).resolve()
    for d in [td, td.parent, td.parent.parent]:
        if d is None or not d.exists():
            continue
        pri = list(d.glob("global_norm_scope-all_anchor-origin.json"))
        if pri:
            return str(pri[0])
        any_json = sorted(d.glob("global_norm_*.json"))
        if any_json:
            return str(any_json[0])

    raise FileNotFoundError(
        "无法自动找到点云 global_norm_*.json。\n"
        "建议：\n"
        "1) 在 motor json 里增加字段 pc_norm_json 指向 global_norm_*.json；或\n"
        "2) 运行时显式传 --eval_norm_json datasets/.../global_norm_*.json"
    )


# ============================================================
# 读取 ctrl：优先 ctrl_norm，否则 ctrl；必要时从 raw 归一化到 [0,1]
# ============================================================
def read_ctrl_norm_from_motor_json(obj: Dict[str, Any]) -> np.ndarray:
    if "ctrl_norm" in obj:
        ctrl = np.asarray(obj["ctrl_norm"], dtype=np.float32).reshape(-1)
        return ctrl
    if "ctrl" not in obj:
        raise KeyError("motor json missing 'ctrl' (or 'ctrl_norm').")
    ctrl = np.asarray(obj["ctrl"], dtype=np.float32).reshape(-1)

    if (np.nanmin(ctrl) < -0.05 or np.nanmax(ctrl) > 1.05) and isinstance(obj.get("norm", None), dict):
        n = obj["norm"]
        if "min" in n and "max" in n:
            mn = np.asarray(n["min"], dtype=np.float32).reshape(-1)
            mx = np.asarray(n["max"], dtype=np.float32).reshape(-1)
            if mn.shape == ctrl.shape and mx.shape == ctrl.shape:
                scale = (mx - mn).astype(np.float32)
                scale[scale < 1e-6] = 1.0
                ctrl = (ctrl - mn) / scale
    return ctrl


def batched(items: List[Any], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def main() -> None:
    ap = argparse.ArgumentParser("Predict point clouds from test/motor/*.json (TDCR)")

    ap.add_argument("--ckpt", type=str, required=True, help="path to latest.pt / epoch_xxxx.pt")
    ap.add_argument("--test_dir", type=str, required=True, help="test dataset root (expects motor/*.json)")
    ap.add_argument("--demo_out", type=str, required=True, help="output folder")

    ap.add_argument("--npoints", type=int, default=20000, help="points to generate (override).")
    ap.add_argument("--cd_points", type=int, default=10000, help="alias of npoints (if npoints<=0).")

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--sample_steps", type=int, default=0, help="override ckpt sample_steps (<=0: no override)")
    ap.add_argument("--prior_std", type=float, default=0.0, help="override ckpt prior_std (<=0: no override)")
    ap.add_argument("--guidance_scale", type=float, default=None, help="override ckpt guidance_scale (if set)")
    ap.add_argument("--sampler", type=str, default="heun", choices=["heun", "euler"])
    ap.add_argument("--no_clamp_rgb", action="store_true", default=False)
    ap.add_argument("--no_ema", action="store_true", default=False)

    ap.add_argument("--color_prior", type=str, default=None, choices=["uniform", "zeros", "gauss"])
    ap.add_argument("--color_prior_std", type=float, default=None)

    ap.add_argument("--eval_norm_json", type=str, default=None, help="explicit global_norm_*.json (optional)")

    args = ap.parse_args()

    # device
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA not available, fallback to CPU.")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # seed
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    np_rng = np.random.RandomState(int(args.seed))

    # load ckpt
    ckpt = torch.load(args.ckpt, map_location="cpu")
    ckpt_args = ckpt.get("args", {}) or {}
    backbone = infer_backbone(ckpt)
    print(f"[DemoTest] Backbone inferred: {backbone}")

    # point_dim / rgb
    use_rgb = bool(ckpt_args.get("use_rgb", False))
    point_dim = int(ckpt_args.get("point_dim", 6 if use_rgb else 3))
    if point_dim not in (3, 6):
        raise ValueError(f"Unsupported point_dim={point_dim} (expect 3 or 6).")

    # sampling hyperparams
    sample_steps = int(ckpt_args.get("sample_steps", 50))
    if int(args.sample_steps) > 0:
        sample_steps = int(args.sample_steps)

    prior_std = ckpt_args.get("prior_std", None)
    if prior_std is None:
        prior_std = ckpt_args.get("point_prior_std", 1.0)
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

    clamp_rgb = not bool(args.no_clamp_rgb)
    sampler_fn = heun_sampler if args.sampler == "heun" else euler_sampler

    # npoints
    npoints = int(args.npoints) if int(args.npoints) > 0 else 0
    if npoints <= 0 and int(args.cd_points) > 0:
        npoints = int(args.cd_points)
    if npoints <= 0:
        te_mp = int(ckpt_args.get("te_max_sample_points", 0) or 0)
        if te_mp > 0:
            npoints = te_mp
    if npoints <= 0:
        npoints = 4096
    print(f"[DemoTest] npoints={npoints}, steps={sample_steps}, prior_std={prior_std}, guidance_scale={guidance_scale}")

    # build model
    net = build_model_from_ckpt_args(backbone, ckpt_args).to(device)
    net.eval()

    # load weights (prefer EMA unless --no_ema)
    if (not args.no_ema) and (ckpt.get("ema", None) is not None):
        state = ckpt["ema"]
        print("[DemoTest] Loading EMA weights.")
    else:
        state = ckpt.get("model", None)
        print("[DemoTest] Loading model weights.")
    if state is None:
        raise KeyError("Checkpoint missing both 'ema' and 'model'.")

    try:
        net.load_state_dict(state, strict=True)
    except Exception as e:
        print(f"[DemoTest][WARN] strict load failed: {e}. Retrying strict=False.")
        net.load_state_dict(state, strict=False)

    # collect motor jsons
    motor_dir = Path(args.test_dir) / "motor"
    if not motor_dir.exists():
        raise FileNotFoundError(f"motor dir not found: {motor_dir}")
    motor_jsons = sorted(glob.glob(str(motor_dir / "*.json")))
    if not motor_jsons:
        raise FileNotFoundError(f"No motor json found under: {motor_dir}")

    # resolve global_norm json (for de-normalization)
    global_norm_json = resolve_global_norm_json(motor_jsons, args.eval_norm_json, args.test_dir)
    center, scale, scale_info = load_global_center_scale(global_norm_json)
    print(f"[DemoTest] Using global_norm_json={global_norm_json}")
    print(f"[DemoTest] Denorm: raw = norm * {scale:.8f} + center {center.tolist()}")

    # cond dim check
    cond_dim = int(ckpt_args.get("cond_dim", 0))
    if cond_dim <= 0:
        print("[DemoTest] cond_dim==0 (unconditional model). Will ignore ctrl in json.")
    else:
        print(f"[DemoTest] cond_dim={cond_dim} (expects ctrl length matches).")

    # output dirs
    out_root = Path(args.demo_out)
    out_norm = out_root / "pred_norm"
    out_den = out_root / "pred_denorm"
    out_root.mkdir(parents=True, exist_ok=True)
    out_norm.mkdir(parents=True, exist_ok=True)
    out_den.mkdir(parents=True, exist_ok=True)

    # predict
    with torch.no_grad():
        for batch_paths in batched(motor_jsons, int(args.batch_size)):
            # read ctrl
            ctrls: List[np.ndarray] = []
            stems: List[str] = []
            for jp in batch_paths:
                obj = json.loads(Path(jp).read_text(encoding="utf-8"))
                stems.append(Path(jp).stem)

                if cond_dim <= 0:
                    # unconditional: still append dummy
                    ctrls.append(np.zeros((0,), dtype=np.float32))
                    continue

                ctrl = read_ctrl_norm_from_motor_json(obj).astype(np.float32).reshape(-1)
                if ctrl.shape[0] != cond_dim:
                    raise ValueError(
                        f"ctrl dim mismatch for {jp}: got {ctrl.shape[0]}, expect {cond_dim}"
                    )
                ctrls.append(ctrl)

            B = len(batch_paths)
            cond_t: Optional[torch.Tensor]
            if cond_dim <= 0:
                cond_t = None
            else:
                cond_np = np.stack(ctrls, axis=0).astype(np.float32)  # (B,D)
                cond_t = torch.from_numpy(cond_np).to(device=device, dtype=torch.float32)

            # prior
            z = make_prior(
                B=B,
                N=npoints,
                C=point_dim,
                device=device,
                prior_std=prior_std,
                color_prior=color_prior,
                color_prior_std=color_prior_std,
                dtype=torch.float32,
            )

            # sample
            pred_t = sampler_fn(
                net,
                z,
                cond_t,
                steps=sample_steps,
                guidance_scale=float(guidance_scale),
                clamp_rgb=clamp_rgb,
            )

            pred_np = pred_t.detach().cpu().numpy().astype(np.float32)  # (B,N,C)

            # save
            for b, stem in enumerate(stems):
                # 1) 保存归一化 pred
                pn = pred_np[b]
                norm_path = str(out_norm / f"{stem}.ply")
                if point_dim == 6:
                    pn2 = pn.copy()
                    pn2[:, 3:6] = np.clip(pn2[:, 3:6], 0.0, 1.0)
                    write_ply_xyzrgb(norm_path, pn2)
                else:
                    write_ply_xyz(norm_path, pn[:, :3])

                # 2) 保存反归一化 pred（只对 xyz 反归一化）
                xyz_den = denorm_xyz(pn[:, :3], center=center, scale=scale)
                den_path = str(out_den / f"{stem}.ply")
                if point_dim == 6:
                    pd = pn.copy()
                    pd[:, :3] = xyz_den
                    pd[:, 3:6] = np.clip(pd[:, 3:6], 0.0, 1.0)
                    write_ply_xyzrgb(den_path, pd)
                else:
                    write_ply_xyz(den_path, xyz_den)

    # write summary
    summary = {
        "ckpt": str(args.ckpt),
        "test_dir": str(Path(args.test_dir).resolve()),
        "demo_out": str(out_root.resolve()),
        "pred_norm_dir": str(out_norm.resolve()),
        "pred_denorm_dir": str(out_den.resolve()),
        "global_norm_json": str(Path(global_norm_json).resolve()),
        "scale_info": scale_info,
        "backbone": backbone,
        "point_dim": int(point_dim),
        "cond_dim": int(cond_dim),
        "npoints": int(npoints),
        "sample_steps": int(sample_steps),
        "prior_std": float(prior_std),
        "guidance_scale": float(guidance_scale),
        "sampler": str(args.sampler),
        "clamp_rgb": bool(clamp_rgb),
        "color_prior": str(color_prior),
        "color_prior_std": float(color_prior_std),
        "seed": int(args.seed),
        "num_samples": int(len(motor_jsons)),
        "device": str(device),
    }
    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[DemoTest] Done. Saved {len(motor_jsons)} preds to:")
    print(f"  - {out_norm}")
    print(f"  - {out_den}")
    print(f"[DemoTest] summary.json written to: {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
