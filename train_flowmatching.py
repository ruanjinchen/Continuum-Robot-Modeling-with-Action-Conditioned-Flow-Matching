from __future__ import annotations
import os, argparse, json, random, re
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch import distributed as dist
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from models import VelocityNet, HybridMLP
from utils import (
    EMA, seed_all, init_distributed, cleanup_distributed, cosine_lr,
    save_point_cloud_ply, count_parameters
)
from datasets import get_datasets, init_np_seed

_USE_NEW_AMP = hasattr(torch, "amp") and hasattr(torch.amp, "autocast")

def make_autocast(enabled: bool, use_bf16: bool):
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    if _USE_NEW_AMP:
        return torch.amp.autocast("cuda", enabled=enabled, dtype=dtype)
    else:
        from torch.cuda.amp import autocast as _autocast
        return _autocast(enabled=enabled, dtype=dtype)

def make_scaler(enabled: bool):
    if _USE_NEW_AMP:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    else:
        from torch.cuda.amp import GradScaler as _GradScaler
        return _GradScaler(enabled=enabled)

def sample_noise_like(x: torch.Tensor, std: float = 1.0) -> torch.Tensor:
    return torch.randn_like(x) * std

def make_point_prior_like(x: torch.Tensor, args) -> torch.Tensor:
    """
    x: (B,N,3) or (B,N,6)
    - xyz prior: N(0, prior_std^2)
    - rgb prior (when 6D): U(0,1)  (or controlled by args.color_prior)
    """
    D = x.shape[-1]
    if D == 3:
        return torch.randn_like(x) * args.prior_std

    if D != 6:
        raise ValueError(f"Unsupported point dim: {D}")

    z = torch.empty_like(x)
    z[..., :3] = torch.randn_like(x[..., :3]) * args.prior_std

    # default: uniform [0,1]
    color_prior = getattr(args, "color_prior", "uniform")
    if color_prior == "uniform":
        z[..., 3:] = torch.rand_like(x[..., 3:])
    elif color_prior == "zeros":
        z[..., 3:] = 0.0
    elif color_prior == "gauss":
        std = float(getattr(args, "color_prior_std", 1.0))
        z[..., 3:] = torch.randn_like(x[..., 3:]) * std
    else:
        raise ValueError(f"Unknown color_prior: {color_prior}")

    return z


def build_model(args) -> nn.Module:
    if getattr(args, "pf_backbone", "mlp") == "mlp":
        m = VelocityNet(
            cond_dim=args.cond_dim,
            point_dim=args.point_dim,
            width=args.width,
            depth=args.depth,
            emb_dim=args.emb_dim,
            cfg_dropout_p=args.cfg_drop_p,
        )
    else:
        # HybridMLP: ContextNet (PVConv pyramid) + VelocityNetWithContext
        if HybridMLP is None:
            raise ImportError("HybridMLP import failed. Please verify models.py (HybridMLP) and "
                              "that third_party/pvcnn is installed correctly.")

        m = HybridMLP(
            cond_dim=args.cond_dim,
            point_dim=args.point_dim,  # 3D or 6D controlled by --use_rgb
            ctx_dim=args.ctx_dim,
            ctx_emb_dim=args.ctx_emb_dim,
            stage_channels=args.ctx_stage_channels,
            stage_blocks=args.ctx_stage_blocks,
            stage_res=args.ctx_stage_res,
            with_se=args.ctx_with_se,
            norm_type=args.ctx_norm,
            gn_groups=args.ctx_gn_groups,
            with_global=args.ctx_with_global,
            voxel_normalize=args.ctx_voxel_normalize,
            # t-gate
            use_t_gate=True,
            t_gate_k=args.ctx_t_gate_k,
            t_gate_tau=args.ctx_t_gate_tau,
            pf_width=args.width,
            pf_depth=args.depth,
            pf_emb_dim=args.emb_dim,
            cfg_dropout_p=args.cfg_drop_p,
        )
    return m

@torch.no_grad()
def heun_sampler(model: nn.Module, x0: torch.Tensor, cond: Optional[torch.Tensor],
                 steps: int = 50, guidance_scale: float = 0.0, use_ema: bool = True) -> torch.Tensor:
    """
    Heun / RK2 sampler for dx/dt = v(x,t,cond), integrate t:0->1
    """
    device = x0.device
    dt = 1.0 / steps
    x = x0
    net = model.module if hasattr(model, "module") else model

    backup = None
    if use_ema and hasattr(net, "ema_shadow"):
        backup = {k: v.detach().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(net.ema_shadow, strict=True)

    B = x.shape[0]
    for k in range(steps):
        t0 = torch.full((B,), k * dt, device=device, dtype=x.dtype)
        v1 = net.guided_velocity(x, t0, cond, guidance_scale=guidance_scale)

        x_hat = x + v1 * dt

        t1 = torch.full((B,), (k + 1) * dt, device=device, dtype=x.dtype)
        v2 = net.guided_velocity(x_hat, t1, cond, guidance_scale=guidance_scale)

        x = x + 0.5 * dt * (v1 + v2)

    if backup is not None:
        net.load_state_dict(backup, strict=True)
    return x

@torch.no_grad()
def euler_sampler(model: nn.Module, x0: torch.Tensor, cond: Optional[torch.Tensor],
                  steps: int = 50, guidance_scale: float = 0.0, use_ema: bool = True) -> torch.Tensor:
    device = x0.device
    dt = 1.0 / steps
    x = x0
    net = model.module if hasattr(model, "module") else model
    backup = None
    if use_ema and hasattr(net, "ema_shadow"):
        backup = {k: v.detach().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(net.ema_shadow, strict=True)
    for i in range(steps):
        t = torch.full((x.shape[0],), (i + 0.5) * dt, device=device, dtype=x.dtype)
        v = net.guided_velocity(x, t, cond, guidance_scale=guidance_scale)  # Note: cond is also used during sampling (CFG-compatible).
        x = x + v * dt
    if backup is not None:
        net.load_state_dict(backup, strict=True)
    return x

# --------- Chamfer Distance: prefer compiled chamfer_3D ---------
_CHAMFER_EXT = None
_CHAMFER_EXT_FAILED = False

def _load_chamfer_ext():
    """Try to load chamfer_3D once; if it fails, always use the torch.cdist fallback."""
    global _CHAMFER_EXT, _CHAMFER_EXT_FAILED
    if _CHAMFER_EXT is not None or _CHAMFER_EXT_FAILED:
        return _CHAMFER_EXT
    try:
        import importlib
        _CHAMFER_EXT = importlib.import_module("chamfer_3D")
        print("[Chamfer] Using compiled chamfer_3D extension.")
    except Exception as e:
        print(f"[Chamfer][WARN] chamfer_3D not available: {e}. Falling back to torch.cdist.")
        _CHAMFER_EXT_FAILED = True
        _CHAMFER_EXT = None
    return _CHAMFER_EXT

@torch.no_grad()
def chamfer_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    pred, target: (B, N, 3)
    Returns: (B,) per-batch Chamfer Distance (CD).
    """
    ext = _load_chamfer_ext()
    if ext is not None:
        B, N, _ = pred.shape
        device = pred.device

        # Compiled op expects float32 + contiguous tensors.
        x = pred.contiguous().to(device=device, dtype=torch.float32)
        y = target.contiguous().to(device=device, dtype=torch.float32)

        d1 = torch.empty(B, N, device=device, dtype=torch.float32)
        d2 = torch.empty(B, N, device=device, dtype=torch.float32)
        i1 = torch.empty(B, N, device=device, dtype=torch.int32)
        i2 = torch.empty(B, N, device=device, dtype=torch.int32)

        _ = ext.forward(x, y, d1, d2, i1, i2)
        cd = d1.mean(dim=1) + d2.mean(dim=1)     # (B,)
        return cd.to(pred.dtype)

    # Fallback: torch.cdist implementation.
    d2 = torch.cdist(pred, target, p=2).pow(2)
    return d2.min(dim=2).values.mean(dim=1) + d2.min(dim=1).values.mean(dim=1)


@torch.no_grad()
def save_vis_samples(args,
                     model,
                     epoch: int,
                     val_batch,
                     out_dir: str,
                     guidance_scale: float = 1.0,
                     use_ema: bool = True,
                     rank: int = 0,
                     writer: Optional[SummaryWriter] = None) -> float:
    """
    Generate a few visualization samples from one validation batch and compute CD.
    Returns cd_mean (float).
    """
    os.makedirs(out_dir, exist_ok=True)
    net = model.module if hasattr(model, "module") else model
    net.eval()
    with torch.inference_mode():
        pts = val_batch["test_points"].to(args.device, non_blocking=True)
        cond = val_batch.get("cond", None)
        if cond is not None:
            cond = cond.to(args.device, non_blocking=True).float()
        z = make_point_prior_like(pts, args)
        pred = heun_sampler(model, z, cond,
                            steps=args.sample_steps,
                            guidance_scale=guidance_scale,
                            use_ema=use_ema)
        cd = chamfer_l2(pred[..., :3], pts[..., :3])  # only xyz for CD
        cd_mean = float(cd.mean().detach().cpu())

        if rank == 0:
            print(f"[epoch {epoch:04d}] CD-L2(cond) mean = {cd_mean:.6f}")
            if writer is not None:
                writer.add_scalar("val/cd", cd_mean, epoch)

        for i in range(min(pts.shape[0], args.vis_count)):
            save_point_cloud_ply(pred[i], os.path.join(out_dir, f"ep{epoch:04d}_cond_{i}.ply"))
            save_point_cloud_ply(pts[i],  os.path.join(out_dir, f"ep{epoch:04d}_gt_{i}.ply"))

    return cd_mean


def train_one_epoch(model, net, opt, scaler, train_loader,
                    epoch, args, ema: EMA,
                    rank: int = 0, world_size: int = 1,
                    writer: Optional[SummaryWriter] = None):
    model.train()
    pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}") if rank == 0 else None
    loss_sum = 0.0
    n_step = 0
    for batch in train_loader:
        gnorm = None
        pts = batch["train_points"].to(args.device, non_blocking=True).float()
        cond = batch.get("cond", None)
        if cond is not None:
            cond = cond.to(args.device, non_blocking=True).float()
        B, N, _ = pts.shape
        z = make_point_prior_like(pts, args)

        if getattr(args, "t_beta_a", 1.0) is None or abs(float(args.t_beta_a) - 1.0) < 1e-8:
            t = torch.rand(B, device=args.device, dtype=pts.dtype)  # uniform
        else:
            a = float(args.t_beta_a)
            assert a > 0.0
            u = torch.rand(B, device=args.device, dtype=pts.dtype)
            t = u.pow(1.0 / a)  # Beta(a,1)

        x_t = (1.0 - t)[:, None, None] * z + t[:, None, None] * pts
        target_v = (pts - z)
        cond_drop_mask = None
        if args.cfg_drop_p > 0.0 and args.cond_dim > 0 and cond is not None:
            drop = (torch.rand(B, device=args.device) < args.cfg_drop_p).to(pts.dtype)
            cond_drop_mask = drop[:, None]  # (B,1)

        with make_autocast(enabled=args.amp, use_bf16=args.use_bf16):
            pred_v = model(x_t, t, cond, cond_drop_mask=cond_drop_mask)
            if pred_v.shape[-1] == 6:
                loss_pos = F.mse_loss(pred_v[..., :3], target_v[..., :3])
                loss_col = F.mse_loss(pred_v[..., 3:], target_v[..., 3:])
                loss = loss_pos + float(args.lambda_color) * loss_col
            else:
                loss = F.mse_loss(pred_v, target_v)

        scaler.scale(loss).backward()
        if args.grad_clip_norm is not None and args.grad_clip_norm > 0:
            scaler.unscale_(opt)
            # clip_grad_norm_ returns the global grad norm BEFORE clipping.
            gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip_norm)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update(net)
        args.global_step += 1
        if args.use_cosine_lr:
            lr_now = cosine_lr(args.global_step, args.total_steps, args.lr,
                               min_lr=args.min_lr, warmup=args.warmup_steps)
            for pg in opt.param_groups:
                pg["lr"] = lr_now
        else:
            lr_now = opt.param_groups[0]["lr"]
        n_step += 1

        loss_v = float(loss.detach().cpu())
        loss_sum += loss_v

        # ---- TensorBoard: train metric ----
        if writer is not None and rank == 0:
            if args.tb_log_every <= 1 or (args.global_step % args.tb_log_every == 0):
                writer.add_scalar("train/loss", loss_v, args.global_step)
                writer.add_scalar("train/lr", float(lr_now), args.global_step)
                if gnorm is not None:
                    g_val = float(gnorm)
                    writer.add_scalar("train/grad_norm", g_val, args.global_step)
                    # >0 means clipping was triggered on this step.
                    if args.grad_clip_norm is not None and args.grad_clip_norm > 0:
                        writer.add_scalar(
                            "train/grad_clipped",
                            float(g_val > args.grad_clip_norm),
                            args.global_step
                        )

        if pbar is not None:
            pbar.set_postfix(loss=float(loss.detach().cpu()),
                             loss_avg=loss_sum / n_step,
                             lr=f"{lr_now:.2e}")
            pbar.update(1)
    if pbar is not None:
        pbar.close()
# =========================
# [Auto-Resume] helper utilities
# =========================

def _find_latest_ckpt(ckpt_dir: str):
    """Return (path, epoch). If none found, returns (None, 0). Matches epoch_XXXX.pt."""
    if not os.path.isdir(ckpt_dir):
        return None, 0
    best_ep, best_path = 0, None
    for fn in os.listdir(ckpt_dir):
        m = re.match(r"epoch_(\d+)\.pt$", fn)
        if m:
            ep = int(m.group(1))
            if ep > best_ep:
                best_ep = ep
                best_path = os.path.join(ckpt_dir, fn)
    return best_path, best_ep

def _find_resume_ckpt(ckpt_dir: str):
    """Prefer latest.pt, otherwise pick the largest epoch_XXXX.pt."""
    latest = os.path.join(ckpt_dir, "latest.pt")
    if os.path.isfile(latest):
        return latest, None  # epoch will be read from the checkpoint dict
    return _find_latest_ckpt(ckpt_dir)

def _move_opt_state_to_device(opt: torch.optim.Optimizer, device: torch.device):
    """Move optimizer state tensors to the target device (prevents device mismatch after resume)."""
    for st in opt.state.values():
        for k, v in list(st.items()):
            if torch.is_tensor(v):
                st[k] = v.to(device)

def _safe_load_ema(ema_obj: EMA, state_dict: dict, ref_model: nn.Module, device: torch.device):
    """
    Overwrite matching EMA keys from ckpt into ema_obj.shadow and move tensors to device.
    Avoids KeyError and supports checkpoints saved on different devices (CPU/GPU).
    """
    cur = ema_obj.shadow
    ref_sd = ref_model.state_dict()
    for k in cur.keys():
        if k in state_dict:
            v = state_dict[k]
            if torch.is_tensor(v) and v.dtype.is_floating_point:
                cur[k] = v.to(device=device, dtype=ref_sd[k].dtype)
    ema_obj.shadow = cur

def atomic_torch_save(obj, path: str):
    """Atomic save: write to .tmp first, then rename to replace."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

def _capture_rng_states():
    """Capture RNG states for Python/NumPy/Torch/CUDA for fully reproducible resume."""
    np_state = np.random.get_state()
    # Convert ndarray to list to avoid deserialization constraints in some environments.
    if isinstance(np_state, tuple) and len(np_state) >= 2 and hasattr(np_state[1], "tolist"):
        np_state = (np_state[0], np_state[1].tolist(), *np_state[2:])
    return {
        "torch": torch.get_rng_state(),
        "cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np_state,
        "python": random.getstate(),
    }

def _restore_rng_states(rng):
    """Restore RNG states; warn on failure but do not interrupt training."""
    if rng is None:
        return
    try:
        if "torch" in rng and rng["torch"] is not None:
            st = rng["torch"]
            # Must be CPU + uint8
            if torch.is_tensor(st):
                st = st.detach()
                if st.is_cuda:
                    st = st.cpu()
                if st.dtype != torch.uint8:
                    st = st.to(torch.uint8)
            else:
                # Fallback: older checkpoints might store this as list/bytes
                st = torch.tensor(st, dtype=torch.uint8, device="cpu")
            torch.set_rng_state(st)

        if "cuda_all" in rng and rng["cuda_all"] is not None and torch.cuda.is_available():
            cuda_states = rng["cuda_all"]
            fixed = []
            for s in cuda_states:
                if torch.is_tensor(s):
                    s = s.detach()
                    if s.is_cuda:
                        s = s.cpu()
                    if s.dtype != torch.uint8:
                        s = s.to(torch.uint8)
                else:
                    s = torch.tensor(s, dtype=torch.uint8, device="cpu")
                fixed.append(s)

            # If the visible GPU count changed, avoid set_rng_state_all failing outright.
            n_dev = torch.cuda.device_count()
            if len(fixed) != n_dev:
                fixed = fixed[:n_dev] if len(fixed) > n_dev else (fixed + [fixed[-1]] * (n_dev - len(fixed)))

            torch.cuda.set_rng_state_all(fixed)

        if "numpy" in rng and rng["numpy"] is not None:
            np_state = rng["numpy"]
            if isinstance(np_state, tuple) and len(np_state) >= 2 and isinstance(np_state[1], list):
                arr = np.array(np_state[1], dtype=np.uint32)
                np_state = (np_state[0], arr, *np_state[2:])
            np.random.set_state(np_state)
        if "python" in rng and rng["python"] is not None:
            random.setstate(rng["python"])
    except Exception as e:
        print(f"[WARN] Failed to restore RNG states: {e}")

def main():
    parser = argparse.ArgumentParser("MeanFlow training (Gaussian -> point cloud)")
    # Data
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset root (contains train/val/test shards).")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)

    # Dataset-specific knobs (forwarded to get_datasets)
    parser.add_argument("--tdcr_use_norm", action="store_true", default=True, help="If set, prefer data_norm when available.")
    parser.add_argument("--tr_max_sample_points", type=int, default=2048)
    parser.add_argument("--te_max_sample_points", type=int, default=2048)
    parser.add_argument("--cond_mode", type=str, default="motors")
    parser.add_argument("--train_fraction", type=float, default=1.0)
    parser.add_argument("--train_subset_seed", type=int, default=0)
    # RGB / 6D points
    parser.add_argument(
        "--use_rgb", action="store_true", default=False,
        help="If set, use 6D xyzrgb points. RGB is read from H5 key `--rgb_key` "
            "and normalized from [0,255] to [-1,1]. Model output will also be 6D."
    )
    parser.add_argument(
        "--rgb_key", type=str, default="rgb",
        help="H5 dataset key for per-point RGB with shape (B,N,3), values in [0,255]. "
            "Only used when --use_rgb."
    )

    # Model
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--cfg_drop_p", type=float, default=0.0)
    # Point-flow backbone: mlp (original VelocityNet) or hybrid (PVConv context + per-point MLP).
    parser.add_argument("--pf_backbone", type=str, default="mlp",
                        choices=["mlp", "hybrid"],
                        help="point-flow backbone: 'mlp' or 'hybrid'")

    # Hybrid context branch (ContextNet) hyperparameters
    parser.add_argument("--ctx_dim", type=int, default=64)
    parser.add_argument("--ctx_emb_dim", type=int, default=256)
    parser.add_argument("--ctx_stage_channels", type=int, nargs="+",
                        default=[128, 256, 256])
    parser.add_argument("--ctx_stage_blocks", type=int, nargs="+",
                        default=[2, 2, 2])
    parser.add_argument("--ctx_stage_res", type=int, nargs="+",
                        default=[32, 16, 8])
    parser.add_argument("--ctx_with_se", action="store_true", default=True)
    parser.add_argument("--ctx_norm", type=str, default="group",
                        choices=["group", "batch", "syncbn", "none"])
    parser.add_argument("--ctx_gn_groups", type=int, default=32)
    parser.add_argument("--ctx_with_global", action="store_true", default=True)
    parser.add_argument("--ctx_voxel_normalize", action="store_true", default=True)

    # t-gate (controls when to rely more on PVConv context)
    parser.add_argument("--ctx_t_gate_tau", type=float, default=0.95)
    parser.add_argument("--ctx_t_gate_k", type=float, default=5.0)

    # Optim
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--use_cosine_lr", action="store_true", default=True)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)

    # Flow / sampling / I/O
    parser.add_argument("--prior_std", type=float, default=1.0)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--t_beta_a", type=float, default=1.0,
                        help="Sample t ~ Beta(a, 1). a=1 -> uniform; a>1 biases t towards 1.")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save a numbered checkpoint every N epochs (epoch_xxxx.pt).")
    parser.add_argument("--val_every", type=int, default=0,
                        help="Run val + visualization every N epochs; <=0 uses save_every.")
    parser.add_argument("--vis_count", type=int, default=16)
    parser.add_argument("--save_uncond", action="store_true", default=True)
    parser.add_argument("--guidance_scale", type=float, default=0.0)
    parser.add_argument("--lambda_color", type=float, default=0.05,
                        help="Weight for RGB loss when point_dim=6 (xyzrgb).")
    parser.add_argument("--color_prior", type=str, default="uniform",
                        choices=["uniform", "zeros", "gauss"],
                        help="RGB prior type when --use_rgb. Default: uniform in [0,1].")
    parser.add_argument("--color_prior_std", type=float, default=1.0,
                        help="RGB prior std when --color_prior=gauss.")
    parser.add_argument("--point_prior_std", type=float, default=None,
                        help="Alias of --prior_std (XYZ gaussian prior std). If set, overrides --prior_std.")

    # System / I/O
    parser.add_argument("--out_dir", type=str, default="./runs/fm_tdcr")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--use_bf16", action="store_true", default=True)
    parser.add_argument("--compile", action="store_true", default=False)

    # TensorBoard
    parser.add_argument("--no_tb", action="store_true",
                        help="Disable TensorBoard logging (enabled by default).")
    parser.add_argument("--tb_log_dir", type=str, default=None,
                        help="TensorBoard log dir (default: out_dir/tb).")
    parser.add_argument("--tb_log_every", type=int, default=50,
                        help="Log training metrics every N global steps; <=1 logs every step.")

    args = parser.parse_args()
    # global point dimension
    args.point_dim = 6 if args.use_rgb else 3
    if args.point_prior_std is not None:
        args.prior_std = float(args.point_prior_std)

    # ddp init
    is_dist, rank, world_size, local_rank = init_distributed()
    args.is_distributed = is_dist
    args.rank = rank
    args.world_size = world_size
    args.local_rank = local_rank
    args.device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    
    # Default: val_every follows save_every
    if args.val_every is None or args.val_every <= 0:
        args.val_every = args.save_every
    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)

    seed_all(args.seed + rank)

    # Datasets (get_datasets will populate args.cond_dim)
    tr_ds, te_ds = get_datasets(args)   # get_datasets also assembles cond and sets args.cond_dim

    # samplers & loaders
    if is_dist:
        tr_sampler = DistributedSampler(tr_ds, shuffle=True, drop_last=True)
        te_sampler = DistributedSampler(te_ds, shuffle=False, drop_last=False)
    else:
        tr_sampler = None
        te_sampler = None

    train_loader = DataLoader(
        tr_ds, batch_size=args.batch_size, shuffle=(tr_sampler is None),
        sampler=tr_sampler, num_workers=args.num_workers, drop_last=True,
        pin_memory=True, worker_init_fn=init_np_seed
    )
    val_loader = DataLoader(
        te_ds, batch_size=args.batch_size, shuffle=False,
        sampler=te_sampler, num_workers=max(1, args.num_workers // 2), drop_last=False,
        pin_memory=True, worker_init_fn=init_np_seed
    )

    net = build_model(args).to(args.device)
    if args.compile:
        net = torch.compile(net)

    ema = EMA(net, decay=0.999)
    net.ema_shadow = ema.shadow

    model = net
    if is_dist:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(net, device_ids=[local_rank], output_device=local_rank,
                    broadcast_buffers=False, find_unused_parameters=False)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = make_scaler(enabled=args.amp)

    if rank == 0:
        print(f"Model params: {count_parameters(net)/1e6:.2f} M")
        print(f"cond_dim={args.cond_dim}  width={args.width} depth={args.depth} emb_dim={args.emb_dim}")
        print(f"WorldSize={world_size}  Device={args.device}")

    args.total_steps = args.epochs * max(1, len(train_loader))
    args.global_step = 0

    ckpt_dir = os.path.join(args.out_dir, "ckpts")
    start_epoch = 1

    ckpt_path, ckpt_ep = _find_resume_ckpt(ckpt_dir)
    if ckpt_path is not None:
        if rank == 0:
            print(f"[Resume] Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # Restore model
        if "model" in ckpt:
            net.load_state_dict(ckpt["model"], strict=True)

        # Restore optimizer
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
            _move_opt_state_to_device(opt, args.device)

        # Restore EMA
        if ema is not None and "ema" in ckpt:
            _safe_load_ema(ema, ckpt["ema"], net, args.device)

        # Restore scaler (if any)
        if "scaler" in ckpt and scaler is not None:
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception as e:
                if rank == 0:
                    print(f"[WARN] Failed to load scaler from ckpt: {e}")

        # epoch / global_step / RNG
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1
        if "global_step" in ckpt:
            args.global_step = int(ckpt["global_step"])
        _restore_rng_states(ckpt.get("rng_state", None))

        if rank == 0:
            print(f"[Resume] Start from epoch {start_epoch}, global_step={args.global_step}")
    else:
        if rank == 0:
            print("[Resume] No checkpoint found, training from scratch.")

    val_iter = iter(val_loader)
    try:
        val_batch = next(val_iter)
    except StopIteration:
        val_batch = next(iter(val_loader))

    writer = None
    if (rank == 0) and (not args.no_tb):
        tb_dir = args.tb_log_dir if args.tb_log_dir is not None else os.path.join(args.out_dir, "tb")
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)

    if rank == 0:
        print(f"Model params: {count_parameters(net)/1e6:.2f} M")
        print(f"cond_dim={args.cond_dim}  width={args.width} depth={args.depth} emb_dim={args.emb_dim}")
        print(f"WorldSize={world_size}  Device={args.device}")


    try:
        for ep in range(start_epoch, args.epochs + 1):
            if is_dist and isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(ep)
            train_one_epoch(model, net, opt, scaler, train_loader, ep, args, ema, rank, world_size, writer=writer)
            if is_dist and dist.is_initialized():
                dist.barrier()
            # ---- 2) Save checkpoints: write latest.pt every epoch, plus history every save_every ----
            if rank == 0:
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_obj = {
                    "epoch": ep,
                    "model": net.state_dict(),
                    "ema": ema.shadow if ema is not None else None,
                    "opt": opt.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "global_step": args.global_step,
                    "rng_state": _capture_rng_states(),
                    "args": vars(args),
                }
                # latest
                atomic_torch_save(ckpt_obj, os.path.join(ckpt_dir, "latest.pt"))
                # History archive
                if (ep % args.save_every == 0) or (ep == args.epochs):
                    atomic_torch_save(ckpt_obj, os.path.join(ckpt_dir, f"epoch_{ep:04d}.pt"))

            if is_dist and dist.is_initialized():
                dist.barrier()

            # ---- 3) val + visualization + write val/CD ----
            do_val = (ep % args.val_every == 0) or (ep == args.epochs)
            if do_val and rank == 0:
                vis_dir = os.path.join(args.out_dir, "samples", f"epoch_{ep:04d}")
                cd_mean = save_vis_samples(
                    args, model, ep, val_batch, vis_dir,
                    guidance_scale=args.guidance_scale,
                    use_ema=True,
                    rank=rank,
                    writer=writer,
                )

            if is_dist and dist.is_initialized():
                dist.barrier()
    finally:
        if writer is not None and rank == 0:
            writer.close()
        cleanup_distributed()

if __name__ == "__main__":
    main()
