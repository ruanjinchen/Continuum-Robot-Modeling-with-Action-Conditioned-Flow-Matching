
# utils.py
from __future__ import annotations
import os, math, random, time, shutil
from typing import Optional, Dict

import torch
import torch.nn as nn
from torch import distributed as dist


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)

    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=True)


def seed_all(seed: int):
    import numpy as np
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def save_point_cloud_xyz(xyz: torch.Tensor, path: str):
    """
    xyz: (N, 3), tensor or np
    """
    import numpy as np
    arr = xyz.detach().cpu().numpy() if isinstance(xyz, torch.Tensor) else xyz
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for p in arr:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def save_point_cloud_ply(xyz: torch.Tensor, path: str):
    import numpy as np
    arr = xyz.detach().cpu().numpy() if isinstance(xyz, torch.Tensor) else xyz
    arr = np.asarray(arr)
    assert arr.ndim == 2 and arr.shape[1] in (3, 6), f"save_point_cloud_ply expects (N,3) or (N,6), got {arr.shape}"

    dirn = os.path.dirname(path)
    if dirn:
        os.makedirs(dirn, exist_ok=True)

    N, D = arr.shape

    if D == 3:
        header = [
            "ply",
            "format ascii 1.0",
            f"element vertex {N}",
            "property float x",
            "property float y",
            "property float z",
            "end_header\n",
        ]
        with open(path, "w") as f:
            f.write("\n".join(header))
            for p in arr:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        return

    # D == 6: xyzrgb, rgb is assumed in [0, 1]
    xyz_part = arr[:, :3]
    rgb_part = arr[:, 3:6]

    # map [0,1] -> [0,255]
    rgb255 = np.clip(rgb_part * 255.0, 0.0, 255.0).astype(np.uint8)


    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {N}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header\n",
    ]
    with open(path, "w") as f:
        f.write("\n".join(header))
        for i in range(N):
            x, y, z = xyz_part[i]
            r, g, b = rgb255[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")




def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def init_distributed():
    """
    Initialize torch.distributed if launched with torchrun.
    Returns (is_distributed, rank, world_size, local_rank).
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return True, rank, world_size, local_rank
    else:
        return False, 0, 1, 0


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


class MetricEMA:
    def __init__(self, alpha: float = 0.98):
        self.a = float(alpha)
        self.value = None

    def update(self, x: float):
        if self.value is None:
            self.value = x
        else:
            self.value = self.a * self.value + (1 - self.a) * x

    def get(self) -> float:
        return float(self.value if self.value is not None else 0.0)


def shard_print(*args, rank: int = 0, **kwargs):
    if rank == 0:
        print(*args, **kwargs)


def cosine_lr(step: int, total: int, base_lr: float, min_lr: float = 1e-6, warmup: int = 0):
    if step < warmup:
        return min_lr + (base_lr - min_lr) * step / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
