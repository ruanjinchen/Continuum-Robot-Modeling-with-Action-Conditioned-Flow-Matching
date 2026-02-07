from __future__ import annotations
import sys, os
from pathlib import Path

def print_color(msg: str):
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_ply_ascii(xyz, path: Path):
    ensure_dir(path.parent)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for p in xyz:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")
