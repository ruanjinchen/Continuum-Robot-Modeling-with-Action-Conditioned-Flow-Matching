from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os, sys

@dataclass
class CollectCfg:
    # I/O & scene
    xml: str = "tdcr.xml"
    nsample: int = 100
    res_w: int = 1280
    res_h: int = 720
    depth_max: float = 0.6
    stride: int = 1
    out_pcd_dir: Path = Path("./pointcloud")
    out_json_dir: Path = Path("./motor")
    # Optional: export RGB images rendered from MuJoCo cameras.
    # - If out_rgb_dir is None: do not save RGB.
    # - If rgb_only is True: do not export pointcloud (still writes motor json).
    out_rgb_dir: Path | None = None
    rgb_format: str = "png"  # {"png","jpg"}
    rgb_only: bool = False
    rgb_cams: list[str] | None = None  # camera names or ids; None=all cameras
    seed: int = 42
    start_index: int = 1
    # sampling
    sampling: str = "continuous"   # {"discrete","continuous"}
    levels_per_motor: int = 10
    unique_tol: float|None = None
    min_gap: float|None = None
    # simulation & stability
    sim_steps: int = 50
    relax_max_steps: int = 4000
    stable_vel_eps: float = 2e-3
    stable_qpos_eps: float = 1e-6
    stable_win: int = 10
    zero_vel_each_ctrl: bool = True
    # rendering backend
    backend: str = "auto"          # {"auto","egl","osmesa","glfw"}
    egl_quiet: bool = True
    # concurrency
    workers: int = 0               # <=0 auto; 1 serial; >=2 parallel
    ctx: str = "spawn"             # {"spawn","fork"}
    resume: bool = True            # skip existing .ply

@dataclass
class H5Cfg:
    pc_dir: Path
    out_root: Path = Path("./data")
    npoints: int = 4096
    voxel_size: float = 0.003
    base_z: float|None = None
    repeat: int = 1

    motor_dir: Path|None = None
    allow_missing_motor: bool = False

    # Optional: also store per-point RGB into H5 as uint8 in [0,255]
    save_rgb: bool = False

    val_frac: float = 0.1
    test_frac: float = 0.1
    shuffle_seed: int = 42

    workers: int = max(2, (os.cpu_count() or 4)//2)
    shard_size: int = 2048
    dtype: str = "float32"  # {"float32","float16"}

    normalize: bool = False
    save_normalized: bool = False

    aug_rotate_z: bool = False
    aug_jitter: bool = False
    jitter_sigma: float = 0.005
    jitter_clip: float = 0.02

    preview: int = 0
    preview_out: Path = Path("./samples")
    preview_only: bool = False
    preview_coords: str = "world"  # {"world","normalized","both"}

@dataclass
class NormCfg:
    root: Path
    splits: list[str] | None = None
    mode: str = "per-sample"    # {"per-sample","global"}
    scope: str = "all"          # {"all","split"}
    dump_global: bool = False
    dtype: str = "float32"
    overwrite: bool = False
    no_inplace: bool = False
    dst_root: Path|None = None
    anchor: str = "centroid"    # {"centroid","origin"}
    export_ply: int = 0
    export_dir: Path = Path("./norm_samples")
    export_seed: int = 42

@dataclass
class MergeCfg:
    src_dir: Path
    start: int
    end: int
    zfill: int = 6
    out_json: Path = Path("motors_all.json")
    out_npz: Path = Path("motors_all.npz")
    fmt: str = "array"          # {"array","dict"}
