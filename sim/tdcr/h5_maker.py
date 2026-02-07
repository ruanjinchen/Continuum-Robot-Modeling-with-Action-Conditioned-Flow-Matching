from __future__ import annotations
import os, sys, math, json
import numpy as np
import multiprocessing as mp
from pathlib import Path
from .config import H5Cfg
from .utils import ensure_dir, print_color, save_ply_ascii

try:
    import open3d as o3d
except Exception:
    o3d = None
try:
    import h5py
except Exception:
    h5py = None

def _to_uint8_rgb(rgb_like: np.ndarray) -> np.ndarray:
    """Convert RGB-like array to uint8 in [0,255].
    - Supports the common Open3D float colors in [0,1]
    - Also supports float/int already in [0,255]
    """
    C = np.asarray(rgb_like)
    if C.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    if C.ndim != 2 or C.shape[1] != 3:
        raise ValueError(f"RGB array shape must be (N,3), got {C.shape}")

    Cf = C.astype(np.float32, copy=False)
    mx = float(np.nanmax(Cf))
    if mx <= 1.0 + 1e-6:      # Treat as [0,1]
        Cf = Cf * 255.0
    return np.clip(np.round(Cf), 0.0, 255.0).astype(np.uint8)


def _load_pc_xyz_rgb(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load point cloud xyz (float32) and optional rgb (uint8 0-255)."""
    ext = path.suffix.lower()
    if ext == ".npy":
        P = np.load(str(path))
        if P.ndim != 2 or P.shape[1] < 3:
            raise ValueError(f"{path} shape {P.shape} not (N,3/>)")
        xyz = P[:, :3].astype(np.float32, copy=False)
        rgb = None
        # Support (N,6): xyzrgb
        if P.shape[1] >= 6:
            rgb = _to_uint8_rgb(P[:, 3:6])
        return xyz, rgb

    if o3d is None:
        raise RuntimeError("open3d not installed; needed for .ply/.pcd")
    pcd = o3d.io.read_point_cloud(str(path))
    xyz = np.asarray(pcd.points, dtype=np.float32)
    rgb = None
    if hasattr(pcd, "has_colors") and pcd.has_colors():
        rgb = _to_uint8_rgb(np.asarray(pcd.colors))
    return xyz, rgb


def _save_ply(xyz: np.ndarray, path: Path):
    save_ply_ascii(xyz, path)

def _crop_base(P: np.ndarray, rgb: np.ndarray|None, base_z: float|None):
    if base_z is None:
        return P, rgb
    m = (P[:, 2] > base_z)
    return P[m], (None if rgb is None else rgb[m])

def _voxel_downsample(P: np.ndarray, voxel_size: float, rgb: np.ndarray|None = None):
    if len(P) == 0 or voxel_size is None or voxel_size <= 0:
        return P, rgb
    vs = float(voxel_size)
    K = np.floor(P / vs).astype(np.int64)
    _, idx = np.unique(K, axis=0, return_index=True)
    idx = np.sort(idx)
    return P[idx], (None if rgb is None else rgb[idx])

def _rand_downsample(P: np.ndarray, rgb: np.ndarray|None, npoints: int, seed: int):
    if len(P) <= npoints:
        return P, rgb
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(P), size=npoints, replace=False)
    return P[idx], (None if rgb is None else rgb[idx])


def _normalize_unit_sphere(P: np.ndarray):
    if len(P) == 0:
        return P, np.zeros(3, np.float32), np.float32(1.0)
    c = P.mean(0, keepdims=True)
    Q = P - c
    s = np.max(np.linalg.norm(Q, axis=1))
    s = max(s, 1e-6)
    return (Q / s).astype(np.float32), c.reshape(-1).astype(np.float32), np.float32(s)

def _aug_rotate_z(P: np.ndarray, rng):
    th = rng.uniform(0, 2*np.pi)
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)
    return (P @ R.T).astype(np.float32)

def _aug_jitter(P: np.ndarray, rng, sigma=0.005, clip=0.02):
    J = rng.normal(0.0, sigma, size=P.shape).astype(np.float32)
    J = np.clip(J, -clip, clip)
    return P + J

def _load_motor_json(motor_dir: Path|None, stem: str) -> np.ndarray|None:
    if motor_dir is None:
        return None
    path = Path(motor_dir) / f"{stem}.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        obj = json.load(f)
    if isinstance(obj, (list, tuple)):
        arr = np.asarray(obj, dtype=np.float32).reshape(-1)
        return arr.astype(np.float32)
    elif isinstance(obj, dict):
        if "ctrl" in obj:
            arr = np.asarray(obj["ctrl"], dtype=np.float32).reshape(-1)
            return arr
        # Dynamically collect motor1..motorK (legacy format compatibility)
        vals = []
        j = 1
        while True:
            key = f"motor{j}"
            if key not in obj:
                break
            vals.append(obj[key])
            j += 1
        if not vals:
            return None
        return np.array(vals, dtype=np.float32).reshape(-1)
    else:
        raise ValueError(f"Unsupported JSON content in {path}")

def _process_one_h5(args):
    path, cfg, seed_offset = args
    rng = np.random.RandomState(seed_offset)

    need_rgb = bool(cfg.get("save_rgb", False))
    P, rgb = _load_pc_xyz_rgb(path)

    # If --save_rgb is requested but the file has no colors, fail fast.
    if need_rgb and rgb is None:
        raise RuntimeError(f"{path}: no RGB colors found (cannot --save_rgb)")

    if cfg['base_z'] is not None:
        P, rgb = _crop_base(P, rgb, cfg['base_z'])
    P, rgb = _voxel_downsample(P, cfg['voxel_size'], rgb=rgb)

    N = cfg['npoints']
    if len(P) == 0:
        P = np.zeros((N, 3), dtype=np.float32)
        if need_rgb:
            rgb = np.zeros((N, 3), dtype=np.uint8)
    elif len(P) < N:
        pad = rng.choice(len(P), size=N-len(P), replace=True)
        P = np.concatenate([P, P[pad]], 0)
        if need_rgb:
            rgb = np.concatenate([rgb, rgb[pad]], 0)
    elif len(P) > N:
        P, rgb = _rand_downsample(P, rgb, N, seed_offset)

    P_world = P.astype(np.float32)

    # Normalize / augment (modify xyz only; keep rgb intact)
    P_norm = None; center = np.zeros(3, np.float32); scale = np.float32(1.0)
    if cfg['need_norm']:
        P_norm, center, scale = _normalize_unit_sphere(P_world.copy())

    if cfg['apply_aug']:
        if cfg['aug_rotate_z']:
            P_world = _aug_rotate_z(P_world, rng)
        if cfg['aug_jitter']:
            P_world = _aug_jitter(P_world, rng, sigma=cfg['jitter_sigma'], clip=cfg['jitter_clip'])

    motor = None
    if cfg.get("motor_dir", None) is not None:
        motor = _load_motor_json(Path(cfg["motor_dir"]), path.stem)
        if motor is None:
            if cfg.get("allow_missing_motor", False):
                motor = None  # Will be filled with NaN(Dmot) when writing shards
            else:
                raise FileNotFoundError(
                    f"Missing motor JSON for {path.stem} under {cfg['motor_dir']}."
                )
    return dict(world=P_world, rgb=(rgb if need_rgb else None),
                norm=P_norm, center=center, scale=scale, motor=motor)


def _write_h5_shards(out_dir: Path, split_name: str, samples, shard_size: int,
                     save_norm: bool, dtype=np.float32, save_motors: bool=False, save_rgb: bool=False):
    split_dir = out_dir / split_name
    ensure_dir(split_dir)
    tot = len(samples)
    if tot == 0:
        return 0
    nshards = math.ceil(tot / shard_size)
    idx = 0
    import h5py
    for si in range(nshards):
        this = samples[idx: idx+shard_size]
        idx += len(this)
        fn = split_dir / f"shard-{si:05d}.h5"
        with h5py.File(str(fn), "w") as f:
            N = this[0]["world"].shape[0]
            d = f.create_dataset("data", shape=(len(this), N, 3), dtype=dtype)
            for i, item in enumerate(this):
                d[i] = item['world'].astype(dtype)
            if save_norm:
                d2 = f.create_dataset("data_norm", shape=(len(this), this[0]['world'].shape[0], 3), dtype=dtype)
                ctr = f.create_dataset("center", shape=(len(this), 3), dtype=np.float32)
                scl = f.create_dataset("scale",  shape=(len(this),),  dtype=np.float32)
                for i, item in enumerate(this):
                    if item['norm'] is None:
                        d2[i] = item['world'].astype(dtype)
                        ctr[i] = np.zeros(3, np.float32)
                        scl[i] = np.float32(1.0)
                    else:
                        d2[i] = item['norm'].astype(dtype)
                        ctr[i] = item['center']
                        scl[i] = item['scale']
            if save_motors:
                # Find a non-None motor sample to determine D
                Dmot = None
                for item in this:
                    mv = item.get('motor', None)
                    if mv is not None:
                        Dmot = int(len(mv))
                        break
                if Dmot is None:
                    # If all motors are None, skip writing the motors dataset
                    Dmot = 0
                if Dmot > 0:
                    mds = f.create_dataset("motors", shape=(len(this), Dmot), dtype=np.float32)
                    for i, item in enumerate(this):
                        mv = item.get('motor', None)
                        if mv is None:
                            mds[i] = np.full(Dmot, np.nan, dtype=np.float32)
                        else:
                            mv = np.asarray(mv, dtype=np.float32).reshape(-1)
                            assert len(mv) == Dmot, f"Inconsistent motor dimensionality: expected {Dmot}, got {len(mv)}"
                            mds[i] = mv
            if save_rgb:
                cds = f.create_dataset("rgb", shape=(len(this), N, 3), dtype=np.uint8)
                for i, item in enumerate(this):
                    rv = item.get('rgb', None)
                    if rv is None:
                        cds[i] = np.zeros((N, 3), dtype=np.uint8)
                    else:
                        cds[i] = np.asarray(rv, dtype=np.uint8)
    return nshards

def h5_stage(cfg: H5Cfg):
    if h5py is None:
        raise RuntimeError("Stage 2 requires the h5py dependency.")
    out_dir = ensure_dir(cfg.out_root)
    files = sorted([p for p in Path(cfg.pc_dir).iterdir() if p.suffix.lower() in (".npy",".ply",".pcd")])
    if not files:
        raise SystemExit("No point clouds found.")
    print_color(f"[H5] Found {len(files)} files. npoints={cfg.npoints} voxel={cfg.voxel_size} repeat={cfg.repeat} out={out_dir}")

    has_motors = (cfg.motor_dir is not None)
    if has_motors:
        md = Path(cfg.motor_dir)
        if not md.exists() or not md.is_dir():
            raise SystemExit(f"--motor_dir does not exist or is not a directory: {md}")
        else:
            print_color(f"[H5] motors: will read motor JSON (same stem as point cloud) from {md}."
            f" missing_policy={'allow_missing(write_NaN)' if cfg.allow_missing_motor else 'strict(error_on_missing)'}")


    if cfg.val_frac < 0 or cfg.test_frac < 0 or cfg.val_frac + cfg.test_frac >= 1.0:
        raise SystemExit("Require: 0 <= val_frac, test_frac and val_frac + test_frac < 1.0")
    rng = np.random.RandomState(cfg.shuffle_seed)
    idx = np.arange(len(files)); rng.shuffle(idx)
    nval  = int(len(files) * cfg.val_frac)
    ntest = int(len(files) * cfg.test_frac)
    val_idx  = set(idx[:nval])
    test_idx = set(idx[nval:nval+ntest])
    train_files = [files[i] for i in range(len(files)) if (i not in val_idx and i not in test_idx)]
    val_files   = [files[i] for i in range(len(files)) if i in val_idx]
    test_files  = [files[i] for i in range(len(files)) if i in test_idx]

    def build_tasks(fs):
        tasks = []
        for rep in range(cfg.repeat):
            for i, p in enumerate(fs):
                tasks.append((p, {
                    "npoints": cfg.npoints,
                    "voxel_size": cfg.voxel_size,
                    "base_z": cfg.base_z,
                    "need_norm": bool(cfg.normalize or cfg.save_normalized or cfg.preview_coords!="world"),
                    "apply_aug": (cfg.aug_rotate_z or cfg.aug_jitter),
                    "aug_rotate_z": cfg.aug_rotate_z,
                    "aug_jitter": cfg.aug_jitter,
                    "jitter_sigma": cfg.jitter_sigma,
                    "jitter_clip": cfg.jitter_clip,
                    "motor_dir": str(cfg.motor_dir) if cfg.motor_dir is not None else None,
                    "allow_missing_motor": bool(cfg.allow_missing_motor),
                    "save_rgb": bool(cfg.save_rgb),
                }, rep*1_000_000 + i))
        return tasks

    def run_pool(tasks):
        if len(tasks) == 0: return []
        with mp.Pool(processes=cfg.workers) as pool:
            it = pool.imap_unordered(_process_one_h5, tasks, chunksize=32)
            out_list = []
            from tqdm import tqdm as _tqdm
            for item in _tqdm(it, total=len(tasks), ncols=120, desc="processing"):
                out_list.append(item)
        return out_list

    if cfg.preview > 0:
        k = min(cfg.preview, len(train_files))
        prev_idx = rng.choice(len(train_files), size=k, replace=False)
        prev_tasks = build_tasks([train_files[i] for i in range(len(train_files)) if i in prev_idx])
        prev_samples = []
        from tqdm import tqdm as _tqdm
        for item in _tqdm(map(_process_one_h5, prev_tasks[:k]), total=k, ncols=120, desc="preview"):
            prev_samples.append(dict(world=item['world'], norm=item['norm']))
        ensure_dir(cfg.preview_out)
        for i, item in enumerate(prev_samples):
            if cfg.preview_coords in ("world","both"):
                np.save(cfg.preview_out / f"sample_{i:02d}_world.npy", item['world'])
                _save_ply(item['world'], cfg.preview_out / f"sample_{i:02d}_world.ply")
            if cfg.preview_coords in ("normalized","both") and item['norm'] is not None:
                np.save(cfg.preview_out / f"sample_{i:02d}_norm.npy", item['norm'])
                _save_ply(item['norm'], cfg.preview_out / f"sample_{i:02d}_norm.ply")
        print_color(f"[preview] wrote {len(prev_samples)} samples ({cfg.preview_coords}) to {cfg.preview_out}")
        if cfg.preview_only:
            print_color("[preview_only] done. Exit without writing H5.")
            return

    dtype_np = np.float16 if cfg.dtype=="float16" else np.float32
    train_list = run_pool(build_tasks(train_files))
    val_list   = run_pool(build_tasks(val_files))
    test_list  = run_pool(build_tasks(test_files))

    nsh_train = _write_h5_shards(out_dir, "train", train_list, cfg.shard_size,
                                 save_norm=cfg.save_normalized, dtype=dtype_np, save_motors=has_motors,
                                 save_rgb=bool(cfg.save_rgb))
    nsh_val   = _write_h5_shards(out_dir, "val",   val_list,   cfg.shard_size,
                                 save_norm=cfg.save_normalized, dtype=dtype_np, save_motors=has_motors,
                                 save_rgb=bool(cfg.save_rgb))
    nsh_test  = _write_h5_shards(out_dir, "test",  test_list,  cfg.shard_size,
                                 save_norm=cfg.save_normalized, dtype=dtype_np, save_motors=has_motors,
                                 save_rgb=bool(cfg.save_rgb))

    meta = {
        "total_raw": len(files),
        "train_augmented": len(train_list),
        "val_augmented": len(val_list),
        "test_augmented": len(test_list),
        "nshards_train": nsh_train,
        "nshards_val": nsh_val,
        "nshards_test": nsh_test,
        "has_motors": bool(has_motors),
        "npoints": cfg.npoints,
        "dtype": cfg.dtype,
        "voxel_size": cfg.voxel_size,
        "base_z": cfg.base_z,
        "repeat": cfg.repeat,
        "val_frac": cfg.val_frac,
        "test_frac": cfg.test_frac,
        "shuffle_seed": cfg.shuffle_seed,
        "normalize": bool(cfg.normalize),
        "save_normalized": bool(cfg.save_normalized),
        "aug_rotate_z": bool(cfg.aug_rotate_z),
        "aug_jitter": bool(cfg.aug_jitter),
        "save_rgb": bool(cfg.save_rgb),
    }
    json.dump(meta, open(out_dir/"meta.json","w"), indent=2)
    print_color("✅ [H5] done. meta written to meta.json")
