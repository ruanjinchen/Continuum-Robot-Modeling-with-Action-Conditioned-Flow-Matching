import os
import glob
from pathlib import Path
from typing import List, Optional, Set, Tuple
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, Subset
from torch.utils import data as torch_data

# ----------------------------- Utilities -----------------------------

def init_np_seed(worker_id: int):
    seed = torch.initial_seed()
    np.random.seed(seed % 4294967296)


class SubsetWithAttrs(Subset):
    """Forward attribute access safely to the base dataset (handles nesting, avoids recursion)."""
    def __getattr__(self, name):
        base = object.__getattribute__(self, 'dataset')
        while isinstance(base, Subset):
            base = object.__getattribute__(base, 'dataset')
        return getattr(base, name)

    def __dir__(self):
        base = object.__getattribute__(self, 'dataset')
        attrs = set(super().__dir__())
        while isinstance(base, Subset):
            base = object.__getattribute__(base, 'dataset')
        attrs.update(dir(base))
        return sorted(attrs)


def _attach_shuffle_idx(ds, sel_idx):
    sel_idx = np.asarray(sel_idx, dtype=np.int64)
    try:
        setattr(ds, "shuffle_idx", sel_idx)
    except Exception:
        pass
    base = getattr(ds, "dataset", None)
    if base is not None:
        try:
            setattr(base, "shuffle_idx", sel_idx)
        except Exception:
            pass


def _pick_subset_indices(args, N: int):
    train_count = getattr(args, "train_count", None)
    train_fraction = getattr(args, "train_fraction", 1.0)
    if train_count is None and (train_fraction is None or not (0.0 < float(train_fraction) < 1.0)):
        return None
    if N <= 1:
        return None
    if train_count is not None:
        n_keep = max(1, min(int(train_count), N))
    else:
        n_keep = max(1, min(int(np.ceil(N * float(train_fraction))), N))
    seed = getattr(args, "train_subset_seed", None)
    if seed is None:
        seed = getattr(args, "seed", 0)
    g = torch.Generator().manual_seed(int(seed))
    idx = torch.randperm(N, generator=g)[:n_keep].tolist()
    idx.sort()
    print(f"[datasets] Use subset of training data: {n_keep}/{N} ({n_keep/N:.2%}) with seed={seed}")
    return np.asarray(idx, dtype=np.int64)

# ----------------------------- TDCR-H5 Dataset -----------------------------
class TDCRH5PointClouds(Dataset):
    """Dataset for TDCR H5 shards with keys: data, data_norm, motors, center, scale."""
    def __init__(self,
                 data_dir: str = None, root_dir: str = None,
                 split: str = "train",
                 use_norm: bool = True,
                 tr_sample_size: int = 2048, te_sample_size: int = 2048,
                 cond_mode: str = "motors",
                 use_rgb: bool = False,
                 rgb_key: str = "rgb",
                 files=None, **kwargs) -> None:
        super().__init__()
        # Early init
        self._handles = {}
        self.gravity_axis = 1
        self.display_axis_order = [0, 2, 1]

        self.split = str(split)
        self.use_norm = bool(use_norm)
        self.tr_n = int(tr_sample_size)
        self.te_n = int(te_sample_size)
        self.cond_mode = str(cond_mode)

        self.use_rgb = bool(use_rgb)
        self.rgb_key = str(rgb_key)
        self.point_dim = 6 if self.use_rgb else 3

        # Resolve directory
        if data_dir is None and root_dir is not None:
            data_dir = root_dir
        if data_dir is None:
            raise ValueError("TDCRH5PointClouds: data_dir/root_dir must be provided.")
        self.data_dir = os.path.abspath(data_dir)

        # Discover shards
        if files is not None:
            if isinstance(files, (list, tuple)):
                self.files = sorted(set([str(x) for x in files]))
            elif isinstance(files, str):
                self.files = sorted(set(glob.glob(files)))
            else:
                raise TypeError("files must be None, list/tuple of paths, or a glob pattern string")
        else:
            patterns = [
                os.path.join(self.data_dir, self.split, "*.h5"),
                os.path.join(self.data_dir, self.split, "*.hdf5"),
                os.path.join(self.data_dir, f"{self.split}*.h5"),
                os.path.join(self.data_dir, "*.h5"),
                os.path.join(self.data_dir, "*.hdf5"),
            ]
            flist = []
            for p in patterns:
                flist.extend(glob.glob(p))
            self.files = sorted(set(flist))
        if not self.files:
            raise FileNotFoundError(
                f"[TDCR-H5] No shard files found under '{self.data_dir}/{self.split}'. Expect shard-*.h5"
            )

        # Build index
        self._index = []
        self._key_points_map = {}
        self._has_motors = False
        self.cond_dim = 0  # initialized to 0; inferred when motors/motors_norm is available

        for fi, fp in enumerate(self.files):
            with h5py.File(fp, "r") as f:
                key = "data_norm" if (self.use_norm and "data_norm" in f) else "data"
                if key not in f:
                    raise KeyError(f"[TDCR-H5] Missing key '{key}' in file: {fp}")
                B = int(f[key].shape[0])

                # --- RGB check (only when use_rgb) ---
                if self.use_rgb:
                    if self.rgb_key not in f:
                        raise KeyError(f"[TDCR-H5] Missing key '{self.rgb_key}' in file: {fp} (required by --use_rgb)")
                    rgb_ds = f[self.rgb_key]
                    pts_ds = f[key]
                    # expect rgb: (B, N, 3) and aligned with points
                    if rgb_ds.ndim != 3 or rgb_ds.shape[0] != pts_ds.shape[0] or rgb_ds.shape[1] != pts_ds.shape[1] or rgb_ds.shape[2] != 3:
                        raise ValueError(
                            f"[TDCR-H5] Bad rgb shape in file {fp}: got {tuple(rgb_ds.shape)}, "
                            f"expect ({pts_ds.shape[0]}, {pts_ds.shape[1]}, 3)"
                        )
                # --- end RGB check ---

                self._index.extend([(fi, i) for i in range(B)])
                self._key_points_map[fi] = key
                if "motors_norm" in f:
                    self._has_motors = True
                    if self.cond_mode == "motors" and self.cond_dim == 0:
                        self.cond_dim = int(f["motors_norm"].shape[1])
                else:
                    if self.cond_mode == "motors":
                        raise KeyError(f"[TDCR-H5] Expected 'motors_norm' in file: {fp}")


        # Dataset-level stats
        self.all_points_mean = np.zeros(self.point_dim, dtype=np.float32)
        self.all_points_std = np.ones(self.point_dim, dtype=np.float32)

        if not self.use_norm:
            try:
                with h5py.File(self.files[0], "r") as f0:
                    if ("center" in f0) and ("scale" in f0):
                        c0 = np.asarray(f0["center"][0], dtype=np.float32)
                        s0 = float(np.asarray(f0["scale"][0], dtype=np.float32))
                        # xyz mean/std
                        self.all_points_mean[:3] = c0
                        self.all_points_std[:3] = np.array([s0, s0, s0], dtype=np.float32)
                        # rgb mean/std keep as 0/1 (already set by zeros/ones)

            except Exception:
                pass

        # default shuffle_idx before subsetting
        self.shuffle_idx = np.arange(len(self._index), dtype=np.int64)

    def __len__(self) -> int:
        return len(self._index)

    def _ensure_open(self, fi: int):
        h = self._handles.get(fi, None)
        if h is None:
            h = h5py.File(self.files[fi], "r")
            self._handles[fi] = h
        return h

    @staticmethod
    def _sample_idx(N: int, K: int) -> np.ndarray:
        if K <= 0: return np.empty((0,), dtype=np.int64)
        if K <= N: return np.random.choice(N, K, replace=False)
        base = np.arange(N, dtype=np.int64)
        extra = np.random.choice(N, K - N, replace=True)
        return np.concatenate([base, extra], axis=0)

    def __getitem__(self, idx: int):
        fi, ri = self._index[idx]
        f = self._ensure_open(fi)
        key = self._key_points_map[fi]

        pts_xyz = f[key][ri].astype(np.float32)  # (N,3)
        N = pts_xyz.shape[0]

        rgb = None
        if self.use_rgb:
            rgb = f[self.rgb_key][ri].astype(np.float32)  # (N,3), usually in [0,255]
            # normalize to [0, 1]
            mx = float(np.max(rgb)) if rgb.size > 0 else 1.0
            if mx > 1.0:
                rgb = rgb / 255.0
            rgb = np.clip(rgb, 0.0, 1.0)


        tr_idx = self._sample_idx(N, self.tr_n)
        te_idx = self._sample_idx(N, self.te_n)

        tr_xyz = pts_xyz[tr_idx]
        te_xyz = pts_xyz[te_idx]

        if self.use_rgb:
            tr_rgb = rgb[tr_idx]
            te_rgb = rgb[te_idx]
            tr_pts = np.concatenate([tr_xyz, tr_rgb], axis=1)  # (K,6)
            te_pts = np.concatenate([te_xyz, te_rgb], axis=1)  # (K,6)
        else:
            tr_pts = tr_xyz
            te_pts = te_xyz


        item = {
            "idx": idx,
            "train_points": torch.from_numpy(tr_pts),
            "test_points": torch.from_numpy(te_pts),
            "mean": torch.from_numpy(self.all_points_mean.reshape(1, self.point_dim)),
            "std": torch.from_numpy(self.all_points_std.reshape(1, self.point_dim)),
        }

        if self.cond_mode == "motors" and self._has_motors:
            if "motors_norm" not in f:
                raise KeyError(f"[TDCR-H5] Expected 'motors_norm' in file: {self.files[fi]}")
            cond = f["motors_norm"][ri].astype(np.float32)
            item["cond"] = torch.from_numpy(cond)

        return item

    def __del__(self):
        handles = getattr(self, "_handles", None)
        if handles:
            for h in list(handles.values()):
                try:
                    h.close()
                except Exception:
                    pass
            handles.clear()


# ----------------------------- Factory & loaders -----------------------------

def get_datasets(args):
    tr_dataset = TDCRH5PointClouds(
        data_dir=args.data_dir, split="train",
        use_norm=getattr(args, "tdcr_use_norm", True),
        tr_sample_size=getattr(args, "tr_max_sample_points", 2048),
        te_sample_size=getattr(args, "te_max_sample_points", 2048),
        cond_mode=getattr(args, "cond_mode", "motors"),
        use_rgb=getattr(args, "use_rgb", False),
        rgb_key=getattr(args, "rgb_key", "rgb"),

    )
    # Prefer val/ if exists, else test/
    val_dir = Path(args.data_dir, "val")
    split = "val" if val_dir.exists() and any(val_dir.glob("shard-*.h5")) else "test"
    te_dataset = TDCRH5PointClouds(
        data_dir=args.data_dir, split=split,
        use_norm=getattr(args, "tdcr_use_norm", True),
        tr_sample_size=getattr(args, "tr_max_sample_points", 2048),
        te_sample_size=getattr(args, "te_max_sample_points", 2048),
        cond_mode=getattr(args, "cond_mode", "motors"),
        use_rgb=getattr(args, "use_rgb", False),
        rgb_key=getattr(args, "rgb_key", "rgb"),

    )

    # Optional: training subset
    sel_idx = _pick_subset_indices(args, len(tr_dataset))
    if sel_idx is not None:
        tr_dataset = SubsetWithAttrs(tr_dataset, sel_idx.tolist())
        _attach_shuffle_idx(tr_dataset, sel_idx)
    else:
        _attach_shuffle_idx(tr_dataset, np.arange(len(tr_dataset), dtype=np.int64))

    # Ensure the eval split exposes shuffle_idx
    if not hasattr(te_dataset, "shuffle_idx"):
        setattr(te_dataset, "shuffle_idx", np.arange(len(te_dataset), dtype=np.int64))

    # Expose cond_dim to the training script via args
    base = getattr(tr_dataset, "dataset", tr_dataset)
    args.cond_dim = getattr(base, "cond_dim", 0)

    return tr_dataset, te_dataset


def get_data_loaders(args):
    tr_dataset, te_dataset = get_datasets(args)

    train_loader = torch_data.DataLoader(
        dataset=tr_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, drop_last=True,
        worker_init_fn=init_np_seed,
    )
    train_unshuffle_loader = torch_data.DataLoader(
        dataset=tr_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, drop_last=True,
        worker_init_fn=init_np_seed,
    )
    test_loader = torch_data.DataLoader(
        dataset=te_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, drop_last=False,
        worker_init_fn=init_np_seed,
    )

    return {
        "test_loader": test_loader,
        "train_loader": train_loader,
        "train_unshuffle_loader": train_unshuffle_loader,
    }
