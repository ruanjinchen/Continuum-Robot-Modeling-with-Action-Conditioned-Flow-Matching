from __future__ import annotations
import os, sys, argparse
from pathlib import Path

from tdcr.config import CollectCfg, H5Cfg, NormCfg, MergeCfg
from tdcr.collect import collect_stage
from tdcr.h5_maker import h5_stage
from tdcr.norm import norm_stage
from tdcr.merge import merge_motors_stage

def build_arg_parser():
    ap = argparse.ArgumentParser(description="TDCR end-to-end pipeline (collect -> H5 -> normalize -> merge motors)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # collect
    ac = sub.add_parser("collect", help="Stage 1: collect MuJoCo point clouds + motor JSON")
    ac.add_argument("--xml", default="tdcr.xml")
    ac.add_argument("--nsample", type=int, default=100)
    ac.add_argument("--res_w", type=int, default=1280)
    ac.add_argument("--res_h", type=int, default=720)
    ac.add_argument("--depth_max", type=float, default=1.0)
    ac.add_argument("--stride", type=int, default=1)
    ac.add_argument("--out_pcd_dir", type=Path, default=Path("./pointcloud"))
    ac.add_argument("--out_json_dir", type=Path, default=Path("./motor"))
    # Optional: export RGB images from MuJoCo cameras
    ac.add_argument("--out_rgb_dir", type=Path, default=None,
                    help="Optional: output directory for RGB images (subdir per camera). If omitted, RGB is not exported.")
    ac.add_argument("--rgb_format", choices=["png", "jpg"], default="png",
                    help="RGB image format: png or jpg")
    ac.add_argument("--rgb_only", action="store_true",
                    help="Export RGB only (still exports motor JSON); do not export point clouds.")
    ac.add_argument("--rgb_cams", nargs="*", default=None,
                    help="Optional: cameras to export (camera name or id). Default: all cameras."
                         "Example: --rgb_cams camera1 camera2 OR --rgb_cams 0 1 2 3")
    ac.add_argument("--seed", type=int, default=42)
    ac.add_argument("--start_index", type=int, default=1)
    ac.add_argument("--sampling", choices=["discrete","continuous"], default="continuous")
    ac.add_argument("--levels_per_motor", type=int, default=10)
    ac.add_argument("--unique_tol", type=float, default=None)
    ac.add_argument("--min_gap", type=float, default=None)
    ac.add_argument("--sim_steps", type=int, default=50)
    # stability & backend & concurrency
    ac.add_argument("--relax_max_steps", type=int, default=10000, help="Max steps for stability early-stop (if still unstable at the limit, proceed anyway).")
    ac.add_argument("--stable_vel_eps", type=float, default=1e-3, help="Stability threshold #1: max |qvel| below this value is considered stationary.")
    ac.add_argument("--stable_qpos_eps", type=float, default=1e-6, help="Stability threshold #2: max |Δqpos| below this value means positions have converged.")
    ac.add_argument("--stable_win", type=int, default=10, help="Number of consecutive steps the two conditions must hold (debounce).")
    ac.add_argument("--zero_vel_each_ctrl", action="store_true")
    ac.add_argument("--backend", choices=["auto","egl","osmesa","glfw"], default="auto")
    ac.add_argument("--no_egl_quiet", action="store_true")
    ac.add_argument("--workers", type=int, default=-1)
    ac.add_argument("--ctx", choices=["spawn","fork"], default="spawn")
    ac.add_argument("--resume", action="store_true")

    # make-h5
    ah = sub.add_parser("make-h5", help="Stage 2: convert point clouds to PointFlow-style H5 (optionally including motors)")
    ah.add_argument("--pc_dir", type=Path, required=True)
    ah.add_argument("--out_root", type=Path, default=Path("./data"))
    ah.add_argument("--npoints", type=int, default=4096)
    ah.add_argument("--voxel_size", type=float, default=0.003)
    ah.add_argument("--base_z", type=float, default=None)
    ah.add_argument("--repeat", type=int, default=1)
    ah.add_argument("--motor_dir", type=Path, default=None)
    ah.add_argument("--allow_missing_motor", action="store_true")
    ah.add_argument("--val_frac", type=float, default=0.1)
    ah.add_argument("--test_frac", type=float, default=0.1)
    ah.add_argument("--shuffle_seed", type=int, default=42)
    ah.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4)//2))
    ah.add_argument("--shard_size", type=int, default=2048)
    ah.add_argument("--dtype", choices=["float32","float16"], default="float32")
    ah.add_argument("--normalize", action="store_true")
    ah.add_argument("--save_normalized", action="store_true")
    ah.add_argument("--aug_rotate_z", action="store_true")
    ah.add_argument("--aug_jitter", action="store_true")
    ah.add_argument("--jitter_sigma", type=float, default=0.005)
    ah.add_argument("--jitter_clip", type=float, default=0.02)
    ah.add_argument("--preview", type=int, default=0)
    ah.add_argument("--preview_out", type=Path, default=Path("./samples"))
    ah.add_argument("--preview_only", action="store_true")
    ah.add_argument("--preview_coords", choices=["world","normalized","both"], default="world")
    ah.add_argument("--save_rgb", action="store_true",help="Optional: write point-cloud RGB into H5 (uint8, 0-255) under dataset name 'rgb'.")

    # add-norm
    an = sub.add_parser("add-norm", help="Stage 3: add data_norm/center/scale to H5 shards")
    an.add_argument("--root", type=Path, required=True)
    an.add_argument("--splits", nargs="*", default=["train","val","test"])
    an.add_argument("--mode", choices=["per-sample","global"], default="per-sample")
    an.add_argument("--scope", choices=["all","split"], default="all")
    an.add_argument("--dump_global", action="store_true")
    an.add_argument("--dtype", choices=["float32","float16"], default="float32")
    an.add_argument("--overwrite", action="store_true")
    an.add_argument("--no_inplace", action="store_true")
    an.add_argument("--dst_root", type=Path, default=None)
    an.add_argument("--anchor", choices=["centroid","origin"], default="centroid")
    an.add_argument("--export_ply", type=int, default=0)
    an.add_argument("--export_dir", type=Path, default=Path("./norm_samples"))
    an.add_argument("--export_seed", type=int, default=42)

    # merge-motors
    am = sub.add_parser("merge-motors", help="Stage 4: merge motor JSON files")
    am.add_argument("--src_dir", type=Path, required=True)
    am.add_argument("--start", type=int, required=True)
    am.add_argument("--end", type=int, required=True)
    am.add_argument("--zfill", type=int, default=6)
    am.add_argument("--out_json", type=Path, default=Path("motors_all.json"))
    am.add_argument("--out_npz", type=Path, default=Path("motors_all.npz"))
    am.add_argument("--format", choices=["array","dict"], default="array")

    # all
    aa = sub.add_parser("all", help="Run all 4 stages sequentially")
    aa.add_argument("--c_xml", default="tdcr.xml")
    aa.add_argument("--c_nsample", type=int, default=100)
    aa.add_argument("--c_res_w", type=int, default=1280)
    aa.add_argument("--c_res_h", type=int, default=720)
    aa.add_argument("--c_depth_max", type=float, default=0.6)
    aa.add_argument("--c_stride", type=int, default=1)
    aa.add_argument("--c_out_pcd_dir", type=Path, default=Path("./pointcloud"))
    aa.add_argument("--c_out_json_dir", type=Path, default=Path("./motor"))
    aa.add_argument("--c_seed", type=int, default=42)
    aa.add_argument("--c_start_index", type=int, default=1)
    aa.add_argument("--c_sampling", choices=["discrete","continuous"], default="continuous")
    aa.add_argument("--c_levels_per_motor", type=int, default=10)
    aa.add_argument("--c_unique_tol", type=float, default=None)
    aa.add_argument("--c_min_gap", type=float, default=None)
    aa.add_argument("--c_sim_steps", type=int, default=50)
    aa.add_argument("--c_relax_max_steps", type=int, default=5000)
    aa.add_argument("--c_stable_vel_eps", type=float, default=2e-3)
    aa.add_argument("--c_stable_qpos_eps", type=float, default=1e-6)
    aa.add_argument("--c_stable_win", type=int, default=10)
    aa.add_argument("--c_zero_vel_each_ctrl", action="store_true")
    aa.add_argument("--c_backend", choices=["auto","egl","osmesa","glfw"], default="auto")
    aa.add_argument("--c_no_egl_quiet", action="store_true")
    aa.add_argument("--c_workers", type=int, default=0)
    aa.add_argument("--c_ctx", choices=["spawn","fork"], default="spawn")
    aa.add_argument("--c_resume", action="store_true")

    aa.add_argument("--h5_out_root", type=Path, default=Path("./data"))
    aa.add_argument("--h5_npoints", type=int, default=4096)
    aa.add_argument("--h5_voxel_size", type=float, default=0.003)
    aa.add_argument("--h5_base_z", type=float, default=None)
    aa.add_argument("--h5_repeat", type=int, default=1)
    aa.add_argument("--h5_motor_dir", type=Path, default=None)
    aa.add_argument("--h5_allow_missing_motor", action="store_true")
    aa.add_argument("--h5_val_frac", type=float, default=0.1)
    aa.add_argument("--h5_test_frac", type=float, default=0.1)
    aa.add_argument("--h5_shuffle_seed", type=int, default=42)
    aa.add_argument("--h5_workers", type=int, default=max(2, (os.cpu_count() or 4)//2))
    aa.add_argument("--h5_shard_size", type=int, default=2048)
    aa.add_argument("--h5_dtype", choices=["float32","float16"], default="float32")
    aa.add_argument("--h5_normalize", action="store_true")
    aa.add_argument("--h5_save_normalized", action="store_true")
    aa.add_argument("--h5_aug_rotate_z", action="store_true")
    aa.add_argument("--h5_aug_jitter", action="store_true")
    aa.add_argument("--h5_jitter_sigma", type=float, default=0.005)
    aa.add_argument("--h5_jitter_clip", type=float, default=0.02)
    aa.add_argument("--h5_preview", type=int, default=0)
    aa.add_argument("--h5_preview_out", type=Path, default=Path("./samples"))
    aa.add_argument("--h5_preview_only", action="store_true")
    aa.add_argument("--h5_preview_coords", choices=["world","normalized","both"], default="world")

    aa.add_argument("--n_enable", action="store_true")
    aa.add_argument("--n_mode", choices=["per-sample","global"], default="per-sample")
    aa.add_argument("--n_scope", choices=["all","split"], default="all")
    aa.add_argument("--n_dump_global", action="store_true")
    aa.add_argument("--n_dtype", choices=["float32","float16"], default="float32")
    aa.add_argument("--n_overwrite", action="store_true")
    aa.add_argument("--n_no_inplace", action="store_true")
    aa.add_argument("--n_dst_root", type=Path, default=None)
    aa.add_argument("--n_anchor", choices=["centroid","origin"], default="centroid")
    aa.add_argument("--n_export_ply", type=int, default=0)
    aa.add_argument("--n_export_dir", type=Path, default=Path("./norm_samples"))
    aa.add_argument("--n_export_seed", type=int, default=42)

    aa.add_argument("--m_enable", action="store_true")
    aa.add_argument("--m_out_json", type=Path, default=Path("motors_all.json"))
    aa.add_argument("--m_out_npz", type=Path, default=Path("motors_all.npz"))
    aa.add_argument("--m_format", choices=["array","dict"], default="array")
    aa.add_argument("--m_zfill", type=int, default=6)

    aa.add_argument("--h5_save_rgb", action="store_true", help="Same as --save_rgb: write point-cloud RGB into H5 (uint8, 0-255) under dataset name 'rgb'.")

    return ap

def main():
    ap = build_arg_parser()
    args = ap.parse_args()

    if args.cmd == "collect":
        cfg = CollectCfg(
            xml=args.xml, nsample=args.nsample, res_w=args.res_w, res_h=args.res_h,
            depth_max=args.depth_max, stride=args.stride, out_pcd_dir=args.out_pcd_dir,
            out_json_dir=args.out_json_dir, seed=args.seed,
            out_rgb_dir=args.out_rgb_dir, rgb_format=args.rgb_format,
            rgb_only=args.rgb_only, rgb_cams=args.rgb_cams,
            start_index=args.start_index,
            sampling=args.sampling, levels_per_motor=args.levels_per_motor, unique_tol=args.unique_tol,
            min_gap=args.min_gap, sim_steps=args.sim_steps, relax_max_steps=args.relax_max_steps,
            stable_vel_eps=args.stable_vel_eps, stable_qpos_eps=args.stable_qpos_eps,
            stable_win=args.stable_win, zero_vel_each_ctrl=args.zero_vel_each_ctrl,
            backend=args.backend, egl_quiet=(not args.no_egl_quiet), workers=args.workers,
            ctx=args.ctx, resume=args.resume
        )
        collect_stage(cfg); return

    if args.cmd == "make-h5":
        cfg = H5Cfg(
            pc_dir=args.pc_dir, out_root=args.out_root, npoints=args.npoints, voxel_size=args.voxel_size,
            base_z=args.base_z, repeat=args.repeat, motor_dir=args.motor_dir,
            allow_missing_motor=args.allow_missing_motor, save_rgb=args.save_rgb, val_frac=args.val_frac, test_frac=args.test_frac,
            shuffle_seed=args.shuffle_seed, workers=args.workers, shard_size=args.shard_size, dtype=args.dtype,
            normalize=args.normalize, save_normalized=args.save_normalized, aug_rotate_z=args.aug_rotate_z,
            aug_jitter=args.aug_jitter, jitter_sigma=args.jitter_sigma, jitter_clip=args.jitter_clip,
            preview=args.preview, preview_out=args.preview_out, preview_only=args.preview_only,
            preview_coords=args.preview_coords
        )
        h5_stage(cfg); return

    if args.cmd == "add-norm":
        cfg = NormCfg(
            root=args.root, splits=args.splits, mode=args.mode, scope=args.scope,
            dump_global=args.dump_global, dtype=args.dtype, overwrite=args.overwrite, 
            no_inplace=args.no_inplace, dst_root=args.dst_root, anchor=args.anchor,
            export_ply=args.export_ply, export_dir=args.export_dir, export_seed=args.export_seed
        )
        norm_stage(cfg); return

    if args.cmd == "merge-motors":
        cfg = MergeCfg(
            src_dir=args.src_dir, start=args.start, end=args.end, zfill=args.zfill,
            out_json=args.out_json, out_npz=args.out_npz, fmt=args.format
        )
        merge_motors_stage(cfg); return

    if args.cmd == "all":
        c = CollectCfg(
            xml=args.c_xml, nsample=args.c_nsample, res_w=args.c_res_w, res_h=args.c_res_h,
            depth_max=args.c_depth_max, stride=args.c_stride, out_pcd_dir=args.c_out_pcd_dir,
            out_json_dir=args.c_out_json_dir, seed=args.c_seed, start_index=args.c_start_index,
            sampling=args.c_sampling, levels_per_motor=args.c_levels_per_motor, unique_tol=args.c_unique_tol,
            min_gap=args.c_min_gap, sim_steps=args.c_sim_steps, relax_max_steps=args.c_relax_max_steps,
            stable_vel_eps=args.c_stable_vel_eps, stable_qpos_eps=args.c_stable_qpos_eps,
            stable_win=args.c_stable_win, zero_vel_each_ctrl=args.c_zero_vel_each_ctrl,
            backend=args.c_backend, egl_quiet=(not args.c_no_egl_quiet), workers=args.c_workers,
            ctx=args.c_ctx, resume=args.c_resume
        )
        h = H5Cfg(
            pc_dir=c.out_pcd_dir, out_root=args.h5_out_root, npoints=args.h5_npoints, voxel_size=args.h5_voxel_size,
            base_z=args.h5_base_z, repeat=args.h5_repeat, motor_dir=args.h5_motor_dir,
            allow_missing_motor=args.h5_allow_missing_motor, save_rgb=args.h5_save_rgb, val_frac=args.h5_val_frac, test_frac=args.h5_test_frac,
            shuffle_seed=args.h5_shuffle_seed, workers=args.h5_workers, shard_size=args.h5_shard_size, dtype=args.h5_dtype,
            normalize=args.h5_normalize, save_normalized=args.h5_save_normalized, aug_rotate_z=args.h5_aug_rotate_z,
            aug_jitter=args.h5_aug_jitter, jitter_sigma=args.h5_jitter_sigma, jitter_clip=args.h5_jitter_clip,
            preview=args.h5_preview, preview_out=args.h5_preview_out, preview_only=args.h5_preview_only,
            preview_coords=args.h5_preview_coords
        )
        n = None
        if args.n_enable:
            n = NormCfg(
                root=h.out_root, splits=["train","val","test"], mode=args.n_mode, scope=args.n_scope, 
                dump_global=args.n_dump_global, dtype=args.n_dtype, overwrite=args.n_overwrite,
                no_inplace=args.n_no_inplace, dst_root=args.n_dst_root, anchor=args.n_anchor,
                export_ply=args.n_export_ply, export_dir=args.n_export_dir, export_seed=args.n_export_seed
            )
        m = None
        if args.m_enable:
            start_idx = c.start_index
            end_idx   = start_idx + c.nsample - 1
            m = MergeCfg(src_dir=c.out_json_dir, start=start_idx, end=end_idx,
                         zfill=args.m_zfill, out_json=args.m_out_json, out_npz=args.m_out_npz, fmt=args.m_format)
        # run
        collect_stage(c); h5_stage(h)
        if n is not None: norm_stage(n)
        if m is not None: merge_motors_stage(m)
        return

if __name__ == "__main__":
    if sys.platform.startswith("win"):
        import multiprocessing as mp
        mp.freeze_support()
    main()
'''
Stage 1: collect data
python tdcr_pipeline.py collect \
  --xml tdcr2_no_base.xml \
  --nsample 5000 \
  --out_pcd_dir "2m_no_base/pointcloud" \
  --out_json_dir "2m_no_base/motor" \
  --out_rgb_dir 2m_no_base/rgb \
  --sampling continuous \
  --seed 42 \
  --start_index 1 \
  --unique_tol 1e-6 \
  --backend auto \
  --workers 8 \
  --zero_vel_each_ctrl \
  --relax_max_steps 10000

python tdcr_pipeline.py collect \
  --xml tdcr5_with_base.xml \
  --nsample 5000 \
  --out_pcd_dir "5m_with_base/pointcloud" \
  --out_json_dir "5m_with_base/motor" \
  --out_rgb_dir 5m_with_base/rgb \
  --sampling continuous \
  --seed 42 \
  --start_index 1 \
  --unique_tol 1e-6 \
  --backend auto \
  --workers 48 \
  --zero_vel_each_ctrl \
  --relax_max_steps 10000

python tdcr_pipeline.py collect \
  --xml tdcr5_no_base.xml \
  --nsample 5000 \
  --out_pcd_dir "5m_no_base/pointcloud" \
  --out_json_dir "5m_no_base/motor" \
  --out_rgb_dir 5m_no_base/rgb \
  --sampling continuous \
  --seed 42 \
  --start_index 1 \
  --unique_tol 1e-6 \
  --backend auto \
  --workers 48 \
  --zero_vel_each_ctrl \
  --relax_max_steps 10000
  

Stage 2: build H5 (with motor_dir)
python tdcr_pipeline.py make-h5 \
  --pc_dir "2m_no_base/pointcloud" \
  --motor_dir "2m_no_base/motor" \
  --out_root 2m_no_base/ \
  --npoints 20000 --voxel_size 0.002 \
  --workers 32 --dtype float32 \
  --val_frac 0.1 --test_frac 0.1 --save_rgb

Stage 3: add normalization (keep origin; scale only, no translation)
python tdcr_pipeline.py add-norm \
  --root 2m_no_base/ --mode global --scope all \
  --anchor origin \
  --dtype float32 --overwrite --dump_global \
  --export_ply 6 --export_dir norm_samples

python tdcr_pipeline.py collect \
  --xml tdcr2.xml \
  --nsample 5000 \
  --out_pcd_dir "2m_no_base/pointcloud" \
  --out_json_dir "2m_no_base/motor" \
  --sampling continuous \
  --seed 42 \
  --start_index 1 \
  --unique_tol 1e-6 \
  --backend auto \
  --workers 8 \
  --zero_vel_each_ctrl \
  --relax_max_steps 10000 \
  --out_rgb_dir 2m_no_base/rgb --rgb_only
'''