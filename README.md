# Continuum-Robot-Modeling-with-Action-Conditioned-Flow-Matching
This is a **TDCR Modeling** project based on **Flow Matching**, implemented in **PyTorch**.

Tested on:
- Windows 11 + CUDA 13.0 + NVIDIA GeForce RTX 5090
- Ubuntu 22.04 + CUDA 12.4 + NVIDIA H100 PCIe
- Ubuntu 22.04 + CUDA 13.0 + NVIDIA GeForce RTX 5090

## Installation

```sh
conda create -y -n tdcr python=3.12
conda activate tdcr
pip install -r requirements.txt
```

### Install PyTorch (pick **one** CUDA build)

**Pick one** matching your CUDA runtime/driver. Do not run all blocks.

```sh
# PyTorch + cu124 (official index)
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
  torch torchvision

# PyTorch + cu130 (official index)
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu130 \
  torch torchvision
```

### Optional: install CUDA toolkit via conda (Linux)

If you do not have a suitable CUDA toolkit installed system-wide, you can install it into the conda env:

```sh
# CUDA 12.4
conda install -y --override-channels --solver=libmamba \
  -c nvidia/label/cuda-12.4.1 -c defaults \
  cuda-toolkit=12.4.*

# CUDA 13.0
conda install -y --override-channels --solver=libmamba \
  -c nvidia/label/cuda-13.0.2 -c defaults \
  cuda-toolkit=13.0.*
```

## Compile

This repo uses several CUDA extensions. Build them once after installing PyTorch.

### PVCNN

```sh
cd third_party/pvcnn

# Set this according to your GPU compute capability (SM).
# Example: 8.0 (A100), 9.0 (H100), etc.
export TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0;12.0"

# If you installed CUDA via conda, these environment variables help CMake/NVCC find headers and libs.
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CONDA_PREFIX/include:$CONDA_PREFIX/targets/x86_64-linux/include:${CPATH}"
export LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}"

# Optional build knobs
export MAX_JOBS=16
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++

python - <<'PY'
import time
start = time.time()
from modules.functional import backend
print("[OK] pvcnn backend built in %.1fs" % (time.time() - start))
PY

cd ../..
```

Sanity check:

```sh
python - <<'PY'
try:
  import os, sys
  here = os.path.dirname(os.path.abspath(__file__))
  tp = os.path.join(here, "third_party", "pvcnn")
  if os.path.isdir(tp) and (tp not in sys.path):
      sys.path.insert(0, tp)
  from modules.pvconv import PVConv
  from modules.shared_mlp import SharedMLP
  print("IMPORT SUCCESS")
except Exception as e:
  raise ImportError(
      f"Cannot import PVConv: {e}. "
      f"Put PVCNN's 'modules/' under third_party/pvcnn/ and ensure a CUDA toolchain is available."
  )
PY
```

### PyTorchEMD

```sh
cd third_party/PyTorchEMD
python -m pip install . --no-build-isolation -v
cd ../..
```

Sanity check:

```sh
python - <<'PY'
import torch, importlib
print("torch", torch.__version__, "cuda", torch.version.cuda)

# Ensure the binary extension can be imported.
m = importlib.import_module("emd_ext")
print("emd_ext ->", m.__file__)

from third_party.PyTorchEMD.emd import earth_mover_distance

x = torch.rand(2, 1024, 3, device='cuda')
y = torch.rand(2, 1024, 3, device='cuda')
d = earth_mover_distance(x, y, transpose=False)
print("EMD OK:", d.shape, "mean:", float(d.mean()))
PY
```

### ChamferDistancePytorch

```sh
cd third_party/ChamferDistancePytorch/chamfer3D
python -m pip install . --no-build-isolation -v
cd ../../..
```

Sanity check (Chamfer distance between identical point clouds should be ~0):

```sh
python - <<'PY'
import torch, importlib
m = importlib.import_module('chamfer_3D')

B, N = 2, 2048
x = torch.randn(B, N, 3, device='cuda', dtype=torch.float32).contiguous()

d1 = torch.empty(B, N, device='cuda', dtype=torch.float32)
d2 = torch.empty(B, N, device='cuda', dtype=torch.float32)
i1 = torch.empty(B, N, device='cuda', dtype=torch.int32)
i2 = torch.empty(B, N, device='cuda', dtype=torch.int32)

ret = m.forward(x, x, d1, d2, i1, i2)
cd = (d1.mean() + d2.mean()).item()
print(f"ret={ret} | CD(same): {cd:.8f}")
PY
```

## Dataset
制作数据集，以sim_2m_with_base为例：
```sh
cd sim
export MUJOCO_GL=egl
export EGL_LOG_LEVEL=fatal
export LIBEGL_DEBUG=fatal

阶段1:采集数据
python tdcr_pipeline.py collect \
  --xml tdcr2_with_base.xml \
  --nsample 5000 \
  --out_pcd_dir "2m_with_base/pointcloud" \
  --out_json_dir "2m_with_base/motor" \
  --out_rgb_dir 2m_with_base/rgb \
  --sampling continuous \
  --seed 42 \
  --start_index 1 \
  --unique_tol 1e-6 \
  --backend auto \
  --workers 48 \
  --zero_vel_each_ctrl \
  --relax_max_steps 10000

阶段2:制作 H5（新增 motor_dir）
python tdcr_pipeline.py make-h5 \
  --pc_dir "2m_with_base/pointcloud" \
  --motor_dir "2m_with_base/motor" \
  --out_root 2m_with_base/ \
  --npoints 20000 --voxel_size 0.002 \
  --workers 32 --dtype float32 \
  --val_frac 0.1 --test_frac 0.1 --save_rgb

阶段3:补写归一化 原点不变 只缩放不平移
python tdcr_pipeline.py add-norm \
  --root 2m_with_base/ --mode global --scope all \
  --anchor origin \
  --dtype float32 --overwrite --dump_global \
  --export_ply 6 --export_dir norm_samples
```

## Training

同样以sim_2m_with_base为例，展示MLP和Hybrid两种Backbone的训练指令：
```sh

# MLP backbone的
export CUDA_VISIBLE_DEVICES=5
python train_flowmatching.py \
  --data_dir datasets/sim/2m_with_base \
  --batch_size 8 --epochs 500 --save_every 20 \
  --tr_max_sample_points 20000 --te_max_sample_points 20000 \
  --cond_mode motors \
  --pf_backbone mlp \
  --use_cosine_lr \
  --use_rgb --rgb_key rgb \
  --lambda_color 0.05 \
  --t_beta_a 3.0 \
  --point_prior_std 0.5 \
  --sample_steps 100 \
  --out_dir runs/sim_2m_with_base_mlp

# Hybrid Backbone的
export CUDA_VISIBLE_DEVICES=1
python train_flowmatching.py \
  --data_dir datasets/sim/2m_with_base \
  --batch_size 32 --lr 8e-4 --warmup_steps 4000 --epochs 500 --save_every 20 \
  --tr_max_sample_points 20000 --te_max_sample_points 20000 \
  --cond_mode motors \
  --pf_backbone hybrid \
  --emb_dim 256 --width 512 --depth 6 --cfg_drop_p 0.0 \
  --ctx_dim 64 \
  --ctx_emb_dim 256 \
  --ctx_stage_channels 80 112 112 \
  --ctx_stage_blocks 2 2 2 \
  --ctx_stage_res 24 16 8 \
  --ctx_with_se --ctx_with_global --ctx_voxel_normalize \
  --ctx_t_gate_tau 0.97 --ctx_t_gate_k 12 \
  --use_cosine_lr \
  --use_rgb --rgb_key rgb \
  --lambda_color 0.05 \
  --t_beta_a 3.0 \
  --point_prior_std 0.5 \
  --sample_steps 50 \
  --out_dir runs/sim_2m_with_base_hybrid
```

## Demo
这个demo展示了输入一个控制序列，让模型预测出每一次控制结果，并保存点云
```sh
export CUDA_VISIBLE_DEVICES=5
SEQ_ROOT=demo

python demo_tdcr_motor_seq_pred.py \
  --ckpt final_results/sim/mlp/sim_2m_with_base_mlp_12_28/ckpts/latest.pt \
  --motor_seq ${SEQ_ROOT}/sim_2m_with_base/motor_seq.json \
  --demo_out demo/sim_2m_with_base_mlp_demo2_seq_pred \
  --eval_norm_json sim/dataset_norm_json/global_norm_scope-all_anchor-origin.json \
  --npoints 20000 \
  --sample_steps 100 \
  --batch_size 1 \
  --seed 123

python demo_tdcr_motor_seq_pred.py \
  --ckpt final_results/sim/hybrid/sim_2m_with_base_hybrid_1_2/ckpts/latest.pt \
  --motor_seq ${SEQ_ROOT}/sim_2m_with_base/motor_seq.json \
  --demo_out demo/sim_2m_with_base_hybrid_demo2_seq_pred \
  --eval_norm_json sim/dataset_norm_json/global_norm_scope-all_anchor-origin.json \
  --npoints 20000 \
  --sample_steps 100 \
  --batch_size 1 \
  --seed 123

```