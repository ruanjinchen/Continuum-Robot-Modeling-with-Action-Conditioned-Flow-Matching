from __future__ import annotations
import os, json
import numpy as np
from pathlib import Path
from .config import MergeCfg
from .utils import ensure_dir, print_color

def _read_one_motor_json(path: str):
    with open(path, 'r') as f:
        obj = json.load(f)
    if isinstance(obj, (list, tuple)):
        arr = np.asarray(obj, dtype=np.float32).reshape(-1)
        return arr
    if isinstance(obj, dict):
        if 'ctrl' in obj:
            arr = np.asarray(obj['ctrl'], dtype=np.float32).reshape(-1)
            return arr
        # fallback: motor1..motorK
        vals = []
        j = 1
        while True:
            key = f'motor{j}'
            if key not in obj:
                break
            vals.append(obj[key])
            j += 1
        if vals:
            return np.asarray(vals, dtype=np.float32).reshape(-1)
    raise ValueError(f'Unsupported JSON format in {path}')

def merge_motors_stage(cfg: MergeCfg):
    N = cfg.end - cfg.start + 1
    # peek first file to determine D
    first = os.path.join(cfg.src_dir, str(cfg.start).zfill(cfg.zfill) + '.json')
    if not os.path.exists(first):
        raise FileNotFoundError(first)
    sample = _read_one_motor_json(first)
    D = int(sample.size)
    motors = np.zeros((N, D), dtype=np.float32)

    from tqdm import tqdm as _tqdm
    for i in _tqdm(range(cfg.start, cfg.end+1), ncols=100, desc='[merge motors]'):
        name = str(i).zfill(cfg.zfill) + '.json'
        path = os.path.join(cfg.src_dir, name)
        arr = _read_one_motor_json(path)
        if arr.size != D:
            raise ValueError(f'{path}: inconsistent motor dim, expect {D}, got {arr.size}')
        motors[i-cfg.start] = arr.astype(np.float32)

    ensure_dir(Path(cfg.out_json).parent)
    if cfg.fmt == 'array':
        out_obj = {'motors': motors.tolist()}
    else:
        out_obj = { str(cfg.start + i).zfill(cfg.zfill):
                    {f'motor{j}': float(motors[i,j-1]) for j in range(1, D+1)}
                    for i in range(N) }
    with open(cfg.out_json, 'w') as f:
        json.dump(out_obj, f)
    print_color(f'[merge] wrote {cfg.out_json}')
    ensure_dir(Path(cfg.out_npz).parent)
    np.savez(cfg.out_npz, motors=motors.astype(np.float32))
    print_color(f'[merge] wrote {cfg.out_npz} with array \'motors\' shape {motors.shape}')
    print_color('✅ [merge] done.')
