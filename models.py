import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys
try:
    here = os.path.dirname(os.path.abspath(__file__))
    tp = os.path.join(here, "third_party", "pvcnn")
    if os.path.isdir(tp) and (tp not in sys.path):
        sys.path.insert(0, tp)
    from modules.pvconv import PVConv
    from modules.shared_mlp import SharedMLP
except Exception:
    PVConv = None
    SharedMLP = None



# ---------------- Time / condition embeddings ----------------

def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """
    Sinusoidal embedding of continuous time t in [0,1].
    Args:
        t: shape (...,) scalar per batch (we'll broadcast)
        dim: embedding dimension (even)
        max_period: controls min frequency
    Returns:
        emb: shape (..., dim)
    """
    assert dim % 2 == 0, "timestep_embedding dim must be even"
    half = dim // 2
    # Map t from [0,1] to [0, max_period] by simple scaling so frequencies are nicely spread.
    # Using log scale like diffusion.
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=t.device, dtype=t.dtype) / half)
    # ensure t shape (..., 1)
    t = t.reshape(*t.shape, 1)
    args = t * freqs
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int = 256, depth: int = 4, act=nn.SiLU, dropout: float = 0.0):
        super().__init__()
        layers = []
        d = in_dim
        for i in range(depth - 1):
            layers += [nn.Linear(d, width), nn.SiLU(), nn.Dropout(dropout)]
            d = width
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

        # Kaiming init for stability
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.0, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FiLMBlock(nn.Module):
    def __init__(self, width: int, emb_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.affine = nn.Linear(emb_dim, width * 2)
        nn.init.zeros_(self.affine.bias)

    def forward(self, h: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        h: (B*N, C)
        emb: (B, E) -> will be broadcast to (B*N, E) by indexing beforehand
        """
        h = self.norm(h)
        gamma, beta = self.affine(emb).chunk(2, dim=-1)
        h = h * (1.0 + gamma) + beta
        return h


class VelocityNet(nn.Module):
    """
    Per-point velocity field v_theta(x, t, cond).
    Design: simple MLP with FiLM conditioning by (time embedding || condition embedding).
    This is purposely light and fast for large N.
    """
    def __init__(self, cond_dim: int, point_dim: int = 3,
                width: int = 512, depth: int = 6, emb_dim: int = 256,
                cfg_dropout_p: float = 0.1):

        super().__init__()
        self.cond_dim = int(cond_dim)
        self.point_dim = int(point_dim)
        self.emb_dim = int(emb_dim)
        self.cfg_dropout_p = float(cfg_dropout_p)

        # encoders
        self.t_proj = nn.Linear(emb_dim, emb_dim)
        self.c_proj = nn.Linear(cond_dim if cond_dim > 0 else 1, emb_dim)
        nn.init.normal_(self.t_proj.weight, std=0.02); nn.init.zeros_(self.t_proj.bias)
        nn.init.normal_(self.c_proj.weight, std=0.02); nn.init.zeros_(self.c_proj.bias)

        # MLP trunk on per-point coordinates
        in_dim = self.point_dim + emb_dim  # concatenate per-point coords with global emb
        self.input = nn.Linear(in_dim, width)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.SiLU(), nn.Linear(width, width)) for _ in range(depth - 1)
        ])
        self.films = nn.ModuleList([FiLMBlock(width, emb_dim) for _ in range(depth - 1)])
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(width, self.point_dim))

        # init
        nn.init.kaiming_normal_(self.input.weight, nonlinearity="relu"); nn.init.zeros_(self.input.bias)
        for m in self.blocks:
            for l in m:
                if isinstance(l, nn.Linear):
                    nn.init.kaiming_normal_(l.weight, nonlinearity="relu"); nn.init.zeros_(l.bias)
        for l in self.out:
            if isinstance(l, nn.Linear):
                nn.init.zeros_(l.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor],
                cond_drop_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, N, 3) current points
            t: (B,) or (B,1) time in [0,1]
            cond: (B, C) or None
            cond_drop_mask: (B,1) booleans; if provided, zeros out cond for classifier-free branch
        Returns:
            v: (B, N, 3)
        """
        B, N, D = x.shape
        assert D == self.point_dim, f"VelocityNet: x has dim {D}, but point_dim={self.point_dim}"

        if t.dim() == 1:
            t = t[:, None]
        # time embedding
        t_emb = timestep_embedding(t.squeeze(-1), self.emb_dim).to(x.dtype)
        t_emb = torch.nn.functional.silu(self.t_proj(t_emb))

        # condition embedding (classifier-free dropout)
        if self.cond_dim > 0 and cond is not None:
            if cond_drop_mask is not None:
                cond = cond * (1.0 - cond_drop_mask)  # mask=1 -> drop
            c_in = cond
        else:
            # If cond_dim > 0 but cond is None (e.g. unconditional sampling),
            # feed a (B, cond_dim) all-zero tensor so Linear sees consistent in_features.
            if self.cond_dim > 0:
                c_in = x.new_zeros((B, self.cond_dim))
            else:
                c_in = x.new_zeros((B, 1))
        # Safety check: fail fast if cond has an unexpected dimensionality (easier to debug).
        assert c_in.shape[1] == (self.cond_dim if self.cond_dim > 0 else 1), \
            f"cond has wrong dim: got {c_in.shape[1]}, expected {(self.cond_dim if self.cond_dim > 0 else 1)}"

        c_emb = torch.nn.functional.silu(self.c_proj(c_in))
        emb = t_emb + c_emb  # fuse

        # broadcast emb to (B*N, E)
        emb_bn = emb[:, None, :].expand(B, N, -1).reshape(B * N, -1)

        # per-point trunk
        h = torch.cat([x, emb[:, None, :].expand(B, N, -1)], dim=-1).reshape(B * N, -1)
        h = self.input(h)
        for blk, fim in zip(self.blocks, self.films):
            h = fim(h, emb_bn)
            h = h + blk(h)  # residual
        v = self.out(h).reshape(B, N, self.point_dim)

        return v

    @torch.no_grad()
    def guided_velocity(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor],
                        guidance_scale: float = 0.0) -> torch.Tensor:
        """
        Classifier-free guidance: v = v(cond) + s * (v(cond) - v(uncond))
        If s==0 -> pure conditional branch.
        """
        if guidance_scale <= 0.0 or cond is None or self.cond_dim == 0:
            return self.forward(x, t, cond, cond_drop_mask=None)
        # cond branch
        v_c = self.forward(x, t, cond, cond_drop_mask=None)
        # uncond branch (drop all cond)
        mask = torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)
        v_u = self.forward(x, t, cond, cond_drop_mask=mask)
        return v_c + guidance_scale * (v_c - v_u)

# =====================================================
# ===============  Hybrid context (PVConv) + per-point MLP  ===========
# =====================================================

def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return max(a, 1)


def _choose_gn_groups(channels: int, prefer: int = 32) -> int:
    prefer = min(prefer, channels)
    g = _gcd(channels, prefer)
    if g == 1 and channels >= 16:
        for cand in [32, 16, 8, 4, 2]:
            if channels % cand == 0 and cand <= channels:
                return cand
    return g


def _make_norm(norm_type: str, channels: int, gn_groups: int):
    if norm_type == "group":
        return nn.GroupNorm(_choose_gn_groups(channels, gn_groups), channels)
    elif norm_type in ("batch", "syncbn"):
        return nn.BatchNorm1d(channels)
    else:
        return nn.Identity()


class _FiLM1d(nn.Module):
    """
    FiLM on (B, C, N): Norm -> (1 + gamma) * x + beta. gamma/beta are zero-initialized so the block starts as identity.
    """
    def __init__(self, channels: int, emb_dim: int, norm_type: str = "group",
                 gn_groups: int = 32, one_plus: bool = True):
        super().__init__()
        self.norm = _make_norm(norm_type, channels, gn_groups)
        self.affine = nn.Linear(emb_dim, channels * 2)
        self.one_plus = bool(one_plus)
        nn.init.zeros_(self.affine.weight); nn.init.zeros_(self.affine.bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        x: (B,C,N)    emb: (B,E)
        """
        B, C, N = x.shape
        y = self.norm(x)
        emb = emb.to(y.dtype)
        gamma, beta = self.affine(emb).chunk(2, dim=-1)
        gamma = gamma.view(B, C, 1); beta = beta.view(B, C, 1)
        if self.one_plus:
            return y * (1.0 + gamma) + beta
        else:
            return y * gamma + beta


class _PVBlock(nn.Module):
    """
    A PVConv block: PVConv -> SharedMLP(1x1) -> FiLM -> residual add.
    - normalize=True stabilizes voxel neighborhoods (recommended).
    """
    def __init__(self, channels: int, resolution: int, emb_dim: int, with_se: bool,
                 norm_type: str = "group", gn_groups: int = 32, voxel_normalize: bool = True,
                 eps: float = 1e-6):
        super().__init__()
        self.pvconv = PVConv(channels, channels, kernel_size=3,
                             resolution=int(resolution), with_se=bool(with_se),
                             normalize=bool(voxel_normalize), eps=eps)
        self.post = SharedMLP(channels, [channels])
        self.film = _FiLM1d(channels, emb_dim, norm_type=norm_type, gn_groups=gn_groups, one_plus=True)

    def forward(self, feat_coords: Tuple[torch.Tensor, torch.Tensor], emb: torch.Tensor):
        f, c = self.pvconv(feat_coords)
        f, c = self.post((f, c))
        f = f + self.film(f, emb)
        return f, c


class _PVStage(nn.Module):
    """
    Stage: 1x1 SharedMLP channel projection -> k × PVBlock
    """
    def __init__(self, in_c: int, out_c: int, num_blocks: int, resolution: int, emb_dim: int,
                 with_se: bool, norm_type: str = "group", gn_groups: int = 32, voxel_normalize: bool = True):
        super().__init__()
        self.proj = SharedMLP(in_c, [out_c])
        self.blocks = nn.ModuleList([
            _PVBlock(out_c, resolution, emb_dim, with_se,
                     norm_type=norm_type, gn_groups=gn_groups, voxel_normalize=voxel_normalize)
            for _ in range(int(num_blocks))
        ])

    def forward(self, feat: torch.Tensor, coords: torch.Tensor, emb: torch.Tensor):
        f, c = self.proj((feat, coords))
        for blk in self.blocks:
            f, c = blk((f, c), emb)
        return f, c


class ContextNet(nn.Module):
    """
    Multi-scale PVConv pyramid that outputs per-point context (ctx) features.
    Includes optional t-gated fusion: small t leans on a global (t,cond) context, large t leans on PVConv multi-scale context.
    """
    def __init__(self,
                 in_point_dim: int,              # 3 / 6
                 cond_dim: int,                  # z (+ joint)
                 emb_dim: int = 256,             # t/cond embedding dim
                 ctx_dim: int = 64,              # output context dim
                 stage_channels: List[int] = (128, 256, 256),
                 stage_blocks:   List[int] = (2, 2, 2),
                 stage_res:      List[int] = (32, 16, 8),
                 with_se: bool = True,
                 norm_type: str = "group",
                 gn_groups: int = 32,
                 with_global: bool = True,
                 voxel_normalize: bool = True,
                 # ---- t-gating parameters ----
                 use_t_gate: bool = True,
                 t_gate_k: float = 10.0,
                 t_gate_tau: float = 0.4):
        super().__init__()
        assert len(stage_channels) == len(stage_blocks) == len(stage_res)
        self.in_point_dim = int(in_point_dim)
        self.emb_dim = int(emb_dim)
        self.ctx_dim = int(ctx_dim)
        self.with_global = bool(with_global)
        self.use_t_gate = bool(use_t_gate)
        self.t_gate_k = float(t_gate_k)
        self.t_gate_tau = float(t_gate_tau)

        self.use_xyz = True
        self.use_rgb = (self.in_point_dim == 6)

        # t/cond -> emb
        self.t_proj = nn.Linear(emb_dim, emb_dim)
        self.c_proj = nn.Linear(cond_dim if cond_dim > 0 else 1, emb_dim)
        nn.init.normal_(self.t_proj.weight, std=0.02); nn.init.zeros_(self.t_proj.bias)
        nn.init.normal_(self.c_proj.weight, std=0.02); nn.init.zeros_(self.c_proj.bias)

        # Stem input channels: emb + xyz (+rgb)
        stem_in_c = emb_dim + (3 if self.use_xyz else 0) + (3 if self.use_rgb else 0)

        # stages
        self.stages = nn.ModuleList()
        in_c = stem_in_c
        for sc, nb, rs in zip(stage_channels, stage_blocks, stage_res):
            self.stages.append(
                _PVStage(in_c, sc, nb, rs, emb_dim, with_se,
                         norm_type=norm_type, gn_groups=gn_groups, voxel_normalize=voxel_normalize)
            )
            in_c = sc

        # Optional global branch
        if self.with_global:
            C_last = stage_channels[-1]
            self.global_mlp = nn.Sequential(
                nn.Linear(C_last, C_last), nn.SiLU(),
                nn.Linear(C_last, C_last)
            )
            for m in self.global_mlp:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    nn.init.zeros_(m.bias)

        # Output head: multi-scale aggregation -> ctx_dim
        self.stage_channels = list(stage_channels)
        head_in = sum(self.stage_channels) + (self.stage_channels[-1] if self.with_global else 0)
        self.head_pre = nn.Conv1d(head_in, self.stage_channels[-1], 1, bias=True)
        self.head_norm = _make_norm(norm_type, self.stage_channels[-1], gn_groups)
        self.head_act  = nn.SiLU()
        self.head_out  = nn.Conv1d(self.stage_channels[-1], ctx_dim, 1, bias=True)

        nn.init.kaiming_normal_(self.head_pre.weight, nonlinearity="relu")
        nn.init.zeros_(self.head_pre.bias)
        nn.init.zeros_(self.head_out.weight); nn.init.zeros_(self.head_out.bias)

        self.norm_type = norm_type
        self.gn_groups = int(gn_groups)

        # ---- Global context generated purely from (t,cond) (used by t-gating) ----
        self.ctx_from_emb = nn.Sequential(
            nn.Linear(self.emb_dim, self.ctx_dim)
        )
        for m in self.ctx_from_emb:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def _t_emb(self, t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if t.dim() == 1:
            t = t[:, None]
        return F.silu(self.t_proj(timestep_embedding(t.squeeze(-1), self.emb_dim).to(dtype)))

    def _c_emb(self, x: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        if cond is None or cond.numel() == 0:
            c_in = x.new_zeros((x.shape[0], 1))
        else:
            c_in = cond
        return F.silu(self.c_proj(c_in))

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        """
        x: (B,N,3/6) -> ctx: (B,N,ctx_dim)
        """
        B, N, D = x.shape
        coords = x[..., :3].permute(0, 2, 1).contiguous()  # (B,3,N)

        # emb (t, cond)
        emb = self._t_emb(t, x.dtype) + self._c_emb(x, cond)  # (B,E)

        # stem feats = [emb | xyz | (rgb)]
        feats = [emb[:, :, None].expand(B, self.emb_dim, N)]
        if self.use_xyz:
            feats.append(coords)
        if self.use_rgb and D == 6:
            feats.append(x[..., 3:].permute(0, 2, 1).contiguous())
        feats = torch.cat(feats, dim=1)  # (B,C_in,N)

        # Run the PVConv stack in FP32 for numerical stability
        with torch.amp.autocast("cuda", enabled=False):
            f, c = feats.float(), coords.float()
            emb32 = emb.float()

            ms_feats = []
            for stage in self.stages:
                f, c = stage(f, c, emb32)
                ms_feats.append(f)                      # (B,C_i,N)

            if self.with_global:
                g = f.max(dim=-1).values                # (B,C_last)
                g = self.global_mlp(g)                  # (B,C_last)
                g = g[:, :, None].expand_as(f)          # (B,C_last,N)
                ms_feats.append(g)

            f_cat = torch.cat(ms_feats, dim=1)          # (B, sum(C_i)+..., N)

            h = self.head_pre(f_cat)
            h = self.head_act(self.head_norm(h))
            ctx_pv = self.head_out(h).permute(0, 2, 1)  # (B,N,ctx_dim)  -- PVConv multi-scale context

            if self.use_t_gate:
                # Build a "global" context from emb and interpolate with PVConv context via a t-gate.
                ctx_glb = self.ctx_from_emb(emb32)                      # (B,ctx_dim)
                ctx_glb = ctx_glb[:, None, :].expand(B, N, -1)          # (B,N,ctx_dim)
                alpha = torch.sigmoid(self.t_gate_k * (t.view(B, 1, 1).float() - self.t_gate_tau))
                ctx = alpha * ctx_pv + (1.0 - alpha) * ctx_glb
            else:
                ctx = ctx_pv

        return ctx.to(x.dtype)


class VelocityNetWithContext(nn.Module):
    """
    Per-point MLP head that maps [x_i | ctx_i | emb(t,cond)] -> v_i.
    Initialization / CFG / FiLM behavior is kept consistent with the base VelocityNet.
    """
    def __init__(self, cond_dim: int, point_dim: int = 3, ctx_dim: int = 64,
                 width: int = 512, depth: int = 6, emb_dim: int = 256,
                 cfg_dropout_p: float = 0.1):
        super().__init__()
        self.cond_dim, self.point_dim = int(cond_dim), int(point_dim)
        self.emb_dim, self.ctx_dim = int(emb_dim), int(ctx_dim)
        self.cfg_dropout_p = float(cfg_dropout_p)

        self.t_proj = nn.Linear(emb_dim, emb_dim)
        self.c_proj = nn.Linear(cond_dim if cond_dim > 0 else 1, emb_dim)
        nn.init.normal_(self.t_proj.weight, std=0.02); nn.init.zeros_(self.t_proj.bias)
        nn.init.normal_(self.c_proj.weight, std=0.02); nn.init.zeros_(self.c_proj.bias)

        in_dim = self.point_dim + self.ctx_dim + emb_dim
        self.input = nn.Linear(in_dim, width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.SiLU(), nn.Linear(width, width)) for _ in range(depth - 1)])
        self.films  = nn.ModuleList([FiLMBlock(width, emb_dim) for _ in range(depth - 1)])
        self.out    = nn.Sequential(nn.SiLU(), nn.Linear(width, self.point_dim))

        nn.init.kaiming_normal_(self.input.weight, nonlinearity="relu"); nn.init.zeros_(self.input.bias)
        for m in self.blocks:
            for l in m:
                if isinstance(l, nn.Linear):
                    nn.init.kaiming_normal_(l.weight, nonlinearity="relu"); nn.init.zeros_(l.bias)
        for l in self.out:
            if isinstance(l, nn.Linear):
                nn.init.zeros_(l.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor],
                ctx: torch.Tensor, cond_drop_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, D = x.shape
        assert ctx.shape[:2] == (B, N), f"ctx shape mismatch: {tuple(ctx.shape)} vs {(B,N,'*')}"
        if t.dim() == 1: t = t[:, None]
        t_emb = torch.nn.functional.silu(self.t_proj(timestep_embedding(t.squeeze(-1), self.emb_dim).to(x.dtype)))
        if self.cond_dim > 0 and cond is not None:
            if cond_drop_mask is not None:
                cond = cond * (1.0 - cond_drop_mask)  # mask=1 -> drop
            c_in = cond
        else:
            c_in = x.new_zeros((B, self.cond_dim if self.cond_dim > 0 else 1))
        c_emb = torch.nn.functional.silu(self.c_proj(c_in))
        emb = t_emb + c_emb

        emb_bn = emb[:, None, :].expand(B, N, -1).reshape(B * N, -1)
        h = torch.cat([x, ctx, emb[:, None, :].expand(B, N, -1)], dim=-1).reshape(B * N, -1)
        h = self.input(h)
        for blk, fim in zip(self.blocks, self.films):
            h = fim(h, emb_bn)
            h = h + blk(h)
        v = self.out(h).reshape(B, N, self.point_dim)
        return v


class HybridMLP(nn.Module):
    """
    Hybrid point-flow backbone:
      - ContextNet (PVConv pyramid) produces per-point context ctx
      - VelocityNetWithContext regresses per-point velocity
      - forward(x, t, cond, cond_drop_mask=None) -> v
      - guided_velocity(x, t, cond, guidance_scale=0.0)
      - set_bn_eval(freeze=True): optionally freeze BatchNorm mid-training
    """
    def __init__(self,
                 cond_dim: int,
                 point_dim: int = 3,
                 # Context branch
                 ctx_dim: int = 64,
                 ctx_emb_dim: int = 256,
                 stage_channels: List[int] = (128, 256, 256),
                 stage_blocks:   List[int] = (2, 2, 2),
                 stage_res:      List[int] = (32, 16, 8),
                 with_se: bool = True,
                 norm_type: str = "group",
                 gn_groups: int = 32,
                 with_global: bool = True,
                 voxel_normalize: bool = True,
                 # t-gate (controls when to rely more on PVConv context)
                 use_t_gate: bool = True,
                 t_gate_k: float = 10.0,
                 t_gate_tau: float = 0.8,
                 # Per-point MLP head
                 pf_width: int = 512,
                 pf_depth: int = 6,
                 pf_emb_dim: int = 256,
                 cfg_dropout_p: float = 0.1):
        super().__init__()
        if PVConv is None or SharedMLP is None:
            raise ImportError(
                "HybridMLP requires PVConv / SharedMLP, but they were not imported successfully.\n"
                "Please ensure third_party/pvcnn is installed correctly, or use --pf_backbone mlp instead."
            )
        self.cond_dim = int(cond_dim)
        self.point_dim = int(point_dim)

        # Context extractor
        self.ctx_net = ContextNet(
            in_point_dim=point_dim, cond_dim=cond_dim, emb_dim=ctx_emb_dim, ctx_dim=ctx_dim,
            stage_channels=list(stage_channels), stage_blocks=list(stage_blocks), stage_res=list(stage_res),
            with_se=with_se, norm_type=norm_type, gn_groups=gn_groups,
            with_global=with_global, voxel_normalize=voxel_normalize,
            # Pass through t-gate params
            use_t_gate=use_t_gate, t_gate_k=t_gate_k, t_gate_tau=t_gate_tau
        )
        # Per-point head
        self.head = VelocityNetWithContext(
            cond_dim=cond_dim, point_dim=point_dim, ctx_dim=ctx_dim,
            width=pf_width, depth=pf_depth, emb_dim=pf_emb_dim, cfg_dropout_p=cfg_dropout_p
        )

    @staticmethod
    def _cond_eff(cond: Optional[torch.Tensor], mask: Optional[torch.Tensor], x: torch.Tensor):
        if cond is None:
            return None if hasattr(x, "new_zeros") is False else x.new_zeros((x.shape[0], 1))
        if mask is None:
            return cond
        return cond * (1.0 - mask.to(cond.dtype))

    def set_bn_eval(self, freeze: bool = True):
        """
        Freeze/unfreeze BatchNorm layers.
        - If vendor SharedMLP/PVConv contains BN/SyncBN, they will be set to eval() when frozen.
        - GroupNorm is unaffected.
        """
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                m.eval() if freeze else m.train()
                m.track_running_stats = True
                m.momentum = 0.0 if freeze else 0.1

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor],
                cond_drop_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B,N,3/6)  t: (B,)  cond: (B,C)
        return: v (B,N,3/6)
        """
        cond_eff = self._cond_eff(cond, cond_drop_mask, x)
        ctx = self.ctx_net(x, t, cond_eff if self.cond_dim > 0 else None)  # (B,N,ctx_dim)
        v = self.head(x, t, cond, ctx, cond_drop_mask=cond_drop_mask)
        return v

    @torch.no_grad()
    def guided_velocity(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor],
                        guidance_scale: float = 0.0) -> torch.Tensor:
        if guidance_scale <= 0.0 or self.cond_dim == 0 or cond is None:
            return self.forward(x, t, cond, cond_drop_mask=None)
        zdim = int(getattr(self, "z_dim", 0))
        if zdim > 0 and cond.shape[1] >= zdim:
            z, j = cond[:, :zdim], cond[:, zdim:]
            j0 = torch.zeros_like(j)
            v_c = self.forward(x, t, torch.cat([z, j],  dim=1), cond_drop_mask=None)
            v_u = self.forward(x, t, torch.cat([z, j0], dim=1), cond_drop_mask=None)
        else:
            v_c = self.forward(x, t, cond,                 cond_drop_mask=None)
            v_u = self.forward(x, t, torch.zeros_like(cond), cond_drop_mask=None)
        return v_c + guidance_scale * (v_c - v_u)