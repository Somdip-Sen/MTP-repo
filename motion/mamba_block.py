"""
Pure PyTorch implementation of a Mamba (Selective State Space Model) block.

Why pure PyTorch?
  The official `mamba_ssm` package requires CUDA. This implementation runs on
  CPU / MPS / CUDA. It is slower than the CUDA kernel, but for motion
  prediction over short trajectories (seq_len <= 30), speed is a non-issue.

Reference:
  Mamba paper (Gu & Dao, 2023): https://arxiv.org/abs/2312.00752
  mamba-minimal repo (John Ma): https://github.com/johnma2006/mamba-minimal
  mamba.py repo (alxndrTL):     https://github.com/alxndrTL/mamba.py

This is a simplified, readable port. It includes:
  - Selective SSM with input-dependent B, C, Delta
  - 1D depthwise conv
  - SiLU gating (Mamba block structure)
  - Sequential scan (works on all devices)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    """A single Mamba (S6) block. Input/output shape: (B, L, d_model)."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
        bias: bool = False,
        conv_bias: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = dt_rank if dt_rank is not None else math.ceil(d_model / 16)

        # Input projection: d_model -> 2 * d_inner (one half for main path, one for gate)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=bias)

        # 1D depthwise conv on the main path
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=conv_bias,
        )

        # x -> (delta, B, C)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        # delta's low-rank -> d_inner projection, with learned bias init
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # S4D-style A init: A = -diag(1..N), stored in log space
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        # Skip connection D (per-channel scalar)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection back to d_model
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

        # dt_proj bias init: softplus(bias) spread in [dt_min, dt_max]
        self._init_dt_bias()

    def _init_dt_bias(self, dt_min: float = 1e-3, dt_max: float = 1e-1) -> None:
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=1e-4)
        # inverse-softplus so softplus(bias) == dt
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) -> (B, L, d_model)."""
        B, L, _ = x.shape

        # Project to 2*d_inner and split into main path (x) and gate (z)
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_path, z = xz.chunk(2, dim=-1)  # each: (B, L, d_inner)

        # Depthwise conv along sequence dim (needs (B, C, L))
        x_path = x_path.transpose(1, 2)  # (B, d_inner, L)
        x_path = self.conv1d(x_path)[:, :, :L]  # trim causal padding tail
        x_path = x_path.transpose(1, 2)  # (B, L, d_inner)
        x_path = F.silu(x_path)

        # Selective SSM: input-dependent delta, B, C
        y = self._selective_scan(x_path)

        # Gate and project out
        y = y * F.silu(z)
        return self.out_proj(y)

    def _selective_scan(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, d_inner) -> y: (B, L, d_inner)
        Selective scan with input-dependent B, C, delta.
        Uses a sequential loop over L (portable across devices).
        """
        B_batch, L, d_inner = x.shape
        N = self.d_state

        # Project x to get delta, B, C
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*N)
        delta, B_proj, C_proj = torch.split(
            x_dbl, [self.dt_rank, N, N], dim=-1
        )
        # delta: (B, L, dt_rank) -> (B, L, d_inner), then softplus
        delta = F.softplus(self.dt_proj(delta))

        A = -torch.exp(self.A_log.float())  # (d_inner, N), negative real
        D = self.D.float()  # (d_inner,)

        # Discretize: dA = exp(delta * A), dB = delta * B
        # Shapes:
        #   delta:  (B, L, d_inner)
        #   A:      (d_inner, N)
        #   B_proj: (B, L, N)
        # Broadcast to (B, L, d_inner, N)
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, d_inner, N)
        deltaB = delta.unsqueeze(-1) * B_proj.unsqueeze(2)  # (B, L, d_inner, N)
        deltaB_x = deltaB * x.unsqueeze(-1)  # (B, L, d_inner, N)

        # Sequential scan over time dim
        h = torch.zeros(B_batch, d_inner, N, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = deltaA[:, t] * h + deltaB_x[:, t]  # (B, d_inner, N)
            y_t = torch.einsum("bdn,bn->bd", h, C_proj[:, t])  # (B, d_inner)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # (B, L, d_inner)

        # Skip connection
        y = y + x * D
        return y


class BiMambaBlock(nn.Module):
    """
    Bidirectional Mamba block used in MambaTrack's MTP.
    Reference: MambaTrack (Bi-Mamba Block), arXiv:2408.09178.
    """

    def __init__(self, d_model: int, d_state: int = 16) -> None:
        super().__init__()
        self.fwd = MambaBlock(d_model=d_model, d_state=d_state)
        self.bwd = MambaBlock(d_model=d_model, d_state=d_state)
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LeakyReLU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        fwd_out = self.fwd(x)
        bwd_out = torch.flip(self.bwd(torch.flip(x, dims=[-2])), dims=[-2])
        y = fwd_out + bwd_out
        y = y + self.mlp(self.norm(y))
        return y
