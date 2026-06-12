"""
Mamba-based Motion Predictor (MTP) — replacement for NSA Kalman filter.

This module provides a drop-in replacement API for the Kalman filter used in
FaceQSORT. The API surface mirrors the old `NSAKalmanFilter`:

    predictor.initiate(xyah)              -> track_state
    predictor.predict(track_state)        -> track_state  (next predicted xyah lives inside)
    predictor.update(track_state, xyah)   -> track_state
    predictor.gating_distance(track_state, measurements_xyah) -> np.ndarray

Internally, per-track state is a small Python object holding:
    - history: deque of past xyah observations (up to max_window)
    - predicted_xyah: last predicted xyah (used by the tracker for bbox + gating)
    - predicted_h: predicted height (used to scale gating distance)

    Architecture (MambaTrack's MTP, arXiv:2408.09178):
    4-dim diff-of-xyah --linear--> d_model
                       --BiMamba x L-->
                       --AdaptivePool + MLP--> 4-dim next-diff prediction
    predicted_xyah = previous_prediction_or_last_xyah + predicted_diff

If no pretrained weights are supplied OR the history is too short to feed the
net, the predictor falls back to linear velocity extrapolation (identical to a
constant-velocity Kalman, just without the covariance).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import numpy as np
import torch
import torch.nn as nn

from motion.mamba_block import BiMambaBlock


# ---------------------------------------------------------------------------
# The Mamba Motion Predictor network (MTP)
# ---------------------------------------------------------------------------
class MTPNet(nn.Module):
    """
    Mamba Motion Predictor Network. Takes a sequence of bbox diffs and
    predicts the next diff. Input/output are raw xyah diffs.

    Input:  (B, L, 4)    sequence of (dxc, dyc, da, dh) diffs
    Output: (B, 4)       predicted next diff
    """

    def __init__(
        self,
        d_model: int = 64,
        d_state: int = 16,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(4, d_model)
        self.blocks = nn.ModuleList(
            [BiMambaBlock(d_model=d_model, d_state=d_state) for _ in range(num_layers)]
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, 4)
        h = self.input_proj(x)
        for blk in self.blocks:
            h = blk(h)
        # Average-pool over time, then MLP to 4-dim
        h = h.mean(dim=1)  # (B, d_model)
        return self.head(h)  # (B, 4)


# ---------------------------------------------------------------------------
# Per-track state (replaces Kalman's (mean, cov))
# ---------------------------------------------------------------------------
@dataclass
class MambaTrackState:
    """
    State carried per track. Replaces Kalman's (mean, cov).

    - history: deque of past xyah observations, newest at the end.
    - predicted_xyah: the predictor's best guess for the current frame
      (set during `predict()`, consumed by the tracker for bbox + gating).
    """

    history: Deque[np.ndarray] = field(default_factory=lambda: deque())
    predicted_xyah: Optional[np.ndarray] = None

    def current_xyah(self) -> np.ndarray:
        """Best known xyah to report as track state."""
        if self.predicted_xyah is not None:
            return self.predicted_xyah.copy()
        if len(self.history) > 0:
            return self.history[-1].copy()
        raise RuntimeError("MambaTrackState has no observations or predictions.")


# ---------------------------------------------------------------------------
# The predictor itself (Kalman drop-in)
# ---------------------------------------------------------------------------
class MambaMotionPredictor:
    """
    Drop-in replacement for NSAKalmanFilter.

    Important behavioral differences from Kalman:
      - No covariance. Gating uses a normalized L2 distance in xyah space,
        scaled by track height (so the threshold `gamma` used upstream still
        has a sensible meaning on the chi-square-ish scale).
      - If no weights are loaded or history is too short, falls back to
        constant-velocity linear extrapolation — same behavior as Kalman with
        a simple velocity model.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        d_model: int = 64,
        d_state: int = 16,
        num_layers: int = 3,
        max_window: int = 20,
        min_history_for_net: int = 10,
        img_size: Optional[tuple] = None,
    ) -> None:
        self.device = torch.device(device if device is not None else "cpu")
        self.max_window = int(max_window)
        self.min_history_for_net = int(min_history_for_net)
        self.img_size = img_size  # Kept for API compatibility; MTP uses raw xyah diffs.
        self.standardize_diffs = False
        self.diff_mean = np.zeros(4, dtype=np.float32)
        self.diff_std = np.ones(4, dtype=np.float32)
        self.diff_clip_z: Optional[float] = None

        self.net = MTPNet(d_model=d_model, d_state=d_state, num_layers=num_layers)
        self.net.to(self.device)
        self.net.eval()

        self._weights_loaded = False
        if weights_path is not None:
            self.load_weights(weights_path)

    # ---- weight loading ----
    def load_weights(self, weights_path: str) -> bool:
        try:
            ckpt = torch.load(weights_path, map_location=self.device)
            state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
            if isinstance(state_dict, dict) and state_dict and all(
                str(k).startswith("_orig_mod.") for k in state_dict
            ):
                state_dict = {str(k)[10:]: v for k, v in state_dict.items()}
            self.net.load_state_dict(state_dict, strict=True)
            if isinstance(ckpt, dict):
                cfg = ckpt.get("config", {})
                if isinstance(cfg, dict) and "window" in cfg:
                    self.min_history_for_net = max(2, int(cfg["window"]))
                    self.max_window = max(self.max_window, self.min_history_for_net)
                if isinstance(cfg, dict) and cfg.get("standardize_diffs", False):
                    self.standardize_diffs = True
                    self.diff_mean = np.asarray(cfg.get("diff_mean", [0, 0, 0, 0]), dtype=np.float32)
                    self.diff_std = np.asarray(cfg.get("diff_std", [1, 1, 1, 1]), dtype=np.float32)
                    self.diff_std = np.maximum(
                        self.diff_std,
                        np.array([1e-3, 1e-3, 1e-6, 1e-3], dtype=np.float32),
                    )
                    clip_z = cfg.get("diff_clip_z", None)
                    self.diff_clip_z = float(clip_z) if clip_z is not None and float(clip_z) > 0 else None
            self._weights_loaded = True
            print(
                f"[MambaMotionPredictor] Loaded weights from '{weights_path}' "
                f"(min_history_for_net={self.min_history_for_net}, "
                f"standardize_diffs={self.standardize_diffs}, "
                f"diff_clip_z={self.diff_clip_z}).",
                flush=True,
            )
            return True
        except Exception as exc:
            print(
                f"[MambaMotionPredictor] Failed to load weights from '{weights_path}': {exc}. "
                f"Falling back to linear extrapolation.",
                flush=True,
            )
            self._weights_loaded = False
            return False

    # ---- Kalman-compatible API ----
    def initiate(self, measurement_xyah: np.ndarray) -> MambaTrackState:
        """Initialize track state with first observation."""
        state = MambaTrackState()
        state.history.append(np.asarray(measurement_xyah, dtype=np.float32).copy())
        state.predicted_xyah = state.history[-1].copy()
        return state

    def predict(self, state: MambaTrackState) -> MambaTrackState:
        """Predict next xyah in-place on `state.predicted_xyah`."""
        n = len(state.history)
        if n == 0:
            # Nothing to do; leave predicted_xyah as-is.
            return state

        last = state.history[-1]
        base = state.predicted_xyah if state.predicted_xyah is not None else last
        if n == 1:
            state.predicted_xyah = last.copy()
            return state

        # Linear fallback: delta from last two observations
        linear_diff = (state.history[-1] - state.history[-2]).astype(np.float32)

        use_net = self._weights_loaded and n >= self.min_history_for_net
        if not use_net:
            state.predicted_xyah = (base + linear_diff).astype(np.float32)
            return state

        # Build fixed-length input matching standalone training:
        # window=10 observations -> 9 raw xyah-diff tuples.
        hist = np.stack(list(state.history)[-self.min_history_for_net:], axis=0).astype(np.float32)
        diffs = np.diff(hist, axis=0)  # (min_history_for_net - 1, 4)
        net_diffs = diffs
        if self.standardize_diffs:
            net_diffs = (diffs - self.diff_mean) / self.diff_std
            if self.diff_clip_z is not None:
                net_diffs = np.clip(net_diffs, -self.diff_clip_z, self.diff_clip_z)

        x = torch.from_numpy(net_diffs).unsqueeze(0).to(self.device)  # (1, L, 4)
        with torch.no_grad():
            pred_diff = self.net(x).squeeze(0).cpu().numpy()  # (4,)
        if self.standardize_diffs:
            pred_diff = pred_diff * self.diff_std + self.diff_mean
        if not np.all(np.isfinite(pred_diff)):
            pred_diff = linear_diff

        state.predicted_xyah = (base + pred_diff).astype(np.float32)
        return state

    def update(self, state: MambaTrackState, measurement_xyah: np.ndarray) -> MambaTrackState:
        """Append a new observation to history."""
        state.history.append(np.asarray(measurement_xyah, dtype=np.float32).copy())
        while len(state.history) > self.max_window:
            state.history.popleft()
        state.predicted_xyah = state.history[-1].copy()
        return state

    def gating_distance(
        self, state: MambaTrackState, measurements_xyah: np.ndarray
    ) -> np.ndarray:
        """
        Return a per-measurement distance compatible with the cost fusion in
        FaceQSORT. We use a normalized squared L2 in xyah space, scaled so that
        the upstream chi-square threshold `gamma` (default 9.4877) behaves
        similarly to Kalman's Mahalanobis gate.

        The scale comes from a fraction of the track's height `h`:
          pos stdev: 0.1 * h,   aspect stdev: 0.05,   h stdev: 0.1 * h
        These match FaceQSORT's NSA Kalman noise weights (1/20, 1/160 → similar magnitude).
        """
        if state.predicted_xyah is None:
            # Fall back to last observed
            ref = state.history[-1] if state.history else np.zeros(4, dtype=np.float32)
        else:
            ref = state.predicted_xyah

        meas = np.asarray(measurements_xyah, dtype=np.float32)  # (M, 4)
        diff = meas - ref[None, :]  # (M, 4)

        h = max(1.0, float(ref[3]))
        std = np.array([0.1 * h, 0.1 * h, 0.05, 0.1 * h], dtype=np.float32)
        z = diff / std[None, :]
        return np.sum(z * z, axis=1).astype(np.float32)
