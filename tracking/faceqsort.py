"""
FaceQSORT (paper-faithful tracker core) — MAMBA-BASED MOTION VARIANT
- Tracking-by-detection
- Two features from same face patch: biometric + appearance
- Cost-level fusion + spatial gating + Hungarian assignment
- Matching cascade + IoU fallback
- EMA feature memory update
- Tentative -> Confirmed tracks, extrapolate missed tracks until they leave the frame

Motion model: Mamba-based Motion Predictor (MTP) instead of NSA Kalman filter.
All other tracker logic is identical to the paper-faithful FaceQSORT.

Ref: FaceQSORT paper (arXiv 2501.11741). See Eq (1)-(5) and feature update text.
     MambaTrack (arXiv 2408.09178) for the MTP motion model.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np

from scipy.optimize import linear_sum_assignment
from faceqsort_mamba.utils.device_utils import get_best_device, resolve_device
from faceqsort_mamba.motion.mamba_motion_predictor import MambaMotionPredictor, MambaTrackState


# -----------------------------
# Utilities
# -----------------------------

def _l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x) + eps
    return x / n


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    # 1 - cosine similarity
    a = _l2norm(a)
    b = _l2norm(b)
    return float(1.0 - np.dot(a, b))


def batch_cosine_distance(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # A: (N,D), B: (M,D) -> (N,M) distances
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    sim = A @ B.T
    return 1.0 - sim


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Intersection-over-Union. Kept for reference / metrics export."""
    # a,b: [x1,y1,x2,y2]
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter + 1e-12
    return float(inter / union)


def iop_xyxy(predicted: np.ndarray, detection: np.ndarray, max_size_ratio: float = 3.0) -> float:
    """
    Intersection-over-Predicted-area.

        IoP(A, B) = |A ∩ B| / |A|       where A = predicted (track) bbox

    Asymmetric: not equal to IoP(B, A). Cheaper than IoU (no union term) and
    favours matches where the predicted bbox is well-covered by a detection,
    even when the detection is somewhat larger than predicted.

    To guard against a huge detection engulfing a tiny predicted bbox and
    spuriously claiming IoP=1.0, returns 0.0 when area(detection) is more
    than `max_size_ratio` times area(predicted).
    """
    x1 = max(predicted[0], detection[0])
    y1 = max(predicted[1], detection[1])
    x2 = min(predicted[2], detection[2])
    y2 = min(predicted[3], detection[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    area_p = max(1e-12, (predicted[2] - predicted[0]) * (predicted[3] - predicted[1]))

    # Size-sanity: reject pairs where detection is far larger than predicted.
    if max_size_ratio > 0:
        area_d = max(1e-12, (detection[2] - detection[0]) * (detection[3] - detection[1]))
        if area_d > max_size_ratio * area_p:
            return 0.0

    return float(inter / area_p)


def to_xyah(xyxy: np.ndarray) -> np.ndarray:
    # [x1,y1,x2,y2] -> [cx, cy, a, h], a=w/h
    x1, y1, x2, y2 = xyxy.astype(np.float32)
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    a = w / h
    return np.array([cx, cy, a, h], dtype=np.float32)


def to_xyxy(xyah: np.ndarray) -> np.ndarray:
    cx, cy, a, h = xyah.astype(np.float32)
    w = a * h
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def xyah_overlaps_frame(xyah: np.ndarray, img_size: Optional[tuple], margin: float = 0.0) -> bool:
    """
    True while the predicted bbox still intersects the image plane.

    img_size follows OpenCV/MOT convention: (height, width). A track is deleted
    only once the whole predicted box has moved outside the frame, so faces
    partially crossing an edge can still reconnect.
    """
    if img_size is None:
        return True

    xyah = np.asarray(xyah, dtype=np.float32)
    if xyah.shape[0] != 4 or not np.all(np.isfinite(xyah)):
        return False
    if float(xyah[2]) <= 0.0 or float(xyah[3]) <= 1.0:
        return False

    h, w = float(img_size[0]), float(img_size[1])
    if h <= 1.0 or w <= 1.0:
        return True

    x1, y1, x2, y2 = to_xyxy(xyah).astype(np.float32).tolist()
    m = max(0.0, float(margin))
    return not (x2 <= -m or y2 <= -m or x1 >= w + m or y1 >= h + m)


# -----------------------------
# Motion model: Mamba-based Motion Predictor (MTP).
# Replaces the previous NSA Kalman filter. Kalman's (mean, cov) state is
# replaced by a per-track MambaTrackState (history deque + predicted xyah).
# See mamba_motion_predictor.py for the drop-in API.
# -----------------------------


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Detection:
    bbox_xyxy: np.ndarray  # shape (4,)
    conf: float
    feat_bio: np.ndarray  # shape (Db,)
    feat_app: np.ndarray  # shape (Da,)
    kps: Optional[np.ndarray] = None  # shape (5,2) RetinaFace landmarks, [0]/[1] = eyes


class TrackState:
    Tentative = 0
    Confirmed = 1
    Deleted = 2


@dataclass
class Track:
    track_id: int
    global_id: int
    motion_state: MambaTrackState  # replaces Kalman's (mean, cov)
    feat_bio: np.ndarray
    feat_app: np.ndarray
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    state: int = TrackState.Tentative
    last_kps: Optional[np.ndarray] = None  # landmarks of last matched detection

    def current_xyah(self) -> np.ndarray:
        return self.motion_state.current_xyah()

    def to_xyxy(self) -> np.ndarray:
        return to_xyxy(self.current_xyah())


@dataclass
class GlobalIdentity:
    global_id: int
    feat_bio: np.ndarray
    feat_app: np.ndarray
    last_xyah: np.ndarray
    last_frame: int
    active_local_id: Optional[int] = None
    hits: int = 1
    tentative: bool = True


# -----------------------------
# FaceQSORT Tracker
# -----------------------------
class FaceQSORTTracker:
    """
    Cost fusion follows FaceQSORT paper (Eq 1-4):
      d_feat = w * (lambda * d_bio + (1-lambda) * d_app)
      gate by Mahalanobis distance <= gamma
      final cost = alpha * d_feat + (1-alpha) * d_spatial
      apply general threshold theta to accept matches
    """

    def __init__(
            self,
            lambda_bio: float = 0.9,  # lambda in paper: weight for biometric vs appearance (0.9 = 90% biometric)
            w_feat: float = 1.0,  # w in paper (scales feature cost)
            alpha: float = 0.5,  # mix feature vs spatial cost
            gamma: float = 9.4877,  # gating threshold in chi-square space (approx df=4, 0.95)
            theta: float = 0.2,  # general association threshold
            iou_thresh: float = 0.3,  # IoU fallback threshold
            max_age: int = 300,
            n_init: int = 3,  # confirm after n_init hits
            ema_momentum: float = 0.9,  # momentum beta in EMA update
            cascade_depth: int = 20,  # matching cascade depth
            device: Optional[str] = None,
            # --- Mamba motion predictor args ---
            mamba_weights: Optional[str] = None,
            mamba_max_window: int = 20,
            mamba_d_model: int = 64,
            mamba_d_state: int = 16,
            mamba_num_layers: int = 3,
            img_size: Optional[tuple] = None,
            delete_on_exit: bool = True,
            exit_margin: float = 0.0,
            motion_gate_growth: float = 0.25,
            motion_gate_max: float = 4.0,
            use_global_id: bool = True,
            global_reid_max_age: int = 300,
            global_reid_appearance_thresh: float = 0.50,
            global_reid_motion_base: float = 3.0,
            global_reid_motion_per_frame: float = 0.5,
            global_reid_motion_cap: float = 20.0,
            global_reid_height_ratio: float = 3.0,
            global_reid_reconsider_hits: int = 8,
            global_id_consistency_gate: bool = True,
            global_id_consistency_thresh: float = 0.75,
            global_id_debug: bool = False,
    ):
        self.device = resolve_device(device)
        self.lambda_bio = float(lambda_bio)
        self.w_feat = float(w_feat)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.theta = float(theta)
        self.iou_thresh = float(iou_thresh)
        self.max_age = int(max_age)
        self.n_init = int(n_init)
        self.ema = float(ema_momentum)
        self.cascade_depth = int(cascade_depth)
        self.img_size = img_size
        self.delete_on_exit = bool(delete_on_exit)
        self.exit_margin = float(exit_margin)
        self.motion_gate_growth = max(0.0, float(motion_gate_growth))
        self.motion_gate_max = max(1.0, float(motion_gate_max))
        self.use_global_id = bool(use_global_id)
        self.global_reid_max_age = max(1, int(global_reid_max_age))
        self.global_reid_appearance_thresh = float(global_reid_appearance_thresh)
        self.global_reid_motion_base = max(0.0, float(global_reid_motion_base))
        self.global_reid_motion_per_frame = max(0.0, float(global_reid_motion_per_frame))
        self.global_reid_motion_cap = max(1.0, float(global_reid_motion_cap))
        self.global_reid_height_ratio = max(1.0, float(global_reid_height_ratio))
        self.global_reid_reconsider_hits = max(1, int(global_reid_reconsider_hits))
        self.global_id_consistency_gate = bool(global_id_consistency_gate)
        self.global_id_consistency_thresh = float(global_id_consistency_thresh)
        self.global_id_debug = bool(global_id_debug)

        # Mamba motion predictor (drop-in for NSAKalmanFilter)
        self.kf = MambaMotionPredictor(
            weights_path=mamba_weights,
            device=str(self.device),
            d_model=mamba_d_model,
            d_state=mamba_d_state,
            num_layers=mamba_num_layers,
            max_window=mamba_max_window,
            img_size=img_size,
        )
        self.tracks: List[Track] = []
        self._next_id = 1
        self._next_global_id = 1
        self._frame_idx = 0
        self.global_identities: Dict[int, GlobalIdentity] = {}

    def predict(self):
        for t in self.tracks:
            if t.state == TrackState.Deleted:
                continue
            self.kf.predict(t.motion_state)
            t.age += 1
            t.time_since_update += 1
            if self.delete_on_exit and not xyah_overlaps_frame(
                t.current_xyah(), self.img_size, self.exit_margin
            ):
                t.state = TrackState.Deleted

    def _start_track(self, det: Detection):
        local_id = self._next_id
        self._next_id += 1
        motion_state = self.kf.initiate(to_xyah(det.bbox_xyxy))
        tr = Track(
            track_id=local_id,
            global_id=local_id,
            motion_state=motion_state,
            feat_bio=_l2norm(det.feat_bio.astype(np.float32)),
            feat_app=_l2norm(det.feat_app.astype(np.float32)),
            last_kps=det.kps.copy() if det.kps is not None else None,
        )
        if self.use_global_id:
            self._assign_global_identity_to_new_track(tr, det)
        self.tracks.append(tr)

    def _ema_update(self, old: np.ndarray, new: np.ndarray) -> np.ndarray:
        # f <- beta f + (1-beta) f_new (then normalize)
        out = self.ema * old + (1.0 - self.ema) * new
        return _l2norm(out.astype(np.float32))

    def _track_by_local_id(self, local_id: Optional[int]) -> Optional[Track]:
        if local_id is None:
            return None
        for t in self.tracks:
            if t.track_id == local_id:
                return t
        return None

    def _is_global_active_now(self, global_id: int, keep_local_id: Optional[int] = None) -> bool:
        for t in self.tracks:
            if t.global_id != global_id or t.track_id == keep_local_id:
                continue
            if t.state != TrackState.Deleted and t.time_since_update == 0:
                return True
        return False

    def _motion_size_gate(self, old_xyah: np.ndarray, new_xyah: np.ndarray, dt: int) -> Tuple[bool, float]:
        old = np.asarray(old_xyah, dtype=np.float32)
        new = np.asarray(new_xyah, dtype=np.float32)
        if old.shape[0] != 4 or new.shape[0] != 4:
            return False, np.inf
        if not np.all(np.isfinite(old)) or not np.all(np.isfinite(new)):
            return False, np.inf
        if float(old[3]) <= 1.0 or float(new[3]) <= 1.0:
            return False, np.inf

        h_ratio = max(float(old[3] / new[3]), float(new[3] / old[3]))
        if h_ratio > self.global_reid_height_ratio:
            return False, np.inf
        if abs(float(old[2] - new[2])) > 0.75:
            return False, np.inf

        h = max(1.0, float(0.5 * (old[3] + new[3])))
        center_norm = float(np.linalg.norm(new[:2] - old[:2]) / h)
        gate = min(
            self.global_reid_motion_cap,
            self.global_reid_motion_base + self.global_reid_motion_per_frame * max(0, dt),
        )
        return center_norm <= gate, center_norm

    def _identity_distance(
        self,
        identity: GlobalIdentity,
        feat_bio: np.ndarray,
        feat_app: np.ndarray,
        xyah: np.ndarray,
    ) -> Optional[Tuple[float, float, float, int]]:
        dt = self._frame_idx - identity.last_frame
        if dt <= 0 or dt > self.global_reid_max_age:
            return None
        if identity.tentative:
            return None
        if self._is_global_active_now(identity.global_id):
            return None

        d_bio = cosine_distance(identity.feat_bio, feat_bio)
        d_app = cosine_distance(identity.feat_app, feat_app)
        d_app_combined = self.lambda_bio * d_bio + (1.0 - self.lambda_bio) * d_app
        if d_app_combined > self.global_reid_appearance_thresh:
            return None

        motion_ok, motion_norm = self._motion_size_gate(identity.last_xyah, xyah, dt)
        if not motion_ok:
            return None

        score = d_app_combined + 0.01 * motion_norm + 0.001 * np.log1p(float(dt))
        return float(score), float(d_app_combined), float(motion_norm), int(dt)

    def _global_feature_distance(
        self,
        identity: GlobalIdentity,
        feat_bio: np.ndarray,
        feat_app: np.ndarray,
    ) -> float:
        d_bio = cosine_distance(identity.feat_bio, feat_bio)
        d_app = cosine_distance(identity.feat_app, feat_app)
        return float(self.lambda_bio * d_bio + (1.0 - self.lambda_bio) * d_app)

    def _find_lost_global_candidate(
        self,
        feat_bio: np.ndarray,
        feat_app: np.ndarray,
        xyah: np.ndarray,
        exclude_global_id: Optional[int] = None,
    ) -> Optional[int]:
        best_gid: Optional[int] = None
        best_score = np.inf
        for gid, identity in self.global_identities.items():
            if exclude_global_id is not None and gid == exclude_global_id:
                continue
            dist = self._identity_distance(identity, feat_bio, feat_app, xyah)
            if dist is None:
                continue
            score = dist[0]
            if score < best_score:
                best_score = score
                best_gid = gid
        return best_gid

    def _new_global_identity(self, tr: Track, xyah: np.ndarray, tentative: bool) -> int:
        gid = self._next_global_id
        self._next_global_id += 1
        self.global_identities[gid] = GlobalIdentity(
            global_id=gid,
            feat_bio=tr.feat_bio.copy(),
            feat_app=tr.feat_app.copy(),
            last_xyah=np.asarray(xyah, dtype=np.float32).copy(),
            last_frame=self._frame_idx,
            active_local_id=tr.track_id,
            hits=tr.hits,
            tentative=tentative,
        )
        return gid

    def _retire_other_tracks_for_global(self, global_id: int, keep_local_id: int) -> None:
        for other in self.tracks:
            if other.track_id == keep_local_id:
                continue
            if other.global_id == global_id and other.state != TrackState.Deleted:
                other.state = TrackState.Deleted

    def _assign_global_identity_to_new_track(self, tr: Track, det: Detection) -> None:
        xyah = to_xyah(det.bbox_xyxy)
        gid = self._find_lost_global_candidate(tr.feat_bio, tr.feat_app, xyah)
        if gid is None:
            tr.global_id = self._new_global_identity(
                tr,
                xyah,
                tentative=tr.state != TrackState.Confirmed,
            )
            if self.global_id_debug:
                print(
                    f"[GlobalID] frame={self._frame_idx} new_global={tr.global_id} local={tr.track_id}",
                    flush=True,
                )
            return

        tr.global_id = gid
        self._retire_other_tracks_for_global(gid, tr.track_id)
        self._update_global_identity_from_track(tr, xyah)
        if self.global_id_debug:
            print(
                f"[GlobalID] frame={self._frame_idx} reuse_global={gid} local={tr.track_id}",
                flush=True,
            )

    def _update_global_identity_from_track(self, tr: Track, xyah: np.ndarray) -> None:
        if not self.use_global_id:
            return
        identity = self.global_identities.get(tr.global_id)
        if identity is None:
            tr.global_id = self._new_global_identity(
                tr,
                xyah,
                tentative=tr.state != TrackState.Confirmed,
            )
            return

        identity.feat_bio = self._ema_update(identity.feat_bio, tr.feat_bio)
        identity.feat_app = self._ema_update(identity.feat_app, tr.feat_app)
        identity.last_xyah = np.asarray(xyah, dtype=np.float32).copy()
        identity.last_frame = self._frame_idx
        identity.active_local_id = tr.track_id
        identity.hits += 1
        if tr.state == TrackState.Confirmed:
            identity.tentative = False

    def _reconsider_global_identity(self, tr: Track, xyah: np.ndarray) -> None:
        if not self.use_global_id:
            return
        identity = self.global_identities.get(tr.global_id)
        if identity is None or identity.hits > self.global_reid_reconsider_hits:
            return

        gid = self._find_lost_global_candidate(
            tr.feat_bio,
            tr.feat_app,
            xyah,
            exclude_global_id=tr.global_id,
        )
        if gid is None:
            return

        old_gid = tr.global_id
        tr.global_id = gid
        self._retire_other_tracks_for_global(gid, tr.track_id)
        self.global_identities.pop(old_gid, None)
        if self.global_id_debug:
            print(
                f"[GlobalID] frame={self._frame_idx} merge_temp={old_gid} -> global={gid} local={tr.track_id}",
                flush=True,
            )

    def _passes_global_consistency(self, tr: Track, det: Detection) -> bool:
        if not self.use_global_id or not self.global_id_consistency_gate:
            return True
        identity = self.global_identities.get(tr.global_id)
        if identity is None or identity.tentative:
            return True

        feat_bio = _l2norm(det.feat_bio.astype(np.float32))
        feat_app = _l2norm(det.feat_app.astype(np.float32))
        d_global = self._global_feature_distance(identity, feat_bio, feat_app)
        if d_global <= self.global_id_consistency_thresh:
            return True

        if self.global_id_debug:
            print(
                f"[GlobalID] frame={self._frame_idx} reject_local_match "
                f"global={tr.global_id} local={tr.track_id} dist={d_global:.3f} "
                f"thresh={self.global_id_consistency_thresh:.3f}",
                flush=True,
            )
        return False

    def _confirmed_tracks(self) -> List[int]:
        return [i for i, t in enumerate(self.tracks) if t.state == TrackState.Confirmed]

    def _active_tracks(self) -> List[int]:
        return [i for i, t in enumerate(self.tracks) if t.state != TrackState.Deleted]

    def _build_cost_matrix(self, track_indices: List[int], detections: List[Detection]) -> np.ndarray:
        if len(track_indices) == 0 or len(detections) == 0:
            return np.zeros((len(track_indices), len(detections)), dtype=np.float32)

        # stack features
        Tb = np.stack([self.tracks[i].feat_bio for i in track_indices], axis=0)
        Ta = np.stack([self.tracks[i].feat_app for i in track_indices], axis=0)
        Db = np.stack([_l2norm(d.feat_bio.astype(np.float32)) for d in detections], axis=0)
        Da = np.stack([_l2norm(d.feat_app.astype(np.float32)) for d in detections], axis=0)

        d_bio = batch_cosine_distance(Tb, Db)  # (T, D)
        d_app = batch_cosine_distance(Ta, Da)

        d_feat = self.w_feat * (self.lambda_bio * d_bio + (1.0 - self.lambda_bio) * d_app)

        # spatial cost: Mahalanobis distances in measurement space
        meas = np.stack([to_xyah(d.bbox_xyxy) for d in detections], axis=0)  # (D,4)
        d_spatial = np.zeros((len(track_indices), len(detections)), dtype=np.float32)
        gated = np.zeros_like(d_spatial, dtype=bool)

        for r, ti in enumerate(track_indices):
            t = self.tracks[ti]
            gd = self.kf.gating_distance(t.motion_state, meas)  # (D,)
            gate_scale = min(
                self.motion_gate_max,
                1.0 + self.motion_gate_growth * max(0, t.time_since_update),
            )
            gate = self.gamma * gate_scale
            # Normalize gated distance so it can be fused with cosine feature distance.
            d_spatial[r, :] = np.minimum(gd / (gate + 1e-12), 1.0).astype(np.float32)
            gated[r, :] = gd <= gate

        # set cost to inf where gated out
        cost = self.alpha * d_feat + (1.0 - self.alpha) * d_spatial
        cost[~gated] = np.inf
        return cost

    def _hungarian(self, cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if cost.size == 0:
            return np.array([], dtype=int), np.array([], dtype=int)
        finite = np.isfinite(cost)
        if not finite.any():
            return np.array([], dtype=int), np.array([], dtype=int)

        # scipy raises "cost matrix is infeasible" when a row/column is all inf.
        # Replace inf with a large finite penalty for assignment, then filter later.
        safe_cost = cost.copy()
        max_finite = float(np.max(safe_cost[finite]))
        penalty = max_finite + 1e6
        safe_cost[~finite] = penalty

        r, c = linear_sum_assignment(safe_cost)
        return r.astype(int), c.astype(int)

    def _match(self, track_indices: List[int], detections: List[Detection]) -> Tuple[
        List[Tuple[int, int]], List[int], List[int]]:
        """
        Returns:
          matches: list of (track_idx_in_tracks, det_idx)
          unmatched_tracks: list of indices in track_indices (positions)
          unmatched_dets: list of det indices
        """
        if len(track_indices) == 0:
            return [], [], list(range(len(detections)))
        if len(detections) == 0:
            return [], list(range(len(track_indices))), []

        cost = self._build_cost_matrix(track_indices, detections)
        row_ind, col_ind = self._hungarian(cost)

        matches = []
        unmatched_t = set(range(len(track_indices)))
        unmatched_d = set(range(len(detections)))

        for r, c in zip(row_ind, col_ind):
            if not np.isfinite(cost[r, c]):
                continue
            if cost[r, c] > self.theta:
                continue
            matches.append((track_indices[r], c))
            unmatched_t.discard(r)
            unmatched_d.discard(c)

        return matches, sorted(list(unmatched_t)), sorted(list(unmatched_d))

    def _iou_fallback(self, track_indices: List[int], det_indices: List[int], detections: List[Detection]) -> Tuple[
        List[Tuple[int, int]], List[int], List[int]]:
        """
        Geometric-only fallback association for tracks unmatched by the
        cosine+Mahalanobis pass. Uses Intersection-over-Predicted-area (IoP)
        rather than symmetric IoU:

            IoP(predicted_bbox_of_track, detection_bbox)
              = |intersection| / |predicted_bbox_of_track|

        IoP skips the union term (cheaper) and is asymmetric in our favour:
        a detection that fully covers the Mamba-predicted bbox wins, even
        when the detection is somewhat larger than predicted. A size-sanity
        guard inside iop_xyxy rejects detections whose area is >3x the
        predicted area, to avoid huge spurious boxes engulfing small tracks.

        Threshold reused: self.iou_thresh (semantically becomes IoP threshold).
        """
        if len(track_indices) == 0 or len(det_indices) == 0:
            return [], track_indices, det_indices

        # greedy IoP matching (replaces symmetric IoU)
        pairs = []
        for ti in track_indices:
            tb = self.tracks[ti].to_xyxy()
            for dj in det_indices:
                db = detections[dj].bbox_xyxy
                pairs.append((ti, dj, iop_xyxy(tb, db)))
        pairs.sort(key=lambda x: x[2], reverse=True)

        matched_t = set()
        matched_d = set()
        matches = []
        for ti, dj, score in pairs:
            if score < self.iou_thresh:
                break
            if ti in matched_t or dj in matched_d:
                continue
            matches.append((ti, dj))
            matched_t.add(ti)
            matched_d.add(dj)

        rem_t = [ti for ti in track_indices if ti not in matched_t]
        rem_d = [dj for dj in det_indices if dj not in matched_d]
        return matches, rem_t, rem_d

    def update(self, detections: List[Detection]) -> List[Dict]:
        """
        Run one frame update. Returns list of active track outputs:
          [{track_id, bbox_xyxy, state, time_since_update}, ...]
        """
        self._frame_idx += 1

        # 1) Predict
        self.predict()

        # 2) Matching cascade (recently updated first)
        # We'll cascade by time_since_update = 0..K
        active = self._active_tracks()
        # prioritize confirmed tracks first (common TbD practice)
        confirmed = [i for i in active if self.tracks[i].state == TrackState.Confirmed]
        tentative = [i for i in active if self.tracks[i].state == TrackState.Tentative]

        remaining_det = list(range(len(detections)))
        matches_all: List[Tuple[int, int]] = []

        # Cascade on confirmed tracks
        for tsu in range(self.cascade_depth + 1):
            if tsu < self.cascade_depth:
                layer = [i for i in confirmed if self.tracks[i].time_since_update == tsu]
            else:
                layer = [i for i in confirmed if self.tracks[i].time_since_update >= tsu]
            if not layer or not remaining_det:
                continue
            layer_dets = [detections[j] for j in remaining_det]
            matches, um_t_pos, um_d = self._match(layer, layer_dets)

            # matches use det indices in layer_dets; map back to remaining_det
            used = set()
            for ti, local_dj in matches:
                dj = remaining_det[local_dj]
                matches_all.append((ti, dj))
                used.add(dj)

            remaining_det = [j for j in remaining_det if j not in used]

        # Match tentative tracks (if any) with remaining dets
        if tentative and remaining_det:
            tent_dets = [detections[j] for j in remaining_det]
            matches, um_t_pos, um_d = self._match(tentative, tent_dets)
            used = set()
            for ti, local_dj in matches:
                dj = remaining_det[local_dj]
                matches_all.append((ti, dj))
                used.add(dj)
            remaining_det = [j for j in remaining_det if j not in used]

        # 3) IoU fallback on remaining confirmed tracks that are unmatched this frame
        matched_track_ids = set(ti for ti, _ in matches_all)
        unmatched_confirmed = [i for i in confirmed if i not in matched_track_ids]
        if unmatched_confirmed and remaining_det:
            m_iou, rem_t, rem_d = self._iou_fallback(unmatched_confirmed, remaining_det, detections)
            matches_all.extend(m_iou)
            remaining_det = rem_d

        # 4) Reject local matches that strongly disagree with the assigned
        # global identity. This prevents an existing local track from absorbing
        # a wrong face before the global-ID layer gets a chance to recover it
        # as a new/reconnected tracklet.
        if self.use_global_id and self.global_id_consistency_gate and matches_all:
            kept_matches: List[Tuple[int, int]] = []
            rejected_dets = []
            for ti, dj in matches_all:
                if self._passes_global_consistency(self.tracks[ti], detections[dj]):
                    kept_matches.append((ti, dj))
                else:
                    rejected_dets.append(dj)
            matches_all = kept_matches
            if rejected_dets:
                existing = set(remaining_det)
                remaining_det.extend([dj for dj in rejected_dets if dj not in existing])

        # 5) Apply matches: update motion state + feature EMA
        for ti, dj in matches_all:
            t = self.tracks[ti]
            det = detections[dj]
            det_xyah = to_xyah(det.bbox_xyxy)
            self.kf.update(t.motion_state, det_xyah)
            t.time_since_update = 0
            t.hits += 1
            t.last_kps = det.kps.copy() if det.kps is not None else None

            # EMA update of both features (paper memory update idea)
            t.feat_bio = self._ema_update(t.feat_bio, det.feat_bio)
            t.feat_app = self._ema_update(t.feat_app, det.feat_app)

            # confirm track after n_init hits
            if t.state == TrackState.Tentative and t.hits >= self.n_init:
                t.state = TrackState.Confirmed

            if self.use_global_id:
                self._reconsider_global_identity(t, det_xyah)
                self._update_global_identity_from_track(t, det_xyah)

        # 6) Mark missed tracks and delete old ones
        for i in self._active_tracks():
            t = self.tracks[i]
            if self.delete_on_exit and self.img_size is not None:
                if not xyah_overlaps_frame(t.current_xyah(), self.img_size, self.exit_margin):
                    t.state = TrackState.Deleted
            elif t.time_since_update > self.max_age:
                t.state = TrackState.Deleted

        # 7) Start new tracks for unmatched detections
        for dj in remaining_det:
            self._start_track(detections[dj])

        # 8) Output active tracks (confirmed recommended)
        outputs = []
        for t in self.tracks:
            if t.state == TrackState.Deleted:
                continue
            outputs.append({
                "track_id": t.global_id if self.use_global_id else t.track_id,
                "local_track_id": t.track_id,
                "global_id": t.global_id if self.use_global_id else t.track_id,
                "bbox_xyxy": t.to_xyxy().copy(),
                "kps": t.last_kps.copy() if t.last_kps is not None else None,
                "state": "confirmed" if t.state == TrackState.Confirmed else "tentative",
                "time_since_update": t.time_since_update,
            })
        return outputs
