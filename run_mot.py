"""
Run FaceQSORT-Mamba over MOTChallenge-format sequences (MOT17, MOT20,
ChokePoint-converted, etc.) and write per-sequence results in MOT format,
ready for TrackEval.

Mirrors FaceQSORT/run_mot.py but uses the Mamba motion predictor instead of
the NSA Kalman filter. By default it loads `./checkpoint_best.pth` (sitting
next to this script) into the predictor; pass `--mamba-weights ""` or
`--mamba-weights none` to force the linear-extrapolation fallback used in the
old Kalman-ablation runs.

MOT sequence layout (input):
    <mot_dir>/<seq>/
        seqinfo.ini
        img1/000001.jpg ... 000NNN.jpg
        gt/gt.txt           # (train only) ground truth

Output layout (TrackEval-ready):
    <output_dir>/<tracker_name>/data/<seq>.txt
"""
from __future__ import annotations

import argparse
import configparser
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

from faceqsort_mamba.data.chokepoint_to_mot import eye_bbox
from faceqsort_mamba.utils.device_utils import get_best_device, resolve_device
from faceqsort_mamba.run_faceqsort import build_tracker_detections


def read_seqinfo(seq_dir: Path) -> dict:
    ini = seq_dir / "seqinfo.ini"
    if not ini.exists():
        raise FileNotFoundError(f"Missing seqinfo.ini in {seq_dir}")
    parser = configparser.ConfigParser()
    parser.read(ini)
    section = parser["Sequence"]
    return {
        "name": section.get("name", seq_dir.name),
        "imDir": section.get("imDir", "img1"),
        "frameRate": float(section.get("frameRate", "30")),
        "seqLength": int(section.get("seqLength", "0")),
        "imWidth": int(section.get("imWidth", "0")),
        "imHeight": int(section.get("imHeight", "0")),
        "imExt": section.get("imExt", ".jpg"),
    }


def list_sequences(mot_dir: Path) -> List[Path]:
    return sorted([p for p in mot_dir.iterdir() if p.is_dir() and (p / "seqinfo.ini").exists()])


def load_gt_boxes(seq_dir: Path) -> Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]]:
    """Load MOT-format GT boxes as frame -> [(gt_id, xyxy), ...]."""
    gt_file = seq_dir / "gt" / "gt.txt"
    if not gt_file.exists():
        raise FileNotFoundError(f"--gt-aware-export needs ground truth file: {gt_file}")

    by_frame: Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]] = {}
    with open(gt_file, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                raise ValueError(f"Invalid GT row in {gt_file}:{line_no}: {line}")
            frame = int(float(parts[0]))
            gt_id = int(float(parts[1]))
            x, y, w, h = [float(v) for v in parts[2:6]]
            if w <= 0.0 or h <= 0.0:
                continue
            if len(parts) >= 7 and float(parts[6]) <= 0.0:
                continue
            by_frame.setdefault(frame, []).append((gt_id, (x, y, x + w, y + h)))
    return by_frame


def iou_xyxy_tuple(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def select_gt_aware_tracks(
    tracks: List[dict],
    gt_boxes: List[Tuple[int, Tuple[float, float, float, float]]],
    min_iou: float,
) -> List[dict]:
    """
    Export-only oracle filter: keep at most one tracker output per GT box, chosen
    by highest IoU. Tracker state and IDs are left untouched.
    """
    if not tracks or not gt_boxes:
        return []

    pairs: List[Tuple[float, int, int]] = []
    for tr_idx, tr in enumerate(tracks):
        x1, y1, x2, y2 = [float(v) for v in tr["bbox_xyxy"].tolist()]
        tr_box = (x1, y1, x2, y2)
        for gt_idx, (_, gt_box) in enumerate(gt_boxes):
            iou = iou_xyxy_tuple(tr_box, gt_box)
            if iou >= min_iou:
                pairs.append((iou, tr_idx, gt_idx))

    pairs.sort(reverse=True)
    used_tracks = set()
    used_gt = set()
    keep_tracks = set()
    for _, tr_idx, gt_idx in pairs:
        if tr_idx in used_tracks or gt_idx in used_gt:
            continue
        used_tracks.add(tr_idx)
        used_gt.add(gt_idx)
        keep_tracks.add(tr_idx)

    return [tr for tr_idx, tr in enumerate(tracks) if tr_idx in keep_tracks]


def make_mot_export_row(frame_idx: int, tr: dict, eye_export: bool = False) -> dict:
    """
    Build one MOT row. With eye_export=True the exported box is rebuilt from
    the track's last matched eye landmarks using the SAME geometry as
    chokepoint_to_mot.py GT generation (w=3.2*IOD, h=4*IOD, eyes at h/3).
    Non-oracle: uses only detector landmarks, never GT. Falls back to the
    tracker bbox when landmarks are unavailable (row['eye_fallback']=True).
    """
    eye_fallback = False
    bbox = None
    if eye_export:
        kps = tr.get("kps")
        if kps is not None and len(kps) >= 2:
            (lx, ly), (rx, ry) = kps[0], kps[1]
            ex, ey, ew, eh = eye_bbox(float(lx), float(ly), float(rx), float(ry))
            bbox = [ex, ey, ex + ew, ey + eh]
        else:
            eye_fallback = True
    if bbox is None:
        bbox = tr["bbox_xyxy"]
        if hasattr(bbox, "tolist"):
            bbox = bbox.tolist()
    x1, y1, x2, y2 = [float(v) for v in bbox]
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return {
        "frame_idx": int(frame_idx),
        "track_id": int(tr["track_id"]),
        "local_track_id": int(tr.get("local_track_id", tr["track_id"])),
        "x1": x1,
        "y1": y1,
        "w": w,
        "h": h,
        "conf": 1.0 if tr["state"] == "confirmed" else 0.5,
        "area": w * h,
        "eye_fallback": eye_fallback,
    }


def format_mot_export_row(row: dict) -> str:
    return (
        f"{row['frame_idx']},{row['track_id']},"
        f"{row['x1']:.3f},{row['y1']:.3f},{row['w']:.3f},{row['h']:.3f},"
        f"{row['conf']:.1f},-1,-1,-1\n"
    )


class MotResultWriter:
    """
    Delayed MOT writer for online global-ID recovery.

    New tracklets can carry a temporary global ID for a few frames before the
    global layer merges them back into an older identity. Keeping a small export
    buffer lets those already-seen rows be committed with the corrected ID.
    """

    def __init__(self, fh, delay_frames: int = 5, eye_export: bool = False):
        self.fh = fh
        self.delay_frames = max(0, int(delay_frames))
        self.eye_export = bool(eye_export)
        self.buffer: List[dict] = []
        self.written = 0
        self.eye_fallbacks = 0

    def _refresh_buffered_track_id(self, local_track_id: int, track_id: int) -> None:
        for row in self.buffer:
            if row["local_track_id"] == local_track_id:
                row["track_id"] = track_id

    @staticmethod
    def _dedupe_rows(rows: List[dict]) -> List[dict]:
        """
        Keep MOT output valid if an ID correction makes two buffered rows share
        the same (frame, track_id). Prefer confirmed/larger boxes deterministically.
        """
        chosen: Dict[Tuple[int, int], dict] = {}
        for row in rows:
            key = (int(row["frame_idx"]), int(row["track_id"]))
            prev = chosen.get(key)
            if prev is None:
                chosen[key] = row
                continue
            if (float(row["conf"]), float(row["area"])) > (float(prev["conf"]), float(prev["area"])):
                chosen[key] = row
        return [chosen[key] for key in sorted(chosen)]

    def _flush_rows(self, rows: List[dict]) -> int:
        rows = self._dedupe_rows(rows)
        for row in rows:
            self.fh.write(format_mot_export_row(row))
        self.written += len(rows)
        return len(rows)

    def flush_ready(self, current_frame: int) -> int:
        cutoff = int(current_frame) - self.delay_frames
        if cutoff < 1:
            return 0
        ready = [row for row in self.buffer if int(row["frame_idx"]) <= cutoff]
        self.buffer = [row for row in self.buffer if int(row["frame_idx"]) > cutoff]
        return self._flush_rows(ready)

    def add_tracks(self, frame_idx: int, tracks: List[dict]) -> int:
        for tr in tracks:
            row = make_mot_export_row(frame_idx, tr, eye_export=self.eye_export)
            if row["eye_fallback"]:
                self.eye_fallbacks += 1
            self._refresh_buffered_track_id(row["local_track_id"], row["track_id"])
            self.buffer.append(row)
        return self.flush_ready(frame_idx)

    def flush_all(self) -> int:
        ready = self.buffer
        self.buffer = []
        return self._flush_rows(ready)


def build_detector_embedders_tracker(args, device, img_size):
    from faceqsort_mamba.embedding.resnet18_appearance import ResNet18Appearance
    from faceqsort_mamba.embedding.arcface_embedder import ArcFaceEmbedder
    from faceqsort_mamba.detection.detector_retinaface import RetinaFaceMobileNet025Detector
    from faceqsort_mamba.tracking.faceqsort import FaceQSORTTracker

    detector = RetinaFaceMobileNet025Detector(
        device=device,
        model_name=args.retinaface_model,
        det_thresh=args.det_thresh,
        input_size=(args.det_size, args.det_size),
        retinaface_repo=args.retinaface_repo,
        network=args.retinaface_network,
    )
    arcface = ArcFaceEmbedder(device=device, model_name=args.arcface_model, input_size=112)
    appearance = ResNet18Appearance(device=device, weights_path=args.resnet18_weights)
    tracker = FaceQSORTTracker(
        lambda_bio=args.lambda_bio,
        alpha=args.alpha,
        theta=args.theta,
        iou_thresh=args.iou_thresh,
        max_age=args.max_age,
        n_init=args.n_init,
        ema_momentum=args.ema_momentum,
        cascade_depth=args.cascade_depth,
        device=device,
        mamba_weights=args.mamba_weights,
        img_size=img_size,
        delete_on_exit=not args.no_delete_on_exit,
        exit_margin=args.exit_margin,
        motion_gate_growth=args.motion_gate_growth,
        motion_gate_max=args.motion_gate_max,
        use_global_id=not args.no_global_id,
        global_reid_max_age=args.global_reid_max_age,
        global_reid_appearance_thresh=args.global_reid_appearance_thresh,
        global_reid_motion_base=args.global_reid_motion_base,
        global_reid_motion_per_frame=args.global_reid_motion_per_frame,
        global_reid_motion_cap=args.global_reid_motion_cap,
        global_reid_height_ratio=args.global_reid_height_ratio,
        global_reid_reconsider_hits=args.global_reid_reconsider_hits,
        global_id_consistency_gate=not args.no_global_id_consistency_gate,
        global_id_consistency_thresh=args.global_id_consistency_thresh,
        global_id_debug=args.global_id_debug,
    )
    return detector, arcface, appearance, tracker


def run_sequence(seq_dir: Path, out_file: Path, args, device) -> int:
    info = read_seqinfo(seq_dir)
    img_dir = seq_dir / info["imDir"]
    ext = info["imExt"]
    num_frames = info["seqLength"] or len(list(img_dir.glob(f"*{ext}")))
    gt_by_frame = load_gt_boxes(seq_dir) if args.gt_aware_export else None

    img_size = (info["imHeight"], info["imWidth"]) if info["imHeight"] and info["imWidth"] else None
    detector, arcface, appearance, tracker = build_detector_embedders_tracker(args, device, img_size)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out_file, "w", encoding="utf-8")
    writer = MotResultWriter(fh, delay_frames=args.export_id_buffer, eye_export=args.export_eye_bbox)
    t0 = time.time()
    frames_read = 0
    try:
        for frame_idx in range(1, num_frames + 1):
            img_path = img_dir / f"{frame_idx:06d}{ext}"
            if not img_path.exists():
                continue
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            frames_read += 1

            raw_dets = detector.detect(frame)
            tracker_dets, _ = build_tracker_detections(
                frame_bgr=frame,
                raw_dets=raw_dets,
                arcface=arcface,
                appearance=appearance,
                min_box_size=args.min_box_size,
                frame_idx=frame_idx,
            )
            tracks = tracker.update(tracker_dets)

            export_tracks = []
            for tr in tracks:
                if args.confirmed_only and tr["state"] != "confirmed":
                    continue
                if int(tr["time_since_update"]) > int(getattr(args, "max_export_tsu", 0)):
                    continue
                export_tracks.append(tr)

            if gt_by_frame is not None:
                export_tracks = select_gt_aware_tracks(
                    export_tracks,
                    gt_by_frame.get(frame_idx, []),
                    args.gt_aware_iou_thresh,
                )

            writer.add_tracks(frame_idx, export_tracks)

            if args.log_every > 0 and frame_idx % args.log_every == 0:
                fps = frame_idx / max(1e-6, time.time() - t0)
                print(f"  [{info['name']}] frame {frame_idx}/{num_frames}  fps={fps:.2f}  wrote={writer.written}", flush=True)
    finally:
        writer.flush_all()
        fh.close()
    dt = time.time() - t0
    eye_note = ""
    if args.export_eye_bbox:
        eye_note = f" eye_fallbacks={writer.eye_fallbacks}"
    print(f"[{info['name']}] done: frames={num_frames} read={frames_read} rows={writer.written} time={dt:.1f}s fps={num_frames/max(dt,1e-6):.2f}{eye_note}")
    if frames_read == 0:
        print(f"[{info['name']}] WARNING: 0 frames read! Expected files like "
              f"{img_dir / (f'{1:06d}' + ext)} — check img dir, naming, extension, symlinks.",
              flush=True)
    return writer.written


def main() -> None:
    p = argparse.ArgumentParser(description="Run FaceQSORT-Mamba over MOTChallenge sequences.")
    p.add_argument("--mot-dir", type=str, required=True,
                   help="Path to MOT split, e.g. /.../chokepoint_mot/ or /.../MOT20/train.")
    p.add_argument("--output-dir", type=str, default="mot_results",
                   help="Root output directory (TrackEval layout).")
    p.add_argument("--tracker-name", type=str, default="FaceQSORT_Mamba",
                   help="Tracker folder name under --output-dir.")
    p.add_argument("--sequence", type=str, default=None,
                   help="Optional single sequence name. Default: all sequences in --mot-dir.")
    p.add_argument("--confirmed-only", action="store_true")
    p.add_argument("--log-every", type=int, default=100)

    p.add_argument("--device", type=str, default=None, choices=["mps", "cuda", "cpu"])
    p.add_argument("--retinaface-repo", type=str, default=None)
    p.add_argument("--retinaface-model", type=str, default="mobilenet0.25_Final.pth")
    p.add_argument("--retinaface-network", type=str, default="mobile0.25", choices=["mobile0.25", "resnet50"])
    p.add_argument("--arcface-model", type=str, default="auto")
    p.add_argument("--resnet18-weights", type=str, default=None)
    p.add_argument("--det-thresh", type=float, default=0.5)
    p.add_argument("--det-size", type=int, default=640)
    p.add_argument("--min-box-size", type=int, default=20)

    p.add_argument("--lambda-bio", type=float, default=0.9)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--theta", type=float, default=0.2)
    p.add_argument("--iou-thresh", type=float, default=0.3)
    p.add_argument("--max-age", type=int, default=300)
    p.add_argument("--n-init", type=int, default=3)
    p.add_argument("--ema-momentum", type=float, default=0.9)
    p.add_argument("--cascade-depth", type=int, default=20)
    p.add_argument(
        "--no-delete-on-exit",
        action="store_true",
        help="Disable frame-exit deletion and fall back to --max-age deletion.",
    )
    p.add_argument(
        "--exit-margin",
        type=float,
        default=0.0,
        help="Extra pixel margin before deleting a track whose predicted box has left the frame.",
    )
    p.add_argument(
        "--motion-gate-growth",
        type=float,
        default=0.25,
        help="Per-missed-frame growth for the motion gate used during occlusion recovery.",
    )
    p.add_argument(
        "--motion-gate-max",
        type=float,
        default=4.0,
        help="Maximum multiplier for the motion gate after repeated missed frames.",
    )
    p.add_argument(
        "--no-global-id",
        action="store_true",
        help="Disable online global identity recovery and write local track IDs.",
    )
    p.add_argument(
        "--global-reid-max-age",
        type=int,
        default=300,
        help="Maximum frames after last sighting for a global ID to be reused.",
    )
    p.add_argument(
        "--global-reid-appearance-thresh",
        type=float,
        default=0.50,
        help="Maximum combined biometric/appearance cosine distance for global re-ID.",
    )
    p.add_argument(
        "--global-reid-motion-base",
        type=float,
        default=3.0,
        help="Base center-distance gate in face-height units for global re-ID.",
    )
    p.add_argument(
        "--global-reid-motion-per-frame",
        type=float,
        default=0.5,
        help="Per-missed-frame growth of the global re-ID motion gate.",
    )
    p.add_argument(
        "--global-reid-motion-cap",
        type=float,
        default=20.0,
        help="Maximum global re-ID motion gate in face-height units.",
    )
    p.add_argument(
        "--global-reid-height-ratio",
        type=float,
        default=3.0,
        help="Maximum height ratio between lost and new bbox for global re-ID.",
    )
    p.add_argument(
        "--global-reid-reconsider-hits",
        type=int,
        default=8,
        help="Keep young global IDs eligible for merge into older IDs for this many matched hits.",
    )
    p.add_argument(
        "--no-global-id-consistency-gate",
        action="store_true",
        help="Disable rejection of local matches that conflict with the assigned global identity.",
    )
    p.add_argument(
        "--global-id-consistency-thresh",
        type=float,
        default=0.75,
        help="Maximum global feature distance allowed before rejecting a local match.",
    )
    p.add_argument(
        "--global-id-debug",
        action="store_true",
        help="Print global-ID reuse, merge, and rejected-match events.",
    )
    p.add_argument("--max-export-tsu", type=int, default=0,
                   help="Max time_since_update for a track to be written. "
                        "0 = only tracks matched on current frame.")
    p.add_argument(
        "--export-id-buffer",
        type=int,
        default=5,
        help="Delay MOT row writing by N frames so temporary global IDs can be corrected before export. 0 disables.",
    )
    p.add_argument(
        "--export-eye-bbox",
        action="store_true",
        help="Export bboxes rebuilt from detector eye landmarks using the same "
             "geometry as chokepoint_to_mot.py GT (non-oracle; for eye-derived-GT benchmarks).",
    )
    p.add_argument(
        "--gt-aware-export",
        action="store_true",
        help="Evaluation-only oracle export: write only tracker boxes that match GT boxes in the same frame.",
    )
    p.add_argument(
        "--gt-aware-iou-thresh",
        type=float,
        default=0.5,
        help="Minimum IoU between tracker output and GT box for --gt-aware-export.",
    )

    # Default: auto-detect ./checkpoint_best.pth next to this script.
    # Pass --mamba-weights "" or --mamba-weights none to force the linear fallback.
    _default_mamba_ckpt = Path(__file__).parent / "checkpoint_best.pth"
    p.add_argument(
        "--mamba-weights",
        type=str,
        default=str(_default_mamba_ckpt) if _default_mamba_ckpt.exists() else None,
        help="Path to trained Mamba MTP checkpoint. Default: ./checkpoint_best.pth if present, else linear fallback.",
    )
    args = p.parse_args()
    if args.mamba_weights and args.mamba_weights.strip().lower() in {"", "none", "null"}:
        args.mamba_weights = None
    if args.export_id_buffer < 0:
        raise ValueError("--export-id-buffer must be >= 0")

    device = resolve_device(args.device) if args.device else get_best_device()
    print(f"[MOT] device={device}", flush=True)
    print(f"[MOT] mamba_weights={args.mamba_weights or 'NONE (linear fallback)'}", flush=True)
    print(f"[MOT] export_id_buffer={args.export_id_buffer}", flush=True)

    mot_dir = Path(args.mot_dir).expanduser().resolve()
    if not mot_dir.exists():
        raise FileNotFoundError(f"MOT directory not found: {mot_dir}")

    if args.sequence:
        seqs = [mot_dir / args.sequence]
        if not seqs[0].exists():
            raise FileNotFoundError(f"Sequence not found: {seqs[0]}")
    else:
        seqs = list_sequences(mot_dir)
        if not seqs:
            raise RuntimeError(f"No sequences with seqinfo.ini found under {mot_dir}")

    out_root = Path(args.output_dir).expanduser().resolve() / args.tracker_name / "data"
    print(f"[MOT] sequences={len(seqs)}  output={out_root}", flush=True)

    for seq_dir in seqs:
        out_file = out_root / f"{seq_dir.name}.txt"
        print(f"[MOT] >>> {seq_dir.name}", flush=True)
        run_sequence(seq_dir, out_file, args, device)

    print(f"[MOT] All done. Results: {out_root}")


if __name__ == "__main__":
    main()
