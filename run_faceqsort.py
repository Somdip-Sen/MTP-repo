from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple, Union

import cv2
import numpy as np

from faceqsort_mamba.utils.device_utils import get_best_device, resolve_device

if TYPE_CHECKING:
    from faceqsort_mamba.embedding.resnet18_appearance import ResNet18Appearance
    from faceqsort_mamba.embedding.arcface_embedder import ArcFaceEmbedder
    from faceqsort_mamba.detection.detector_retinaface import FaceDetection


def parse_source(src: str) -> Union[int, str]:
    src = str(src).strip()
    return int(src) if src.isdigit() else src


def clip_bbox_xyxy(bbox_xyxy: np.ndarray, image_shape: Tuple[int, int, int]) -> np.ndarray:
    h, w = image_shape[:2]
    x1, y1, x2, y2 = bbox_xyxy.astype(np.float32).tolist()
    x1 = max(0.0, min(x1, float(w - 1)))
    y1 = max(0.0, min(y1, float(h - 1)))
    x2 = max(0.0, min(x2, float(w)))
    y2 = max(0.0, min(y2, float(h)))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def crop_face(frame_bgr: np.ndarray, bbox_xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = bbox_xyxy.astype(int).tolist()
    return frame_bgr[y1:y2, x1:x2]


def id_color(track_id: int) -> Tuple[int, int, int]:
    palette = [
        (0, 255, 255),    # yellow
        (255, 0, 255),    # magenta
        (0, 180, 255),    # orange
        (80, 255, 80),    # neon green
        (255, 80, 80),    # bright blue
        (255, 255, 0),    # cyan
        (180, 80, 255),   # purple
        (40, 120, 255),   # vivid orange-red
        (255, 120, 40),   # sky blue
        (120, 255, 40),   # lime
    ]
    return palette[(int(track_id) - 1) % len(palette)]


def ui_scale(frame_height: int) -> float:
    """
    Resolution-relative scale for overlay text/lines. UI constants below are
    tuned for 720p; low-res videos (e.g. 144p) otherwise get oversized,
    clipped overlays. Floored so strokes never vanish on tiny frames.
    """
    return max(0.25, float(frame_height) / 720.0)


def draw_tracks(
    frame_bgr: np.ndarray,
    tracks: List[dict],
    confirmed_only: bool = True,
    max_tsu: int = 0,
) -> np.ndarray:
    """
    Draw bboxes ONLY for tracks updated within the last `max_tsu` frames.
    Default max_tsu=0 means draw only tracks matched on the current frame, so
    tracks the tracker is still internally holding through occlusion (alive
    but unmatched, with predicted bboxes) do not accumulate as ghost rectangles
    over the scene.
    """
    vis = frame_bgr.copy()
    s = ui_scale(vis.shape[0])
    box_thick = max(1, int(round(2 * s)))
    txt_thick = max(1, int(round(s)))
    txt_offset = max(2, int(round(8 * s)))
    for tr in tracks:
        state = tr["state"]
        if confirmed_only and state != "confirmed":
            continue
        if int(tr["time_since_update"]) > max_tsu:
            continue

        x1, y1, x2, y2 = tr["bbox_xyxy"].astype(int).tolist()
        tid = int(tr["track_id"])
        color = id_color(tid)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, box_thick)
        label = str(tid)
        cv2.putText(vis, label, (x1, max(0, y1 - txt_offset)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45 * s, color, txt_thick, cv2.LINE_AA)
    return vis


def build_tracker_detections(
        frame_bgr: np.ndarray,
        raw_dets: List["FaceDetection"],
        arcface: "ArcFaceEmbedder",
        appearance: "ResNet18Appearance",
        min_box_size: int,
        frame_idx: int = 0,
) -> Tuple[List["Detection"], int]:
    from faceqsort_mamba.tracking.faceqsort import Detection

    out: List[Detection] = []
    feature_failures = 0
    for det_idx, det in enumerate(raw_dets):
        bbox = clip_bbox_xyxy(det.bbox_xyxy, frame_bgr.shape)
        x1, y1, x2, y2 = bbox.astype(int).tolist()
        if (x2 - x1) < min_box_size or (y2 - y1) < min_box_size:
            continue

        crop = crop_face(frame_bgr, bbox)
        if crop.size == 0:
            continue

        try:
            feat_bio = arcface(frame_bgr, det.kps, bbox)
            feat_app = appearance(crop)
            
            # Validate feature dimensions
            feat_bio = np.asarray(feat_bio, dtype=np.float32)
            feat_app = np.asarray(feat_app, dtype=np.float32)
            if feat_bio.ndim != 1:
                raise ValueError(f"ArcFace output shape {feat_bio.shape} is not 1D")
            if feat_app.ndim != 1:
                raise ValueError(f"ResNet18 output shape {feat_app.shape} is not 1D")
        except Exception as e:
            # Skip detections that fail feature extraction to keep tracker state valid.
            feature_failures += 1
            continue

        out.append(
            Detection(
                bbox_xyxy=bbox,
                conf=float(det.conf),
                feat_bio=feat_bio,
                feat_app=feat_app,
                kps=np.asarray(det.kps, dtype=np.float32) if det.kps is not None else None,
            )
        )
    return out, feature_failures


def main() -> None:
    parser = argparse.ArgumentParser(description="FaceQSORT full pipeline with RetinaFace MobileNet-0.25 + ArcFace.")
    parser.add_argument("--input", type=str, required=True, help="Video path or camera index (e.g., 0).")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory for outputs.")
    parser.add_argument("--output-video", type=str, default=None, help="Output tracked video path (.mp4).")
    parser.add_argument("--output-mot", type=str, default=None, help="Output MOT text path.")
    parser.add_argument("--output-jsonl", type=str, default=None, help="Output JSONL path.")
    parser.add_argument("--no-video", action="store_true", help="Disable writing output video.")
    parser.add_argument("--show", action="store_true", help="Show live preview window.")
    parser.add_argument("--confirmed-only", action="store_true", help="Render/export only confirmed tracks.")
    parser.add_argument("--log-every", type=int, default=0,
                        help="Optional periodic progress every N frames (0 disables).")

    parser.add_argument("--device", type=str, default=None, choices=["mps", "cuda", "cpu"], help="Execution device.")
    parser.add_argument("--model-root", type=str, default=None, help="InsightFace model cache root directory.")
    parser.add_argument(
        "--retinaface-repo",
        type=str,
        default=None,
        help="Path to local Pytorch_Retinaface repo (needed for .pth detector weights).",
    )

    parser.add_argument("--det-thresh", type=float, default=0.5, help="Detector confidence threshold.")
    parser.add_argument("--det-size", type=int, default=640, help="Detector input size (square).")
    parser.add_argument("--min-box-size", type=int, default=20, help="Minimum face size in pixels.")
    parser.add_argument(
        "--retinaface-model",
        type=str,
        default="mobilenet0.25_Final.pth",
        help="RetinaFace model (.pth for local Pytorch_Retinaface backend, or InsightFace ONNX model id/path).",
    )
    parser.add_argument(
        "--retinaface-network",
        type=str,
        default="mobile0.25",
        choices=["mobile0.25", "resnet50"],
        help="RetinaFace backbone for .pth weights.",
    )
    parser.add_argument("--arcface-model", type=str, default="auto", help="ArcFace model id or .onnx path.")
    parser.add_argument(
        "--resnet18-weights",
        type=str,
        default=None,
        help="Optional local ResNet18 weights file for appearance features.",
    )

    parser.add_argument("--lambda-bio", type=float, default=0.9, help="Weight for biometric vs appearance (0.9 = 90%% biometric).")
    parser.add_argument("--w-feat", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=9.4877)
    parser.add_argument("--theta", type=float, default=0.2)
    parser.add_argument("--iou-thresh", type=float, default=0.3)
    parser.add_argument("--max-age", type=int, default=300)
    parser.add_argument("--n-init", type=int, default=3)
    parser.add_argument("--ema-momentum", type=float, default=0.9)
    parser.add_argument("--cascade-depth", type=int, default=20, help="Matching cascade depth for confirmed tracks.")
    parser.add_argument(
        "--no-delete-on-exit",
        action="store_true",
        help="Disable frame-exit deletion and fall back to --max-age deletion.",
    )
    parser.add_argument(
        "--exit-margin",
        type=float,
        default=0.0,
        help="Extra pixel margin before deleting a track whose predicted box has left the frame.",
    )
    parser.add_argument(
        "--motion-gate-growth",
        type=float,
        default=0.25,
        help="Per-missed-frame growth for the motion gate used during occlusion recovery.",
    )
    parser.add_argument(
        "--motion-gate-max",
        type=float,
        default=4.0,
        help="Maximum multiplier for the motion gate after repeated missed frames.",
    )
    parser.add_argument(
        "--no-global-id",
        action="store_true",
        help="Disable online global identity recovery and write local track IDs.",
    )
    parser.add_argument(
        "--global-reid-max-age",
        type=int,
        default=300,
        help="Maximum frames after last sighting for a global ID to be reused.",
    )
    parser.add_argument(
        "--global-reid-appearance-thresh",
        type=float,
        default=0.50,
        help="Maximum combined biometric/appearance cosine distance for global re-ID.",
    )
    parser.add_argument(
        "--global-reid-motion-base",
        type=float,
        default=3.0,
        help="Base center-distance gate in face-height units for global re-ID.",
    )
    parser.add_argument(
        "--global-reid-motion-per-frame",
        type=float,
        default=0.5,
        help="Per-missed-frame growth of the global re-ID motion gate.",
    )
    parser.add_argument(
        "--global-reid-motion-cap",
        type=float,
        default=20.0,
        help="Maximum global re-ID motion gate in face-height units.",
    )
    parser.add_argument(
        "--global-reid-height-ratio",
        type=float,
        default=3.0,
        help="Maximum height ratio between lost and new bbox for global re-ID.",
    )
    parser.add_argument(
        "--global-reid-reconsider-hits",
        type=int,
        default=8,
        help="Keep young global IDs eligible for merge into older IDs for this many matched hits.",
    )
    parser.add_argument(
        "--no-global-id-consistency-gate",
        action="store_true",
        help="Disable rejection of local matches that conflict with the assigned global identity.",
    )
    parser.add_argument(
        "--global-id-consistency-thresh",
        type=float,
        default=0.75,
        help="Maximum global feature distance allowed before rejecting a local match.",
    )
    parser.add_argument(
        "--global-id-debug",
        action="store_true",
        help="Print global-ID reuse, merge, and rejected-match events.",
    )
    parser.add_argument(
        "--max-export-tsu",
        type=int,
        default=0,
        help="Max time_since_update for a track to be drawn/written. "
             "0 (default) = only tracks matched on the current frame. "
             "Increase to render tracks predicted through brief occlusion (at the cost of ghost boxes when faces leave the frame).",
    )
    # Default: auto-detect ./checkpoint_best.pth next to this script.
    # Pass --mamba-weights "" or --mamba-weights none to force the linear fallback.
    _default_mamba_ckpt = Path(__file__).parent / "checkpoint_best.pth"
    parser.add_argument(
        "--mamba-weights",
        type=str,
        default=str(_default_mamba_ckpt) if _default_mamba_ckpt.exists() else None,
        help="Path to trained Mamba MTP checkpoint. Default: ./checkpoint_best.pth if present, else linear fallback.",
    )
    args = parser.parse_args()
    # Allow disabling via empty string or 'none'
    if args.mamba_weights and args.mamba_weights.strip().lower() in {"", "none", "null"}:
        args.mamba_weights = None

    device = resolve_device(args.device) if args.device else get_best_device()
    print(f"[FaceQSORT] Device selected: {device}", flush=True)
    print(f"[FaceQSORT] Mamba weights: {args.mamba_weights or 'NONE (linear fallback)'}", flush=True)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_video = Path(args.output_video).expanduser().resolve() if args.output_video else output_dir / "tracked.mp4"
    output_mot = Path(args.output_mot).expanduser().resolve() if args.output_mot else output_dir / "tracks_mot.txt"
    output_jsonl = Path(args.output_jsonl).expanduser().resolve() if args.output_jsonl else output_dir / "tracks.jsonl"
    output_mot.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_video:
        output_video.parent.mkdir(parents=True, exist_ok=True)

    from faceqsort_mamba.embedding.resnet18_appearance import ResNet18Appearance
    from faceqsort_mamba.embedding.arcface_embedder import ArcFaceEmbedder
    from faceqsort_mamba.detection.detector_retinaface import RetinaFaceMobileNet025Detector
    from faceqsort_mamba.tracking.faceqsort import FaceQSORTTracker

    source = parse_source(args.input)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input source: {args.input}")
    print(f"[FaceQSORT] Input opened: {args.input}", flush=True)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0
    print(f"[FaceQSORT] Input FPS (reported): {fps:.2f}", flush=True)

    writer = None

    try:
        print("[FaceQSORT] Initializing RetinaFace detector...", flush=True)
        detector = RetinaFaceMobileNet025Detector(
            device=device,
            model_name=args.retinaface_model,
            model_root=args.model_root,
            det_thresh=args.det_thresh,
            input_size=(args.det_size, args.det_size),
            retinaface_repo=args.retinaface_repo,
            network=args.retinaface_network,
        )
        print(f"[FaceQSORT] Detector ready: {detector.model_name}", flush=True)
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize RetinaFace MobileNet-0.25 detector. "
            "If using .pth weights, pass --retinaface-repo to your local Pytorch_Retinaface repo."
        ) from exc

    try:
        print("[FaceQSORT] Initializing ArcFace embedder...", flush=True)
        arcface = ArcFaceEmbedder(
            device=device,
            model_name=args.arcface_model,
            model_root=args.model_root,
            input_size=112,
        )
        print("[FaceQSORT] ArcFace ready.", flush=True)
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize ArcFace embedder. "
            "Provide a valid ArcFace ONNX path via --arcface-model or ensure InsightFace models are available."
        ) from exc
    print("[FaceQSORT] Initializing ResNet18 appearance embedder...", flush=True)
    appearance = ResNet18Appearance(device=device, weights_path=args.resnet18_weights)
    print("[FaceQSORT] ResNet18 ready.", flush=True)
    tracker = FaceQSORTTracker(
        lambda_bio=args.lambda_bio,
        w_feat=args.w_feat,
        alpha=args.alpha,
        gamma=args.gamma,
        theta=args.theta,
        iou_thresh=args.iou_thresh,
        max_age=args.max_age,
        n_init=args.n_init,
        ema_momentum=args.ema_momentum,
        cascade_depth=args.cascade_depth,
        device=device,
        mamba_weights=args.mamba_weights,
        img_size=(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
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

    mot_fh = open(output_mot, "w", encoding="utf-8")
    jsonl_fh = open(output_jsonl, "w", encoding="utf-8")
    print("[FaceQSORT] Tracking loop started.", flush=True)

    frame_idx = 0
    total_tracks_written = 0
    start_time = time.time()
    prev_raw_det: Optional[int] = None
    prev_used_det: Optional[int] = None
    prev_feature_failures: Optional[int] = None
    prev_export_tracks: Optional[int] = None
    prev_track_ids: set[int] = set()
    prev_track_states: Dict[int, str] = {}
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            raw_dets = detector.detect(frame)
            tracker_dets, feature_failures = build_tracker_detections(
                frame_bgr=frame,
                raw_dets=raw_dets,
                arcface=arcface,
                appearance=appearance,
                min_box_size=args.min_box_size,
                frame_idx=frame_idx,
            )
            tracks = tracker.update(tracker_dets)

            # Filter by state and by recency. Tracks the tracker is internally
            # holding through occlusion (time_since_update > max_export_tsu) are
            # excluded from rendering and MOT output to prevent ghost boxes
            # accumulating after a face leaves the frame.
            _max_tsu = int(getattr(args, "max_export_tsu", 0))
            if args.confirmed_only:
                tracks_export = [
                    t for t in tracks
                    if t["state"] == "confirmed" and int(t["time_since_update"]) <= _max_tsu
                ]
            else:
                tracks_export = [t for t in tracks if int(t["time_since_update"]) <= _max_tsu]

            raw_det_count = len(raw_dets)
            used_det_count = len(tracker_dets)
            export_track_count = len(tracks_export)

            cur_track_ids = {int(t["track_id"]) for t in tracks}
            cur_track_states = {int(t["track_id"]): t["state"] for t in tracks}

            if prev_raw_det is None or raw_det_count != prev_raw_det:
                print(f"[FaceQSORT][event] frame={frame_idx} raw_detections {prev_raw_det} -> {raw_det_count}",
                      flush=True)
            if prev_used_det is None or used_det_count != prev_used_det:
                print(f"[FaceQSORT][event] frame={frame_idx} used_detections {prev_used_det} -> {used_det_count}",
                      flush=True)
            if prev_feature_failures is None or feature_failures != prev_feature_failures:
                if feature_failures > 0 or (prev_feature_failures is not None and prev_feature_failures > 0):
                    print(
                        f"[FaceQSORT][event] frame={frame_idx} feature_failures {prev_feature_failures} -> {feature_failures}",
                        flush=True,
                    )
            if prev_export_tracks is None or export_track_count != prev_export_tracks:
                print(
                    f"[FaceQSORT][event] frame={frame_idx} output_tracks {prev_export_tracks} -> {export_track_count}",
                    flush=True)

            new_ids = sorted(cur_track_ids - prev_track_ids)
            if new_ids:
                details = ",".join([f"{tid}:{cur_track_states.get(tid, 'unknown')}" for tid in new_ids])
                print(f"[FaceQSORT][event] frame={frame_idx} new_tracks [{details}]", flush=True)

            removed_ids = sorted(prev_track_ids - cur_track_ids)
            if removed_ids:
                details = ",".join(str(tid) for tid in removed_ids)
                print(f"[FaceQSORT][event] frame={frame_idx} removed_tracks [{details}]", flush=True)

            common_ids = sorted(cur_track_ids & prev_track_ids)
            for tid in common_ids:
                prev_state = prev_track_states.get(tid)
                cur_state = cur_track_states.get(tid)
                if prev_state != cur_state:
                    print(
                        f"[FaceQSORT][event] frame={frame_idx} track_state_change id={tid} {prev_state} -> {cur_state}",
                        flush=True,
                    )

            prev_raw_det = raw_det_count
            prev_used_det = used_det_count
            prev_feature_failures = feature_failures
            prev_export_tracks = export_track_count
            prev_track_ids = cur_track_ids
            prev_track_states = cur_track_states

            for tr in tracks_export:
                x1, y1, x2, y2 = tr["bbox_xyxy"].astype(np.float32).tolist()
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                # Use track confidence: 1.0 for confirmed, 0.5 for tentative
                conf = 1.0 if tr["state"] == "confirmed" else 0.5
                mot_fh.write(
                    f"{frame_idx},{int(tr['track_id'])},{x1:.3f},{y1:.3f},{w:.3f},{h:.3f},{conf},-1,-1,-1\n"
                )
                total_tracks_written += 1

            frame_log = {
                "frame_idx": frame_idx,
                "raw_detections": len(raw_dets),
                "used_detections": len(tracker_dets),
                "tracks": [
                    {
                        "track_id": int(tr["track_id"]),
                        "local_track_id": int(tr.get("local_track_id", tr["track_id"])),
                        "global_id": int(tr.get("global_id", tr["track_id"])),
                        "state": tr["state"],
                        "time_since_update": int(tr["time_since_update"]),
                        "bbox_xyxy": [float(v) for v in tr["bbox_xyxy"].tolist()],
                    }
                    for tr in tracks_export
                ],
            }
            jsonl_fh.write(json.dumps(frame_log) + "\n")

            vis = draw_tracks(frame, tracks, confirmed_only=args.confirmed_only, max_tsu=_max_tsu)
            hud_s = ui_scale(vis.shape[0])
            cv2.putText(
                vis,
                f"frame={frame_idx} det={len(raw_dets)} used={len(tracker_dets)} track={len(tracks_export)}",
                (max(2, int(round(10 * hud_s))), max(10, int(round(24 * hud_s)))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65 * hud_s,
                (50, 220, 50),
                max(1, int(round(2 * hud_s))),
                cv2.LINE_AA,
            )

            if not args.no_video and writer is None:
                h, w = vis.shape[:2]
                # Try codec fallback: mp4v -> MJPG -> H264 -> DIVX
                for codec_str in ["mp4v", "MJPG", "H264", "DIVX"]:
                    try:
                        fourcc = cv2.VideoWriter_fourcc(*codec_str)
                        writer = cv2.VideoWriter(str(output_video), fourcc, fps, (w, h))
                        if writer.isOpened():
                            print(f"[FaceQSORT] Video codec '{codec_str}' initialized successfully.", flush=True)
                            break
                        writer = None
                    except Exception:
                        pass
                
                if writer is None or not writer.isOpened():
                    raise RuntimeError(f"Failed to open output video writer with any codec: {output_video}")
            if writer is not None:
                writer.write(vis)

            if args.show:
                cv2.imshow("FaceQSORT", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.log_every > 0 and (frame_idx == 1 or frame_idx % args.log_every == 0):
                elapsed = max(1e-6, time.time() - start_time)
                avg_fps = frame_idx / elapsed
                print(
                    f"[FaceQSORT] Progress frame={frame_idx} raw_det={len(raw_dets)} "
                    f"used_det={len(tracker_dets)} tracks={len(tracks_export)} avg_fps={avg_fps:.2f}",
                    flush=True,
                )
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        mot_fh.close()
        jsonl_fh.close()
        if args.show:
            cv2.destroyAllWindows()

    dt = max(1e-6, time.time() - start_time)
    print(
        f"[FaceQSORT] Finished: frames={frame_idx}, tracks_written={total_tracks_written}, "
        f"fps={frame_idx / dt:.2f}"
    )
    if not args.no_video:
        print(f"[FaceQSORT] Video: {output_video}")
    print(f"[FaceQSORT] MOT:   {output_mot}")
    print(f"[FaceQSORT] JSONL: {output_jsonl}")


if __name__ == "__main__":
    main()
