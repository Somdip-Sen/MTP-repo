"""
Convert ChokePoint single-person XML ground truth to MOTChallenge format.

ChokePoint GT gives left-eye + right-eye pixel coordinates per (frame, person).
We synthesize a face bbox from eye positions using the standard heuristic:
    inter = |right.x - left.x|
    cx, cy = midpoint
    w = 3.2 * inter      # box width ≈ 3× eye distance (conservative face width)
    h = 4.0 * inter      # box height ≈ 4× eye distance (forehead + chin)
    top-left at (cx - w/2, cy - h/3)   # eyes sit ~1/3 from top

Output layout (MOTChallenge / TrackEval MotChallenge2DBox-compatible):
    <out>/<seq>/
        seqinfo.ini
        img1/   (symlinks to original frames, renamed 000001.jpg .. 00NNNN.jpg)
        gt/gt.txt       rows:  frame,id,x,y,w,h,1,1,1.0

Frame numbering: ChokePoint frames are 0-indexed ("00000000.jpg"), MOT is
1-indexed. We remap: chokepoint_frame N -> mot_frame N+1, and the symlinked
filename in img1/ becomes zero-padded N+1 for compatibility with
run_mot.py which globs img1/{frame:06d}.jpg.
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import cv2


def eye_bbox(lx: float, ly: float, rx: float, ry: float) -> Tuple[float, float, float, float]:
    inter = max(1.0, abs(rx - lx))
    cx = (lx + rx) / 2.0
    cy = (ly + ry) / 2.0
    w = 3.2 * inter
    h = 4.0 * inter
    x = cx - w / 2.0
    y = cy - h / 3.0
    return x, y, w, h


def clamp_bbox(x: float, y: float, w: float, h: float, W: int, H: int) -> Tuple[float, float, float, float]:
    x1 = max(0.0, min(x, W - 1.0))
    y1 = max(0.0, min(y, H - 1.0))
    x2 = max(0.0, min(x + w, float(W)))
    y2 = max(0.0, min(y + h, float(H)))
    return x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)


def parse_xml(xml_path: Path) -> Tuple[List[str], Dict[int, List[Tuple[int, float, float, float, float, float, float]]]]:
    """
    Returns:
      frames_order: list of ChokePoint frame names in order (e.g. "00000000")
      gt_by_frame:  {chokepoint_frame_int -> [(person_id, lx, ly, rx, ry, ... )]}
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    frames_order: List[str] = []
    gt: Dict[int, List[Tuple[int, float, float, float, float]]] = {}
    for frame in root.findall("frame"):
        num_str = frame.get("number")
        frames_order.append(num_str)
        num_int = int(num_str)
        persons: List[Tuple[int, float, float, float, float]] = []
        for person in frame.findall("person"):
            pid_str = person.get("id")
            pid = int(pid_str)
            le = person.find("leftEye")
            re = person.find("rightEye")
            if le is None or re is None:
                continue
            persons.append((pid, float(le.get("x")), float(le.get("y")), float(re.get("x")), float(re.get("y"))))
        if persons:
            gt[num_int] = persons
    return frames_order, gt


def convert_one(src_frames_dir: Path, xml_path: Path, out_seq_dir: Path, frame_rate: float, symlink: bool = True) -> dict:
    frames_order, gt_by_frame = parse_xml(xml_path)
    if not frames_order:
        raise RuntimeError(f"No frames parsed from {xml_path}")

    probe = src_frames_dir / f"{frames_order[0]}.jpg"
    if not probe.exists():
        raise FileNotFoundError(f"First frame missing: {probe}")
    im = cv2.imread(str(probe))
    if im is None:
        raise RuntimeError(f"cv2 failed to read {probe}")
    H, W = im.shape[:2]

    img1_dir = out_seq_dir / "img1"
    gt_dir = out_seq_dir / "gt"
    img1_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    mot_frame = 0
    linked = 0
    gt_rows = 0
    with open(gt_dir / "gt.txt", "w", encoding="utf-8") as fh_gt:
        for cp_name in frames_order:
            mot_frame += 1  # 1-indexed
            src_jpg = src_frames_dir / f"{cp_name}.jpg"
            dst_jpg = img1_dir / f"{mot_frame:06d}.jpg"
            if not dst_jpg.exists():
                if symlink:
                    dst_jpg.symlink_to(src_jpg.resolve())
                else:
                    dst_jpg.write_bytes(src_jpg.read_bytes())
                linked += 1
            cp_int = int(cp_name)
            if cp_int in gt_by_frame:
                for pid, lx, ly, rx, ry in gt_by_frame[cp_int]:
                    x, y, w, h = eye_bbox(lx, ly, rx, ry)
                    x, y, w, h = clamp_bbox(x, y, w, h, W, H)
                    if w <= 0 or h <= 0:
                        continue
                    # MOT gt.txt format: frame,id,x,y,w,h,conf,class,visibility
                    # conf=1, class=1 (pedestrian in MOT17/20; repurposed here for face), visibility=1.0
                    fh_gt.write(f"{mot_frame},{pid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1,1.0\n")
                    gt_rows += 1

    seq_name = out_seq_dir.name
    with open(out_seq_dir / "seqinfo.ini", "w", encoding="utf-8") as fh:
        fh.write("[Sequence]\n")
        fh.write(f"name={seq_name}\n")
        fh.write("imDir=img1\n")
        fh.write(f"frameRate={frame_rate}\n")
        fh.write(f"seqLength={mot_frame}\n")
        fh.write(f"imWidth={W}\n")
        fh.write(f"imHeight={H}\n")
        fh.write("imExt=.jpg\n")

    return {"seq": seq_name, "frames": mot_frame, "linked": linked, "gt_rows": gt_rows, "W": W, "H": H}


def main() -> None:
    p = argparse.ArgumentParser(description="Convert ChokePoint single-person sequences to MOT format.")
    p.add_argument("--chokepoint-root", type=str, default=".",
                   help="Directory that contains P?E_S?_Cx/ frame folders (defaults to CWD).")
    p.add_argument("--gt-dir", type=str, required=True,
                   help="Directory containing ChokePoint groundtruth XMLs (e.g. ./groundtruth).")
    p.add_argument("--sequences", type=str, nargs="+", required=True,
                   help="Sequence identifiers to convert, e.g. P1E_S1_C1 P1E_S1_C2 P1E_S1_C3.")
    p.add_argument("--output-dir", type=str, default="chokepoint_mot",
                   help="Where to write the MOT-format workspace.")
    p.add_argument("--frame-rate", type=float, default=30.0)
    p.add_argument("--copy", action="store_true",
                   help="Copy frames instead of symlinking (larger disk usage).")
    args = p.parse_args()

    cp_root = Path(args.chokepoint_root).resolve()
    gt_dir = Path(args.gt_dir).resolve()
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[convert] chokepoint_root={cp_root}")
    print(f"[convert] gt_dir={gt_dir}")
    print(f"[convert] output_dir={out_root}")

    for seq in args.sequences:
        # Frames may live at cp_root/<seq>/ OR cp_root/P1E_S1_extract/<seq>/ etc.; try common layouts.
        candidates = [
            cp_root / seq,
            cp_root / f"{seq.split('_C')[0]}_extract" / seq,
        ]
        src = next((c for c in candidates if c.exists()), None)
        if src is None:
            raise FileNotFoundError(f"Could not find frames for {seq} under {candidates}")
        xml_path = gt_dir / f"{seq}.xml"
        if not xml_path.exists():
            raise FileNotFoundError(f"Missing GT: {xml_path}")
        out_seq_dir = out_root / seq
        print(f"[convert] {seq}: src={src} xml={xml_path}")
        stats = convert_one(src, xml_path, out_seq_dir, args.frame_rate, symlink=not args.copy)
        print(
            f"  -> frames={stats['frames']} linked={stats['linked']} gt_rows={stats['gt_rows']} "
            f"resolution={stats['W']}x{stats['H']}"
        )

    print("[convert] done.")


if __name__ == "__main__":
    main()
