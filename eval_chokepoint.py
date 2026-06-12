"""
Score FaceQSORT on ChokePoint (converted to MOT format) using TrackEval.

Computes HOTA, CLEAR (MOTA/MOTP/IDSW), and Identity (IDF1/IDR/IDP) metrics.

Expects:
  - chokepoint_mot/<seq>/{seqinfo.ini, img1/, gt/gt.txt}   (from chokepoint_to_mot.py)
  - mot_results/<tracker>/data/<seq>.txt                    (from run_mot.py)

Restructures into TrackEval's MotChallenge2DBox layout:
  <gt_root>/<BENCHMARK>/<SPLIT>/<seq>/
  <gt_root>/seqmaps/<BENCHMARK>-<SPLIT>.txt
  <trackers_root>/<BENCHMARK>-<SPLIT>/<tracker>/data/<seq>.txt

then invokes trackeval programmatically.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import List

# TrackEval (current release) uses np.float / np.int / np.bool which were removed
# in numpy 2.x. Monkey-patch before trackeval is imported.
import numpy as _np
for _alias, _real in (("float", float), ("int", int), ("bool", bool)):
    if not hasattr(_np, _alias):
        setattr(_np, _alias, _real)


def load_gt_frames(seq_dir: Path) -> set:
    """Frame indices that carry at least one valid GT row."""
    frames = set()
    gt_file = seq_dir / "gt" / "gt.txt"
    with open(gt_file, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            if len(parts) >= 7 and float(parts[6]) <= 0.0:
                continue
            frames.add(int(float(parts[0])))
    return frames


def write_filtered_result(src: Path, dst: Path, keep_frames: set) -> tuple:
    """Copy tracker result keeping only rows on annotated frames. Returns (kept, dropped)."""
    kept = dropped = 0
    with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            if int(float(parts[0])) in keep_frames:
                fout.write(line if line.endswith("\n") else line + "\n")
                kept += 1
            else:
                dropped += 1
    return kept, dropped


def ensure_layout(cp_mot_dir: Path, mot_results_dir: Path, benchmark: str, split: str,
                  tracker: str, sequences: List[str],
                  annotated_frames_only: bool = False) -> dict:
    """Materialize the TrackEval-expected folder layout via symlinks (no data copy)."""
    gt_root = cp_mot_dir
    tr_root = mot_results_dir

    bench_split_dir = gt_root / f"{benchmark}-{split}"
    bench_split_dir.mkdir(parents=True, exist_ok=True)
    for seq in sequences:
        src = cp_mot_dir / seq
        dst = bench_split_dir / seq
        if not src.exists():
            raise FileNotFoundError(f"Missing converted seq: {src}")
        if dst.is_symlink():
            dst.unlink()
        elif dst.is_dir():
            # Materialized copy (e.g. symlink-dereferencing rsync); replace with fresh symlink.
            shutil.rmtree(dst)
        elif dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve(), target_is_directory=True)

    seqmaps_dir = gt_root / "seqmaps"
    seqmaps_dir.mkdir(parents=True, exist_ok=True)
    seqmap_file = seqmaps_dir / f"{benchmark}-{split}.txt"
    with open(seqmap_file, "w", encoding="utf-8") as fh:
        fh.write("name\n")
        for seq in sequences:
            fh.write(f"{seq}\n")

    tr_bench_dir = tr_root / f"{benchmark}-{split}" / tracker / "data"
    tr_bench_dir.mkdir(parents=True, exist_ok=True)
    for seq in sequences:
        src = tr_root / tracker / "data" / f"{seq}.txt"
        dst = tr_bench_dir / f"{seq}.txt"
        if not src.exists():
            raise FileNotFoundError(f"Missing tracker output: {src}")
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        if annotated_frames_only:
            keep = load_gt_frames(cp_mot_dir / seq)
            kept, dropped = write_filtered_result(src, dst, keep)
            print(f"[eval] {seq}: annotated-frames-only filter kept {kept}, "
                  f"dropped {dropped} rows ({len(keep)} GT frames)")
        else:
            dst.symlink_to(src.resolve())

    return {
        "GT_FOLDER": str(gt_root.resolve()),
        "TRACKERS_FOLDER": str(tr_root.resolve()),
        "SEQMAP_FILE": str(seqmap_file.resolve()),
    }


def run_eval(paths: dict, benchmark: str, split: str, tracker: str, sequences: List[str]) -> None:
    import trackeval

    eval_config = {
        "USE_PARALLEL": False,
        "NUM_PARALLEL_CORES": 1,
        "BREAK_ON_ERROR": True,
        "RETURN_ON_ERROR": False,
        "LOG_ON_ERROR": os.path.join(paths["TRACKERS_FOLDER"], "trackeval_error.log"),
        "PRINT_RESULTS": True,
        "PRINT_ONLY_COMBINED": False,
        "PRINT_CONFIG": False,
        "TIME_PROGRESS": True,
        "DISPLAY_LESS_PROGRESS": True,
        "OUTPUT_SUMMARY": True,
        "OUTPUT_EMPTY_CLASSES": False,
        "OUTPUT_DETAILED": True,
        "PLOT_CURVES": False,
    }

    dataset_config = {
        "GT_FOLDER": paths["GT_FOLDER"],
        "TRACKERS_FOLDER": paths["TRACKERS_FOLDER"],
        "OUTPUT_FOLDER": None,
        "TRACKERS_TO_EVAL": [tracker],
        "CLASSES_TO_EVAL": ["pedestrian"],
        "BENCHMARK": benchmark,
        "SPLIT_TO_EVAL": split,
        "INPUT_AS_ZIP": False,
        "PRINT_CONFIG": False,
        "DO_PREPROC": False,
        "TRACKER_SUB_FOLDER": "data",
        "OUTPUT_SUB_FOLDER": "",
        "TRACKER_DISPLAY_NAMES": None,
        "SEQMAP_FOLDER": None,
        "SEQMAP_FILE": paths["SEQMAP_FILE"],
        "SEQ_INFO": None,
        "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt/gt.txt",
        "SKIP_SPLIT_FOL": False,
    }

    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    evaluator.evaluate([dataset], metrics)


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate FaceQSORT on ChokePoint using TrackEval.")
    p.add_argument("--cp-mot-dir", type=str, default="chokepoint_mot")
    p.add_argument("--mot-results-dir", type=str, default="mot_results")
    p.add_argument("--tracker", type=str, default="FaceQSORT")
    p.add_argument("--benchmark", type=str, default="ChokePoint")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--sequences", type=str, nargs="+",
                   default=["P1E_S1_C1", "P1E_S1_C2", "P1E_S1_C3"])
    p.add_argument("--annotated-frames-only", action="store_true",
                   help="Protocol option: evaluate only on frames that have GT annotations "
                        "(eye-coordinate GT covers only near-frontal views). Uses GT frame "
                        "indices, never GT boxes. Apply uniformly to all trackers.")
    args = p.parse_args()

    cp_mot_dir = Path(args.cp_mot_dir).resolve()
    mot_results_dir = Path(args.mot_results_dir).resolve()

    paths = ensure_layout(cp_mot_dir, mot_results_dir, args.benchmark, args.split,
                          args.tracker, args.sequences,
                          annotated_frames_only=args.annotated_frames_only)
    print("[eval] layout ready:")
    for k, v in paths.items():
        print(f"  {k} = {v}")
    run_eval(paths, args.benchmark, args.split, args.tracker, args.sequences)


if __name__ == "__main__":
    main()
