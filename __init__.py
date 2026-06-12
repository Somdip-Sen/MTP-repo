"""
faceqsort_mamba — FaceQSORT-style multi-face tracker with a Mamba motion predictor.

Code-only release (Python files only). Trained weights, datasets, and two
third-party detector/baseline repositories are NOT bundled here.

PACKAGE LAYOUT (by pipeline component)
    detection/   detector_retinaface.py      RetinaFace face detector wrapper
    embedding/   arcface_embedder.py          ArcFace biometric embedding (ONNX)
                 resnet18_appearance.py       ResNet18 appearance embedding
    motion/      mamba_block.py               pure-PyTorch (Bi)Mamba S6 block
                 mamba_motion_predictor.py    Mamba Motion Predictor (Kalman drop-in)
    tracking/    faceqsort.py                 FaceQSORT association + global-ID layer
    fq_data/     chokepoint_to_mot.py         ChokePoint eye-coords -> MOT GT converter
    fq_utils/    device_utils.py              device / provider resolution
                 (fq_ prefix avoids clashing with the RetinaFace repo's own
                  data/ and utils/ packages injected onto sys.path at runtime)
    run_faceqsort.py    single-video pipeline runner
    run_mot.py          MOTChallenge-format sequence runner + exporter
    eval_chokepoint.py  TrackEval scoring for ChokePoint

RUNNING (from INSIDE this folder; imports are rooted at the folder itself):
    python run_faceqsort.py   --help
    python run_mot.py         --help
    python eval_chokepoint.py --help
    # equivalently: python -m run_faceqsort --help

EXTERNAL DEPENDENCIES NOT INCLUDED (clone/obtain separately):
    - Pytorch_Retinaface  (detector backbone code; pass via --retinaface-repo;
      injected onto sys.path at runtime by detection/detector_retinaface.py)
    - BoT-FaceSORT        (used only for MovieShot TrackEval scoring)
    - RetinaFace weights (mobilenet0.25_Final.pth / Resnet50_Final.pth),
      ArcFace ONNX (w600k_r50.onnx), trained MTP checkpoint (checkpoint_best.pth)
    - Python deps: torch, torchvision, opencv-python, numpy, scipy,
      onnxruntime, insightface, trackeval
"""
