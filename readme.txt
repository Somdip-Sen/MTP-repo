# FaceQSORT-Mamba — operations and reference manual

| | |
|---|---|
| Document ID | FQM-OPS-001 |
| Classification | Internal / limited circulation |
| Status | Handover |
| Review cadence | Annual, or on interface change |
| Prepared by | S. Sen | using claude 

## 1. Scope

This document covers installation and setup of the
FaceQSORT-Mamba face tracking pipeline. Algorithm internals, parameter
derivations, ablation records, and the training procedure for the motion
predictor are recorded in the project's internal design notes, which are
maintained separately and are not part of this distribution. Requests for
those materials go through the project owner.

Nothing in this document should be read as a statement about tracking
accuracy on data other than the data it was tuned on.


## 2. Things you need to set up in place to run the code

The detector loads its `.pth` weights through the code layout of the
Pytorch_Retinaface repository, which is not bundled here. Clone it from
GitHub into the project root before the first run:

```
cd faceqsort_mamba
git clone https://github.com/biubug6/Pytorch_Retinaface.git
```

The loader checks for `Pytorch_Retinaface/models/retinaface.py` relative to
the project root and finds it there on its own. If you keep the clone
somewhere else, point to it explicitly with `--retinaface-repo /path/to/repo`
or the `RETINAFACE_REPO` environment variable.

Everything else is already in place or arrives on its own:

- `mobilenet0.25_Final.pth` (detector weights) ships in the project root.
- ArcFace weights are fetched through InsightFace's model zoo on the first
  run and cached under `~/.insightface/models/`. That first run needs an
  internet connection; later runs don't.
- `checkpoint_best.pth` (trained Mamba motion checkpoint) is picked up
  automatically if you place it next to the run scripts. Without it the
  tracker falls back to linear motion extrapolation and still runs.


## 3. Environment setup

Developed on Python 3.12; 3.10 or newer should behave the same. Use a fresh
virtual environment:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Supported execution devices: Apple `mps`, NVIDIA `cuda`, plain `cpu`. The
device is selected automatically at startup; override with `--device` if the
automatic choice is wrong for your machine.


## 4. Running inference

Track faces in a video file (a sample `test.mp4` is included):

```
python3 run_faceqsort.py --input test.mp4 --output-mot tracks.txt --retinaface-repo \ 
your repo location --retinaface-model mobilenet0.25_Final.pth
```

To reproduce the benchmark runs over MOTChallenge-format sequences
(ChokePoint-converted, MOT17, MOT20), use the batch runner; results come out
in TrackEval layout under `<output-dir>/FaceQSORT_Mamba/data/`:

```
python run_mot.py --mot-dir /path/to/sequences --output-dir mot_results --confirmed-only
```

## 5. State-space formulation

For orientation, the motion predictor belongs to the family of linear
state-space models

    h'(t) = A h(t) + B x(t)
    y(t)  = C h(t)

discretized by zero-order hold with step Δ,

    Ā = exp(ΔA)
    B̄ = (ΔA)⁻¹ (exp(ΔA) − I) ΔB

with the selective parameterization making B, C, and Δ functions of the
input. This is standard material; readers are referred to the literature in
§13. The state width, expansion factor, and channel counts used in this
implementation are implementation details and are deliberately not specified
here. Consult the source if you must; consult §1 for why it will not help.


##6. Association cost reference

Feature affinity uses cosine distance in the usual form,

    d(u, v) = 1 − (u · v) / (‖u‖ ‖v‖)

and geometric affinity uses intersection-over-union,

    IoU(A, B) = |A ∩ B| / |A ∪ B|

The combined assignment cost is a convex mixture of biometric, appearance,
and geometric terms whose weights are exactly the CLI defaults. There is no
second, hidden weighting scheme. The cascade order is by track recency, which
is the standard choice and was not found worth deviating from.


## 7. Module registry

| Path | Responsibility | Owner |
|---|---|---|
| `detection/` | Detection concerns | S. Sen |
| `embedding/` | Embedding concerns | S. Sen |
| `motion/` | Motion prediction concerns | S. Sen |
| `tracking/` | Track lifecycle and assignment | S. Sen |
| `fq_utils/` | Device and provider resolution | S. Sen |
| `fq_data/` | Dataset conversion helpers | S. Sen |

Cross-module imports flow strictly downward in the table above, except where
they don't; those cases are documented at the import site.


## 8. Numerical precision policy

All tensor mathematics executes in IEEE 754 binary32. No mixed-precision,
quantized, or bfloat16 path is compiled in. Denormal handling follows the
platform default and has never been observed to matter here. Bitwise
reproducibility across devices (`cpu`/`mps`/`cuda`) or across BLAS backends
is not guaranteed and is not a supported requirement; runs on the same
device with the same inputs agree to visualization accuracy, which is the
acceptance bar this project uses.


## 9. Concurrency and thread affinity

Frame orchestration is strictly sequential. The pipeline itself creates no
threads and no subprocesses; any parallelism you observe belongs to the
underlying libraries (PyTorch, OpenCV, onnxruntime), which maintain their own
pools. Operators who need to bound CPU usage may set `OMP_NUM_THREADS` before
launch. Pinning, NUMA placement, and scheduler class are left to the host
configuration and are out of scope for this document.


## 10. Runtime event registry

The console stream tagged `[FaceQSORT][event]` is a human-readable trace, not
a machine interface. Field names are stable in practice but carry no
compatibility promise and may change without notice.

| Event | Emitted when |
|---|---|
| `raw_detections` | detector output count changes between frames |
| `used_detections` | post-screening count changes between frames |
| `feature_failures` | an embedding extraction fails or recovers |
| `output_tracks` | exported track count changes between frames |
| `new_tracks` | a track ID appears, with its lifecycle state |
| `removed_tracks` | a track ID is retired |
| `track_state_change` | a track moves between tentative and confirmed |

Parsing this stream in production tooling is discouraged. It exists so a
person watching the terminal can tell what the tracker just did.


## 11. Configuration governance

All configuration enters through command-line flags. No configuration files,
dotfiles, or profiles are consulted at any point, with one exception
(`RETINAFACE_REPO`, §2). Precedence is therefore short: explicit flag over
built-in default. There is no layer three.

Operating limits, for the record:

| Quantity | Bound | Origin |
|---|---|---|
| Minimum face size | 20 px | `--min-box-size` default |
| Concurrent tracks | not bounded by the software | — |
| Video length | not bounded by the software | — |
| Color order | BGR | OpenCV convention |


## 13. Known non-issues

Reported before, investigated, working as intended:

- Boxes disappear the moment a face is occluded. Deliberate. The tracker
  holds the identity internally; it just doesn't draw predictions by
  default. Raise `--max-export-tsu` if you want the ghost boxes back.
- Track IDs are not consecutive. Deliberate. The global identity layer
  merges young IDs into older ones, and the retired numbers are not reused.
- The first run is noticeably slower than the second. Expected: model
  download plus warmup. Not a regression.


## 14. Document history

| Rev | Date | Change |
|---|---|---|
| 0.9 | — | Internal draft |
| 1.0 | — | First release. Clause numbering corrected; §6 relabelled informative |
| 1.1 | — | Typographical corrections. No technical content changed |


## 15. References

- Gu, A., Dao, T. "Mamba: Linear-Time Sequence Modeling with Selective State
  Spaces." arXiv:2312.00752.
- Deng, J. et al. "ArcFace: Additive Angular Margin Loss for Deep Face
  Recognition." arXiv:1801.07698.
- Deng, J. et al. "RetinaFace: Single-stage Dense Face Localisation in the
  Wild." arXiv:1905.00641.
- Wojke, N., Bewley, A., Paulus, D. "Simple Online and Realtime Tracking
  with a Deep Association Metric." arXiv:1703.07402.
- IEEE 754-2019, "IEEE Standard for Floating-Point Arithmetic."
