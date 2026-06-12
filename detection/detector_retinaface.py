from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from fq_utils.device_utils import insightface_ctx_id, insightface_providers, resolve_device


@dataclass
class FaceDetection:
    bbox_xyxy: np.ndarray
    conf: float
    kps: Optional[np.ndarray]


class _TorchRetinaFaceBackend:
    """
    Pytorch_Retinaface (.pth) backend for RetinaFace MobileNet-0.25 / ResNet50.
    Uses local repo code from biubug6/Pytorch_Retinaface-compatible layout.
    """

    def __init__(
        self,
        weights_path: str,
        device: str,
        det_thresh: float,
        nms_thresh: float,
        top_k: int,
        keep_top_k: int,
        network: Optional[str],
        retinaface_repo: Optional[str],
    ):
        self.weights_path = str(Path(weights_path).expanduser().resolve())
        self.device = resolve_device(device)
        self.det_thresh = float(det_thresh)
        self.nms_thresh = float(nms_thresh)
        self.top_k = int(top_k)
        self.keep_top_k = int(keep_top_k)
        self.repo_root = self._resolve_repo_root(retinaface_repo)

        self._load_repo_modules()
        self._load_model(network=network)
        self._priors_cache: Dict[Tuple[int, int], object] = {}

    def _resolve_repo_root(self, retinaface_repo: Optional[str]) -> str:
        candidates: List[Path] = []
        if retinaface_repo:
            candidates.append(Path(retinaface_repo).expanduser())

        env_repo = os.environ.get("RETINAFACE_REPO")
        if env_repo:
            candidates.append(Path(env_repo).expanduser())

        workspace_root = Path(__file__).resolve().parent.parent
        # Common local layouts used around this project.
        candidates.extend(
            [
                workspace_root / "Pytorch_Retinaface",
                workspace_root / "My_work" / "Pytorch_Retinaface_my_work",
                workspace_root.parent / "Extra" / "My_work" / "Pytorch_Retinaface_my_work",
            ]
        )

        for cand in candidates:
            if (cand / "models" / "retinaface.py").exists() and (cand / "data" / "__init__.py").exists():
                return str(cand.resolve())

        raise RuntimeError(
            "Could not find Pytorch_Retinaface repo. Set --retinaface-repo or RETINAFACE_REPO "
            "to a directory containing models/retinaface.py and data/__init__.py."
        )

    def _load_repo_modules(self) -> None:
        repo = self.repo_root
        if repo not in sys.path:
            sys.path.insert(0, repo)

        try:
            import torch
            from data import cfg_mnet, cfg_re50
            from layers.functions.prior_box import PriorBox
            from models.retinaface import RetinaFace
            from utils.box_utils import decode, decode_landm
            from utils.nms.py_cpu_nms import py_cpu_nms
        except Exception as exc:
            raise RuntimeError(
                f"Failed importing RetinaFace modules from repo: {repo}. "
                "Verify the repository and dependencies."
            ) from exc

        self.torch = torch
        self.cfg_mnet = cfg_mnet
        self.cfg_re50 = cfg_re50
        self.PriorBox = PriorBox
        self.RetinaFace = RetinaFace
        self.decode = decode
        self.decode_landm = decode_landm
        self.py_cpu_nms = py_cpu_nms

    def _get_torch_device(self):
        torch = self.torch
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if self.device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _remove_prefix(state_dict: Dict, prefix: str) -> Dict:
        out: Dict = {}
        for key, value in state_dict.items():
            if key.startswith(prefix):
                out[key[len(prefix):]] = value
            else:
                out[key] = value
        return out

    def _infer_network_from_weights(self) -> str:
        name = Path(self.weights_path).name.lower()
        if "mobilenet" in name or "mobile0.25" in name or "mnet" in name:
            return "mobile0.25"
        if "resnet" in name or "re50" in name:
            return "resnet50"
        return "mobile0.25"

    def _load_model(self, network: Optional[str]) -> None:
        torch = self.torch
        if not Path(self.weights_path).exists():
            raise FileNotFoundError(f"RetinaFace weights file not found: {self.weights_path}")

        net_name = network if network else self._infer_network_from_weights()
        if net_name not in {"mobile0.25", "resnet50"}:
            raise ValueError(f"Unsupported retinaface network '{net_name}'. Use mobile0.25 or resnet50.")

        cfg_src = self.cfg_mnet if net_name == "mobile0.25" else self.cfg_re50
        cfg = dict(cfg_src)
        cfg["pretrain"] = False
        model = self.RetinaFace(cfg=cfg, phase="test")

        pretrained = torch.load(self.weights_path, map_location="cpu")
        if isinstance(pretrained, dict) and "state_dict" in pretrained:
            pretrained = pretrained["state_dict"]
        if not isinstance(pretrained, dict):
            raise RuntimeError(f"Unexpected checkpoint format for RetinaFace weights: {type(pretrained)}")
        pretrained = self._remove_prefix(pretrained, "module.")
        model.load_state_dict(pretrained, strict=False)
        model.eval()

        self.cfg = cfg
        self.torch_device = self._get_torch_device()
        self.model = model.to(self.torch_device)

    def _get_priors(self, h: int, w: int):
        key = (h, w)
        if key not in self._priors_cache:
            priors = self.PriorBox(self.cfg, image_size=(h, w)).forward().to(self.torch_device)
            self._priors_cache[key] = priors
        return self._priors_cache[key]

    def detect(self, frame_bgr: np.ndarray) -> List[FaceDetection]:
        torch = self.torch
        img = np.float32(frame_bgr)
        im_height, im_width, _ = img.shape

        scale = torch.tensor([im_width, im_height, im_width, im_height], dtype=torch.float32, device=self.torch_device)
        img -= (104.0, 117.0, 123.0)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0).to(self.torch_device)

        with torch.no_grad():
            loc, conf, landms = self.model(img)

        priors = self._get_priors(im_height, im_width)
        prior_data = priors.data

        boxes = self.decode(loc.data.squeeze(0), prior_data, self.cfg["variance"])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()

        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        landms = self.decode_landm(landms.data.squeeze(0), prior_data, self.cfg["variance"])
        scale1 = torch.tensor(
            [im_width, im_height, im_width, im_height, im_width, im_height, im_width, im_height, im_width, im_height],
            dtype=torch.float32,
            device=self.torch_device,
        )
        landms = landms * scale1
        landms = landms.cpu().numpy()

        inds = np.where(scores > self.det_thresh)[0]
        boxes = boxes[inds]
        scores = scores[inds]
        landms = landms[inds]
        if len(scores) == 0:
            return []

        order = scores.argsort()[::-1][: self.top_k]
        boxes = boxes[order]
        scores = scores[order]
        landms = landms[order]

        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = self.py_cpu_nms(dets, self.nms_thresh)
        dets = dets[keep, :]
        landms = landms[keep]

        dets = dets[: self.keep_top_k, :]
        landms = landms[: self.keep_top_k, :]

        out: List[FaceDetection] = []
        for idx in range(len(dets)):
            box = dets[idx]
            bbox = box[:4].astype(np.float32)
            conf_score = float(box[4])
            kps = landms[idx].reshape(5, 2).astype(np.float32) if idx < len(landms) else None
            out.append(FaceDetection(bbox_xyxy=bbox, conf=conf_score, kps=kps))
        return out


class RetinaFaceMobileNet025Detector:
    """
    RetinaFace detector wrapper.
    Preferred path: local .pth weights (Pytorch_Retinaface backend).
    Optional fallback: InsightFace ONNX model ID/path.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        model_name: str = "mobilenet0.25_Final.pth",
        model_root: Optional[str] = None,
        det_thresh: float = 0.5,
        nms_thresh: float = 0.4,
        input_size: Tuple[int, int] = (640, 640),
        retinaface_repo: Optional[str] = None,
        network: str = "mobile0.25",
        top_k: int = 5000,
        keep_top_k: int = 750,
    ):
        self.device = resolve_device(device)
        self.det_thresh = float(det_thresh)
        self.nms_thresh = float(nms_thresh)
        self.input_size = tuple(int(v) for v in input_size)
        self.model_root = str(Path(model_root).expanduser()) if model_root else None

        resolved_model = self._resolve_local_model_path(model_name)
        if resolved_model is not None and resolved_model.suffix.lower() == ".pth":
            self.backend = "retinaface_pth"
            self._torch_backend = _TorchRetinaFaceBackend(
                weights_path=str(resolved_model),
                device=self.device,
                det_thresh=self.det_thresh,
                nms_thresh=self.nms_thresh,
                top_k=top_k,
                keep_top_k=keep_top_k,
                network=network,
                retinaface_repo=retinaface_repo,
            )
            self.model_name = str(resolved_model)
            return

        self.backend = "insightface_onnx"
        self._detector = self._load_insightface_model(model_name)
        self.model_name = self._detector.model_file if hasattr(self._detector, "model_file") else model_name
        self._prepare_insightface_model()

    @staticmethod
    def _resolve_local_model_path(model_name: str) -> Optional[Path]:
        if not model_name:
            return None
        candidate = Path(model_name).expanduser()
        if candidate.exists():
            return candidate.resolve()
        cwd_candidate = Path.cwd() / model_name
        if cwd_candidate.exists():
            return cwd_candidate.resolve()
        return None

    def _load_insightface_model(self, model_name: str):
        try:
            from insightface.model_zoo import get_model
        except Exception as exc:
            raise RuntimeError(
                "InsightFace is required for ONNX detector mode. Install dependencies from requirements.txt."
            ) from exc

        kwargs: Dict = {"providers": insightface_providers(self.device)}
        if self.model_root is not None:
            kwargs["root"] = self.model_root
        model = get_model(model_name, **kwargs)
        if model is None:
            raise RuntimeError(
                f"Could not load detector model '{model_name}'. "
                "Provide a local RetinaFace .pth (mobilenet0.25_Final.pth) "
                "or a valid InsightFace ONNX model id/path."
            )
        if not hasattr(model, "prepare") or not hasattr(model, "detect"):
            raise RuntimeError(f"Loaded detector is invalid type: {type(model).__name__}")
        return model

    def _prepare_insightface_model(self) -> None:
        ctx_id = insightface_ctx_id(self.device)
        try:
            self._detector.prepare(
                ctx_id=ctx_id,
                nms=self.nms_thresh,
                det_thresh=self.det_thresh,
                input_size=self.input_size,
            )
        except TypeError:
            self._detector.prepare(ctx_id=ctx_id, nms=self.nms_thresh, input_size=self.input_size)

    def detect(self, frame_bgr: np.ndarray) -> List[FaceDetection]:
        if self.backend == "retinaface_pth":
            return self._torch_backend.detect(frame_bgr)

        try:
            bboxes, kpss = self._detector.detect(
                frame_bgr,
                threshold=self.det_thresh,
                max_num=0,
                metric="default",
            )
        except TypeError:
            bboxes, kpss = self._detector.detect(
                frame_bgr,
                threshold=self.det_thresh,
                max_num=0,
            )
        if bboxes is None or len(bboxes) == 0:
            return []

        out: List[FaceDetection] = []
        for idx in range(len(bboxes)):
            box = bboxes[idx]
            bbox = box[:4].astype(np.float32)
            conf = float(box[4]) if box.shape[0] > 4 else 1.0
            kps = None
            if kpss is not None and idx < len(kpss):
                kps = kpss[idx].astype(np.float32)
            out.append(FaceDetection(bbox_xyxy=bbox, conf=conf, kps=kps))
        return out
