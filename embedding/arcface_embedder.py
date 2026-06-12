from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from faceqsort_mamba.utils.device_utils import insightface_ctx_id, insightface_providers, resolve_device


def _l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x) + eps)


class ArcFaceEmbedder:
    """
    ArcFace biometric feature extractor using InsightFace ONNX models.
    """

    DEFAULT_CANDIDATES: Sequence[str] = (
        "w600k_r50",
        "arcface_r100_v1",
    )
    LOCAL_AUTO_CANDIDATES: Sequence[str] = (
        "~/.insightface/models/buffalo_l/w600k_r50.onnx",
        "~/.insightface/models/buffalo_sc/w600k_mbf.onnx",
    )

    def __init__(
        self,
        device: Optional[str] = None,
        model_name: str = "auto",
        model_root: Optional[str] = None,
        input_size: int = 112,
    ):
        self.device = resolve_device(device)
        self.input_size = int(input_size)
        self.model_root = str(Path(model_root).expanduser()) if model_root else None

        resolved_name = self._resolve_model_name(model_name)
        self._model = self._load_model(resolved_name)
        self._prepare_model()

    def _resolve_model_name(self, model_name: str) -> str:
        name = str(model_name or "auto").strip()
        if name.lower() != "auto":
            return name

        for candidate in self.LOCAL_AUTO_CANDIDATES:
            p = Path(candidate).expanduser()
            if p.exists():
                return str(p)

        return self.DEFAULT_CANDIDATES[0]

    def _looks_like_retinaface_checkpoint(self, pth_path: Path) -> bool:
        try:
            import torch
        except Exception:
            return False

        ckpt = torch.load(str(pth_path), map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            keys = list(ckpt["state_dict"].keys())
        elif isinstance(ckpt, dict):
            keys = list(ckpt.keys())
        else:
            return False

        markers = ("ClassHead", "BboxHead", "LandmarkHead", "fpn.", "ssh.", "module.fpn.")
        return any(any(marker in key for marker in markers) for key in keys)

    def _load_model(self, model_name: str):
        try:
            from insightface.model_zoo import get_model
        except Exception as exc:
            raise RuntimeError(
                "InsightFace is required for ArcFace embedder. Install dependencies from requirements.txt."
            ) from exc

        path_obj = Path(model_name).expanduser()
        if path_obj.suffix.lower() == ".pth":
            if path_obj.exists() and self._looks_like_retinaface_checkpoint(path_obj):
                raise RuntimeError(
                    f"Provided ArcFace model '{path_obj}' looks like a RetinaFace detector checkpoint. "
                    "Use an ArcFace ONNX model (e.g. ~/.insightface/models/buffalo_l/w600k_r50.onnx)."
                )
            raise RuntimeError(
                "ArcFace .pth checkpoints are not supported by this wrapper. "
                "Provide an ArcFace ONNX model path or an InsightFace model id."
            )

        providers = insightface_providers(self.device)
        candidates: List[str] = [model_name]
        for name in self.DEFAULT_CANDIDATES:
            if name not in candidates:
                candidates.append(name)

        errors: List[str] = []
        for name in candidates:
            try:
                kwargs: Dict = {"providers": providers}
                if self.model_root is not None:
                    kwargs["root"] = self.model_root
                model = get_model(name, **kwargs)
                if model is None:
                    errors.append(f"{name}: get_model returned None")
                    continue
                if not hasattr(model, "prepare") or not hasattr(model, "get_feat"):
                    errors.append(f"{name}: not an ArcFace recognizer ({type(model).__name__})")
                    continue
                return model
            except Exception as exc:  # noqa: PERF203
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
                continue

        raise RuntimeError(
            "Failed to load ArcFace model. "
            "Tried candidates: "
            + ", ".join(candidates)
            + ". Errors: "
            + " | ".join(errors)
        )

    def _prepare_model(self) -> None:
        ctx_id = insightface_ctx_id(self.device)
        self._model.prepare(ctx_id=ctx_id)

    def align_crop(self, frame_bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
        if kps is None or len(kps) < 5:
            raise ValueError("ArcFace requires 5-point landmarks from detector for alignment.")
        try:
            from insightface.utils import face_align
        except Exception as exc:
            raise RuntimeError("InsightFace face_align utility is not available.") from exc

        return face_align.norm_crop(frame_bgr, landmark=kps, image_size=self.input_size)

    def extract_from_aligned(self, aligned_bgr: np.ndarray) -> np.ndarray:
        feat = self._model.get_feat(aligned_bgr)
        if feat.ndim == 2:
            feat = feat[0]
        feat = feat.astype(np.float32)
        return _l2norm(feat)

    def extract_from_crop(self, crop_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(crop_bgr, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        return self.extract_from_aligned(resized)

    def __call__(self, frame_bgr: np.ndarray, kps: Optional[np.ndarray], bbox_xyxy: np.ndarray) -> np.ndarray:
        if kps is not None and len(kps) >= 5:
            aligned = self.align_crop(frame_bgr, kps)
            return self.extract_from_aligned(aligned)

        x1, y1, x2, y2 = bbox_xyxy.astype(int).tolist()
        h, w = frame_bgr.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid bounding box for ArcFace crop extraction.")
        crop = frame_bgr[y1:y2, x1:x2]
        return self.extract_from_crop(crop)
