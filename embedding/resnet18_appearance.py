from pathlib import Path
from typing import Optional

import torch
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights

from fq_utils.device_utils import resolve_device

class ResNet18Appearance:
    def __init__(self, device=None, weights_path: Optional[str] = None):
        self.device = resolve_device(device)

        weights_enum = ResNet18_Weights.IMAGENET1K_V1
        try:
            m = resnet18(weights=weights_enum)
        except Exception:
            # Offline/permission fallback: initialize model without downloading weights.
            m = resnet18(weights=None)

        if weights_path is not None:
            p = Path(weights_path).expanduser()
            if p.exists():
                state_dict = torch.load(str(p), map_location="cpu")
                if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
                    state_dict = state_dict["state_dict"]
                if isinstance(state_dict, dict):
                    m.load_state_dict(state_dict, strict=False)

        meta = getattr(weights_enum, "meta", {}) if weights_enum is not None else {}
        mean = meta.get("mean", [0.485, 0.456, 0.406])
        std = meta.get("std", [0.229, 0.224, 0.225])

        self.backbone = torch.nn.Sequential(*list(m.children())[:-1]).to(self.device).eval()  # pool output
        self.tf = T.Compose([
            T.ToPILImage(),
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    @torch.no_grad()
    def __call__(self, face_crop_bgr):
        # face_crop_bgr: HxWx3 uint8 (OpenCV)
        x = self.tf(face_crop_bgr[..., ::-1].copy()).unsqueeze(0).to(self.device)  # BGR->RGB
        feat = self.backbone(x).flatten(1)  # (1,512)
        return feat[0].detach().cpu().numpy().astype("float32")
