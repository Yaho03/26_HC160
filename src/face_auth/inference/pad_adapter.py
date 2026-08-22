from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class TorchScriptPADConfig:
    model_path: str
    model_version: str
    input_size: int = 224
    live_class_index: int = 1
    output_kind: str = "logits"
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


class TorchScriptPADScorer:
    """Adapter for a calibrated TorchScript binary PAD model.

    The model and threshold must be validated on the target cameras. This
    adapter deliberately does not provide a heuristic fallback.
    """

    def __init__(self, config: TorchScriptPADConfig, *, device: str = "cpu") -> None:
        import torch

        if config.output_kind not in {"logits", "probability"}:
            raise ValueError("output_kind must be 'logits' or 'probability'")
        self.config = config
        self.device = torch.device(device)
        self.model = torch.jit.load(config.model_path, map_location=self.device)
        self.model.eval()

    @property
    def model_version(self) -> str:
        return self.config.model_version

    def score(self, crops: list[Image.Image]) -> list[float]:
        if not crops:
            return []
        import torch

        batch = torch.stack([self._tensor(image) for image in crops]).to(self.device)
        with torch.no_grad():
            output = self.model(batch)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if output.ndim == 1:
            output = output[:, None]

        if self.config.output_kind == "logits":
            if output.shape[1] == 1:
                scores = torch.sigmoid(output[:, 0])
            else:
                scores = torch.softmax(output, dim=1)[:, self.config.live_class_index]
        else:
            index = 0 if output.shape[1] == 1 else self.config.live_class_index
            scores = output[:, index]
        values = [float(value) for value in scores.detach().cpu().tolist()]
        if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
            raise ValueError("PAD model output must be a finite probability in [0, 1]")
        return values

    def _tensor(self, image: Image.Image):
        import torch

        resized = image.convert("RGB").resize(
            (self.config.input_size, self.config.input_size), Image.BILINEAR
        )
        array = np.asarray(resized, dtype=np.float32) / 255.0
        array = (array - np.asarray(self.config.mean)) / np.asarray(self.config.std)
        return torch.from_numpy(array.transpose(2, 0, 1)).float()
