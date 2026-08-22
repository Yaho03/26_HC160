from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

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


@dataclass(frozen=True)
class ONNXPADConfig:
    """Input/output contract for an ONNX binary PAD model.

    Defaults match Open Model Zoo's original anti-spoof-mn3 ONNX artifact.
    Its class 0 is bona fide and class 1 is spoof.
    """

    model_path: str
    model_version: str
    input_size: int = 128
    live_class_index: int = 0
    output_kind: str = "probability"
    input_name: str = "actual_input_1"
    output_name: str | None = None
    mean: tuple[float, float, float] = (151.2405, 119.5950, 107.8395)
    scale: tuple[float, float, float] = (63.0105, 56.4570, 55.0035)


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

    def metadata(self) -> dict[str, Any]:
        return {
            "runtime": "torchscript",
            "version": self.config.model_version,
            "input_size": self.config.input_size,
            "output_kind": self.config.output_kind,
            "live_class_index": self.config.live_class_index,
            "preprocessing": {
                "color_order": "RGB",
                "pixel_scale": 255.0,
                "mean": list(self.config.mean),
                "divisor": list(self.config.std),
            },
        }


class ONNXPADScorer:
    """Fail-closed ONNX adapter for calibrated binary PAD models."""

    def __init__(
        self,
        config: ONNXPADConfig,
        *,
        providers: Sequence[str] | None = None,
        session=None,
    ) -> None:
        _validate_config(config)
        self.config = config
        if session is None:
            try:
                import onnxruntime as ort
            except ImportError as error:
                raise RuntimeError(
                    "ONNX PAD runtime is unavailable; install requirements-pad-onnx.txt"
                ) from error
            selected = list(providers or ("CPUExecutionProvider",))
            session = ort.InferenceSession(config.model_path, providers=selected)
        self.session = session
        self.providers = tuple(
            self.session.get_providers()
            if hasattr(self.session, "get_providers")
            else providers or ()
        )
        inputs = {item.name for item in self.session.get_inputs()}
        if config.input_name not in inputs:
            raise ValueError(
                f"PAD model input {config.input_name!r} not found; available={sorted(inputs)}"
            )
        outputs = [item.name for item in self.session.get_outputs()]
        if not outputs:
            raise ValueError("PAD model has no outputs")
        self.output_name = config.output_name or outputs[0]
        if self.output_name not in outputs:
            raise ValueError(
                f"PAD model output {self.output_name!r} not found; available={outputs}"
            )

    @property
    def model_version(self) -> str:
        return self.config.model_version

    def score(self, crops: list[Image.Image]) -> list[float]:
        values = []
        for crop in crops:
            output = self.session.run(
                [self.output_name],
                {self.config.input_name: self._array(crop)[None, ...]},
            )[0]
            values.append(_live_probability(output, self.config))
        return values

    def _array(self, image: Image.Image) -> np.ndarray:
        resized = image.convert("RGB").resize(
            (self.config.input_size, self.config.input_size), Image.BILINEAR
        )
        array = np.asarray(resized, dtype=np.float32)
        array = (array - np.asarray(self.config.mean, dtype=np.float32)) / np.asarray(
            self.config.scale, dtype=np.float32
        )
        return np.ascontiguousarray(array.transpose(2, 0, 1), dtype=np.float32)

    def metadata(self) -> dict[str, Any]:
        return {
            "runtime": "onnx",
            "version": self.config.model_version,
            "input_size": self.config.input_size,
            "input_name": self.config.input_name,
            "output_name": self.output_name,
            "output_kind": self.config.output_kind,
            "live_class_index": self.config.live_class_index,
            "providers": list(self.providers),
            "preprocessing": {
                "color_order": "RGB",
                "pixel_scale": 1.0,
                "mean": list(self.config.mean),
                "divisor": list(self.config.scale),
            },
        }


def create_pad_scorer(
    *,
    runtime: str,
    model_path: str,
    model_version: str,
    input_size: int | None = None,
    live_class_index: int | None = None,
    output_kind: str | None = None,
    device: str = "cpu",
    providers: Sequence[str] | None = None,
):
    """Create a PAD scorer with runtime-specific, explicit safe defaults."""

    if runtime == "torchscript":
        return TorchScriptPADScorer(
            TorchScriptPADConfig(
                model_path=model_path,
                model_version=model_version,
                input_size=224 if input_size is None else input_size,
                live_class_index=1
                if live_class_index is None
                else live_class_index,
                output_kind="logits" if output_kind is None else output_kind,
            ),
            device=device,
        )
    if runtime == "onnx":
        return ONNXPADScorer(
            ONNXPADConfig(
                model_path=model_path,
                model_version=model_version,
                input_size=128 if input_size is None else input_size,
                live_class_index=0
                if live_class_index is None
                else live_class_index,
                output_kind="probability" if output_kind is None else output_kind,
            ),
            providers=providers,
        )
    raise ValueError(f"Unsupported PAD runtime: {runtime}")


def _validate_config(config: ONNXPADConfig) -> None:
    if config.output_kind not in {"logits", "probability"}:
        raise ValueError("output_kind must be 'logits' or 'probability'")
    if config.input_size <= 0:
        raise ValueError("input_size must be positive")
    if config.live_class_index < 0:
        raise ValueError("live_class_index must not be negative")
    if any(value == 0 for value in config.scale):
        raise ValueError("PAD preprocessing scale values must be non-zero")


def _live_probability(output, config: ONNXPADConfig) -> float:
    values = np.asarray(output, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("PAD model output is empty")
    if values.size == 1:
        if config.live_class_index != 0:
            raise ValueError("Single-output PAD model requires live_class_index=0")
        value = float(values[0])
        if config.output_kind == "logits":
            value = float(1.0 / (1.0 + np.exp(-value)))
    else:
        if config.live_class_index >= values.size:
            raise ValueError("PAD live_class_index is outside the model output")
        if config.output_kind == "logits":
            shifted = values - np.max(values)
            probabilities = np.exp(shifted) / np.exp(shifted).sum()
            value = float(probabilities[config.live_class_index])
        else:
            value = float(values[config.live_class_index])
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("PAD model output must be a finite probability in [0, 1]")
    return value
