import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from src.face_auth.inference.pad_adapter import (
    ONNXPADConfig,
    ONNXPADScorer,
    TorchScriptPADConfig,
    TorchScriptPADScorer,
    create_pad_scorer,
)


class _FixedPAD(torch.nn.Module):
    def forward(self, batch):
        scores = torch.tensor([[-2.0, 2.0]], dtype=batch.dtype, device=batch.device)
        return scores.repeat(batch.shape[0], 1)


class _Node:
    def __init__(self, name):
        self.name = name


class _FakeONNXSession:
    def __init__(self, output, *, input_name="actual_input_1", output_name="output1"):
        self.output = np.asarray(output, dtype=np.float32)
        self.input_name = input_name
        self.output_name = output_name
        self.feeds = []

    def get_inputs(self):
        return [_Node(self.input_name)]

    def get_outputs(self):
        return [_Node(self.output_name)]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, output_names, feed):
        self.feeds.append((output_names, feed))
        return [self.output]


class TorchScriptPADScorerTest(unittest.TestCase):
    def test_binary_logits_are_converted_to_live_probabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pad.ts"
            traced = torch.jit.trace(_FixedPAD(), torch.zeros(1, 3, 16, 16))
            traced.save(str(path))
            scorer = TorchScriptPADScorer(
                TorchScriptPADConfig(
                    model_path=str(path),
                    model_version="test-pad-v1",
                    input_size=16,
                )
            )
            values = scorer.score([Image.new("RGB", (24, 24))])
            self.assertEqual(len(values), 1)
            self.assertGreater(values[0], 0.95)

    def test_metadata_records_the_full_preprocessing_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pad.ts"
            traced = torch.jit.trace(_FixedPAD(), torch.zeros(1, 3, 16, 16))
            traced.save(str(path))
            scorer = TorchScriptPADScorer(
                TorchScriptPADConfig(str(path), "test-pad-v1", input_size=16)
            )
            metadata = scorer.metadata()
            self.assertEqual(metadata["runtime"], "torchscript")
            self.assertEqual(metadata["preprocessing"]["pixel_scale"], 255.0)


class ONNXPADScorerTest(unittest.TestCase):
    def test_open_model_zoo_defaults_select_class_zero_as_live(self):
        session = _FakeONNXSession([[0.91, 0.09]])
        scorer = ONNXPADScorer(
            ONNXPADConfig("anti-spoof-mn3.onnx", "omz-2022.1"),
            session=session,
        )

        values = scorer.score([Image.new("RGB", (32, 48), (255, 128, 0))])

        self.assertAlmostEqual(values[0], 0.91, places=5)
        _, feed = session.feeds[0]
        model_input = feed["actual_input_1"]
        self.assertEqual(model_input.shape, (1, 3, 128, 128))
        self.assertEqual(model_input.dtype, np.float32)
        self.assertAlmostEqual(
            float(model_input[0, 0, 0, 0]),
            (255.0 - 151.2405) / 63.0105,
            places=5,
        )
        self.assertEqual(scorer.metadata()["providers"], ["CPUExecutionProvider"])

    def test_logits_are_softmaxed_before_selecting_live_class(self):
        scorer = ONNXPADScorer(
            ONNXPADConfig(
                "pad.onnx", "test-v1", output_kind="logits", live_class_index=0
            ),
            session=_FakeONNXSession([[2.0, -2.0]]),
        )
        self.assertGreater(scorer.score([Image.new("RGB", (16, 16))])[0], 0.98)

    def test_invalid_probability_fails_closed(self):
        scorer = ONNXPADScorer(
            ONNXPADConfig("pad.onnx", "test-v1"),
            session=_FakeONNXSession([[1.2, -0.2]]),
        )
        with self.assertRaisesRegex(ValueError, "finite probability"):
            scorer.score([Image.new("RGB", (16, 16))])

    def test_missing_named_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not found"):
            ONNXPADScorer(
                ONNXPADConfig("pad.onnx", "test-v1"),
                session=_FakeONNXSession([[0.8, 0.2]], input_name="images"),
            )

    @patch("src.face_auth.inference.pad_adapter.ONNXPADScorer")
    def test_factory_uses_runtime_specific_onnx_defaults(self, scorer_class):
        create_pad_scorer(
            runtime="onnx",
            model_path="anti-spoof-mn3.onnx",
            model_version="omz-2022.1",
        )
        config = scorer_class.call_args.args[0]
        self.assertEqual(config.input_size, 128)
        self.assertEqual(config.live_class_index, 0)
        self.assertEqual(config.output_kind, "probability")


if __name__ == "__main__":
    unittest.main()
