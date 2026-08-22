import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from src.face_auth.inference.pad_adapter import (
    TorchScriptPADConfig,
    TorchScriptPADScorer,
)


class _FixedPAD(torch.nn.Module):
    def forward(self, batch):
        scores = torch.tensor([[-2.0, 2.0]], dtype=batch.dtype, device=batch.device)
        return scores.repeat(batch.shape[0], 1)


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


if __name__ == "__main__":
    unittest.main()
