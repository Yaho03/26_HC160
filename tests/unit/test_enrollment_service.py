import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.face_auth.application.enrollment_service import (
    EnrollmentTemplate,
    load_template,
    save_template,
)


class EnrollmentTemplateStorageTest(unittest.TestCase):
    def test_round_trip_preserves_embedding_and_versions(self):
        template = EnrollmentTemplate(
            embedding=np.array([0.6, 0.8], dtype=np.float32),
            template_version="template-test",
            model_version="model-test",
            alignment_version="alignment-test",
            created_at=datetime(2026, 8, 22, tzinfo=timezone.utc),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "template.npz"
            save_template(template, path)
            loaded = load_template(path)
        np.testing.assert_allclose(loaded.embedding, template.embedding)
        self.assertEqual(loaded.template_version, template.template_version)
        self.assertEqual(loaded.model_version, template.model_version)
        self.assertEqual(loaded.alignment_version, template.alignment_version)


if __name__ == "__main__":
    unittest.main()
