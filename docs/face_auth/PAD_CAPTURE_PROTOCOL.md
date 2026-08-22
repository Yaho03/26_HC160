# Physical PAD Capture Protocol

## Purpose

This protocol produces pseudonymous, source-bound videos for calibrating and testing presentation attack detection. It does not make `anti-spoof-mn3` a validated financial-security model by itself. The model remains `candidate_unvalidated` until held-out physical results are reviewed.

Raw videos and model weights stay under ignored `local_*` directories. Do not commit faces, names, student IDs, phone numbers, or a table that maps tokens back to people.

## 1. Verify the candidate model

Download the artifact from the HTTPS URL recorded in `configs/models/anti-spoof-mn3.json`, then verify it before inference:

```bash
python -m src.face_auth.inference.pad_model_cli \
  --registry configs/models/anti-spoof-mn3.json \
  --model local_models/anti-spoof-mn3/anti-spoof-mn3.onnx
```

The command must report `verified: true`. The registry fixes the source, MIT license reference, byte count, SHA-256, RGB `128x128` input preprocessing, and class `0` as bona fide. A checksum mismatch stops the run.

## 2. Split subjects before recording

Assign opaque tokens before opening the camera:

- `subject_*`: one person; a token must appear in calibration or test, never both.
- `session_*`: one recorded clip; never reuse it.
- `device_*`: the capture camera, not the display used for an attack.
- `sample_*`: one manifest row and one video.

Use different capture devices between splits when evaluating device generalization and pass `--require-device-disjoint`. Keep the private token-to-consent record outside the repository.

## 3. Capture matrix

Record each cell in separate sessions. Start with a pilot, inspect exclusions and confidence intervals, then expand the weak cells rather than claiming accuracy from a tiny balanced sample.

| Label | Attack species | Presentation |
|---|---|---|
| `bona_fide` | `none` | Live person under normal pose and illumination variation. |
| `attack` | `print` | Printed face presented flat and with natural hand motion. |
| `attack` | `screen_static` | Static face image displayed on a phone or monitor. |
| `attack` | `screen_replay` | Previously recorded face video replayed on a display. |
| `attack` | `adversarial_print` | Approved adversarial sample transferred to print and recaptured. |
| `attack` | `adversarial_screen` | Approved adversarial sample displayed and recaptured. |

Vary distance, angle, print finish, display brightness, ambient light, and capture device deliberately. Keep those conditions in a separate non-identifying experiment log; do not encode them into the fixed manifest columns.

## 4. Record one clip

Example calibration print capture:

```bash
python -m src.face_auth.evaluation.pad_capture_cli \
  --artifact-root local_pad_dataset/pad-v1 \
  --manifest local_pad_dataset/pad-v1/manifest.csv \
  --sample-id sample_cal_print_0001 \
  --label attack \
  --attack-species print \
  --subject-token subject_cal_0001 \
  --session-token session_cal_print_0001 \
  --device-token device_webcam_0001 \
  --split calibration \
  --camera 0 \
  --duration-seconds 5 \
  --fps 15
```

The tool creates a deterministic relative path such as `calibration/print/sample_cal_print_0001.mp4`. It refuses overwrites and split leakage. A short or interrupted recording leaves no manifest row.

For bona fide clips, use `--label bona_fide --attack-species none`. Test captures must use test-only subject tokens and `--split test`.

## 5. Evaluation order

1. Run model verification and preserve its output with the experiment record.
2. Finish the calibration split before choosing a threshold.
3. Freeze the threshold artifact under a versioned name.
4. Evaluate the test split once without retuning.
5. Report per-species APCER, BPCER, ACER, Wilson intervals, latency, `NOT_EVALUATED`, and `ERROR` counts.

Do not describe a low APCER as meaningful when many attacks were excluded before PAD scoring. Keep adversarial print and screen transfer results separate from digital-only attacks.
