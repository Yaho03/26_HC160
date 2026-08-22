# Face Authentication Smoke Test - 2026-08-22

## Inputs

- `outputs/test_videos/dynamic_normal.mp4`: synthetic moving-face baseline;
- `outputs/test_videos/static_attack.mp4`: one face image repeated as a static replay;
- CPU execution with cached MTCNN and FaceNet VGGFace2 weights;
- identity threshold `0.60` and blur threshold `10` were smoke-test values, not calibrated release thresholds.

## Results

| Check | Dynamic normal | Static replay |
|---|---:|---:|
| Single-face detection in sampled frames | pass | pass |
| FaceNet identity similarity | 0.9854 | 0.9846 |
| BASELINE_ONLY decision | VERIFIED | VERIFIED |
| Content-replay score | 0, PASS | 19, FAIL |
| Camera-motion score | 0.0426 | 0.00001 |
| Five-point yaw range | 2.95 to 10.22 | 6.44 to 6.78 |

## Interpretation

Identity similarity alone cannot separate this static replay from the matching user. `BASELINE_ONLY` is therefore only a development baseline. The repeated-content gate detects this exact synthetic replay, while random active liveness and a validated PAD model remain required for physical print/screen attacks that contain natural capture noise.

The old blur threshold `40` rejected every sampled frame in `dynamic_normal.mp4` (observed variance approximately 9.5 to 25.5). The CLI now exposes quality thresholds explicitly, but the release value must be selected from target-device validation data. The synthetic dynamic video also moves the whole source image by augmentation; its camera-motion result must not be treated as a real-device calibration result.

The scenario builder inserted eight static-attack frames at frame 20 and produced a 53-frame MP4. After codec-tolerant fingerprinting (`content-replay-v2`), the inserted segment produced a repeated-content run of 7 and was denied. The earlier exact-frame threshold missed this re-encoded segment, so codec transfer must remain part of validation.

## Remaining Blocker For FULL Claim

No validated PAD model artifact is present in the repository. The `FULL` CLI requires a TorchScript PAD model and fails closed without one. APCER/BPCER and clean false-reject measurements on subject/session/device-disjoint physical captures are required before the team reports a final security accuracy.
