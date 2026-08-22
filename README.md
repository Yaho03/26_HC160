# 26_HC160

한이음 HC160 프로젝트: 얼굴 인증/인식 모델을 대상으로 targeted adversarial attack,
방어, 탐지, 재현 가능한 실험 계약을 정리하는 연구 저장소다.

## 협업 및 문서 작성 규칙

문서, 이슈, PR 본문은 기본적으로 한국어로 작성한다. 코드 식별자, 명령어, 파일명,
모델명, 지표명(FAR, FRR, ASR 등), 요구사항 ID(FR-XXX, EXP-XXX)는 원문 영어를
유지한다.

- 협업 가이드: [CONTRIBUTING.md](./CONTRIBUTING.md)
- 이슈 템플릿: [.github/ISSUE_TEMPLATE](./.github/ISSUE_TEMPLATE)
- PR 템플릿: [.github/PULL_REQUEST_TEMPLATE.md](./.github/PULL_REQUEST_TEMPLATE.md)

이 저장소는 얼굴 이미지, 임베딩, 인증 템플릿, 모델 가중치 같은 민감 산출물을 Git에
커밋하지 않는다. 실험 결과를 보고할 때는 데이터 split, seed, 모델/threshold/policy
버전, 실패/오류/제외 표본 수를 함께 적는다.

## Session-based face authentication prototype

The new implementation lives under `src/face_auth/`. It separates enrollment from authentication and adds session state, fail-closed gate policy, challenge/nonce issuance, and a purpose/context-bound one-time result token.

Run the current tests:

```bash
python -m unittest discover -s tests -v
```

Create a multi-frame prototype template from a video:

```bash
python -m src.face_auth.cli enroll \
  --video path/to/enrollment.mp4 \
  --frames 30 \
  --min-valid-frames 5 \
  --output local_templates/user-1.npz
```

Run the first recorded-video authentication slice:

```bash
python -m src.face_auth.cli authenticate \
  --video path/to/probe.mp4 \
  --template local_templates/user-1.npz \
  --threshold 0.60 \
  --threshold-version local-validation-v1 \
  --user-id user-1 \
  --context-hash demo-context-a
```

The threshold above is an example only. It must be calibrated on a validation split. The CLI currently reports `BASELINE_ONLY`; it is not the complete PAD/liveness security profile. Prototype templates are not encrypted and must not be committed.

Run the `FULL` reference profile with a separately validated TorchScript PAD model:

```bash
python -m src.face_auth.cli authenticate \
  --video path/to/probe.mp4 \
  --template local_templates/user-1.npz \
  --profile FULL \
  --threshold 0.60 \
  --threshold-version identity-validation-v1 \
  --pad-model local_models/pad-v1.ts \
  --pad-model-version pad-v1 \
  --pad-live-threshold 0.80 \
  --pad-threshold-version pad-validation-v1 \
  --user-id user-1
```

`FULL` additionally requires camera-motion, content-replay, passive PAD, randomized head-turn liveness, and identity-continuity gates. It refuses to start without a PAD model. Feature-squeezing inspection can be enabled with a validation-derived `--adversarial-threshold`.

An approved original Open Model Zoo `anti-spoof-mn3` ONNX artifact can instead use `--pad-runtime onnx` after installing `requirements-pad-onnx.txt`. This adds a runtime adapter, not a validation claim; calibrate its threshold and evaluate it on held-out physical attacks before reporting FULL-profile security.

Build a deterministic mid-session insertion video:

```bash
python -m src.attack_scenarios.cli \
  --manifest configs/scenarios/mid_frame_insertion.example.json
```

Calibrate thresholds on validation data only:

```bash
python -m src.face_auth.evaluation.calibrate_cli \
  --input configs/thresholds.validation.example.csv \
  --output local_thresholds/validation-v1.json \
  --version validation-v1
```

The camera and quality defaults are starting values, not accuracy claims. Use `--min-blur-variance` and the other explicit threshold options only with a recorded validation artifact.

Start with the [face-auth documentation](docs/face_auth/README.md) for the module map and security profiles. See the [repository implementation status](docs/13_IMPLEMENTATION_STATUS.md) and [local runbook](docs/14_LOCAL_RUNBOOK.md) for verified scope, remaining gaps, and copy-ready commands.

## Attack pipeline

Attack-side Colab notebook:

```text
notebooks/colab_targeted_attack_pipeline.ipynb
```

Main attack scripts:

- `python -m src.attacks.targeted_fgsm_face`
- `python -m src.attacks.targeted_pgd_face`
- `python -m src.attacks.targeted_square_face`
- `python -m src.attacks.targeted_jsma_face`
- `python -m src.attacks.targeted_zoo_face`

Supporting modules:

- `src/common/`: shared model/device/attack utilities
- `src/datasets/`: LFW dataset preparation
- `src/training/`: ResNet-50 face identity training
- `src/reports/`: attack summary, index, representative panel, and plot generation

Attack handoff documents:

- `docs/attack_results_final_2026-05-02.md`
- `docs/attack_index_handoff.md`
- `docs/team_update_attack.md`

Defense modules should use `attack_index.csv` and read adversarial images from the `adv_file` column. Keep `sample_id` in every defense result row so attack and defense results can be joined later.

Final attack result files are not committed to Git because they contain generated images and model outputs. They are shared separately through Google Drive or zip handoff files.

## Verification baseline

The first verification baseline reuses the trained ResNet-50 identity model as a feature extractor. It takes the feature vector before the final classification layer, compares two face images with cosine similarity, and reports verification metrics such as FAR, FRR, EER, and ROC-AUC.

This is a bridge step from identity classification to face-authentication verification. A later version can replace the ResNet feature extractor with an ArcFace/InsightFace embedding model while keeping the same pair CSV and metric format.

The commands below are the historical bridge workflow. Its threshold lifecycle does not satisfy the new calibration/test separation contract. New reportable experiments must create provenance-bound disjoint score exports with `python -m src.evaluation.facenet_score_export_cli`, then use `python -m src.evaluation.verification_baseline_cli`; see the [face verification specification](docs/05_FACE_VERIFICATION_SPEC.md).

Build test pairs:

```bash
python -m src.verification.build_lfw_verification_pairs \
  --data-dir data/processed/lfw_identity_10 \
  --split test \
  --out outputs/verification/lfw_test_pairs.csv
```

Evaluate clean verification:

```bash
python -m src.verification.evaluate_face_verification \
  --pairs outputs/verification/lfw_test_pairs.csv \
  --checkpoint checkpoints/face_resnet50_lfw10/best.pt
```

Run the first targeted verification PGD attack:

```bash
python -m src.verification.targeted_pgd_verification \
  --pairs outputs/verification/lfw_test_pairs.csv \
  --metrics outputs/verification/verification_metrics.json \
  --checkpoint checkpoints/face_resnet50_lfw10/best.pt \
  --epsilon 0.03 \
  --alpha 0.003 \
  --steps 10 \
  --limit 100
```

Generated verification outputs are written under `outputs/verification/` and are not committed to Git.

Typical attack result workflow:

```bash
python -m src.datasets.prepare_lfw_identity_dataset
python -m src.training.train_face_resnet50 --epochs 12 --batch-size 64 --num-workers 2
python -m src.attacks.targeted_fgsm_face --epsilon 0.005 --limit 300
python -m src.attacks.targeted_pgd_face --epsilon 0.03 --alpha 0.003 --steps 10 --limit 300
python -m src.attacks.targeted_square_face --epsilon 0.05 --max-queries 300 --limit 300
python -m src.attacks.targeted_jsma_face --theta 0.05 --steps 20 --pixels-per-step 200 --limit 300
python -m src.attacks.targeted_zoo_face --epsilon 0.05 --max-queries 2000 --coords-per-iter 128 --learning-rate 0.02 --limit 300
python -m src.reports.summarize_face_attack
python -m src.reports.build_attack_index
```

## Defense pipeline

Open `notebooks/colab_defense_pipeline.ipynb` in Colab and run all cells.

### Google Drive에 올려둘 파일

```
내 드라이브/hanium-aml-defense/
  hanium_attack_outputs.zip   ← 공격 결과 (attack_index.csv + 적대적 이미지)
  lfw_data.zip                ← LFW raw 데이터 (Adversarial Training 시 필요)
  best.pt                     ← 학습된 ResNet-50 체크포인트
```

## 현재 구현된 방어 기법

| 방어 기법 | 스크립트 | 주요 파라미터 | 방식 |
|-----------|----------|--------------|------|
| JPEG Compression | `src/defenses/defense_jpeg.py` | `--quality 75` | 전처리 |
| Gaussian Blur | `src/defenses/defense_smoothing.py` | `--radius 3` (PIL GaussianBlur) | 전처리 |
| Bit-depth Reduction | `src/defenses/defense_bitdepth.py` | `--bits 4` | 전처리 |
| ROI-first | `src/defenses/defense_roi.py` | `--mode attenuate --attenuate-factor 0.3 --margin 0.15` | 전처리 (공격 표면 축소) |
| Adversarial Training | `src/defenses/defense_adv_training.py` | `--attack-family pgd --epochs 5 --mix-ratio 0.5` | 모델 재학습 |

각 스크립트는 `--attack-family` 옵션으로 특정 공격만 필터링할 수 있으며, 생략하면 5종 공격 전체에 적용됩니다.

## 실행

```bash
# 전처리 방어 4종 일괄 실행
python -m src.defenses.defense_jpeg      --quality 75
python -m src.defenses.defense_smoothing --radius 3
python -m src.defenses.defense_bitdepth  --bits 4
python -m src.defenses.defense_roi       --mode attenuate --attenuate-factor 0.3 --margin 0.15

# 적대적 학습 (시간 오래 걸림)
python -m src.defenses.defense_adv_training \
    --attack-family pgd --epochs 5 --mix-ratio 0.5

python -m src.reports.summarize_defense
```

Defense source modules:

- `src/defenses/run_preprocessing_defenses.py`: 전처리 방어 4종 일괄 실행 오케스트레이터
- `src/reports/plot_defense_results.py`: 결과 로드 / 집계 / 시각화 / 보고서 생성

Verification 방어 평가 모듈은 `src/verification/defenses/`로 분리되어 있습니다 (FaceNet 임베딩 기반 cosine similarity 평가, classification 방어와 별도 트랙).

## 과거 분류 방어 결과

아래 표는 기존 classification 방어 파이프라인이 남긴 역사적 결과다. 새 얼굴 검증 프로토콜이나 실시간 인증의 보안 성능으로 해석하면 안 된다. 적대적 학습의 held-out 분리, clean 성능 보존, 성공률 분모를 새 계약으로 재검증하기 전에는 공식 비교 수치로 사용하지 않는다.

| 방어 기법 | 방어 성공률 | 복원율 |
|-----------|-----------|--------|
| Adversarial Training | **99.8%** | **99.8%** |
| Gaussian Blur | 90.2% | 48.3% |
| Bit-depth Reduction | 86.0% | 48.3% |
| JPEG Compression | 83.6% | 51.9% |

자세한 결과: `outputs/defenses/defense_report.md`
