# Colab 실행 가이드: 생성형 AI 방어 (DAE + DiffPure)

작성일: 2026-05-30  
목적: DAE(Denoising Autoencoder) 학습 및 DiffPure 실행, 방어팀 handoff 포맷으로 결과 전달

---

## 개요

| 방어 | 방식 | 필요한 것 | 방어팀 handoff 가능 |
|---|---|---|---|
| DAE | 학습된 U-Net이 perturbation 제거 | LFW 데이터 + 학습 30에폭 (~30분) | ✅ |
| DiffPure | DDPM forward+reverse로 perturbation 제거 | pretrained DDPM 모델 (~1.5GB) | ✅ |
| DiffPure-fallback | noise추가+Gaussian denoising (proxy) | 없음 | ✅ (빠른 테스트용) |

---

## 사전 준비

```bash
%%bash
cd /content/26_HC160
git pull
pip install -q facenet-pytorch diffusers accelerate

# outputs 복원
mkdir -p outputs/verification_facenet outputs/handoff
cp -r /content/drive/MyDrive/hanium-aml/results/verification_facenet/* outputs/verification_facenet/ || true
# PGD PNG handoff 패키지 있으면 복원
if [ -f /content/drive/MyDrive/hanium-aml/results/handoff/facenet_pgd_png_package.zip ]; then
  unzip -q /content/drive/MyDrive/hanium-aml/results/handoff/facenet_pgd_png_package.zip \
    -d outputs/handoff/
  echo "PNG handoff 복원 완료"
fi
ls outputs/handoff/
```

---

## STEP 1 — DAE 학습

### 1-a. 기본 학습 (noise augmentation만)

```bash
%%bash
set -e
cd /content/26_HC160

python -m src.training.train_face_dae \
  --data-dir data/raw/lfw \
  --out-dir checkpoints/face_dae \
  --epochs 30 \
  --batch-size 32 \
  --base-ch 32 \
  --noise-sigma-min 0.005 \
  --noise-sigma-max 0.05

echo "=== 학습 결과 ==="
ls -lh checkpoints/face_dae/
```

예상 시간: Colab T4 기준 약 20~30분.

### 1-b. adversarial pair 포함 학습 (더 강력, PNG handoff 있을 때)

```bash
%%bash
set -e
cd /content/26_HC160

python -m src.training.train_face_dae \
  --data-dir data/raw/lfw \
  --adv-dir outputs/handoff/facenet_pgd_png_package \
  --out-dir checkpoints/face_dae_with_adv \
  --epochs 40 \
  --batch-size 32 \
  --base-ch 32

echo "=== 학습 결과 (adv pair 포함) ==="
ls -lh checkpoints/face_dae_with_adv/
```

### 1-c. 체크포인트 Drive 저장

```bash
%%bash
mkdir -p /content/drive/MyDrive/hanium-aml/checkpoints
cp -r checkpoints/face_dae /content/drive/MyDrive/hanium-aml/checkpoints/ || true
cp -r checkpoints/face_dae_with_adv /content/drive/MyDrive/hanium-aml/checkpoints/ || true
echo "체크포인트 저장 완료"
ls /content/drive/MyDrive/hanium-aml/checkpoints/
```

---

## STEP 2 — DAE 방어 평가

```bash
%%bash
set -e
cd /content/26_HC160

python -m src.defenses.verification_defense_dae \
  --handoff-index outputs/handoff/facenet_pgd_png_package/attack_handoff_index.csv \
  --handoff-root outputs/handoff/facenet_pgd_png_package \
  --checkpoint checkpoints/face_dae/best.pt \
  --pretrained vggface2 \
  --out-dir outputs/defenses/verification/dae \
  --save-images

echo "=== DAE 방어 결과 ==="
cat outputs/defenses/verification/dae/verification_defense_dae.csv | head -5
```

결과에서 확인:
- `defense_success_rate` — smoothing(78.8%), JPEG(0%)와 비교
- `similarity_after_defense` — threshold(0.4797) 이하로 내려가는지
- `defense_time_sec` — smoothing보다 느릴 것 (U-Net 추론)

---

## STEP 3 — DiffPure 방어 평가

### 3-a. DiffPure-fallback (빠른 테스트, diffusers 불필요)

```bash
%%bash
set -e
cd /content/26_HC160

python -m src.defenses.verification_defense_diffpure \
  --handoff-index outputs/handoff/facenet_pgd_png_package/attack_handoff_index.csv \
  --handoff-root outputs/handoff/facenet_pgd_png_package \
  --use-fallback \
  --fallback-noise-sigma 0.03 \
  --fallback-denoise-sigma 1.5 \
  --pretrained vggface2 \
  --out-dir outputs/defenses/verification/diffpure_fallback
```

### 3-b. DiffPure 실제 (DDPM, t_diff sweep)

```bash
%%bash
set -e
cd /content/26_HC160

# t_diff 크기별 방어 강도 비교
for t_diff in 0.05 0.10 0.15; do
  echo "=== DiffPure t_diff=${t_diff} ==="
  python -m src.defenses.verification_defense_diffpure \
    --handoff-index outputs/handoff/facenet_pgd_png_package/attack_handoff_index.csv \
    --handoff-root outputs/handoff/facenet_pgd_png_package \
    --model-id google/ddpm-celebahq-256 \
    --t-diff "$t_diff" \
    --pretrained vggface2 \
    --out-dir "outputs/defenses/verification/diffpure_t${t_diff/./p}"
done
```

확인할 것:
- t_diff 증가 → 방어 성공률 ↑, identity 보존 ↓ (FAR/FRR 변화)
- 적절한 trade-off 지점 찾기

---

## STEP 4 — Adaptive Attack vs DAE

DAE 방어를 뚫는 adaptive PGD.  
DAE는 미분 가능한 신경망이라 gradient가 그대로 흐름 → BPDA 없이 직접 공격 가능.

```bash
%%bash
set -e
cd /content/26_HC160

for eps in 0.010 0.020; do
  echo "=== Adaptive vs DAE eps=${eps} ==="
  python -m src.verification.targeted_pgd_facenet_adaptive \
    --pairs outputs/verification/lfw_test_pairs.csv \
    --metrics outputs/verification_facenet/verification_metrics.json \
    --pretrained vggface2 \
    --epsilon "$eps" \
    --alpha 0.001 \
    --steps 20 \
    --limit 100 \
    --only-initial-rejects \
    --defense-transform dae \
    --dae-checkpoint checkpoints/face_dae/best.pt \
    --dae-base-ch 32 \
    --image-format png \
    --out-dir "outputs/verification_attacks_facenet/pgd_adaptive_dae"
done

echo "=== summarize ==="
python -m src.verification.summarize_verification_attacks \
  --metadata-root outputs/verification_attacks_facenet/pgd_adaptive_dae \
  --out outputs/verification_attacks_facenet/pgd_adaptive_dae/summary.csv

cat outputs/verification_attacks_facenet/pgd_adaptive_dae/summary.csv
```

---

## STEP 5 — 전체 비교 그래프 생성

```bash
%%bash
set -e
cd /content/26_HC160
pip install -q matplotlib pandas

python -m src.reports.plot_verification_attack_defense \
  --attack-summaries \
    outputs/verification_attacks_facenet/pgd_png/verification_attack_summary_pgd_png.csv \
    outputs/verification_attacks_facenet/fgsm/verification_attack_summary_fgsm.csv \
    outputs/verification_attacks_facenet/pgd_adaptive/verification_attack_summary_adaptive.csv \
    outputs/verification_attacks_facenet/pgd_adaptive_dae/summary.csv \
  --out-dir outputs/figures/verification_attack_defense

echo "=== 생성된 그래프 ==="
ls outputs/figures/verification_attack_defense/
```

---

## STEP 6 — Drive 저장 및 방어팀 전달

```bash
%%bash
cd /content/26_HC160

mkdir -p /content/drive/MyDrive/hanium-aml/results/defenses/verification

# DAE 방어 결과
cp -r outputs/defenses/verification/dae \
  /content/drive/MyDrive/hanium-aml/results/defenses/verification/ || true

# DiffPure 방어 결과
for d in outputs/defenses/verification/diffpure*; do
  cp -r "$d" /content/drive/MyDrive/hanium-aml/results/defenses/verification/ || true
done

# 비교 그래프
cp -r outputs/figures \
  /content/drive/MyDrive/hanium-aml/results/ || true

echo "=== Drive 저장 완료 ==="
ls /content/drive/MyDrive/hanium-aml/results/defenses/verification/
```

방어팀에게 전달할 결과 파일:
- `outputs/defenses/verification/dae/verification_defense_dae.csv`
- `outputs/defenses/verification/diffpure_t0p10/verification_defense_diffpure_t0.10.csv`

---

## 나에게 보내줄 숫자

실험이 끝나면 아래 출력을 보내주세요:

```text
=== DAE 방어 결과 ===
Defense success rate: X%
Still accepted (ASR): X%
Avg similarity drop: X.XXXX

=== DiffPure t_diff=0.10 방어 결과 ===
Defense success rate: X%
Still accepted (ASR): X%
Avg similarity drop: X.XXXX

=== Adaptive vs DAE (eps=0.010) ===
adaptive_success (DAE 방어 후 accept 유지): X%
```
