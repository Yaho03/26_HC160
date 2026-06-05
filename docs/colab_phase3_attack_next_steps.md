# Colab 실행 가이드: Phase 3 공격 확장

작성일: 2026-05-30  
목적: 방어팀 5/29 요청 3가지 처리 — PNG handoff 재생성, FGSM 추가, adv training 데이터 500쌍

---

## 사전 확인

```bash
%%bash
cd /content/26_HC160
git pull
echo "=== branch ==="
git branch
echo "=== verification outputs ==="
ls outputs/verification_facenet/ 2>/dev/null || echo "없음 — Drive에서 복원 필요"
ls outputs/verification_attacks_facenet/ 2>/dev/null || echo "없음 — Drive에서 복원 필요"
```

만약 outputs가 없으면 아래 셀로 Drive에서 복원:

```bash
%%bash
mkdir -p outputs/verification_facenet outputs/verification_attacks_facenet
cp -r /content/drive/MyDrive/hanium-aml/results/verification_facenet/* outputs/verification_facenet/ || true
cp -r /content/drive/MyDrive/hanium-aml/results/verification_attacks_facenet/* outputs/verification_attacks_facenet/ || true
echo "=== 복원 결과 ==="
ls outputs/verification_facenet/
ls outputs/verification_attacks_facenet/
```

---

## STEP 1 — PGD PNG 재실행 (최우선: 방어팀 요청 #1)

기존 PGD 공격을 PNG 포맷으로 다시 돌린다.  
이전 결과(.jpg)는 덮어쓰지 않기 위해 `--out-dir`을 `pgd_png`로 분리한다.

```bash
%%bash
set -e
cd /content/26_HC160

echo "=== PGD PNG sweep: eps 0.005, 0.010 ==="
for eps in 0.005 0.010; do
  echo "--- eps=${eps} ---"
  python -m src.verification.targeted_pgd_facenet_verification \
    --pairs outputs/verification/lfw_test_pairs.csv \
    --metrics outputs/verification_facenet/verification_metrics.json \
    --pretrained vggface2 \
    --epsilon "$eps" \
    --alpha 0.001 \
    --steps 10 \
    --limit 100 \
    --only-initial-rejects \
    --image-format png \
    --out-dir "outputs/verification_attacks_facenet/pgd_png"
done

echo "=== summarize ==="
python -m src.verification.summarize_verification_attacks \
  --metadata-root outputs/verification_attacks_facenet/pgd_png \
  --out outputs/verification_attacks_facenet/pgd_png/verification_attack_summary_pgd_png.csv

echo "=== summary ==="
cat outputs/verification_attacks_facenet/pgd_png/verification_attack_summary_pgd_png.csv
```

---

## STEP 2 — FGSM verification 공격 실행 (방어팀 요청 #2)

FGSM은 gradient 한 번만 계산해서 단순하지만, verification 기준으로 처음 돌리는 것이다.  
epsilon sweep으로 성공률을 먼저 확인한다.

```bash
%%bash
set -e
cd /content/26_HC160

echo "=== FGSM verification sweep ==="
for eps in 0.005 0.010 0.020 0.030; do
  echo "--- fgsm eps=${eps} ---"
  python -m src.verification.targeted_fgsm_facenet_verification \
    --pairs outputs/verification/lfw_test_pairs.csv \
    --metrics outputs/verification_facenet/verification_metrics.json \
    --pretrained vggface2 \
    --epsilon "$eps" \
    --limit 100 \
    --only-initial-rejects \
    --image-format png \
    --out-dir "outputs/verification_attacks_facenet/fgsm"
done

echo "=== summarize FGSM ==="
python -m src.verification.summarize_verification_attacks \
  --metadata-root outputs/verification_attacks_facenet/fgsm \
  --out outputs/verification_attacks_facenet/fgsm/verification_attack_summary_fgsm.csv

echo "=== FGSM summary ==="
cat outputs/verification_attacks_facenet/fgsm/verification_attack_summary_fgsm.csv
```

결과에서 확인할 것:

- FGSM ASR (target accept rate after attack) — PGD보다 낮을 것
- 어떤 epsilon에서 FGSM이 처음으로 40~50% 이상 넘는지
- L2/Linf — PGD와 비교

---

## STEP 3 — PGD 추가 실행 (방어팀 요청 #3: adv training 데이터 500쌍)

adv training에 필요한 (source, adv) 쌍을 500개 이상으로 늘린다.  
limit=200으로 올리고 eps=0.005~0.020 범위로 sweep하면 성공 샘플 합산 500쌍 이상이 나온다.

```bash
%%bash
set -e
cd /content/26_HC160

echo "=== PGD adv training data sweep ==="
for eps in 0.005 0.010 0.015 0.020; do
  echo "--- adv training data eps=${eps} ---"
  python -m src.verification.targeted_pgd_facenet_verification \
    --pairs outputs/verification/lfw_test_pairs.csv \
    --metrics outputs/verification_facenet/verification_metrics.json \
    --pretrained vggface2 \
    --epsilon "$eps" \
    --alpha 0.001 \
    --steps 10 \
    --limit 200 \
    --only-initial-rejects \
    --image-format png \
    --out-dir "outputs/verification_attacks_facenet/pgd_adv_training"
done

echo "=== summarize adv training data ==="
python -m src.verification.summarize_verification_attacks \
  --metadata-root outputs/verification_attacks_facenet/pgd_adv_training \
  --out outputs/verification_attacks_facenet/pgd_adv_training/verification_attack_summary_adv_training.csv

echo "=== adv training summary ==="
cat outputs/verification_attacks_facenet/pgd_adv_training/verification_attack_summary_adv_training.csv
```

성공 샘플 수 확인 (attack_success=True인 행 개수):

```bash
%%bash
cd /content/26_HC160
python3 - <<'EOF'
import csv, glob
total_success = 0
for path in sorted(glob.glob("outputs/verification_attacks_facenet/pgd_adv_training/metadata_*.csv")):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    successes = [r for r in rows if r["attack_success"].strip().lower() in {"true","1","yes"}]
    print(f"{path.split('/')[-1]}: {len(successes)}/{len(rows)} success")
    total_success += len(successes)
print(f"\n총 성공 샘플: {total_success}")
EOF
```

총 성공 샘플이 500 이상이면 STEP 4로 진행.

---

## STEP 4 — Handoff 패키지 3종 빌드 & Drive 저장

PGD-PNG, FGSM, adv training 데이터를 각각 zip으로 만들어서 Drive에 올린다.

```bash
%%bash
set -e
cd /content/26_HC160
git pull

# (1) PGD PNG handoff — 방어팀에 JPEG 방어 재평가 요청용
echo "=== build PGD-PNG handoff ==="
python -m src.verification.build_verification_attack_handoff \
  --metadata-root outputs/verification_attacks_facenet/pgd_png \
  --verification-metrics outputs/verification_facenet/verification_metrics.json \
  --attack-summary outputs/verification_attacks_facenet/pgd_png/verification_attack_summary_pgd_png.csv \
  --epsilons 0.005,0.010 \
  --successful-only \
  --out-dir outputs/handoff/facenet_pgd_png_package \
  --zip-out outputs/handoff/facenet_pgd_png_package.zip

# (2) FGSM handoff — 방어팀 공격 다양화 요청용
echo "=== build FGSM handoff ==="
python -m src.verification.build_verification_attack_handoff \
  --metadata-root outputs/verification_attacks_facenet/fgsm \
  --verification-metrics outputs/verification_facenet/verification_metrics.json \
  --attack-summary outputs/verification_attacks_facenet/fgsm/verification_attack_summary_fgsm.csv \
  --epsilons ALL \
  --successful-only \
  --out-dir outputs/handoff/facenet_fgsm_package \
  --zip-out outputs/handoff/facenet_fgsm_package.zip

# (3) adv training 데이터 패키지 — 성공 샘플만, 전 epsilon 포함
echo "=== build adv training handoff ==="
python -m src.verification.build_verification_attack_handoff \
  --metadata-root outputs/verification_attacks_facenet/pgd_adv_training \
  --verification-metrics outputs/verification_facenet/verification_metrics.json \
  --attack-summary outputs/verification_attacks_facenet/pgd_adv_training/verification_attack_summary_adv_training.csv \
  --epsilons ALL \
  --successful-only \
  --out-dir outputs/handoff/facenet_adv_training_package \
  --zip-out outputs/handoff/facenet_adv_training_package.zip

echo "=== handoff 파일 크기 ==="
ls -lh outputs/handoff/*.zip
```

Drive에 저장:

```bash
%%bash
set -e
cd /content/26_HC160
mkdir -p /content/drive/MyDrive/hanium-aml/results/handoff

cp outputs/handoff/facenet_pgd_png_package.zip \
   /content/drive/MyDrive/hanium-aml/results/handoff/

cp outputs/handoff/facenet_fgsm_package.zip \
   /content/drive/MyDrive/hanium-aml/results/handoff/

cp outputs/handoff/facenet_adv_training_package.zip \
   /content/drive/MyDrive/hanium-aml/results/handoff/

# 공격 결과도 Drive에 백업
mkdir -p /content/drive/MyDrive/hanium-aml/results/verification_attacks_facenet
cp -r outputs/verification_attacks_facenet/pgd_png \
   /content/drive/MyDrive/hanium-aml/results/verification_attacks_facenet/ || true
cp -r outputs/verification_attacks_facenet/fgsm \
   /content/drive/MyDrive/hanium-aml/results/verification_attacks_facenet/ || true
cp -r outputs/verification_attacks_facenet/pgd_adv_training \
   /content/drive/MyDrive/hanium-aml/results/verification_attacks_facenet/ || true

echo "=== Drive 저장 완료 ==="
ls -lh /content/drive/MyDrive/hanium-aml/results/handoff/
```

---

## STEP 4-b — Adaptive Attack 실행 (smoothing 78.8% 뚫기)

방어팀 smoothing이 78.8% 방어 성공 중이다.
Adaptive PGD는 PGD 루프 안에서 smoothing을 직접 통과하며 gradient를 계산하기 때문에
perturbation이 smoothing 후에도 살아남도록 최적화된다.

```bash
%%bash
set -e
cd /content/26_HC160

echo "=== Adaptive PGD (vs smoothing) sweep ==="
for eps in 0.005 0.010 0.020; do
  echo "--- adaptive eps=${eps} ---"
  python -m src.verification.targeted_pgd_facenet_adaptive \
    --pairs outputs/verification/lfw_test_pairs.csv \
    --metrics outputs/verification_facenet/verification_metrics.json \
    --pretrained vggface2 \
    --epsilon "$eps" \
    --alpha 0.001 \
    --steps 20 \
    --limit 100 \
    --only-initial-rejects \
    --defense-transform smoothing \
    --smoothing-kernel 13 \
    --smoothing-sigma 3.0 \
    --image-format png \
    --out-dir "outputs/verification_attacks_facenet/pgd_adaptive"
done

echo "=== summarize adaptive ==="
python -m src.verification.summarize_verification_attacks \
  --metadata-root outputs/verification_attacks_facenet/pgd_adaptive \
  --out outputs/verification_attacks_facenet/pgd_adaptive/verification_attack_summary_adaptive.csv

echo "=== adaptive summary ==="
cat outputs/verification_attacks_facenet/pgd_adaptive/verification_attack_summary_adaptive.csv
```

결과에서 확인할 것:
- `adaptive_success` 컬럼 — smoothing 후에도 accept 유지되는 비율 (일반 PGD보다 높아야 의미 있음)
- 일반 PGD eps=0.010: smoothing 후 ASR ≈ 21% (= 100%-78.8%) 예상
- Adaptive PGD eps=0.010: smoothing 후 ASR 40%+ 목표

Drive 저장:
```bash
%%bash
cd /content/26_HC160
mkdir -p /content/drive/MyDrive/hanium-aml/results/verification_attacks_facenet
cp -r outputs/verification_attacks_facenet/pgd_adaptive \
   /content/drive/MyDrive/hanium-aml/results/verification_attacks_facenet/ || true
```

---

## STEP 5 — 방어팀에게 보낼 메시지

STEP 1~4가 완료되면 이 메시지를 방어팀에 보내면 된다:

```text
안녕하세요, 요청하신 내용 처리해서 3개 패키지 올려드렸습니다.

1. facenet_pgd_png_package.zip
   - 기존 PGD handoff를 PNG 포맷으로 재생성했습니다.
   - eps=0.005 / eps=0.010 성공 샘플 포함.
   - 이제 JPEG 방어 평가가 공정하게 될 것 같습니다.

2. facenet_fgsm_package.zip
   - FGSM verification 공격(targeted, FaceNet 기준) 결과입니다.
   - 동일한 attack_handoff_index.csv 포맷으로 맞췄습니다.

3. facenet_adv_training_package.zip
   - adv training용 (source, adv) 쌍입니다.
   - PGD, eps=0.005~0.020, 성공 샘플만 포함, 500쌍 이상입니다.
   - source_file + adv_file 쌍을 학습 데이터로 쓰시면 됩니다.

Drive 경로: hanium-aml/results/handoff/
방어 평가 포맷은 기존과 동일합니다 (attack_handoff_index.csv 기준).
```

---

## STEP 6 — 비교 그래프 생성 (결과 나온 후)

STEP 1~4-b가 끝나고 summary CSV들이 생겼으면 아래로 비교 그래프를 만든다.

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
  --out-dir outputs/figures/verification_attack_defense

echo "=== 생성된 그래프 ==="
ls outputs/figures/verification_attack_defense/
```

방어팀 summary도 있으면 `--defense-summary` 옵션으로 같이 넣으면 공격 vs 방어 비교 그래프까지 나온다:

```bash
python -m src.reports.plot_verification_attack_defense \
  --attack-summaries \
    outputs/verification_attacks_facenet/pgd_png/verification_attack_summary_pgd_png.csv \
    outputs/verification_attacks_facenet/fgsm/verification_attack_summary_fgsm.csv \
    outputs/verification_attacks_facenet/pgd_adaptive/verification_attack_summary_adaptive.csv \
  --defense-summary outputs/handoff/defense_results/verification_defense_summary.csv \
  --out-dir outputs/figures/verification_attack_defense
```

---

## 결과에서 확인할 숫자 (나한테 보내주면 됩니다)

```text
=== PGD PNG summary ===
(verification_attack_summary_pgd_png.csv 내용)

=== FGSM summary ===
(verification_attack_summary_fgsm.csv 내용)

=== adv training 총 성공 샘플 수 ===
(총 500쌍 이상인지 확인)
```
