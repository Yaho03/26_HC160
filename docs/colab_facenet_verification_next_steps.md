# Colab 실행 셀: FaceNet 기반 Verification 고도화

작성일: 2026-05-14

이 셀은 기존 ResNet feature verification baseline 다음 단계로, `facenet-pytorch`의 pretrained FaceNet-style embedding 모델을 사용해 clean verification과 targeted PGD impersonation attack을 실행한다.

전제:

- `/content/26_HC160` repo가 있음
- `codex/add-verification-baseline` 브랜치 checkout 완료
- `outputs/verification/lfw_test_pairs.csv` 생성 완료

---

## 실행 셀

```bash
%%bash
set -e

cd /content/26_HC160

echo "=== pull latest FaceNet verification code ==="
git pull

echo "=== install facenet-pytorch ==="
pip install -q facenet-pytorch

echo "=== evaluate clean FaceNet verification ==="
python -m src.verification.evaluate_facenet_verification \
  --pairs outputs/verification/lfw_test_pairs.csv \
  --pretrained vggface2

echo "=== targeted FaceNet verification PGD epsilon sweep ==="
for eps in 0.001 0.003 0.005 0.010; do
  echo "=== facenet eps=${eps} ==="
  python -m src.verification.targeted_pgd_facenet_verification \
    --pairs outputs/verification/lfw_test_pairs.csv \
    --metrics outputs/verification_facenet/verification_metrics.json \
    --pretrained vggface2 \
    --epsilon "$eps" \
    --alpha 0.001 \
    --steps 10 \
    --limit 100 \
    --only-initial-rejects
done

echo "=== summarize FaceNet verification attack sweep ==="
python -m src.verification.summarize_verification_attacks \
  --metadata-root outputs/verification_attacks_facenet \
  --out outputs/verification_attacks_facenet/verification_attack_summary.csv

echo "=== save FaceNet verification outputs to Drive ==="
mkdir -p /content/drive/MyDrive/hanium-aml/results/verification_facenet
mkdir -p /content/drive/MyDrive/hanium-aml/results/verification_attacks_facenet
cp -r outputs/verification_facenet/* /content/drive/MyDrive/hanium-aml/results/verification_facenet/
cp -r outputs/verification_attacks_facenet/* /content/drive/MyDrive/hanium-aml/results/verification_attacks_facenet/

echo "=== FaceNet clean verification metrics ==="
cat outputs/verification_facenet/verification_metrics.json

echo "=== FaceNet attack summary ==="
cat outputs/verification_attacks_facenet/verification_attack_summary.csv
```

---

## 나에게 보내야 할 출력

다 끝나면 아래 두 부분을 보내면 된다.

```text
=== FaceNet clean verification metrics ===
...
=== FaceNet attack summary ===
...
```

확인할 핵심:

- ROC-AUC
- EER
- FAR
- FRR
- threshold
- epsilon별 attack success rate
- 평균 similarity gain
- 평균 L2/Linf

---

## 추가 실험: 더 강한 FaceNet PGD sweep

위 기본 sweep에서 eps=0.010의 성공률이 높게 나오면, 아래 셀로 eps와 step 수를 확장한다.

```bash
%%bash
set -e

cd /content/26_HC160

echo "=== stronger FaceNet PGD sweep ==="
for steps in 10 20 40; do
  for eps in 0.010 0.015 0.020 0.030; do
    echo "=== facenet eps=${eps}, steps=${steps} ==="
    python -m src.verification.targeted_pgd_facenet_verification \
      --pairs outputs/verification/lfw_test_pairs.csv \
      --metrics outputs/verification_facenet/verification_metrics.json \
      --pretrained vggface2 \
      --epsilon "$eps" \
      --alpha 0.001 \
      --steps "$steps" \
      --limit 100 \
      --only-initial-rejects
  done
done

echo "=== summarize stronger FaceNet PGD sweep ==="
python -m src.verification.summarize_verification_attacks \
  --metadata-root outputs/verification_attacks_facenet \
  --out outputs/verification_attacks_facenet/verification_attack_summary.csv

echo "=== save updated stronger FaceNet outputs to Drive ==="
mkdir -p /content/drive/MyDrive/hanium-aml/results/verification_attacks_facenet
cp -r outputs/verification_attacks_facenet/* /content/drive/MyDrive/hanium-aml/results/verification_attacks_facenet/

echo "=== FaceNet attack summary ==="
cat outputs/verification_attacks_facenet/verification_attack_summary.csv
```

확인 목표:

- 90% 이상 성공하는 최소 epsilon
- steps 증가에 따른 성공률 변화
- epsilon 대비 L2/Linf 증가량
- 이후 방어 실험에 사용할 대표 설정 선택

---

## 방어팀 전달 패키지 생성

FaceNet attack metadata와 이미지가 생성되어 있고 Drive에 저장되어 있다면, 아래 셀로 방어팀 전달용 zip을 만든다.

```bash
%%bash
set -e

cd /content/26_HC160

echo "=== pull latest handoff script ==="
git pull

echo "=== restore FaceNet outputs from Drive if needed ==="
mkdir -p outputs/verification_facenet outputs/verification_attacks_facenet
if [ -d /content/drive/MyDrive/hanium-aml/results/verification_facenet ]; then
  cp -r /content/drive/MyDrive/hanium-aml/results/verification_facenet/* outputs/verification_facenet/ || true
fi
if [ -d /content/drive/MyDrive/hanium-aml/results/verification_attacks_facenet ]; then
  cp -r /content/drive/MyDrive/hanium-aml/results/verification_attacks_facenet/* outputs/verification_attacks_facenet/ || true
fi

echo "=== build FaceNet verification attack handoff package ==="
python -m src.verification.build_verification_attack_handoff \
  --metadata-root outputs/verification_attacks_facenet \
  --verification-metrics outputs/verification_facenet/verification_metrics.json \
  --attack-summary outputs/verification_attacks_facenet/verification_attack_summary.csv \
  --epsilons 0.005,0.010 \
  --successful-only \
  --out-dir outputs/handoff/facenet_verification_attack_package \
  --zip-out outputs/handoff/facenet_verification_attack_package.zip

echo "=== validate handoff package ==="
python -m src.verification.validate_facenet_handoff_package \
  --package-dir outputs/handoff/facenet_verification_attack_package

echo "=== create defense result template ==="
python -m src.verification.create_facenet_defense_result_template \
  --handoff-index outputs/handoff/facenet_verification_attack_package/attack_handoff_index.csv \
  --out outputs/handoff/facenet_verification_defense_results_template.csv

echo "=== create representative attack panels ==="
python -m src.verification.make_verification_attack_panels \
  --package-dir outputs/handoff/facenet_verification_attack_package \
  --out-dir outputs/handoff/facenet_verification_attack_panels \
  --per-epsilon 6

echo "=== save handoff package to Drive ==="
mkdir -p /content/drive/MyDrive/hanium-aml/results/handoff
cp outputs/handoff/facenet_verification_attack_package.zip \
  /content/drive/MyDrive/hanium-aml/results/handoff/
cp outputs/handoff/facenet_verification_defense_results_template.csv \
  /content/drive/MyDrive/hanium-aml/results/handoff/
rm -rf /content/drive/MyDrive/hanium-aml/results/handoff/facenet_verification_attack_panels
cp -r outputs/handoff/facenet_verification_attack_panels \
  /content/drive/MyDrive/hanium-aml/results/handoff/

echo "=== handoff package ==="
ls -lh outputs/handoff/facenet_verification_attack_package.zip
ls -lh /content/drive/MyDrive/hanium-aml/results/handoff/facenet_verification_attack_package.zip
ls -lh /content/drive/MyDrive/hanium-aml/results/handoff/facenet_verification_defense_results_template.csv
find /content/drive/MyDrive/hanium-aml/results/handoff/facenet_verification_attack_panels -type f | head -20
```
