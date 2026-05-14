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

