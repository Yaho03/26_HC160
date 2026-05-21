# FaceNet Verification Attack Handoff Guide

작성일: 2026-05-22  
담당: 공격팀  
목적: 방어팀이 FaceNet verification adversarial images에 방어를 적용하고, 방어 성공 여부를 같은 기준으로 평가할 수 있도록 전달 포맷을 정의한다.

---

## 1. 배경

기존 공격/방어 실험은 10-class identity classification 기준이었다.

```text
공격 성공: 공격 이미지가 target class로 분류됨
방어 성공: 방어 후 target class가 아니게 됨
복원 성공: 방어 후 true class로 돌아옴
```

이제 공격팀은 실제 금융 얼굴인증에 가까운 verification 기준으로 확장했다.

```text
source 얼굴 + target enrollment 얼굴
    ↓
FaceNet embedding cosine similarity
    ↓
threshold 이상이면 target 사용자로 accept
threshold 미만이면 reject
```

따라서 방어팀은 classification label이 아니라 FaceNet similarity와 threshold 기준으로 방어 성공 여부를 계산해야 한다.

---

## 2. 현재 FaceNet baseline

모델:

```text
facenet-pytorch/InceptionResnetV1 pretrained=vggface2
```

Clean verification 성능:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.9996 |
| EER | 0.50% |
| Threshold | 0.4797 |
| Accuracy | 99.50% |
| FAR | 0.67% |
| FRR | 0.34% |

대표 공격 결과:

| Attack | Epsilon | Steps | Alpha | Target Accept Rate |
|---|---:|---:|---:|---:|
| targeted_pgd_facenet_verification | 0.005 | 10 | 0.001 | 45% |
| targeted_pgd_facenet_verification | 0.010 | 10 | 0.001 | 80% |

---

## 3. 방어팀에 전달할 패키지

패키지 이름:

```text
facenet_verification_attack_package.zip
```

포함 파일:

```text
facenet_verification_attack_package/
  attack_handoff_index.csv
  verification_metrics.json
  verification_attack_summary.csv
  manifest.json
  README.md
  samples/
    <epsilon>/
      <sample_id>/
        source/
        target_enroll/
        adversarial/
        perturbation/
```

핵심 파일은 `attack_handoff_index.csv`이다.

---

## 4. attack_handoff_index.csv 컬럼

| Column | Meaning |
|---|---|
| sample_id | 방어 결과 join용 고유 ID |
| pair_id | 원본 verification pair ID |
| attack | 공격 이름 |
| model | 공격 대상 모델 |
| pretrained | pretrained weight 이름 |
| source_file | 공격자 원본 얼굴 |
| target_enroll_file | 타겟 등록 얼굴 |
| adv_file | 방어팀이 입력으로 써야 할 adversarial image |
| perturbation_file | perturbation 시각화 이미지 |
| source_name | source identity |
| target_name | target identity |
| threshold | verification accept/reject 기준 |
| similarity_before | clean source-target similarity |
| similarity_after_attack | 공격 후 source-target similarity |
| accepted_before | 공격 전 accept 여부 |
| accepted_after_attack | 공격 후 accept 여부 |
| attack_success_before_defense | 방어 전 공격 성공 여부 |
| epsilon | PGD epsilon |
| alpha | PGD alpha |
| steps | PGD steps |
| l2 | perturbation L2 |
| linf | perturbation Linf |

---

## 5. 방어팀 평가 방법

각 row에 대해:

1. `adv_file` 이미지를 읽는다.
2. 방어 기법을 적용한다.
3. 방어 후 이미지를 `defended_file`로 저장한다.
4. FaceNet으로 `defended_file` embedding을 추출한다.
5. FaceNet으로 `target_enroll_file` embedding을 추출한다.
6. cosine similarity를 계산한다.
7. threshold와 비교한다.

판정:

```text
accepted_after_defense = similarity_after_defense >= threshold
```

방어 성공:

```text
defense_success = attack_success_before_defense and not accepted_after_defense
```

즉 공격 전에는 reject, 공격 후에는 accept, 방어 후에는 다시 reject되면 방어 성공이다.

---

## 6. 방어팀 결과 CSV 제안

방어팀이 반환하면 좋은 컬럼:

```text
sample_id
attack
defense
defense_params
model
pretrained
source_file
target_enroll_file
adv_file
defended_file
source_name
target_name
threshold
similarity_before
similarity_after_attack
similarity_after_defense
accepted_before
accepted_after_attack
accepted_after_defense
attack_success_before_defense
attack_success_after_defense
defense_success
epsilon
alpha
steps
l2
linf
defense_time_sec
status
```

---

## 7. 공격팀에서 패키지 만드는 명령

Colab에서 FaceNet attack 결과가 있는 상태에서:

```bash
python -m src.verification.build_verification_attack_handoff \
  --metadata-root outputs/verification_attacks_facenet \
  --verification-metrics outputs/verification_facenet/verification_metrics.json \
  --attack-summary outputs/verification_attacks_facenet/verification_attack_summary.csv \
  --epsilons 0.005,0.010 \
  --successful-only \
  --out-dir outputs/handoff/facenet_verification_attack_package \
  --zip-out outputs/handoff/facenet_verification_attack_package.zip
```

Drive에 저장:

```bash
mkdir -p /content/drive/MyDrive/hanium-aml/results/handoff
cp outputs/handoff/facenet_verification_attack_package.zip \
  /content/drive/MyDrive/hanium-aml/results/handoff/
```

패키지 검증:

```bash
python -m src.verification.validate_facenet_handoff_package \
  --package-dir outputs/handoff/facenet_verification_attack_package
```

방어팀 반환용 템플릿 생성:

```bash
python -m src.verification.create_facenet_defense_result_template \
  --handoff-index outputs/handoff/facenet_verification_attack_package/attack_handoff_index.csv \
  --out outputs/handoff/facenet_verification_defense_results_template.csv
```

---

## 8. 방어팀에게 전달할 메시지

```text
FaceNet verification attack handoff package를 전달드립니다.

기존 classification attack이 아니라, source 얼굴이 target enrollment 얼굴로 인증 통과하는 impersonation attack 기준입니다.

attack_handoff_index.csv의 adv_file을 방어 입력으로 사용해 주세요.
방어 후 defended_file과 target_enroll_file의 FaceNet cosine similarity를 계산하고,
threshold 이상이면 공격 유지, threshold 미만이면 방어 성공으로 보면 됩니다.

우선 eps=0.005와 eps=0.010의 successful attack samples를 포함했습니다.
```
