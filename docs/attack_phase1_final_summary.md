# 공격 파트 1차 완료 정리

## 1. 완료 상태

수행계획서 기준 공격 파트의 1차 구현과 소규모 실험을 완료했다.

구현 완료 공격:

- FGSM: white-box 1-step targeted attack
- PGD: white-box iterative targeted baseline
- Square Attack: black-box query-based targeted attack
- JSMA 변형: saliency 기반 multi-pixel targeted JSMA
- ZOO-style attack: finite difference 기반 black-box targeted attack

수행계획서의 필수 공격 항목 기준:

| 계획서 공격 | 현재 상태 |
|---|---|
| FGSM | 구현 및 실험 완료 |
| JSMA | multi-pixel 변형 구현 및 실험 완료 |
| Square | 구현 및 실험 완료 |
| ZOO | ZOO-style finite difference 구현 및 소규모 실험 완료 |

## 2. 모델 기준선

- Dataset: LFW deepfunneled
- Task: 10-class face identity classification
- Model: ImageNet pretrained ResNet-50 fine-tuning
- Colab Pro CUDA 환경에서 학습
- Test accuracy: 76.23%
- 공격 평가 subset clean accuracy: 91.00%

공격 성공률은 전체 샘플 기준과 원본 정분류 샘플 기준을 모두 기록했다. 주요 비교에는 `target_success_rate_on_clean`을 사용한다.

## 3. 공격 결과 핵심 비교

| 공격 | 대표 설정 | 원본 정분류 기준 Target ASR | 특징 |
|---|---|---:|---|
| FGSM | eps=0.005 | 46.15% | 매우 빠르지만 targeted setting에서 불안정 |
| PGD | eps=0.005~0.050 | 100.00% | 가장 강한 white-box iterative baseline |
| Square | eps=0.050, queries=300 | 76.92% | 효율적인 black-box baseline |
| multi-pixel JSMA | theta=0.05, steps=20, k=200 | 100.00% | 희소 perturbation, 프로젝트 독창성 포인트 |
| ZOO-style | eps=0.050, queries=2000 | 38.89% | finite difference black-box, 고비용/저효율 |

## 4. ZOO 결과 해석

ZOO-style attack은 두 설정으로 실험했다.

| epsilon | max_queries | coords_per_iter | learning_rate | Target ASR on clean | avg_queries | avg_time_sec |
|---:|---:|---:|---:|---:|---:|---:|
| 0.03 | 1000 | 64 | 0.01 | 16.67% | 936.65 | 3.14 |
| 0.05 | 2000 | 128 | 0.02 | 38.89% | 1615.45 | 5.96 |

해석:

- query budget과 epsilon을 늘리면 성공률이 상승했다.
- 평균 query 사용량이 매우 높아 Square Attack보다 비용이 크다.
- 현재 구현은 원 논문 전체 최적화 기법을 모두 구현한 것이 아니라, finite difference 기반 ZOO-style targeted black-box baseline이다.
- 수행계획서의 ZOO 항목은 1차 구현 및 소규모 실험 완료로 보고 가능하다.

## 5. 현재 결론

현재 공격 강도와 실용성을 함께 보면 다음과 같이 정리할 수 있다.

```text
White-box 강도: PGD ≈ multi-pixel JSMA > FGSM
Black-box 효율: Square > ZOO-style
```

주요 결론:

- PGD는 가장 강력한 white-box 기준선이다.
- multi-pixel JSMA는 희소 perturbation 형태를 보이며, 수행계획서의 독창성 포인트로 활용 가능하다.
- Square Attack은 ZOO보다 query 효율이 좋고 black-box 대표 공격으로 사용하기 적절하다.
- ZOO-style attack은 성공률은 낮지만 finite difference 기반 black-box 공격의 고비용 특성을 보여주는 비교 항목으로 의미가 있다.

## 6. 결과 파일

Colab 기준 주요 결과 파일:

```text
outputs/attacks/face_attack_summary.csv
outputs/attacks/attack_index.csv
outputs/attack_panels/
outputs/attacks/fgsm_face/
outputs/attacks/pgd_face/
outputs/attacks/square_face/
outputs/attacks/jsma_face/
outputs/attacks/zoo_face/
```

개인 Drive 백업 권장 위치:

```text
MyDrive/hanium-aml/results/
```

현재 attack_index 생성 결과:

```text
Metadata files: 17
Indexed attacks: 1620
Rows by attack_family:
fgsm      400
jsma      100
pgd       700
square    400
zoo        20
```

## 7. 팀/멘토 보고용 문장

공격 파트는 LFW 기반 ResNet-50 얼굴 신원 분류 모델을 구축하고, 수행계획서에 포함된 FGSM, JSMA, Square, ZOO 공격을 targeted attack 형태로 1차 구현 및 실험했다. 추가로 PGD를 강한 white-box iterative baseline으로 포함했다. 실험 결과 PGD와 multi-pixel JSMA 변형은 원본 정분류 샘플 기준 100% target ASR을 보였고, Square Attack은 epsilon 0.05, query budget 300 조건에서 76.92% target ASR을 기록했다. ZOO-style finite difference 공격은 epsilon 0.05, query budget 2000 조건에서 38.89% target ASR을 보였으며, Square 대비 높은 query 비용과 낮은 효율을 확인했다. 공격 결과는 attack_index.csv로 통합하여 방어 파트가 adv_file을 바로 입력으로 사용할 수 있도록 정리했다.

## 8. 다음 작업

공격 파트 다음 우선순위:

1. 대표 성공/실패 샘플 선정
   - FGSM 성공/실패
   - PGD 성공
   - Square 성공/실패
   - JSMA 성공
   - ZOO 성공/실패

2. 결과표 정제
   - 보고용 핵심 컬럼만 남긴 compact summary 생성
   - 공격별 ASR, L2, Linf, time, query 중심 정리

3. 방어 파트 연동 확인
   - attack_index.csv 기반으로 방어 파트가 adv_file을 읽을 수 있는지 확인
   - defense_results.csv에 sample_id 유지

4. ZOO는 필요 시 추가 튜닝
   - 더 큰 query budget
   - 더 많은 샘플
   - 단, 시간 비용이 크므로 우선순위는 낮음
