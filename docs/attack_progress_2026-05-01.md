# 공격 모듈 1차 진행상황 정리 (2026-05-01)

## 1. 현재까지 진행한 작업

공격 담당 파트에서 얼굴인식 모델 기준선 구축과 targeted attack 실험 파이프라인을 1차로 구성했다.

진행 완료 항목:

- LFW 데이터셋 Colab 연동
- LFW 상위 10명 identity classification 데이터셋 구성
- ResNet-50 기반 얼굴 신원 분류 모델 학습
- targeted FGSM 공격 구현 및 실험
- targeted PGD 공격 구현 및 실험
- targeted Square Attack 구현 및 실험
- targeted multi-pixel JSMA 변형 공격 구현 및 실험
- 공격별 metadata CSV 저장
- 공격 결과 summary CSV 생성
- 원본 / adversarial / perturbation 시각화 패널 생성
- 주요 결과 Google Drive 백업

## 2. 실험 환경

- 실행 환경: Google Colab Pro
- 사용 장치: CUDA GPU
- 데이터셋: LFW deepfunneled
- 분류 대상: LFW 이미지 수 상위 10명
- 모델: ImageNet pretrained ResNet-50 fine-tuning
- 입력 크기: 224 x 224
- 공격 방식: targeted attack
- target 설정: 기본적으로 true label의 다음 class로 유도하는 cyclic target 방식

## 3. 얼굴 분류 모델 학습 결과

ResNet-50을 LFW 10-class identity classifier로 fine-tuning했다.

학습 결과:

- validation accuracy: 76.61%
- test accuracy: 76.23%
- 공격 평가 subset clean accuracy: 91.00% (100장 중 91장 정분류)

해석:

- 전체 test accuracy는 약 76% 수준이지만, 공격 평가에 사용한 100장 subset에서는 91장이 정상 분류되었다.
- 공격 성공률은 전체 샘플 기준과 원본 정분류 샘플 기준을 함께 기록했다.
- 보고 및 비교에는 `target_success_rate_on_clean`을 주요 ASR 지표로 사용하는 것이 적절하다.

## 4. 공격별 결과 요약

### 4.1 Targeted FGSM

FGSM은 1-step gradient 기반 targeted attack으로 구현했다.

| epsilon | target_success_rate_on_clean | avg_l2 | avg_linf | avg_time_sec |
|---:|---:|---:|---:|---:|
| 0.005 | 46.15% | 1.8839 | 0.005 | 0.0316 |
| 0.010 | 41.76% | 3.7541 | 0.010 | 0.0317 |
| 0.030 | 16.48% | 11.1646 | 0.030 | 0.0314 |
| 0.050 | 21.98% | 18.4937 | 0.050 | 0.0321 |

해석:

- FGSM은 매우 빠르지만 targeted setting에서는 성공률이 안정적이지 않았다.
- epsilon을 키운다고 target identity로 더 잘 유도되지는 않았다.
- 1-step 방식이라 target class 방향으로 정교하게 최적화하는 데 한계가 있다.

### 4.2 Targeted PGD

PGD는 epsilon ball 안에서 여러 step 동안 target loss를 줄이는 iterative targeted attack으로 구현했다.

대표 결과:

| epsilon | alpha | steps | target_success_rate_on_clean | avg_l2 | avg_linf | avg_time_sec |
|---:|---:|---:|---:|---:|---:|---:|
| 0.005 | 0.0005 | 10 | 100.00% | 0.6160 | 0.005 | 0.1670 |
| 0.010 | 0.0010 | 10 | 100.00% | 1.1974 | 0.010 | 0.1692 |
| 0.030 | 0.0030 | 10 | 100.00% | 3.5556 | 0.030 | 0.1853 |
| 0.050 | 0.0050 | 10 | 100.00% | 5.9130 | 0.050 | 0.1939 |

해석:

- PGD는 모든 주요 설정에서 원본 정분류 샘플 기준 100% target success를 보였다.
- FGSM보다 느리지만 여전히 이미지 1장당 0.17~0.19초 수준으로 실험 가능하다.
- 동일 epsilon 제약 안에서 target confidence를 크게 높여 가장 강한 화이트박스 공격 기준선으로 사용할 수 있다.

주의:

- `epsilon=0.05, alpha=0.003, steps=10` 설정은 실제 최대 이동량이 `alpha * steps = 0.03`이라 avg_linf가 0.03에 머물렀다.
- 이후 비교에는 epsilon별로 alpha를 epsilon / steps 수준으로 맞춘 결과를 우선 사용한다.

### 4.3 Targeted Square Attack

Square Attack은 gradient를 사용하지 않는 black-box 방식으로 구현했다.

| epsilon | max_queries | target_success_rate_on_clean | avg_queries_used | avg_l2 | avg_linf | avg_time_sec |
|---:|---:|---:|---:|---:|---:|---:|
| 0.005 | 300 | 25.27% | 235.12 | 1.0889 | 0.005 | 1.4217 |
| 0.010 | 300 | 37.36% | 212.56 | 2.1712 | 0.010 | 1.2585 |
| 0.030 | 300 | 61.54% | 161.82 | 6.4642 | 0.030 | 0.9715 |
| 0.050 | 300 | 76.92% | 125.16 | 10.7210 | 0.050 | 0.7404 |

해석:

- Square Attack은 black-box 공격이므로 PGD보다 성공률은 낮지만, epsilon 증가에 따라 ASR이 자연스럽게 상승했다.
- epsilon이 커질수록 평균 query 사용량과 시간은 감소했다.
- 수행계획서의 black-box 공격 비교축으로 활용 가능하다.

### 4.4 Targeted multi-pixel JSMA 변형

고전 JSMA는 고해상도 이미지에서 픽셀 단위 탐색 비용이 크기 때문에, saliency가 큰 여러 channel을 한 step에서 동시에 수정하는 multi-pixel 변형으로 구현했다.

설정:

- theta: 0.05
- steps: 20
- pixels_per_step: 200
- max_pixel_ratio: 0.05

결과:

| attack | target_success_rate_on_clean | avg_changed_channels | avg_l2 | avg_linf | avg_time_sec |
|---|---:|---:|---:|---:|---:|
| targeted multi-pixel JSMA | 100.00% | 1304.0 | 1.6428 | 0.048 | 0.1758 |

해석:

- 원본 정분류 샘플 91장 전체를 target identity로 유도했다.
- perturbation 패널에서 PGD보다 희소한 점 형태의 변화가 관찰된다.
- 수행계획서에 명시한 “복수 픽셀 동시 수정 변형 JSMA” 방향의 1차 구현으로 볼 수 있다.

## 5. 시각화 결과

생성한 패널은 다음 형식이다.

```text
original | adversarial | perturbation
```

확인된 특징:

- 원본과 adversarial 이미지는 육안상 거의 동일하다.
- PGD perturbation은 이미지 전체에 넓게 분포한다.
- JSMA perturbation은 상대적으로 희소한 점 형태로 나타난다.
- target prediction이 실제로 목표 class로 변경된 성공 사례를 확인했다.

저장 위치:

```text
outputs/attack_panels/fgsm_eps0005/
outputs/attack_panels/pgd_eps003/
outputs/attack_panels/jsma_theta005/
```

Google Drive 백업 위치:

```text
MyDrive/hanium-aml/results/
```

## 6. 현재 결론

1차 공격 실험에서는 ResNet-50 기반 LFW 얼굴 신원 분류 모델에 대해 targeted attack 파이프라인을 구축했다.

현재 결과 기준 공격 강도는 다음과 같이 정리할 수 있다.

```text
PGD ≈ multi-pixel JSMA > Square Attack > FGSM
```

- PGD: 가장 강한 white-box 기준선
- JSMA 변형: 희소 perturbation 기반 targeted attack으로 프로젝트 독창성 포인트
- Square Attack: black-box 조건에서 의미 있는 공격 성공률 확보
- FGSM: 빠른 baseline이나 targeted setting에서는 성공률이 불안정

## 7. 팀 공유용 한 줄 요약

공격 파트는 LFW 기반 ResNet-50 얼굴 분류 모델을 학습하고, targeted FGSM, PGD, Square, multi-pixel JSMA 공격을 구현 및 실험했다. 현재 PGD와 JSMA 변형은 원본 정분류 샘플 기준 100% target success를 달성했고, Square Attack은 black-box 조건에서 epsilon 0.05 기준 76.92% target success를 보였다.

## 8. 다음 작업

우선순위는 다음과 같다.

1. 공격 코드 구조 정리
   - 공통 attack interface 정리
   - metadata column 통일
   - output path 규칙 정리

2. 방어 담당과 연동할 출력 포맷 확정
   - adversarial image 경로
   - perturbation image 경로
   - metadata CSV
   - success / clean_correct / target label / L0/L2/Linf / time

3. 공격 결과 시각화 보강
   - 공격별 대표 성공 사례 3~5개 선정
   - 실패 사례도 일부 확인
   - FGSM vs PGD vs Square vs JSMA 비교 패널 구성

4. 이후 공격 확장 후보
   - ZOO: 계획서의 black-box 공격 항목을 채우기 위한 후보
   - 단, 구현 및 실행 비용이 높으므로 현재 4종 결과 정리 후 진행 권장

## 9. 팀원에게 공유할 메시지 초안

공격 파트 1차 구현 진행상황 공유합니다.

- LFW 데이터셋 기반으로 10명 identity classification용 ResNet-50 모델을 학습했습니다.
- Colab Pro CUDA 환경에서 12 epoch 학습했고, test accuracy는 76.23%, 공격 평가 subset clean accuracy는 91%입니다.
- targeted FGSM, targeted PGD, targeted Square Attack, targeted multi-pixel JSMA 변형을 구현했습니다.
- PGD와 JSMA 변형은 원본을 맞힌 샘플 기준 target success 100%를 달성했습니다.
- Square Attack은 black-box 설정에서 epsilon 0.05 기준 target success 76.92%를 보였습니다.
- 공격 결과는 metadata CSV와 original/adversarial/perturbation 패널로 저장했습니다.
- 다음 단계는 방어 파트와 연동할 metadata/output 포맷을 맞추고, 공격별 대표 성공/실패 사례를 정리하는 것입니다.

## 10. 멘토/회의 보고용 짧은 버전

이번 주 공격 파트에서는 LFW 기반 ResNet-50 얼굴 신원 분류 모델을 구축하고 targeted adversarial attack 실험 파이프라인을 구성했습니다. 화이트박스 공격으로 FGSM, PGD, JSMA 변형을 구현했고, 블랙박스 공격으로 Square Attack을 구현했습니다. 실험 결과 PGD와 multi-pixel JSMA 변형은 원본 정분류 샘플 기준 100% target success를 보였고, Square Attack은 epsilon 0.05, query budget 300 조건에서 76.92% target success를 기록했습니다. 다음 주에는 방어 파이프라인과 연동 가능한 출력 포맷을 정리하고, 공격별 정성/정량 비교 자료를 보강할 예정입니다.
