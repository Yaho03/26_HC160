# Verification 기반 Targeted PGD 공격 1차 결과

작성일: 2026-05-14  
실험 목적: 기존 identity classification 공격에서 금융 얼굴인증에 가까운 face verification 공격으로 전환 가능성 확인

---

## 1. 실험 요약

기존 공격 실험은 얼굴 이미지를 10개 identity class 중 하나로 분류하는 ResNet-50 classifier를 대상으로 수행했다. 이번 실험은 같은 checkpoint를 feature extractor처럼 사용하여 두 얼굴 이미지의 cosine similarity를 계산하고, threshold 기반으로 accept/reject를 판단하는 verification 구조를 추가했다.

이번 실험의 핵심은 다음과 같다.

```text
source 얼굴 이미지가 target enrollment 얼굴 이미지로 인증 통과하도록
source 이미지를 PGD 방식으로 미세 변형한다.
```

즉 기존 공격이 다음 목표였다면:

```text
model(x_adv) = target_class
```

이번 verification 공격은 다음 목표다.

```text
cosine_similarity(embedding(x_adv), embedding(target_enroll)) >= threshold
```

---

## 2. Clean verification baseline

사용 데이터:

- LFW identity 10-class test split
- test image 수: 223
- positive pair: 298
- negative pair: 298
- 총 pair 수: 596

사용 모델:

- 기존 ResNet-50 face identity classifier checkpoint
- 마지막 classification layer 직전 feature를 embedding처럼 사용
- cosine similarity 기반 verification

Clean verification 결과:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.6540 |
| EER | 39.26% |
| Threshold | 0.2713 |
| Accuracy | 60.74% |
| FAR | 39.26% |
| FRR | 39.26% |

해석:

현재 baseline은 실제 face verification 전용 모델이 아니라 identity classifier의 feature를 임시 embedding으로 사용한 것이다. 따라서 clean verification 성능은 높지 않다. 특히 FAR이 39.26%로 높기 때문에, 금융 얼굴인증 시스템으로 사용하기에는 부적절하다.

하지만 이 baseline은 다음 단계로 넘어가기 위한 실험 구조를 검증하는 데 의미가 있다.

```text
pair 생성 → similarity 계산 → threshold 결정 → accept/reject → FAR/FRR/EER 측정
```

위 verification 평가 파이프라인이 정상 동작함을 확인했다.

---

## 3. Targeted verification PGD 공격 설정

공격 대상:

- negative pair 중 공격 전에는 reject되는 pair만 선택
- 즉 source와 target이 서로 다른 사람이고, clean 상태에서는 인증 통과하지 못하는 경우만 공격

공격 설정:

| Parameter | Value |
|---|---:|
| Attack | targeted_pgd_verification |
| Steps | 10 |
| Alpha | 0.003 |
| Limit | 100 |
| Target threshold | 0.2713 |
| Pair filter | only_initial_rejects=True |

공격 loss:

```text
loss = 1 - cosine_similarity(embedding(x_adv), embedding(target_enroll))
```

공격은 이 loss를 줄이는 방향으로 진행된다. 즉 source adversarial image의 embedding이 target enrollment image의 embedding과 가까워지도록 픽셀을 업데이트한다.

---

## 4. Epsilon sweep 결과

공격 전에는 모두 reject되던 100개 negative pair만 대상으로 실험했다.

| Epsilon | Samples | Success Rate | Avg Similarity Before | Avg Similarity After | Avg Similarity Gain | Avg L2 | Avg Linf | Avg Time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.005 | 100 | 100.00% | 0.2256 | 0.5033 | 0.2777 | 1.3934 | 0.0050 | 0.2106s |
| 0.010 | 100 | 100.00% | 0.2256 | 0.5543 | 0.3286 | 2.4499 | 0.0100 | 0.2087s |
| 0.020 | 100 | 100.00% | 0.2256 | 0.5702 | 0.3445 | 3.4468 | 0.0200 | 0.2123s |
| 0.030 | 100 | 100.00% | 0.2256 | 0.5755 | 0.3498 | 3.5444 | 0.0300 | 0.2123s |

---

## 5. 결과 해석

이번 실험에서 가장 중요한 결과는 다음이다.

```text
공격 전 reject되던 negative pair만 대상으로 했는데도
epsilon 0.005부터 0.030까지 모두 100% target accept에 성공했다.
```

이는 현재 ResNet feature 기반 verification baseline이 targeted PGD impersonation attack에 매우 취약하다는 의미다.

특히 epsilon 0.005에서도 평균 similarity가 다음처럼 증가했다.

```text
0.2256 → 0.5033
```

threshold가 0.2713이므로, 공격 전에는 threshold 아래였던 source-target pair가 공격 후에는 threshold를 크게 넘어 인증 통과했다.

---

## 6. 중요한 한계

이 결과를 해석할 때 반드시 다음 한계를 같이 언급해야 한다.

현재 모델은 ArcFace, FaceNet, InsightFace 같은 얼굴 인증 전용 embedding 모델이 아니다. 기존 ResNet-50 identity classifier의 feature를 임시로 embedding처럼 사용했다.

따라서 이번 결과는 다음을 의미한다.

```text
verification 공격 파이프라인 구현과 동작 검증에는 성공했다.
다만 실제 금융 얼굴인증 수준의 강한 모델에 대한 공격 성능은 아직 검증하지 않았다.
```

즉, 이번 실험은 최종 결론이 아니라 다음 단계로 넘어가기 위한 proof-of-concept이다.

---

## 7. 다음 단계

### 7.1 ArcFace 또는 InsightFace 기반 verification 모델로 교체

가장 중요한 다음 작업은 얼굴 인증 전용 embedding 모델을 사용하는 것이다.

후보:

- ArcFace
- InsightFace
- FaceNet
- DeepFace

추천:

```text
InsightFace / ArcFace 계열 우선 검토
```

이유:

- face verification에서 널리 사용된다.
- embedding similarity 기반 인증 구조와 잘 맞는다.
- FAR, FRR, EER 기준 평가가 자연스럽다.

실행상 고려:

```text
InsightFace/ArcFace는 실무적으로 좋은 선택이지만 Colab에서 ONNX 기반으로 동작하는 경우가 많아
gradient 기반 PGD 공격을 바로 연결하기 어렵다.
따라서 다음 실험은 우선 facenet-pytorch의 InceptionResnetV1 pretrained embedding 모델로 진행하고,
그 후 InsightFace/ArcFace는 clean verification 및 black-box/transfer attack 평가로 확장한다.
```

### 7.2 Verification PGD를 전용 embedding 모델에 다시 적용

현재 구현한 `targeted_pgd_verification.py`의 구조는 그대로 유지할 수 있다. 바뀌는 것은 embedding extractor이다.

현재:

```text
ResNet-50 classifier feature
```

다음:

```text
ArcFace / InsightFace embedding
```

공격 loss는 동일하게 사용할 수 있다.

```text
loss = 1 - cosine_similarity(embedding(x_adv), embedding(target_enroll))
```

### 7.3 기존 방어를 verification 기준으로 재평가

기존 방어:

- JPEG compression
- Gaussian smoothing
- Bit-depth reduction

새 평가 기준:

```text
공격 전: reject
공격 후: accept
방어 후: reject
```

즉 방어 성공은 다음 조건으로 정의한다.

```text
attack_success_before_defense == True
attack_success_after_defense == False
```

### 7.4 생성형 AI 방어로 확장

verification 공격과 방어 평가가 안정화되면 생성형 AI 기반 방어를 추가한다.

추천 방향:

```text
Generative AI Purification Defense
```

구조:

```text
adversarial image
    ↓
generative denoising / diffusion purification
    ↓
purified image
    ↓
verification model
```

목표:

- adversarial perturbation 제거
- 정상 얼굴 identity 보존
- 공격 성공률 감소
- clean verification 성능 저하 최소화

---

## 8. 팀 공유용 요약 문장

회의나 팀 공유 시 다음처럼 설명할 수 있다.

```text
기존 10-class 얼굴 분류 공격에서 한 단계 확장하여,
두 얼굴의 embedding similarity와 threshold로 accept/reject를 판단하는 face verification baseline을 구현했습니다.

ResNet-50 classifier feature를 임시 embedding으로 사용한 1차 baseline에서는 ROC-AUC 0.6540, EER 39.26%로 clean verification 성능은 낮았습니다.

하지만 이 구조를 대상으로 targeted PGD impersonation attack을 수행한 결과,
공격 전 reject되던 negative pair 100개에 대해 epsilon 0.005부터 0.030까지 모두 100% target accept에 성공했습니다.

이는 verification 공격 파이프라인이 정상 동작함을 보여주며,
다음 단계에서는 ArcFace/InsightFace 같은 얼굴 인증 전용 embedding 모델로 교체해 같은 공격과 방어 실험을 재수행할 예정입니다.
```
