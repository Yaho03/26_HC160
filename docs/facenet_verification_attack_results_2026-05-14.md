# FaceNet 기반 Verification Targeted PGD 공격 결과

작성일: 2026-05-14  
실험 목적: ResNet classifier feature 기반 임시 verification baseline을 넘어, pretrained FaceNet-style embedding 모델에서 targeted impersonation attack 성능 확인

---

## 1. 실험 의의

이전 verification 실험은 기존 ResNet-50 identity classifier의 마지막 분류층 직전 feature를 embedding처럼 사용했다. 해당 baseline은 clean verification 성능이 낮아, 실제 얼굴 인증 모델로 보기에는 한계가 있었다.

이번 실험은 `facenet-pytorch`의 `InceptionResnetV1(pretrained=vggface2)` 모델을 사용했다. 이 모델은 얼굴 embedding을 직접 출력하는 pretrained face verification 계열 모델이므로, 금융 얼굴인증 시나리오에 더 가까운 평가가 가능하다.

---

## 2. Clean FaceNet verification 결과

사용 데이터:

- LFW identity 10-class test split
- test image 수: 223
- positive pair: 298
- negative pair: 298
- 총 pair 수: 596

사용 모델:

- `facenet-pytorch/InceptionResnetV1`
- pretrained: `vggface2`
- cosine similarity 기반 verification

Clean verification 결과:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.9996 |
| EER | 0.50% |
| Threshold | 0.4797 |
| Accuracy | 99.50% |
| FAR | 0.67% |
| FRR | 0.34% |
| TP | 297 |
| TN | 296 |
| FP | 2 |
| FN | 1 |

해석:

FaceNet baseline은 ResNet feature baseline보다 훨씬 강하다.

기존 ResNet feature baseline:

| Model | ROC-AUC | EER | Accuracy | FAR | FRR |
|---|---:|---:|---:|---:|---:|
| ResNet classifier feature | 0.6540 | 39.26% | 60.74% | 39.26% | 39.26% |
| FaceNet pretrained embedding | 0.9996 | 0.50% | 99.50% | 0.67% | 0.34% |

즉 FaceNet은 clean 상태에서는 positive/negative pair를 거의 완벽하게 구분한다.

---

## 3. Targeted FaceNet verification PGD 설정

공격 목표:

```text
source 얼굴 이미지의 embedding을 target enrollment 얼굴 embedding에 가깝게 만들어,
source가 target 사용자로 인증 통과하도록 만든다.
```

공격 성공 기준:

```text
cosine_similarity(embedding(x_adv), embedding(target_enroll)) >= threshold
```

공격 loss:

```text
loss = 1 - cosine_similarity(embedding(x_adv), embedding(target_enroll))
```

실험 설정:

| Parameter | Value |
|---|---:|
| Attack | targeted_pgd_facenet_verification |
| Model | InceptionResnetV1 pretrained=vggface2 |
| Steps | 10 |
| Alpha | 0.001 |
| Samples | 100 |
| Pair filter | only_initial_rejects=True |
| Threshold | 0.4797 |

공격 대상은 negative pair 중 clean 상태에서 reject되는 pair만 사용했다. 따라서 공격 성공률은 실제 impersonation 성공률로 해석할 수 있다.

---

## 4. Epsilon sweep 결과

| Epsilon | Samples | Target Accept Rate | Avg Similarity Before | Avg Similarity After | Avg Similarity Gain | Avg L2 | Avg Linf | Avg Time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.001 | 100 | 3.00% | 0.1344 | 0.1988 | 0.0644 | 0.2566 | 0.0010 | 0.5373s |
| 0.003 | 100 | 14.00% | 0.1344 | 0.3235 | 0.1891 | 0.7484 | 0.0030 | 0.5378s |
| 0.005 | 100 | 45.00% | 0.1344 | 0.4347 | 0.3003 | 1.1801 | 0.0050 | 0.5432s |
| 0.010 | 100 | 80.00% | 0.1344 | 0.6090 | 0.4746 | 1.9272 | 0.0100 | 0.5407s |

---

## 5. 결과 해석

FaceNet baseline은 clean verification 성능이 매우 높다.

```text
ROC-AUC 0.9996
EER 0.50%
FAR 0.67%
FRR 0.34%
```

그럼에도 targeted PGD 공격을 적용하면 epsilon 증가에 따라 target accept rate가 빠르게 증가했다.

```text
eps=0.001 → 3%
eps=0.003 → 14%
eps=0.005 → 45%
eps=0.010 → 80%
```

이는 더 강한 얼굴 embedding 모델에서도 targeted impersonation attack이 실제로 동작함을 보여준다.

특히 eps=0.010에서 평균 similarity가 다음처럼 증가했다.

```text
0.1344 → 0.6090
```

threshold가 0.4797이므로, 평균적으로 target 인증 기준을 넘는 수준까지 embedding이 이동했다.

---

## 6. ResNet baseline과 FaceNet baseline 비교

| 항목 | ResNet feature baseline | FaceNet baseline |
|---|---:|---:|
| Clean ROC-AUC | 0.6540 | 0.9996 |
| Clean EER | 39.26% | 0.50% |
| Clean FAR | 39.26% | 0.67% |
| Clean FRR | 39.26% | 0.34% |
| PGD eps=0.005 ASR | 100.00% | 45.00% |
| PGD eps=0.010 ASR | 100.00% | 80.00% |

해석:

- ResNet feature baseline은 인증 모델로 약해서 작은 epsilon에도 쉽게 뚫렸다.
- FaceNet baseline은 clean 성능이 매우 높지만, white-box PGD 공격에는 여전히 취약하다.
- 따라서 이제 프로젝트는 “임시 분류 모델 공격”에서 “pretrained face verification 모델 공격” 단계로 올라왔다.

---

## 7. 다음 실험 계획

### 7.1 더 강한 PGD 설정 확인

현재 eps=0.010, steps=10에서 80% 성공했다. 다음은 공격 강도를 조금 올려 성공률이 어디까지 올라가는지 확인한다.

추천 sweep:

```text
eps: 0.010, 0.015, 0.020, 0.030
steps: 10, 20, 40
alpha: 0.001 또는 eps / steps * 2
```

목표:

- 90% 이상 성공하는 최소 epsilon 찾기
- steps 증가가 성공률에 미치는 영향 확인
- perturbation 크기와 성공률 trade-off 정리

### 7.2 방어 평가로 전환

FaceNet verification 공격 metadata를 기준으로 방어팀/방어 모듈이 평가해야 할 항목:

```text
공격 전: reject
공격 후: accept
방어 후: reject
```

즉 방어 성공은 다음과 같다.

```text
defense_success = attack_success_before_defense and not attack_success_after_defense
```

우선 적용할 방어:

- JPEG compression
- Gaussian smoothing
- Bit-depth reduction

그 다음 확장:

- Generative AI purification
- Denoising autoencoder
- Diffusion purification

### 7.3 생성형 AI 방어 접목

FaceNet 기반 PGD 공격 결과가 확보되었으므로, 생성형 AI는 공격 이미지 정화 방어로 붙이는 것이 자연스럽다.

구조:

```text
targeted PGD adversarial face
    ↓
generative purification
    ↓
purified face
    ↓
FaceNet verification
```

비교 지표:

- attack success rate before defense
- attack success rate after defense
- defense success rate
- clean verification accuracy degradation
- FAR / FRR / EER 변화
- similarity drop
- image quality 변화

---

## 8. 팀 공유용 요약

```text
기존 ResNet feature 기반 verification은 clean EER가 39.26%로 약했기 때문에,
pretrained FaceNet embedding 모델을 추가로 적용했습니다.

FaceNet clean verification에서는 ROC-AUC 0.9996, EER 0.50%, FAR 0.67%, FRR 0.34%로
얼굴 인증 모델로서 훨씬 안정적인 baseline을 얻었습니다.

이 강한 baseline에서도 targeted PGD impersonation attack을 수행한 결과,
eps=0.001에서는 3%, eps=0.003에서는 14%, eps=0.005에서는 45%, eps=0.010에서는 80%의 target accept 성공률을 보였습니다.

따라서 프로젝트는 단순 얼굴 분류 공격에서 실제 face verification embedding 모델에 대한 impersonation attack 단계로 확장되었습니다.
다음 단계는 더 강한 PGD sweep으로 최소 성공 epsilon을 찾고, JPEG/smoothing/bit-depth 및 생성형 AI purification 방어를 FaceNet verification 기준으로 평가하는 것입니다.
```

