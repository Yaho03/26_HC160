# 방어 Verification 전환 로드맵

작성일: 2026-05-19  
참고: `docs/generative_ai_adversarial_extension_plan.md` 3번, 4번, 8번

---

## 1. Verification 전환 시 핵심 지표

금융 생체인증 관점에서 방어를 평가할 때 아래 지표를 사용한다.

| 지표 | 정의 | 중요도 |
|------|------|--------|
| **FAR** (False Acceptance Rate) | 타인인데 본인으로 잘못 통과한 비율 | ⭐⭐⭐ 최우선 |
| **FRR** (False Rejection Rate) | 본인인데 거절된 비율 | ⭐⭐ |
| **EER** (Equal Error Rate) | FAR = FRR 이 되는 지점 | ⭐⭐ |
| **ROC-AUC** | threshold 변화에 따른 전체 인증 성능 | ⭐⭐ |
| **ASR** (Attack Success Rate) | 공격 이미지가 타겟 사용자로 인증 통과한 비율 | ⭐⭐⭐ |

> 금융 생체인증에서 **FAR이 가장 중요**하다.  
> 공격자가 타인의 계정으로 통과하는 상황이 가장 위험하기 때문이다.

### 방어 성공의 재정의

```
기존 (classification):
  attack_success_after_defense = (pred_after == target_label)

변경 (verification):
  attack_success_after_defense = (cosine_sim(defended, target_enroll) >= threshold)
  defense_success = NOT attack_success_after_defense
```

좋은 방어의 조건:

```
① 공격 이미지는 reject  → FAR 감소, ASR 감소
② 정상 이미지는 accept  → FRR 유지
```

② 가 깨지면 정상 사용자도 인증이 안 되므로 좋은 방어가 아니다.

---

## 2. 기존 방어 기법 재평가 / 수정 방향

### 전처리 3종 (JPEG / Gaussian / Bit-depth)

**수정 범위**: 핵심 변환 로직은 유지, 평가 로직만 추가

```python
# 추가해야 할 평가 로직
similarity_defended = cosine_similarity(
    embedding(defended_image),
    embedding(target_enroll_image)
)
attack_success_verification = (similarity_defended >= threshold)
defense_success_verification = not attack_success_verification
```

**추가할 출력 컬럼**:
- `similarity_clean` — 원본 이미지 similarity
- `similarity_adv` — 공격 후 similarity
- `similarity_defended` — 방어 후 similarity
- `threshold` — EER 기준 threshold
- `attack_success_verification`
- `defense_success_verification`

### Adversarial Training

**수정 범위**: 학습 loss 변경 필요

```python
# 현재 (classification loss)
loss = CrossEntropyLoss(pred, true_label)

# 변경 (verification loss)
loss = 1 - cosine_similarity(
    embedding(adv_image),
    embedding(clean_image)
)
# 목표: adv 이미지의 embedding이 clean과 같아지도록
```

단, verification loss 기반 학습은 공격팀의 verification 공격 완성 후 진행한다.

---

## 3. 생성형 AI 방어 (우선순위 낮음, 고려 사항 정리)

> 구현 전 고려할 내용을 미리 정리해둔다.

### 핵심 아이디어

```
adversarial image
    ↓
generative purification  ← 자연 이미지 분포로 복원
    ↓
purified image
    ↓
face verification model
    ↓
cosine_similarity(purified, target_enroll) → accept/reject
```

생성형 모델은 자연 이미지의 분포를 학습하므로,  
그 분포에서 벗어난 adversarial perturbation을 복원 과정에서 제거할 수 있다.

**적용 범위: 모든 입력 이미지**

실제 인증 시스템에서는 어떤 이미지가 공격인지 알 수 없기 때문에,  
정화는 공격 이미지뿐 아니라 **모든 입력 이미지에 무조건 적용**된다.

```
정상 사용자 이미지  ─┐
                    ├→ 생성형 정화 → verification
공격자 이미지       ─┘
```

이로 인해 핵심 tradeoff가 발생한다:

| 상황 | 결과 |
|------|------|
| 공격 이미지에 정화 | perturbation 제거 → FAR 감소 ✅ |
| 정상 이미지에 정화 | 얼굴 특징 미세 변형 → similarity 저하 → FRR 증가 가능 ⚠️ |

```
정화 강도 너무 강함 → 정상 사용자도 거절됨 (FRR ↑)
정화 강도 너무 약함 → 공격이 여전히 통과함 (FAR ↑)
```

따라서 생성형 방어 평가 시 반드시 다음 두 조건을 모두 확인해야 한다:
- 공격 이미지의 ASR이 낮아졌는가 (FAR 감소)
- 정상 이미지의 인증 성능이 유지되는가 (FRR 변화 없음)

### 사용 가능한 모델 비교

#### A. Denoising Autoencoder (DAE)

```
adv image → encoder → latent → decoder → purified image
```

| 항목 | 내용 |
|------|------|
| 장점 | 가볍고 빠름, 학습/추론 비용 낮음, 구현 쉬움 |
| 단점 | 강한 perturbation 제거 한계, 얼굴 디테일 손실 가능 |
| 적합성 | **1순위 후보** — 실험 구조 빠르게 검증 가능 |

#### B. VAE (Variational Autoencoder)

```
adv image → encoder → 확률 분포 샘플링 → decoder → purified image
```

| 항목 | 내용 |
|------|------|
| 장점 | latent space에서 자연 이미지 분포로 projection |
| 단점 | 얼굴 정체성 변형 가능성 (identity drift) |
| Tradeoff | perturbation 제거 ↑ vs. 얼굴 유사도 ↓ |
| 적합성 | 보조 실험 후보 |

#### C. DiffPure (Diffusion Purification)

```
adv image → noise 추가 (forward) → denoising (reverse) → purified image
```

| 항목 | 내용 |
|------|------|
| 장점 | 이론적으로 가장 강력, perturbation 제거 효과 높음 |
| 단점 | 매우 느림 (샘플당 수 초), GPU 메모리 많이 사용 |
| Tradeoff | 성능 ↑ vs. 속도 ↓, 금융 실시간 인증엔 부적합 |
| 적합성 | 성능 상한선 확인용 실험 |

#### D. GAN 기반 복원 (pix2pix 등)

| 항목 | 내용 |
|------|------|
| 장점 | 학습 후 추론 빠름, 이미지 품질 높음 |
| 단점 | 학습 불안정, mode collapse 위험 |
| Tradeoff | 품질 ↑ vs. 학습 어려움 ↑ |
| 적합성 | 구현 난이도 고려 시 후순위 |

### 프로젝트에 가장 적합한 모델

```
1순위: Denoising Autoencoder (DAE)
  → 빠르게 실험 구조 검증 + 기존 방어와 비교 가능

2순위: DiffPure
  → 시간 여유 있을 때 성능 상한선 확인
```

### 생성형 방어 파이프라인 / 아키텍처

```
┌─────────────────────────────────────┐
│         Input                       │
│  adversarial face image             │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│    Generative Purification          │
│                                     │
│  Option A: DAE                      │
│    adv → encoder → decoder → clean  │
│                                     │
│  Option B: DiffPure                 │
│    adv → add noise → denoise        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│    Face Verification Model          │
│    ResNet-50 feature extractor      │
│    → embedding vector               │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│    Similarity Comparison            │
│    cosine_sim(purified, enrolled)   │
│    >= threshold → accept            │
│    <  threshold → reject            │
└─────────────────────────────────────┘
```

**평가 시 반드시 확인할 것**:
- 공격 이미지 → reject 됐는가 (FAR 감소)
- 정상 이미지 → accept 유지되는가 (FRR 변화 없음)

---

## 4. 방어팀이 구현/고민해야 할 것

### 단기 (공격팀 verification 완성 대기 중)

- [ ] 기존 defense 스크립트 4종에 verification 평가 로직 추가 준비
- [ ] `target_enroll_file`, `threshold` 컬럼 받을 포맷 공격팀과 협의
- [ ] verification 기준 FAR/FRR/EER 계산 함수 구현

### 중기 (공격팀 verification 공격 완성 후)

- [ ] 전처리 방어 3종 → verification 기준 재평가
- [ ] Adversarial Training → verification loss 기반 재학습
- [ ] FAR/FRR/EER 변화 측정 및 시각화
- [ ] 7번 비교 실험 표 채우기 (ASR 기준)

### 생성형 방어 고려 시

- [ ] DAE 구조 설계 (encoder-decoder, LFW 클린 이미지로 학습)
- [ ] 정화 후 verification 성능 유지 여부 확인
- [ ] 기존 방어 4종 + 생성형 방어 비교표 구성

### 설계 시 항상 고려해야 할 tradeoff

```
방어 강도 ↑  →  FRR ↑  (정상 사용자도 거절될 수 있음)
방어 강도 ↓  →  FAR ↑  (공격자가 통과할 수 있음)

금융 인증에서는 FAR을 최소화하되 FRR이 과도하게 높아지지 않는 지점을 찾아야 한다.
```
