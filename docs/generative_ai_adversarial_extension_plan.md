# 생성형 AI 접목을 통한 Adversarial Face Authentication 확장 제안

작성일: 2026-05-13  
작성 목적: 기존 Adversarial AI 중심 프로젝트를 유지하면서, 생성형 AI를 보안 목적의 확장 모듈로 접목하는 방안 정리

---

## 1. 결론 요약

현재 프로젝트는 LFW 기반 얼굴 인식 모델에 대해 FGSM, PGD, Square, JSMA, ZOO 등 적대적 공격을 수행하고, JPEG compression, Gaussian smoothing, bit-depth reduction 등의 방어 기법을 비교하는 단계까지 진행되었다.

다음 단계에서는 프로젝트를 단순한 얼굴 분류 공격에서 실제 금융 생체인증에 가까운 얼굴 인증 verification 구조로 전환하고, 생성형 AI는 보조 확장 모듈로 도입하는 것이 가장 적절하다.

추천 방향은 다음과 같다.

```text
Adversarial Face Verification Platform
+ Generative AI Purification Defense
```

즉, 생성형 AI를 공격 이미지 생성 자체에 먼저 쓰기보다는, adversarial perturbation을 제거하는 diffusion 기반 정화 방어로 활용한다.

이 방향이 좋은 이유는 다음과 같다.

- 기존 수행계획서의 핵심인 Adversarial AI 공격/방어 연구를 유지할 수 있다.
- 생성형 AI는 “다층적 방어 기술”의 한 레이어로 자연스럽게 추가된다.
- 딥페이크나 얼굴 사칭 생성처럼 위험하게 보일 수 있는 방향을 피하고, 방어·탐지 중심의 보안 프로젝트로 설명할 수 있다.
- 최종 산출물이 단순 실험 코드가 아니라 “금융 얼굴인증 공격·방어·탐지 플랫폼”으로 확장된다.

---

## 2. 현재 프로젝트 상태

현재까지 진행된 공격 실험은 얼굴 인증 verification이 아니라 identity classification 구조에 가깝다.

현재 구조:

```text
입력 얼굴 이미지
    ↓
ResNet-50 identity classifier
    ↓
10명 중 하나로 분류
```

현재 공격 목표:

```text
Ariel_Sharon 이미지를 Colin_Powell 클래스로 분류하게 만들기
```

현재 공격 기법:

- Targeted FGSM
- Targeted PGD
- Targeted Square Attack
- Targeted JSMA variant
- Targeted ZOO-style attack

현재 방어 기법:

- JPEG compression
- Gaussian smoothing
- Bit-depth reduction

현재까지의 의미:

- adversarial attack의 기본 원리와 구현 흐름을 확보했다.
- 공격 결과와 방어 결과를 `sample_id` 기준으로 통합 분석하는 구조를 만들었다.
- 팀원 간 공격/방어 결과를 CSV로 주고받는 협업 포맷이 생겼다.

하지만 금융 생체인증 시스템으로 보려면 다음 전환이 필요하다.

```text
classification: 이 얼굴은 10명 중 누구인가?
verification: 이 얼굴은 등록된 사용자 본인이 맞는가?
```

---

## 3. 왜 Verification으로 전환해야 하는가

금융 얼굴인증은 보통 classification 문제가 아니다.

금융 앱에서 사용자가 얼굴 인증을 할 때 시스템은 “이 사람이 전체 사용자 중 누구인가?”를 묻는 것이 아니라, “현재 얼굴이 등록된 계정 주인의 얼굴과 충분히 비슷한가?”를 판단한다.

따라서 실제 인증 시스템은 보통 다음 구조를 가진다.

```text
등록 얼굴 enroll image
    ↓
face embedding model
    ↓
embedding vector A

검증 얼굴 verify image
    ↓
face embedding model
    ↓
embedding vector B

cosine similarity(A, B)
    ↓
threshold 이상이면 accept
threshold 미만이면 reject
```

이 구조에서 중요한 지표는 단순 accuracy가 아니라 다음과 같다.

- FAR, False Acceptance Rate
  - 타인인데 본인으로 잘못 통과한 비율
- FRR, False Rejection Rate
  - 본인인데 거절된 비율
- EER, Equal Error Rate
  - FAR와 FRR이 같아지는 지점
- ROC-AUC
  - threshold 변화에 따른 인증 성능
- Attack Success Rate
  - 공격 이미지가 타겟 사용자로 인증 통과한 비율

금융 생체인증에서는 특히 FAR이 중요하다. 공격자가 타인의 계정으로 통과하는 상황이 가장 위험하기 때문이다.

---

## 4. 생성형 AI를 어디에 붙일 수 있는가

생성형 AI는 프로젝트에 크게 세 방향으로 접목할 수 있다.

### 4.1 공격 쪽: 생성형 얼굴 공격

가능한 방향:

```text
source face
    ↓
generative model
    ↓
target identity와 더 비슷한 자연스러운 공격 얼굴 생성
```

예시:

- diffusion model을 이용해 target identity와 embedding similarity가 높은 이미지 생성
- latent space에서 identity-preserving 또는 target-impersonation 방향으로 이미지 수정
- adversarial perturbation이 아니라 자연스러운 얼굴 변형으로 인증 통과 시도

장점:

- 연구적으로 흥미롭다.
- 기존 픽셀 노이즈 기반 공격보다 현실적인 공격 시나리오를 만들 수 있다.
- “생성형 AI 기반 face impersonation attack”으로 확장성이 크다.

단점:

- 구현 난이도가 높다.
- 딥페이크·사칭 이미지 생성으로 보일 수 있어 윤리적 설명이 필요하다.
- 현재 팀의 1순위인 Adversarial AI 학습보다 생성형 AI 비중이 커질 위험이 있다.

판단:

```text
1차 확장으로는 비추천.
후반 고도화 또는 실험 아이디어로 남기는 것이 적절하다.
```

### 4.2 방어 쪽: 생성형 AI 기반 정화 방어

가능한 방향:

```text
adversarial image
    ↓
diffusion / generative restoration model
    ↓
purified image
    ↓
face verification model
```

핵심 아이디어:

```text
생성형 모델은 자연 이미지의 분포를 학습한다.
adversarial perturbation은 자연 이미지 분포에서 벗어난 작은 노이즈다.
따라서 생성형 복원 과정을 거치면 perturbation은 줄이고 얼굴 정체성은 보존할 수 있다.
```

구현 후보:

- diffusion denoising
- image-to-image restoration
- super-resolution/restoration 모델 활용
- autoencoder 기반 reconstruction
- diffusion purification을 흉내낸 noise-add + denoise pipeline

장점:

- 기존 다층 방어 구조에 자연스럽게 추가된다.
- 생성형 AI를 보안 목적에 맞게 활용한다.
- 기존 JPEG, smoothing, bit-depth와 같은 입력 변환 방어와 비교하기 쉽다.
- 결과표와 시각화가 명확하다.

단점:

- diffusion 모델이 무겁다.
- 얼굴 정체성까지 변형하면 정상 인증 성능이 떨어질 수 있다.
- Colab GPU 사용량과 실행 시간이 늘어날 수 있다.

판단:

```text
가장 추천하는 생성형 AI 접목 방향.
Adversarial AI를 중심에 두면서 생성형 AI를 방어 레이어로 추가할 수 있다.
```

### 4.3 탐지/리포트 쪽: 생성형 AI 기반 설명 보조

가능한 방향:

```text
공격 결과 CSV + 방어 결과 CSV + 시각화 패널
    ↓
LLM 기반 요약
    ↓
공격 위험도/방어 효과 리포트 생성
```

예시:

- “PGD 공격은 성공률이 높고 perturbation이 작아 가장 위험함”
- “Gaussian smoothing은 전체 방어 성공률은 높지만 recovery 관점에서는 JPEG와 비교 필요”
- “ZOO는 query 수가 많지만 현재 설정에서는 성공률이 낮음”

장점:

- 구현 난이도가 낮다.
- 플랫폼 완성도를 높일 수 있다.
- 교수님/팀원에게 결과를 설명하기 좋다.

단점:

- Adversarial AI 연구의 핵심 기여로 보기는 어렵다.
- 생성형 AI를 단순 문서 요약 도구로만 쓰면 프로젝트 기술성이 약해 보일 수 있다.

판단:

```text
보조 기능으로는 좋지만, 메인 생성형 AI 접목안으로는 부족하다.
```

---

## 5. 최종 추천 아키텍처

최종적으로 다음 구조를 목표로 한다.

```text
                 ┌────────────────────────┐
                 │  Clean face image       │
                 └───────────┬────────────┘
                             │
                             ▼
                 ┌────────────────────────┐
                 │ Face verification model │
                 │ ArcFace / InsightFace   │
                 └───────────┬────────────┘
                             │
                             ▼
                 ┌────────────────────────┐
                 │ Clean FAR / FRR / EER   │
                 └────────────────────────┘


                 ┌────────────────────────┐
                 │ Source attacker face    │
                 └───────────┬────────────┘
                             │
                             ▼
                 ┌────────────────────────┐
                 │ Targeted attack module  │
                 │ FGSM / PGD / Square ... │
                 └───────────┬────────────┘
                             │
                             ▼
                 ┌────────────────────────┐
                 │ Adversarial face image  │
                 └───────────┬────────────┘
                             │
              ┌──────────────┼────────────────┐
              ▼              ▼                ▼
       ┌───────────┐  ┌────────────┐  ┌────────────────────┐
       │ JPEG      │  │ Smoothing  │  │ Generative purifier │
       └─────┬─────┘  └─────┬──────┘  └──────────┬─────────┘
             │              │                    │
             ▼              ▼                    ▼
       ┌────────────────────────────────────────────────────┐
       │ Verification after defense                         │
       │ similarity, accept/reject, attack success, recovery │
       └────────────────────────────────────────────────────┘
```

---

## 6. 구현 단계 제안

### Phase 1. Verification baseline 구축

목표:

```text
얼굴 분류 모델이 아니라 얼굴 인증 모델 기준의 평가 체계를 만든다.
```

구현할 것:

- LFW에서 positive pair 생성
  - 같은 사람 이미지 2장
- LFW에서 negative pair 생성
  - 서로 다른 사람 이미지 2장
- embedding 추출
- cosine similarity 계산
- threshold 기반 accept/reject
- FAR, FRR, EER, ROC-AUC 계산

산출물:

- `outputs/verification/lfw_test_pairs.csv`
- `outputs/verification/verification_scores.csv`
- `outputs/verification/verification_metrics.json`

이 단계가 중요한 이유:

```text
이후 모든 공격과 방어는 classification success가 아니라
verification accept/reject 기준으로 평가되어야 한다.
```

### Phase 2. Verification targeted attack 구현

목표:

```text
공격자 얼굴이 타겟 사용자의 등록 얼굴로 인증 통과하도록 만든다.
```

기존 classification targeted attack:

```text
model(x_adv) = target_class
```

새로운 verification targeted attack:

```text
cosine_similarity(embedding(x_adv), embedding(target_enroll)) >= threshold
```

공격 loss 예시:

```text
loss = 1 - cosine_similarity(embedding(x_adv), embedding(target_enroll))
```

공격자는 이 loss를 줄인다.

구현할 공격:

- Targeted FGSM verification
- Targeted PGD verification
- Query-based attack은 후순위

산출물:

- `outputs/verification_attacks/metadata_targeted_pgd_verification.csv`
- 공격 전 similarity
- 공격 후 similarity
- threshold
- attack_success
- l2, linf

### Phase 3. 기존 방어를 verification 기준으로 재평가

목표:

```text
JPEG, smoothing, bit-depth가 인증 공격에 얼마나 효과적인지 재평가한다.
```

기존 방어 성공 기준:

```text
공격 후 분류 결과가 target class가 아니면 성공
```

새 방어 성공 기준:

```text
공격 이미지는 target 사용자로 accept되었는데,
방어 후 이미지는 reject되면 방어 성공
```

필요 컬럼:

```csv
sample_id,
attack,
source_name,
target_name,
source_file,
target_enroll_file,
adv_file,
defended_file,
similarity_clean,
similarity_adv,
similarity_defended,
threshold,
attack_success,
defense_success,
l2,
linf
```

### Phase 4. 생성형 AI 방어 추가

목표:

```text
Diffusion 또는 generative restoration을 이용해 adversarial perturbation을 제거한다.
```

현실적인 구현 순서:

1. 먼저 lightweight restoration 모델 또는 denoising autoencoder 사용
2. 그다음 diffusion purification으로 확장
3. 최종적으로 기존 방어들과 비교

실험 방식:

```text
adv image
    ↓
generative purification
    ↓
purified image
    ↓
face verification
```

평가 기준:

- attack success before defense
- attack success after defense
- defense success rate
- recovery rate
- clean verification degradation
- FAR/FRR/EER 변화
- average similarity drop
- 이미지 품질 변화

중요한 점:

```text
방어가 공격을 막더라도 정상 사용자의 얼굴 인증까지 망가뜨리면 좋은 방어가 아니다.
```

따라서 생성형 AI 방어는 다음 두 조건을 모두 만족해야 한다.

- 공격 이미지는 reject하게 만든다.
- 정상 이미지는 계속 accept하게 만든다.

### Phase 5. 플랫폼화

목표:

```text
공격, 방어, 탐지 결과를 한 번에 확인할 수 있는 실험 플랫폼으로 정리한다.
```

입력:

- source attacker image
- target enrolled image
- attack method
- defense method

출력:

- 원본 이미지
- 공격 이미지
- 방어 후 이미지
- perturbation 시각화
- similarity before/after
- accept/reject 결과
- 위험도 점수
- CSV/Markdown 리포트

---

## 7. 생성형 AI 방어를 선택했을 때 기대되는 비교 실험

기존 방어와 생성형 방어를 다음처럼 비교한다.

| Attack | No Defense ASR | JPEG ASR | Smoothing ASR | Bit-depth ASR | GenAI Purification ASR |
|---|---:|---:|---:|---:|---:|
| FGSM | 측정 예정 | 측정 예정 | 측정 예정 | 측정 예정 | 측정 예정 |
| PGD | 측정 예정 | 측정 예정 | 측정 예정 | 측정 예정 | 측정 예정 |
| Square | 측정 예정 | 측정 예정 | 측정 예정 | 측정 예정 | 측정 예정 |
| JSMA | 측정 예정 | 측정 예정 | 측정 예정 | 측정 예정 | 측정 예정 |
| ZOO | 측정 예정 | 측정 예정 | 측정 예정 | 측정 예정 | 측정 예정 |

여기서 ASR은 Attack Success Rate이다.

좋은 방어는 다음 조건을 만족한다.

```text
ASR after defense 낮음
clean verification accuracy 유지
FAR 낮음
FRR 과도하게 증가하지 않음
이미지 품질 손상 적음
```

---

## 8. 팀 역할 분담 제안

### 공격 담당

- verification pair 생성
- clean verification baseline 측정
- verification targeted FGSM/PGD 구현
- 공격 이미지와 metadata 생성
- 공격 전후 similarity, l2, linf 기록

### 방어 담당

- 기존 JPEG/smoothing/bit-depth를 verification 공격 결과에 적용
- 방어 후 similarity 재계산
- defense_success, recovery 계산
- 생성형 AI purification 후보 조사 및 적용

### 분석/플랫폼 담당

- 공격/방어 결과 join
- FAR, FRR, EER, ASR 계산
- 대표 이미지 패널 생성
- Markdown/PDF 리포트 생성
- 최종 데모 UI 또는 notebook 정리

---

## 9. 주간회의에서 논의할 질문

1. 생성형 AI를 공격 쪽에 먼저 붙일지, 방어 쪽에 먼저 붙일지
2. verification 모델을 기존 ResNet feature로 임시 진행할지, ArcFace/InsightFace로 바로 전환할지
3. 생성형 AI 방어를 diffusion purification으로 할지, lightweight restoration으로 먼저 할지
4. 최종 산출물을 notebook 중심으로 할지, 간단한 웹 플랫폼 형태로 할지
5. 생성형 AI 사용 범위를 어디까지 허용할지

현재 추천 답변:

```text
1. 방어 쪽 우선
2. ResNet feature baseline 후 ArcFace/InsightFace 전환
3. lightweight restoration으로 실험 구조 확정 후 diffusion purification 확장
4. 처음은 notebook, 최종은 간단한 demo UI
5. 얼굴 생성/사칭보다는 perturbation 제거 중심
```

---

## 10. 우리가 당장 해야 할 일

### 이번 주 목표

```text
Classification attack 프로젝트를 Verification attack 프로젝트로 전환하는 기반 만들기
```

구체적 작업:

1. LFW verification pair CSV 생성
2. clean verification 성능 측정
3. EER threshold 결정
4. verification targeted PGD 설계
5. 기존 공격 CSV 포맷과 verification 공격 CSV 포맷 차이 정리

### 다음 주 목표

```text
Verification targeted PGD를 실제로 성공시키기
```

구체적 작업:

1. source image와 target enroll image 선택
2. embedding similarity loss 구현
3. PGD로 source image를 target embedding에 가깝게 변형
4. attack_success 계산
5. 대표 성공/실패 패널 생성

### 그 다음 목표

```text
기존 방어 및 생성형 AI 방어를 verification 기준으로 비교
```

구체적 작업:

1. JPEG/smoothing/bit-depth를 verification 공격 이미지에 적용
2. 방어 후 similarity 계산
3. 생성형 purification 후보 적용
4. 전체 방어 비교표 생성
5. 최종 보고서 구조 확정

---

## 11. 최종 프로젝트 목표안

최종 목표 문장은 다음처럼 정리할 수 있다.

```text
본 프로젝트는 금융 생체인증 환경을 가정하여 얼굴 verification 모델에 대한 targeted adversarial impersonation attack을 구현하고,
기존 입력 변환 기반 방어와 생성형 AI 기반 purification 방어를 비교 분석하는 공격·방어·탐지 플랫폼을 개발한다.
```

핵심 키워드:

- Face verification
- Targeted adversarial impersonation
- FAR / FRR / EER
- PGD / FGSM / Square / ZOO
- Multi-layer defense
- Generative AI purification
- Diffusion denoising
- Attack-defense integrated analysis

---

## 12. 참고 자료

- MIT 6.S184, Flow Matching and Diffusion Models 2026  
  https://diffusion.csail.mit.edu/2026/

- MIT Professional Education, Deep Learning for AI and Computer Vision  
  https://professional.mit.edu/course-catalog/deep-learning-ai-and-computer-vision

- MIT 6.S191, Introduction to Deep Learning  
  https://introtodeeplearning.com/

