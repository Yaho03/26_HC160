# 방어 파트 진행 기록

작성일: 2026-05-19  
참고: `docs/generative_ai_adversarial_extension_plan.md`

---

## 최종 프로젝트 목표

> 금융 생체인증 환경을 가정하여 얼굴 verification 모델에 대한 targeted adversarial impersonation attack을 구현하고,  
> 기존 입력 변환 기반 방어와 다층 방어 기법을 비교 분석하는 공격·방어·탐지 플랫폼을 개발한다.

---

## 완료 사항 (Classification 기반)

### 방어 기법 구현 (4종)

| 방어 | 파라미터 | 방어 성공률 | 복원율 |
|------|----------|-----------|--------|
| JPEG 압축 | quality = 75 | 83.6% | 51.9% |
| Gaussian Blur | radius = 3 | 90.2% | 48.3% |
| Bit-depth 축소 | bits = 4 | 86.0% | 48.3% |
| Adversarial Training | PGD, epochs=5, mix_ratio=0.5 | **99.8%** | **99.8%** |

- 평가 기준: classification (`pred_after == target_label` 여부)
- 평가 대상: 5종 공격(FGSM, PGD, SQUARE, JSMA, ZOO) × 499 샘플

> ZOO 공격에 대해 전처리 방어 성능이 가장 낮음 (JPEG 33.3%, Bitdepth에서 신뢰도 역효과 발생)  
> 적대적 학습은 PGD만으로 학습했음에도 5종 공격 전체에 일반화됨

### 구현된 소스 코드

```
src/defenses/
  defense_jpeg.py
  defense_smoothing.py
  defense_bitdepth.py
  defense_adv_training.py
  run_preprocessing_defenses.py
  plot_results.py
  summarize_defense.py
```

### Colab 파이프라인

- `notebooks/colab_defense_pipeline.ipynb` 단일 노트북으로 전체 실행
- Drive 캐시 — 결과 저장 및 재실행 시 재사용
- tqdm 진행바 — 실시간 진행도 표시

### 결과물 (GitHub 커밋 완료)

```
outputs/defenses/
  defense_summary.csv
  defense_report.md
  adv_training_results.csv
  figures/ (heatmap, bar_by_attack, boxplot_conf_drop)
```

---

## 결과 분석

### 새로 구현한 방어 기법 — Adversarial Training 결과

적대적 학습은 PGD 적대적 예제만으로 fine-tune 했음에도 5종 공격 전체에서 압도적인 성능을 보였다.

| 공격 | 방어 성공률 | 복원율 | 신뢰도 감소 |
|------|-----------|--------|-----------|
| FGSM | **100.0%** | **100.0%** | 0.485 |
| PGD | 99.4% | 99.4% | 0.977 |
| SQUARE | **100.0%** | **100.0%** | 0.297 |
| JSMA | **100.0%** | **100.0%** | 0.344 |
| ZOO | **100.0%** | **100.0%** | 0.298 |
| **전체** | **99.8%** | **99.8%** | 0.562 |

주목할 점:
- **학습에 사용하지 않은 공격(FGSM, SQUARE, JSMA, ZOO)에도 100%** 방어 성공률 달성
- **복원율이 방어 성공률과 동일** — 공격을 막는 것을 넘어 원래 레이블로 정확히 복원
- PGD 공격에서 신뢰도 감소가 0.977로 가장 높음 — 학습에 사용한 공격이므로 당연

**한계점**

100% 수치는 과장된 결과일 수 있다.

현재 공격 이미지는 전부 **원본 모델(`best.pt`) 기준**으로 생성되었고, 적대적 학습은 **새 모델(`best_adv.pt`)** 을 만든다. 즉, 공격 이미지가 새 모델을 타겟으로 생성된 것이 아니기 때문에 잘 안 먹히는 것이다.

```
공격 생성 기준: best.pt
방어 평가 기준: best_adv.pt  ← 다른 모델
```

진정한 강건성을 측정하려면 `best_adv.pt` 를 타겟으로 새로 공격을 생성해야 하지만, 이는 끝없는 공격-방어 반복(군비 경쟁)으로 이어지므로 실제 연구에서는 **평가 조건을 고정**하는 것이 일반적이다.

따라서 본 실험의 적대적 학습 결과는 **"원본 모델 기준 공격에 대해 다른 방어 기법 대비 얼마나 강한가"** 의 상대적 비교로 해석하는 것이 적절하다.

### 기존 전처리 방어와 비교

**방어 성공률 비교**

| 공격 | JPEG | Gaussian | Bit-depth | Adv.Training |
|------|------|---------|-----------|-------------|
| FGSM | 55.3% | 76.6% | 63.8% | **100.0%** |
| PGD | 87.6% | 93.5% | 88.8% | **99.4%** |
| SQUARE | 84.2% | 88.1% | 87.1% | **100.0%** |
| JSMA | 90.5% | 93.5% | 91.1% | **100.0%** |
| ZOO | 33.3% | 66.7% | 50.0% | **100.0%** |
| **전체** | 83.6% | 90.2% | 86.0% | **99.8%** |

**핵심 차이점**

1. **ZOO 공격** — 전처리 방어가 가장 취약한 구간. JPEG 33.3%, Bit-depth 신뢰도 감소 -0.033(역효과). 적대적 학습은 100% 방어
2. **FGSM 공격** — 전처리 방어 중 JPEG가 55.3%로 가장 낮음. 적대적 학습은 100% 방어
3. **복원율** — 전처리 방어는 방어 성공률 대비 복원율이 절반 수준(48~52%)에 불과하지만, 적대적 학습은 방어 성공 = 복원으로 동일

**처리 시간 비교**

| 방어 | 학습 시간 | 추론 시간(샘플당) |
|------|---------|----------------|
| JPEG | - | 약 200ms |
| Gaussian | - | 약 210ms |
| Bit-depth | - | 약 220ms |
| Adv.Training | 30분~1시간 (최초 1회) | 약 9ms |

> 적대적 학습은 초기 학습 비용이 크지만, 추론 시간은 오히려 가장 빠름

---

## 다음 진행 사항

### 단기 — 공격팀 Verification 전환 대응

공격팀이 verification 기반으로 전환하는 동안, 방어팀은 연동 준비를 진행한다.

- [ ] 공격팀 verification 공격 결과 CSV 포맷 확인
- [ ] 기존 방어 스크립트가 수용해야 할 새 컬럼 정리

```
필요 컬럼 (공격팀으로부터 받아야 할 것)
  target_enroll_file    ← 등록된 타겟 얼굴 이미지 경로
  similarity_adv        ← 공격 후 cosine similarity
  threshold             ← EER 기준 threshold
```

### 중기 — Verification 기준 방어 재평가

공격팀의 verification 공격 완성 후 기존 방어 4종을 재적용한다.

**방어 성공 기준 변경**

```
기존: pred_after != target_label
변경: cosine_similarity(defended, target_enroll) < threshold
```

**구현할 것**

- [ ] 기존 defense 스크립트에 verification 평가 로직 추가
- [ ] 방어 후 cosine similarity 재계산
- [ ] 출력 CSV에 컬럼 추가
  - `similarity_defended`
  - `attack_success_verification`
  - `defense_success_verification`
- [ ] FAR, FRR, EER 변화 측정

**채워야 할 비교 실험 표** (계획서 7번)

| Attack | No Defense ASR | JPEG ASR | Smoothing ASR | Bit-depth ASR | Adv.Training ASR |
|--------|---------------|----------|--------------|--------------|-----------------|
| FGSM | - | - | - | - | - |
| PGD | - | - | - | - | - |
| SQUARE | - | - | - | - | - |
| JSMA | - | - | - | - | - |
| ZOO | - | - | - | - | - |

> ASR = Attack Success Rate (verification 기준)

### 방어 기법 수정 필요 사항

Phase 3 결과를 반영해 각 방어 기법을 아래와 같이 수정해야 한다.

**전처리 3종 (JPEG / Gaussian / Bit-depth)**

핵심 변환 로직은 그대로 유지하되, **평가 로직만 수정** 필요.

| 현재 | 수정 후 |
|------|--------|
| `pred_after_defense != target_label` 로 성공 판단 | `cosine_similarity(defended, target_enroll) < threshold` 로 성공 판단 |
| 출력: `pred_after_defense`, `recovered` | 추가: `similarity_defended`, `attack_success_verification`, `defense_success_verification` |

→ `defense_jpeg.py` / `defense_smoothing.py` / `defense_bitdepth.py` 에 verification 평가 모드 추가

---

**Adversarial Training**

가장 큰 수정이 필요하다. 현재는 **classification loss** 기준으로 학습했기 때문에 verification 공격에는 효과가 불확실하다.

| 현재 | 수정 후 |
|------|--------|
| `CrossEntropyLoss(pred, true_label)` | `1 - cosine_similarity(embedding(adv), embedding(clean))` |
| 목표: 분류 정확도 유지 | 목표: embedding 공간에서 적대적 이미지와 원본의 유사도 유지 |
| 학습 데이터: clean + PGD adv 이미지 | 학습 데이터: (clean, adv) 쌍 — embedding이 같아지도록 |

→ `defense_adv_training.py` 의 학습 loss 변경 필요  
→ 단, 공격팀의 verification 공격 완성 후 verification loss 기준 재학습 진행

---

### 최종 산출물 방향

계획서 9번 추천안 기준:
- verification 기준 방어 비교표 + FAR/FRR/EER 변화 시각화
- notebook 중심으로 먼저 완성 → 최종 demo UI 확장
- 최종 보고서에 classification 결과 + verification 결과 통합 비교 포함
