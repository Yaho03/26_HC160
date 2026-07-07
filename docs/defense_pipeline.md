# 방어 파이프라인 설계

## 1. 전체 구조

```
입력 이미지 (카메라 or API)
        │
        ▼
┌───────────────────────────────┐
│  1단계: 입력 검증              │
│  Temporal Consistency         │
│  - 연속 프레임 embedding 분석  │
│  - 비정상 패턴 → 즉시 거부     │
└───────────────┬───────────────┘
                │ 통과
                ▼
┌───────────────────────────────┐
│  2단계: 전처리 방어 (병렬)     │
│                               │
│  ROI-first                    │
│  + Gaussian Smoothing         │
│  + Randomized Smoothing       │
│         │                     │
│         ▼                     │
│  Ensemble Voting              │
│  (과반수 reject → 거부)        │
└───────────────┬───────────────┘
                │ 통과
                ▼
┌───────────────────────────────┐
│  3단계: 인증                   │
│  FaceNet cosine similarity    │
│  (Adversarial Training 모델)  │
│  threshold: 0.47966           │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  4단계: 사후 포렌식            │
│  Feature Squeezing 탐지       │
│  risk_score 산출              │
│  탐지 룰 적용 (FA-R001~R008)  │
│  대시보드 전송                 │
└───────────────────────────────┘
```

---

## 2. 기법별 역할 및 구현 상태

| 기법 | 역할 | 단계 | 구현 상태 | 구현 파일 | 난이도 |
|------|------|------|-----------|-----------|--------|
| Temporal Consistency | 카메라 연속 프레임 분석, 정지 이미지·화면 재생 차단 | 1단계 | ✅ 완료 | `src/verification/defenses/verification_defense_temporal.py` | 중 |
| ROI-first | Otsu 이진화로 얼굴 영역만 추출, 배경 perturbation 제거 | 2단계 | ✅ 완료 | `src/verification/defenses/verification_defense_roi.py` | 중 |
| Gaussian Smoothing | 고주파 perturbation 억제 | 2단계 | ✅ 완료 | `src/verification/defenses/verification_defense_smoothing.py` | 하 |
| Randomized Smoothing | 가우시안 노이즈 반복 주입 후 평균 embedding, certified defense | 2단계 | ✅ 완료 | `src/verification/defenses/verification_defense_randomized_smoothing.py` | 상 |
| Ensemble Voting | ROI-first·Smoothing·Randomized Smoothing 결과 voting으로 최종 판정 | 2단계 | ✅ 완료 | `src/verification/defenses/verification_defense_ensemble.py` | 중 |
| Adversarial Training | FaceNet verification loss 기반 모델 재학습 | 3단계 | ✅ 완료 | `src/verification/defenses/verification_defense_adv_training.py` | 상 |
| Feature Squeezing | 여러 해상도로 squeezing 후 prediction 차이로 공격 탐지 | 4단계 | 미구현 | - | 중 |
| JPEG 재압축 | 실험 baseline | baseline | ✅ 완료 | `src/verification/defenses/verification_defense_jpeg.py` | 하 |
| Bit-depth Reduction | 실험 baseline | baseline | ✅ 완료 | `src/verification/defenses/verification_defense_bitdepth.py` | 하 |

---

## 3. 실험 결과 (212개 샘플, 방어 전 공격 성공 171개 기준)

| 기법 | 방어 성공 | 공격 차단률 | 방어 후 공격 성공률 | 결과 파일 |
|------|----------|------------|-------------------|-----------|
| JPEG 재압축 (baseline) | - | - | - | `outputs/verification_defense/jpeg/` |
| Bit-depth Reduction (baseline) | - | - | - | `outputs/verification_defense/bitdepth/` |
| Gaussian Smoothing | - | - | - | `outputs/verification_defense/smoothing/` |
| Temporal Consistency (1단계) | 171/171 | 100.0% | 0.0% | `outputs/verification_defense/temporal/` |
| ROI-first (2단계) | 127/171 | 74.3% | 19.3% | `outputs/verification_defense/roi_first/` |
| Randomized Smoothing (2단계) | 69/171 | 40.4% | 48.1% | `outputs/verification_defense/randomized_smoothing/` |
| **Ensemble Voting (2단계)** | **131/171** | **76.6%** | **18.9%** | `outputs/verification_defense/ensemble/` |
| **1+2단계 파이프라인** | **171/171** | **100.0%** | **0.0%** | - |
| **Adversarial Training (3단계)** | **208/212** | **98.1%** | **1.9%** | `outputs/verification_defense/adv_training/` |

> Temporal Consistency 100% 탐지는 테스트셋 전체가 JPEG 정지 이미지이기 때문.
> 실제 서비스 환경(카메라 입력) 기준으로는 static_thresh 재조정 필요.
> 실질적 방어 성능 지표: Ensemble Voting 76.6% / Adversarial Training 98.1% 차단 기준.

---

## 4. 기법별 상세 설명

### 1단계: Temporal Consistency

#### 구현 방식: B. 시뮬레이션 기반 (현재 적용)

현재 데이터(212개 JPEG 샘플)로 즉시 테스트 가능한 시뮬레이션 방식으로 구현됨.

- 단일 이미지에 미세한 회전(±5°)·밝기(±15%)·대비(±10%) 변화를 N회 적용해 연속 프레임을 생성함
- 정상 이미지: 자연스럽게 embedding이 변하므로 std가 상대적으로 높음
- 공격 이미지(JPEG 단일 최적화 이미지): 동일 이미지를 augmentation해도 embedding 변화가 거의 없음 → 정지 이미지 특성 그대로 탐지됨
- `embedding_std < static_thresh(0.015)` → 정지 이미지로 판정, 즉시 거부
- 파라미터: `n_frames=10`, `static_thresh=0.015`, `threshold=0.47966`
- 대응 공격: 정지 이미지, 화면 재생, 프린트 어택

#### 향후 전환 필요: A. 실제 카메라 스트림 기반

시뮬레이션 방식은 검증 목적이며, 실서비스 적용 시 아래와 같이 전환해야 함.

- 카메라에서 실제 연속 N프레임을 촬영하여 각 프레임의 FaceNet embedding을 추출함
- 실제 사용자는 촬영 중 자연스러운 미세 움직임(고개 각도, 조명 변화)이 발생하므로 embedding std가 시뮬레이션보다 높게 측정됨
- 이를 기준으로 `static_thresh` 재캘리브레이션 필요 (현재 0.015는 시뮬레이션 전용 값)
- 실제 환경에서는 정상 사용자 false positive 발생 가능성 제거를 위해 threshold 상향 조정 예정

### 2단계: ROI-first
- Otsu 이진화로 얼굴 영역 자동 마스킹
- 배경 영역을 평균값으로 대체
- 배경에 분산된 perturbation 차단
- 한계: 얼굴 영역 내 집중 공격에는 효과 제한적

### 2단계: Randomized Smoothing
- 입력 이미지에 가우시안 노이즈를 N회 반복 주입
- N개 embedding의 평균으로 cosine similarity 계산
- 수학적으로 certified defense 가능 (특정 반경 내 공격 보장 방어)
- 파라미터: `n_samples=50`, `sigma=0.05`
- 한계: 연산량 증가 (N회 forward pass), JPEG 이미지 기준 차단률 40.4%

### 2단계: Ensemble Voting
- ROI-first, Gaussian Smoothing, Randomized Smoothing 3종 결과를 voting
- 과반수(2/3 이상) accept이어야 최종 인증 통과, 그 외 거부
- 단일 방어 우회해도 나머지로 차단 가능
- 실험 결과 차단률 76.6% (단독 최고인 ROI-first 74.3%보다 향상)

### 3단계: Adversarial Training
- (adv, source, target_enroll) 트리플릿으로 InceptionResnetV1 fine-tune
- Loss: `max(0, cos(adv, target) - cos(adv, source) + margin)` — adv를 source처럼, target에서 멀어지게 학습
- block8 + last_linear + last_bn만 unfreeze (전체 9%, 212개 샘플 과적합 방지)
- 파라미터: `epochs=5`, `lr=1e-5`, `batch=8`, `margin=0.15`
- 학습 데이터: 공격팀 handoff 패키지 212개 JPEG 트리플릿
- 실험 결과: ASR 80.7% → 1.9%, 방어 성공 208/212 (98.1%)
- checkpoint: `outputs/verification_defense/adv_training/best_adv_trained.pt`

### 4단계: Feature Squeezing
- 원본 이미지와 squeezed 이미지(저해상도, 색상 축소 등)의 prediction 차이 측정
- 차이가 크면 공격으로 탐지
- 방어가 아닌 탐지 목적 → risk_score에 반영

---

## 5. 실험 baseline vs 실무 파이프라인 비교

| 구분 | 실험 baseline | 실무 파이프라인 |
|------|--------------|----------------|
| 목적 | 기법별 성능 비교 | 실시간 인증 서버 적용 |
| 방식 | 3종 독립 실행 후 결과 비교 | 레이어별 순차 적용 |
| 속도 | 무관 | 최우선 고려 |
| 기법 수 | 전체 실행 | 단계별 1~2개 |

---

## 6. 데이터 스키마 설계

전체 파이프라인 구현 시 필요한 테이블 구조. `DB_SCHEMA.md`의 기존 테이블에 추가되는 내용.

### 5-1. temporal_consistency_results (1단계 신규)
카메라 연속 프레임 기반 이상 탐지 결과.

```sql
CREATE TABLE temporal_consistency_results (
    id                  BIGSERIAL       PRIMARY KEY,
    session_id          VARCHAR(64)     NOT NULL REFERENCES attack_sessions(session_id),
    frame_count         INT             NOT NULL,
    avg_embedding_diff  NUMERIC(12,8)   NOT NULL,
    max_embedding_diff  NUMERIC(12,8)   NOT NULL,
    is_anomaly          BOOLEAN         NOT NULL,
    reject_reason       VARCHAR(64),
    defense_time_sec    NUMERIC(10,6),
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);
```

| 컬럼 | 설명 |
|------|------|
| frame_count | 촬영된 연속 프레임 수 |
| avg_embedding_diff | 프레임 간 평균 embedding 변화량 |
| max_embedding_diff | 프레임 간 최대 embedding 변화량 |
| is_anomaly | 비정상 패턴 감지 여부 |
| reject_reason | 거부 사유 (static_image / screen_replay / unnatural_movement) |

---

### 5-2. defense_results 컬럼 확장 (2단계 기존 테이블 확장)
기존 `defense_results` 테이블에 신규 기법 컬럼 추가.

```sql
ALTER TABLE defense_results
    ADD COLUMN roi_mask_applied      BOOLEAN,
    ADD COLUMN randomized_n_samples  INT,
    ADD COLUMN ensemble_votes        JSONB,
    ADD COLUMN ensemble_decision     VARCHAR(16);
```

| 컬럼 | 설명 | 예시 |
|------|------|------|
| roi_mask_applied | ROI 마스킹 적용 여부 | True |
| randomized_n_samples | Randomized Smoothing 반복 횟수 | 100 |
| ensemble_votes | 각 방어 기법별 accept/reject 결과 | {"roi": "reject", "smoothing": "reject", "randomized": "accept"} |
| ensemble_decision | Ensemble 최종 판정 | accept / reject |

---

### 5-3. adversarial_training_runs (3단계 신규)
Adversarial Training 학습 이력 관리.

```sql
CREATE TABLE adversarial_training_runs (
    id              BIGSERIAL       PRIMARY KEY,
    model_version   VARCHAR(64)     NOT NULL,
    base_model      VARCHAR(64)     NOT NULL DEFAULT 'InceptionResnetV1',
    loss_fn         VARCHAR(64)     NOT NULL DEFAULT 'cosine_verification_loss',
    train_pairs     INT             NOT NULL,
    epochs          INT             NOT NULL,
    mix_ratio       NUMERIC(4,2)    NOT NULL,
    eer_before      NUMERIC(8,6),
    eer_after       NUMERIC(8,6),
    asr_before      NUMERIC(6,4),
    asr_after       NUMERIC(6,4),
    checkpoint_path TEXT,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);
```

| 컬럼 | 설명 |
|------|------|
| model_version | 학습된 모델 버전 ID |
| loss_fn | 사용한 loss 함수 (cosine_verification_loss) |
| train_pairs | 학습에 사용된 (source, adv) 쌍 수 |
| mix_ratio | clean:adv 학습 데이터 비율 |
| eer_before / eer_after | 학습 전후 EER 비교 |
| asr_before / asr_after | 학습 전후 공격 성공률 비교 |

---

### 5-4. feature_squeezing_results (4단계 신규)
Feature Squeezing 기반 공격 탐지 결과.

```sql
CREATE TABLE feature_squeezing_results (
    id                  BIGSERIAL       PRIMARY KEY,
    sample_id           VARCHAR(64)     NOT NULL REFERENCES attack_samples(sample_id),
    squeezer            VARCHAR(32)     NOT NULL,
    sim_original        NUMERIC(12,8)   NOT NULL,
    sim_squeezed        NUMERIC(12,8)   NOT NULL,
    sim_diff            NUMERIC(12,8)   GENERATED ALWAYS AS (ABS(sim_original - sim_squeezed)) STORED,
    is_attack_detected  BOOLEAN         NOT NULL,
    detection_threshold NUMERIC(12,8)   NOT NULL,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);
```

| 컬럼 | 설명 | 예시 |
|------|------|------|
| squeezer | squeezing 방식 | low_resolution / color_depth / median_filter |
| sim_original | 원본 이미지 similarity | 0.6231 |
| sim_squeezed | squeezed 이미지 similarity | 0.4102 |
| sim_diff | 두 similarity 차이 (자동 계산) | 0.2129 |
| is_attack_detected | 공격 탐지 여부 | True |

---

### 5-5. 전체 테이블 관계 (추가분 포함)

```
attack_samples
    │
    ├──→ defense_results (확장)
    │         └── ensemble_votes, roi_mask_applied, randomized_n_samples
    │
    ├──→ feature_squeezing_results
    │
    └──→ attack_sessions
              └──→ temporal_consistency_results
              └──→ session_rule_hits

adversarial_training_runs (독립 테이블, 모델 이력 관리)
```
