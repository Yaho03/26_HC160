# Defense Pipeline 출력 파일 명세

`notebooks/colab_defense_pipeline.ipynb` 실행 시 생성되는 파일 목록과 각 파일의 내용을 정리합니다.

---

## 생성 파일 구조

```
/content/defense_results/
│
├── jpeg/
│   ├── jpeg_results_q75.csv
│   └── q75/images/
│       └── {sample_id}_jpeg_q75.jpg
│
├── smoothing/
│   ├── smoothing_results_r3p0.csv
│   └── r3p0/images/
│       └── {sample_id}_smoothing_r3p0.png
│
├── bitdepth/
│   ├── bitdepth_results_4bit.csv
│   └── 4bit/images/
│       └── {sample_id}_bitdepth_4bit.png
│
├── adv_training/                          ← Phase 1-B (선택)
│   ├── adv_training_results.csv
│   └── best_adv.pt
│
├── defense_summary.csv
├── defense_report.md
└── figures/
    ├── heatmap.png
    ├── bar_by_attack.png
    └── boxplot_conf_drop.png
```

> Drive 캐시 경로: `내 드라이브/hanium-aml-defense/outputs/defenses/`  
> 전처리 방어 CSV 3개는 Phase 1 완료 후 Drive에 자동 저장됩니다.  
> `best_adv.pt` 는 Drive `adv_checkpoints/` 에 저장되며 재실행 시 학습을 생략합니다.

---

## Phase 1 — 전처리 방어 3종

### `jpeg/jpeg_results_q75.csv` / `smoothing/smoothing_results_r3p0.csv` / `bitdepth/bitdepth_results_4bit.csv`

방어 기법별 샘플 단위 평가 결과입니다.  
행 수 = 방어 대상 샘플 수 (`clean_correct=True & success_on_clean=True`)

| 컬럼 | 타입 | 내용 |
|------|------|------|
| `sample_id` | str | 공격 샘플 고유 ID — `attack_index.csv` 와 join 키 |
| `attack_family` | str | fgsm / pgd / square / jsma / zoo |
| `attack` | str | 세부 공격 이름 |
| `defense` | str | jpeg / gaussian_smoothing / bit_depth |
| `defense_params` | JSON str | 방어 파라미터 |
| `input_adv_file` | str | 입력 적대적 이미지 경로 |
| `defended_file` | str | 방어 적용 후 저장된 이미지 경로 |
| `pred_before_defense` | int | 방어 전 모델 예측 레이블 |
| `pred_after_defense` | int | 방어 후 모델 예측 레이블 |
| `pred_after_defense_name` | str | 방어 후 예측 클래스 이름 |
| `true_label` / `target_label` | int | 실제 레이블 / 공격 목표 레이블 |
| `attack_success_before_defense` | bool | 방어 전 공격 성공 여부 |
| `attack_success_after_defense` | bool | 방어 후에도 공격 성공 여부 |
| `recovered` | bool | 방어 후 원래 레이블로 복원 여부 |
| `target_conf_before_defense` | float | 방어 전 목표 클래스 신뢰도 |
| `target_conf_after_defense` | float | 방어 후 목표 클래스 신뢰도 |
| `true_conf_after_defense` | float | 방어 후 실제 클래스 신뢰도 |
| `defense_time_sec` | float | 샘플 방어 처리 시간 (초) |
| `status` | str | ok / missing_adv_file |

---

## Phase 1-B — 적대적 학습 (선택)

### `adv_training/adv_training_results.csv`

fine-tune 된 모델로 전체 공격 샘플을 재분류한 결과입니다.  
컬럼 구조는 전처리 방어 CSV와 동일하며 `defense = "adv_training"` 으로 구분됩니다.

| 컬럼 | 내용 |
|------|------|
| `defense` | `adv_training` 고정 |
| `defense_params` | `{"attack_family": "pgd", "mix_ratio": 0.5, "epochs": 5}` |
| `defended_file` | fine-tune 체크포인트 경로 (`best_adv.pt`) |

### `adv_training/best_adv.pt`

fine-tune 된 ResNet-50 체크포인트입니다.  
Drive `adv_checkpoints/best_adv.pt` 에도 동일한 파일이 백업됩니다.

---

## Phase 2 — 집계

### `defense_summary.csv`

공격 × 방어 조합별 집계 결과입니다.  
행 수 = (5종 공격 + ALL) × 4종 방어 = **24행**

| 컬럼 | 타입 | 내용 |
|------|------|------|
| `result_file` | str | 원본 결과 CSV 경로 |
| `attack_family` | str | fgsm / pgd / square / jsma / zoo / ALL |
| `defense` | str | jpeg / gaussian_smoothing / bit_depth / adv_training |
| `defense_params` | JSON str | 방어 파라미터 |
| `samples` | int | 해당 조합의 샘플 수 |
| `defense_success_rate` | float | 방어 성공률 (0~1) |
| `recovery_rate` | float | 복원율 (0~1) |
| `avg_target_conf_drop` | float | 목표 클래스 신뢰도 평균 감소량 |
| `avg_defense_time_sec` | float | 평균 방어 처리 시간 (초) |

---

## Phase 3 — 시각화 및 보고서

### `figures/heatmap.png`

공격 × 방어 조합을 행/열로 놓은 히트맵 2개 (방어 성공률 / 복원율)

### `figures/bar_by_attack.png`

공격별 방어 4종 성능 비교 그룹 막대 차트 (방어 성공률 / 복원율)

### `figures/boxplot_conf_drop.png`

방어 전후 목표 클래스 신뢰도 감소량 분포 박스 플롯 (공격 × 방어 4종)

### `defense_report.md`

실험 조건 + 방어 성공률 / 복원율 / 신뢰도 감소 테이블 + 분석을 담은 마크다운 보고서
