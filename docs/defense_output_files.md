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
├── defense_summary.csv
├── defense_report.md
└── figures/
    ├── heatmap.png
    ├── bar_by_attack.png
    └── boxplot_conf_drop.png
```

> Drive 캐시 경로: `내 드라이브/hanium-aml-defense/outputs/defenses/`  
> CSV 3개는 Phase 1 완료 후 Drive에 자동 저장됩니다.

---

## Phase 1 — 방어 기법 실행 결과

### `jpeg/jpeg_results_q75.csv`
### `smoothing/smoothing_results_r3p0.csv`
### `bitdepth/bitdepth_results_4bit.csv`

방어 기법별 샘플 단위 평가 결과입니다.  
행 수 = 방어 대상 샘플 수 (`clean_correct=True & success_on_clean=True`)

| 컬럼 | 타입 | 내용 |
|------|------|------|
| `sample_id` | str | 공격 샘플 고유 ID — `attack_index.csv` 와 join 키 |
| `attack_family` | str | fgsm / pgd / square / jsma / zoo |
| `attack` | str | 세부 공격 이름 (예: `targeted_fgsm`) |
| `defense` | str | jpeg / gaussian_smoothing / bit_depth |
| `defense_params` | JSON str | 방어 파라미터 (예: `{"quality": 75}`) |
| `input_adv_file` | str | 입력된 적대적 이미지 경로 |
| `defended_file` | str | 방어 적용 후 저장된 이미지 경로 |
| `pred_before_defense` | int | 방어 전 모델 예측 레이블 번호 |
| `pred_after_defense` | int | 방어 후 모델 예측 레이블 번호 |
| `pred_after_defense_name` | str | 방어 후 예측 클래스 이름 |
| `true_label` | int | 실제 레이블 번호 |
| `target_label` | int | 공격 목표 레이블 번호 |
| `attack_success_before_defense` | bool | 방어 전 공격 성공 여부 |
| `attack_success_after_defense` | bool | 방어 후에도 공격 성공 여부 |
| `recovered` | bool | 방어 후 원래 레이블로 복원 여부 |
| `target_conf_before_defense` | float | 방어 전 목표 클래스 신뢰도 (0~1) |
| `target_conf_after_defense` | float | 방어 후 목표 클래스 신뢰도 (0~1) |
| `true_conf_after_defense` | float | 방어 후 실제 클래스 신뢰도 (0~1) |
| `defense_time_sec` | float | 해당 샘플 방어 처리 시간 (초) |
| `status` | str | ok / missing_adv_file |

### `{defense}/images/`

방어 기법이 적용된 이미지 파일 모음입니다.

| 방어 | 파일명 형식 | 포맷 |
|------|------------|------|
| JPEG | `{sample_id}_jpeg_q75.jpg` | JPEG |
| Gaussian | `{sample_id}_smoothing_r3p0.png` | PNG |
| Bit-depth | `{sample_id}_bitdepth_4bit.png` | PNG |

---

## Phase 2 — 집계

### `defense_summary.csv`

공격 × 방어 조합별 집계 결과입니다.  
행 수 = (5종 공격 + ALL) × 3종 방어 = **18행**

| 컬럼 | 타입 | 내용 |
|------|------|------|
| `result_file` | str | 원본 결과 CSV 경로 |
| `attack_family` | str | fgsm / pgd / square / jsma / zoo / ALL |
| `defense` | str | jpeg / gaussian_smoothing / bit_depth |
| `defense_params` | JSON str | 방어 파라미터 |
| `samples` | int | 해당 조합의 샘플 수 |
| `defense_success_rate` | float | 방어 성공률 (0~1) |
| `recovery_rate` | float | 복원율 (0~1) |
| `avg_target_conf_drop` | float | 목표 클래스 신뢰도 평균 감소량 |
| `avg_defense_time_sec` | float | 평균 방어 처리 시간 (초) |

---

## Phase 3 — 시각화 및 보고서

### `figures/heatmap.png`

공격 × 방어 조합을 행/열로 놓은 히트맵 2개를 나란히 표시합니다.

- 왼쪽: Defense Success Rate (방어 성공률 %)
- 오른쪽: Recovery Rate (복원율 %)
- 색상: 높을수록 진함 (YlGn / Blues)

### `figures/bar_by_attack.png`

공격 5종을 x축, 방어 3종을 그룹 막대로 나타낸 차트 2개입니다.

- 왼쪽: Defense Success Rate
- 오른쪽: Recovery Rate

### `figures/boxplot_conf_drop.png`

방어 전후 목표 클래스 신뢰도 감소량(`conf_drop = target_conf_before − target_conf_after`)의 분포를 나타낸 박스 플롯입니다.

- x축: 공격 종류
- y축: 신뢰도 감소량
- 색상: 방어 기법 3종 구분

### `defense_report.md`

실험 조건과 집계 결과를 마크다운 표로 정리한 보고서입니다.

- 실험 조건 (공격 파라미터 / 방어 파라미터)
- 공격 × 방어 방어 성공률 표
- 공격 × 방어 복원율 표
- 공격 × 방어 평균 신뢰도 감소 표
