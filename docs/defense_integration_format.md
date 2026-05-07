# 공격-방어 연동 포맷 정리

## 목적

공격 파트에서 생성한 adversarial image와 metadata를 방어 파트가 바로 사용할 수 있도록 공통 입력 포맷을 정리한다.

방어 파트는 공격별 CSV를 직접 읽을 필요 없이 아래 파일 하나만 사용한다.

```text
outputs/attacks/attack_index.csv
```

이 파일은 `src/reports/build_attack_index.py`로 생성한다.

```bash
python -m src.reports.build_attack_index
```

## attack_index.csv 핵심 컬럼

| 컬럼 | 의미 | 방어 파트 사용 여부 |
|---|---|---|
| sample_id | 공격 샘플 고유 ID | 필수 |
| attack | 세부 공격명 | 필수 |
| attack_family | 공격 계열: fgsm, pgd, square, jsma | 필수 |
| file | 원본 이미지 경로 | 선택 |
| adv_file | 공격 이미지 경로 | 필수 |
| perturbation_file | perturbation 시각화 이미지 경로 | 선택 |
| success | target 공격 성공 여부 | 필수 |
| clean_correct | 원본을 모델이 맞혔는지 여부 | 권장 |
| success_on_clean | 원본 정분류 샘플 기준 공격 성공 여부 | 권장 |
| true_label | 원본 정답 class id | 필수 |
| true_name | 원본 정답 이름 | 필수 |
| target_label | target class id | 필수 |
| target_name | target 이름 | 필수 |
| pred_before_name | 공격 전 예측 이름 | 권장 |
| pred_after_name | 공격 후 예측 이름 | 권장 |
| epsilon | L-infinity budget 계열 공격 파라미터 | 공격별 선택 |
| theta | JSMA 픽셀 수정량 | 공격별 선택 |
| alpha | PGD step size | 공격별 선택 |
| steps | PGD/JSMA 반복 횟수 | 공격별 선택 |
| max_queries | Square query budget | 공격별 선택 |
| queries_used | Square 실제 사용 query 수 | 공격별 선택 |
| l0 | 변경된 channel 수 | 권장 |
| l2 | L2 perturbation 크기 | 권장 |
| linf | L-infinity perturbation 크기 | 권장 |
| time_sec | 공격 생성 시간 | 권장 |
| target_conf_gain | target confidence 증가량 | 권장 |

## 방어 파트 입력 방식

방어 파트는 기본적으로 `success_on_clean=True`인 샘플을 우선 사용한다.

권장 필터:

```python
import pandas as pd

attack_index = pd.read_csv("outputs/attacks/attack_index.csv")
attack_inputs = attack_index[
    (attack_index["clean_correct"] == True) &
    (attack_index["success_on_clean"] == True)
]
```

방어 파이프라인 입력 이미지는 다음 컬럼을 사용한다.

```text
adv_file
```

원본 비교가 필요하면 다음 컬럼을 사용한다.

```text
file
```

## 방어 결과 CSV 권장 포맷

방어 파트는 결과를 아래 경로에 저장하는 것을 권장한다.

```text
outputs/defenses/defense_results.csv
```

권장 컬럼:

| 컬럼 | 의미 |
|---|---|
| sample_id | attack_index.csv의 sample_id 그대로 사용 |
| attack_family | 공격 계열 |
| attack | 공격명 |
| defense | 방어 기법명: jpeg, gaussian, bit_depth, roi_first 등 |
| defense_params | 방어 파라미터 JSON 문자열 |
| defended_file | 방어 적용 후 이미지 경로 |
| pred_before_defense | 방어 전 모델 예측 |
| pred_after_defense | 방어 후 모델 예측 |
| target_label | 공격 target label |
| true_label | 원본 true label |
| attack_success_before_defense | 방어 전 공격 성공 여부 |
| attack_success_after_defense | 방어 후에도 target 성공인지 여부 |
| recovered | 방어 후 true label로 복구되었는지 여부 |
| target_conf_before_defense | 방어 전 target confidence |
| target_conf_after_defense | 방어 후 target confidence |
| defense_time_sec | 방어 처리 시간 |

## 방어 성능 지표

방어 파트에서 최소로 계산할 지표는 다음과 같다.

```text
Defense Success Rate = attack_success_before_defense=True 중 attack_success_after_defense=False 비율
Recovery Rate = attack_success_before_defense=True 중 recovered=True 비율
Target Confidence Drop = target_conf_before_defense - target_conf_after_defense 평균
Defense Time = defense_time_sec 평균
```

## 방어 담당에게 전달할 내용

공격 파트에서 `outputs/attacks/attack_index.csv`를 생성해두었으니, 방어 파트는 이 파일의 `adv_file`을 입력으로 사용하면 된다. 결과 저장 시 `sample_id`를 반드시 유지하면 공격 결과와 방어 결과를 나중에 join해서 공격별 방어율, target confidence drop, 처리 시간 등을 계산할 수 있다.
