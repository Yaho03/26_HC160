# 팀 공유용 공격 파트 진행상황

공격 파트 1차 구현 및 실험 완료했습니다.

## 완료한 작업

- LFW 데이터셋 기반 10명 identity classification용 ResNet-50 모델 학습
- targeted FGSM 구현 및 실험
- targeted PGD 구현 및 실험
- targeted Square Attack 구현 및 실험
- targeted multi-pixel JSMA 변형 구현 및 실험
- targeted ZOO-style finite difference 공격 구현 및 소규모 실험
- 공격별 metadata CSV 생성
- original/adversarial/perturbation 패널 생성
- `attack_index.csv`로 공격 결과 통합

## 모델 결과

- ResNet-50 test accuracy: 76.23%
- 공격 평가 subset clean accuracy: 91.00%

## 공격 결과 요약

| 공격 | 대표 설정 | 원본 정분류 기준 Target ASR |
|---|---|---:|
| FGSM | eps=0.005 | 46.15% |
| PGD | eps=0.005~0.050 | 100.00% |
| Square | eps=0.050, queries=300 | 76.92% |
| multi-pixel JSMA | theta=0.05 | 100.00% |
| ZOO-style | eps=0.050, queries=2000 | 38.89% |

## 방어 파트 전달사항

방어 파트는 아래 파일을 사용하면 됩니다.

```text
outputs/attacks/attack_index.csv
```

현재 index 구성:

```text
fgsm      400
jsma      100
pgd       700
square    400
zoo        20
total    1620
```

방어 파트에서는 `adv_file`을 입력 이미지로 사용하고, 결과 저장 시 `sample_id`를 유지하면 됩니다.

## 다음 작업

- 공격별 대표 성공/실패 샘플 선정
- 보고용 compact summary 정리
- 방어 파트와 attack_index.csv 연동 확인
