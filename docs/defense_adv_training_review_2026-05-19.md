# 방어팀 Adversarial Training 결과 검토

작성일: 2026-05-19  
자료: `/Users/ehgus/Downloads/defense_results.zip`  
검토 목적: 방어팀이 전달한 adversarial training 방어 결과를 공격팀 관점에서 해석하고, 다음 협업 지점을 정리한다.

---

## 1. 전달받은 파일 구성

압축 파일 내부에는 다음 결과가 포함되어 있었다.

```text
content/defense_results/
  adv_training/
    adv_training_results.csv
    best_adv.pt
  bitdepth/
    bitdepth_results_4bit.csv
  jpeg/
    jpeg_results_q75.csv
  smoothing/
    smoothing_results_r3p0.csv
  defense_summary.csv
  defense_report.md
  figures/
    heatmap.png
    bar_by_attack.png
    boxplot_conf_drop.png
```

새로 추가된 핵심 파일은 다음이다.

```text
adv_training/adv_training_results.csv
adv_training/best_adv.pt
```

`best_adv.pt`는 adversarial training으로 학습된 방어 모델 checkpoint로 보인다. `adv_training_results.csv`는 해당 모델을 사용해 기존 공격 이미지에 대해 방어 후 예측을 다시 수행한 결과이다.

---

## 2. 실험 조건

방어팀 보고서 기준 공격 설정은 다음과 같다.

| 공격 | 파라미터 |
|---|---|
| FGSM | epsilon = 0.005 |
| PGD | epsilon = 0.03, alpha = 0.003, steps = 10 |
| Square | epsilon = 0.05, max_queries = 300 |
| JSMA | theta = 0.05, steps = 20, pixels_per_step = 200 |
| ZOO | epsilon = 0.05, max_queries = 2000, coords_per_iter = 128, lr = 0.02 |

방어 설정:

| 방어 | 파라미터 |
|---|---|
| Adversarial Training | attack_family = pgd, mix_ratio = 0.5, epochs = 5 |
| JPEG | quality = 75 |
| Gaussian smoothing | radius = 3 |
| Bit-depth reduction | bits = 4 |

---

## 3. 전체 성능 요약

전체 499개 공격 성공 샘플 기준 방어 성공률은 다음과 같다.

| Defense | Samples | Defense Success Rate | Recovery Rate | Avg Target Conf Drop | Avg Time |
|---|---:|---:|---:|---:|---:|
| Adv. Training | 499 | 99.80% | 99.80% | 0.5623 | 0.0083s |
| Gaussian smoothing | 499 | 90.18% | 48.30% | 0.4485 | 0.2178s |
| Bit-depth 4bit | 499 | 85.97% | 48.30% | 0.4420 | 0.2231s |
| JPEG q75 | 499 | 83.57% | 51.90% | 0.4441 | 0.2020s |

핵심:

```text
Adversarial training이 기존 입력 변환 방어보다 압도적으로 높은 방어 성공률과 복원율을 보였다.
```

특히 기존 입력 변환 방어들은 target class만 피하게 만들 수는 있어도 true class로 복원하는 비율은 50% 안팎이었다. 반면 adversarial training은 대부분 샘플을 true class로 되돌렸다.

---

## 4. 공격별 방어 성공률

| Attack | Adv. Training | Bit-depth | Gaussian | JPEG |
|---|---:|---:|---:|---:|
| FGSM | 100.00% | 63.83% | 76.60% | 55.32% |
| PGD | 99.41% | 88.82% | 93.53% | 87.65% |
| Square | 100.00% | 87.13% | 88.12% | 84.16% |
| JSMA | 100.00% | 91.12% | 93.49% | 90.53% |
| ZOO | 100.00% | 50.00% | 66.67% | 33.33% |

해석:

- Adv. Training은 모든 공격군에서 거의 100% 방어 성공률을 보였다.
- PGD 기반 adversarial training인데도 FGSM, Square, JSMA, ZOO까지 높은 방어 성능을 보였다.
- 이는 학습에 사용한 PGD adversarial examples가 다른 공격에도 어느 정도 일반화된 것으로 볼 수 있다.

---

## 5. 공격별 복원율

복원율은 단순히 target class를 피하는 것이 아니라 true class로 되돌아간 비율이다.

| Attack | Adv. Training | Bit-depth | Gaussian | JPEG |
|---|---:|---:|---:|---:|
| FGSM | 100.00% | 17.02% | 23.40% | 27.66% |
| PGD | 99.41% | 52.94% | 53.53% | 55.29% |
| Square | 100.00% | 50.50% | 46.53% | 50.50% |
| JSMA | 100.00% | 53.85% | 53.85% | 58.58% |
| ZOO | 100.00% | 8.33% | 8.33% | 16.67% |

해석:

```text
Adversarial training은 target 회피뿐 아니라 true identity 복원 측면에서도 가장 좋다.
```

기존 입력 변환 방어는 공격을 깨뜨려도 다른 class로 오분류되는 경우가 많았다. 공격팀 관점에서는 이 차이를 중요하게 봐야 한다.

---

## 6. 공격팀 관점에서 중요한 점

이번 방어 결과는 좋은 성과지만, 아직 다음 한계가 있다.

### 6.1 현재 결과는 classification 기준이다

전달받은 CSV 컬럼은 다음과 같은 classification 기반 구조이다.

```text
pred_before_defense
pred_after_defense
target_label
true_label
attack_success_before_defense
attack_success_after_defense
recovered
```

즉 기준은 다음이다.

```text
공격 후 target class로 분류되었는가?
방어 후 target class가 아니게 되었는가?
방어 후 true class로 돌아왔는가?
```

하지만 우리가 지금 새로 전환한 FaceNet verification 공격 기준은 다르다.

```text
source 얼굴이 target enrollment 얼굴로 accept되는가?
방어 후 다시 reject되는가?
```

따라서 이번 결과는 기존 classification attack 방어 결과로 해석해야 한다. FaceNet verification attack 방어 결과는 별도로 다시 만들어야 한다.

### 6.2 Adversarial training checkpoint는 추가 공격 대상이 될 수 있다

`best_adv.pt`가 같이 전달되었으므로 공격팀은 다음 실험도 할 수 있다.

```text
기존 clean ResNet checkpoint 공격
vs
adversarially trained ResNet checkpoint 공격
```

즉 같은 FGSM/PGD/Square/JSMA/ZOO 공격을 `best_adv.pt` 모델에 다시 걸어보면, adversarial training이 공격 자체에 얼마나 강한지 공격팀 관점에서도 확인할 수 있다.

다만 현재 우리 메인 방향은 FaceNet verification으로 전환 중이므로, 이 실험은 classification branch의 추가 검증으로 두는 것이 좋다.

---

## 7. 방어팀에게 전달할 피드백

방어팀에게는 다음처럼 말하면 된다.

```text
전달해주신 adversarial training 결과 확인했습니다.
기존 JPEG, smoothing, bit-depth보다 전체 방어 성공률과 복원율이 크게 높고,
전체 기준 defense success/recovery가 약 99.8%로 나왔습니다.

특히 PGD 기반 adversarial training인데 FGSM, Square, JSMA, ZOO에도 거의 100%에 가까운 방어 성능이 나온 점이 좋습니다.

다만 현재 결과는 기존 10-class classification attack 기준이므로,
저희 공격팀에서 새로 만든 FaceNet verification attack 결과를 별도로 넘기면
그 기준으로도 방어 파이프라인을 다시 평가하면 좋겠습니다.
```

---

## 8. 공격팀의 다음 작업

공격팀은 방어를 직접 구현하기보다 방어팀이 사용할 수 있는 공격 패키지를 정리해야 한다.

우선순위:

1. FaceNet verification attack 결과 중 대표 설정 확정
   - 현재 eps=0.005, steps=10: ASR 45%
   - 현재 eps=0.010, steps=10: ASR 80%
2. FaceNet attack metadata CSV 정리
3. adversarial image 경로와 target enrollment image 경로 포함
4. 방어팀용 handoff 문서 작성
5. 방어팀이 방어 후 계산해야 할 컬럼 정의

방어팀용 verification 결과 컬럼 제안:

```text
sample_id
attack
model
source_file
target_enroll_file
adv_file
defended_file
source_name
target_name
threshold
similarity_before
similarity_after_attack
similarity_after_defense
accepted_before
accepted_after_attack
accepted_after_defense
attack_success_before_defense
attack_success_after_defense
defense_success
epsilon
alpha
steps
l2
linf
```

---

## 9. 현재 결론

이번 방어팀 결과는 classification 기준에서는 매우 강한 방어 결과이다.

```text
Adv. Training defense success: 99.80%
Adv. Training recovery rate: 99.80%
```

그러나 프로젝트가 이제 FaceNet verification 기반 impersonation attack으로 확장되었기 때문에, 다음 협업은 다음 구조가 되어야 한다.

```text
공격팀:
FaceNet verification adversarial images + metadata 제공

방어팀:
해당 adversarial images에 방어 적용
방어 후 FaceNet similarity 재계산
accepted_after_defense / defense_success 산출
```

즉 이번 자료는 기존 단계의 방어 성과로 정리하고, 다음 단계는 verification 기준으로 공격·방어 결과 포맷을 새로 맞추는 것이다.

