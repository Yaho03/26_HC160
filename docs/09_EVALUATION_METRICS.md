# EVALUATION METRICS — 평가 지표

| 항목 | 내용 |
|---|---|
| 문서명 | 평가 지표 정의서 |
| 버전 | v1.0 |
| 상태 | 확정 |
| 최종 수정일 | 2026-08-23 |

---

## 1. Verification confusion count

`same_identity` ground truth와 `accepted` decision에 대해 다음을 정의한다.

- TP: genuine pair accept
- FN: genuine pair reject
- FP: impostor pair accept
- TN: impostor pair reject

```text
FAR/FMR = FP / (FP + TN)
FRR/FNMR = FN / (TP + FN)
TAR      = TP / (TP + FN)
TNR      = TN / (TN + FP)
accuracy = (TP + TN) / all pairs
```

분모가 0이면 0이 아니라 명시적인 undefined 값을 반환한다.

## 2. 공격 지표

Eligible targeted attempt는 공격 전 거부된 다른 identity pair다.

```text
targeted ASR = reject-to-accept success / eligible targeted attempt
```

L0/L2/L-infinity, query 수, elapsed time, budget별 success와 failure/error 수를 함께 보고한다.

## 3. 방어 지표

```text
conditional defense success rate
  = 방어 전 accepted attack 중 방어 후 reject된 수 / 방어 전 accepted attack 수

conditional ASR after defense
  = 방어 후에도 accepted인 attack / 방어 전 accepted attack 수

population ASR after defense
  = 방어 후 accepted인 attack / 전체 eligible attack attempt
```

이 지표들은 분모가 다르므로 같은 label을 사용하면 안 된다.

Clean 성능 보존은 다음과 같이 계산한다.

```text
clean TAR delta = TAR_after_defense - TAR_before_defense
clean FRR delta = FRR_after_defense - FRR_before_defense
```

## 4. Detector 지표

TPR, FPR, precision, recall, ROC-AUC, threshold, attack 종류와 sample 수를 보고한다.
Detector failure와 authentication decision은 분리한다.

## 5. EER와 operating point

Calibration score에서 `abs(FAR - FRR)`를 최소화하는 threshold로 EER을 추정한 뒤
threshold를 고정한다. 최종 test EER는 설명용으로 보고할 수 있지만 사전 선택한 threshold
평가를 대체하지 않는다.

TAR@FAR는 impostor pair 수가 해당 FAR을 지원할 때만 보고한다. 작은 dataset으로 운영
수준의 확실성을 암시하지 않고 confidence interval을 포함한다.

## 6. Runtime

- Attack/defense 시간의 시작과 끝을 명확히 정의한다.
- Model load·warm-up과 steady-state inference를 분리한다.
- 명시한 hardware에서 p50, p95와 sample 수를 보고한다.
- Randomized/temporal method는 forward-pass 또는 frame 수를 기록한다.

## 7. Legacy result 주의사항

커밋된 classification report와 verification report는 서로 다른 task 정의를 사용한다.
현재 verification summary의 `defense_success_rate` 분모는 전체 212 row다. 향후 중앙
evaluation은 이 역사적 값을 보존하면서 conditional denominator를 별도로 제공해야 한다.

## 8. 경험적 FAR 지원 규칙

Impostor pair가 `N`개면 관찰 가능한 최소 non-zero empirical FAR은 `1/N`이다. 따라서
다음을 만족할 때만 target FAR을 지원하는 것으로 본다.

```text
N * target_far >= 1
```

이는 최소 보고 기준이며 운영 수준의 확실성을 증명하지 않는다. Threshold artifact는
impostor denominator, achieved FAR numerator/denominator, calibration manifest hash와
pair-ID-set hash를 저장한다. 최종 test는 고정 numeric threshold를 사용하고 artifact를
변경하지 않은 채 자체 count를 보고한다.
