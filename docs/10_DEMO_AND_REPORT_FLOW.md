# DEMO AND REPORT FLOW — 데모·보고 흐름

| 항목 | 내용 |
|---|---|
| 문서명 | 데모 및 결과 보고 흐름 |
| 버전 | v1.0 |
| 상태 | 확정 |
| 최종 수정일 | 2026-08-23 |

---

## 1. 데모 원칙

Demo는 검증된 experiment run을 설명한다. 발표 중 검토되지 않은 보안 주장을 새로
생성하지 않는다.

## 2. 권장 시나리오

```text
1. 완료된 run ID를 선택한다.
2. Dataset/model/threshold provenance를 보여준다.
3. Clean genuine와 clean impostor decision을 보여준다.
4. Source, target enrollment와 lossless adversarial probe를 보여준다.
5. Reject-to-accept 전이와 attack budget을 보여준다.
6. 하나의 defense를 적용하고 defended decision을 보여준다.
7. Attack 감소 옆에 clean-performance cost를 표시한다.
8. Detector evidence를 authentication decision과 분리해 표시한다.
9. 한계와 reproduction command로 마무리한다.
```

## 3. Report 필수 절

- 범위와 threat model
- Dataset, split, model, preprocessing과 threshold version
- Clean baseline
- Attack과 budget
- Defense와 clean trade-off
- 해당하는 경우 adaptive 및 transfer 평가
- Runtime과 재현성
- Sample 수, error와 confidence interval
- 개인정보, 윤리와 한계
- Run ID와 artifact hash

## 4. 실패 시나리오

Demo는 다음 중 하나 이상의 예상 실패를 포함한다.

- Artifact 누락 또는 schema mismatch
- Low-quality retryable capture
- Fail closed하는 model error
- Threshold를 넘지 못한 attack
- ASR을 낮추지만 clean TAR을 훼손한 defense
- 별도 reference prototype의 token replay 또는 context mismatch

## 5. 금지하는 표현과 시연

- “100% 안전” 또는 같은 의미의 주장
- Training-set adversarial 성능을 held-out robustness로 제시
- Temporal still-image 결과를 camera 배포 증거로 사용
- Certificate 없이 certified robustness 주장
- 승인 없이 식별 가능한 face artifact 표시
