# ATTACK SPEC — 공격 사양

| 항목 | 내용 |
|---|---|
| 문서명 | 적대적 공격 사양서 |
| 버전 | v1.0 |
| 상태 | 확정 |
| 최종 수정일 | 2026-08-23 |

---

## 1. 목적과 허용 범위

Attack module은 승인된 연구 model의 robustness를 평가한다. 제3자 시스템 공격을 위한
지침이 아니다. 모든 실험은 프로젝트가 통제하는 model, data와 동의된 demo environment로
제한한다.

## 2. Threat model

| 유형 | 공격자가 아는 정보 | 현재 예시 |
|---|---|---|
| White-box | Model, preprocessing, gradient와 threshold | FGSM, PGD, multi-pixel JSMA variant |
| Score-based black-box | Output score/probability와 query access | Square-style, ZOO-style attack |
| Transfer | Source model에서 생성하고 다른 target model에서 평가 | 계획된 cross-model 실험 |
| Adaptive defense-aware | 알려진 defense 또는 defended model을 공격에 포함 | 학습형·미분 가능 방어의 필수 평가 |

## 3. 공통 interface

Attack input:

- Source/probe 및 target-enrollment artifact ID
- Verification protocol 및 threshold artifact ID
- Attack configuration과 threat model
- Seed와 query/time budget

Output은 schema-valid attack result와 선택적 정본 PNG/tensor artifact다. Attack
implementation은 aggregate report metric을 계산하지 않는다.

## 4. 필수 측정값

- Attack 전후 score와 decision
- Reject-to-accept success
- 의미가 있는 경우 L0, L2, L-infinity
- Perturbation scale과 norm space
- Step, query 수, elapsed time
- Random seed와 early-stop reason
- 정본 artifact hash와 serialization format
- 실패한 attack과 별도로 기록한 error

## 5. 공정한 비교 규칙

- 비교하는 attack은 동일한 eligible pair manifest를 사용한다.
- Target-selection policy를 고정한다.
- Attack 고유 constraint를 보고하며 JSMA sparsity와 L-infinity budget을 같은 값처럼 취급하지 않는다.
- Black-box ASR은 query budget에 따른 함수로 보고한다.
- Protocol에 다른 규칙을 사전 등록하지 않았다면 failure와 timeout을 분모에 포함한다.
- 최종 test set으로 attack parameter를 조정하지 않는다.

## 6. Legacy 매핑

`src/attacks/`의 5개 script는 classification baseline이다.
`src/verification/targeted_pgd_verification.py`는 ResNet-feature verification bridge다.
커밋된 FaceNet defense artifact를 end-to-end로 재현하려면 FaceNet batch attack
producer가 추가로 필요하다.

Legacy result file은 변경하지 않으며 원래의 success definition을 표시한다.
