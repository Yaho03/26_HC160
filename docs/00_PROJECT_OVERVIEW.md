# PROJECT OVERVIEW — 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 문서명 | 프로젝트 개요서 |
| 버전 | v1.0 |
| 상태 | 확정 |
| 최종 수정일 | 2026-08-23 |

---

## 1. 프로젝트 정체성

HC160은 금융 얼굴 인증 시나리오에서 targeted adversarial impersonation attack,
다층 방어와 탐지 신호를 평가하는 재현 가능한 연구·시연 프로젝트다.

이 저장소는 운영용 금융 인증 시스템이 아니며 그렇게 소개해서도 안 된다. 정본 산출물은
데이터, 모델, threshold, configuration과 artifact metadata가 고정된 실험 run이다.
향후 UI는 완료된 run을 표시할 수 있지만 연구 지표를 재정의해서는 안 된다.

## 2. 핵심 연구 질문

독립적으로 calibration하고 고정한 threshold에서, 방어 기법이 clean verification 성능을
보존하면서 targeted reject-to-accept 공격 성공률을 낮출 수 있는가?

주요 평가 우선순위는 다음과 같다.

1. 공격 전 거부된 pair 중 targeted attack 성공률
2. 변경하지 않은 clean test pair의 FAR/FMR과 FRR/FNMR
3. 방어 적용 후 clean TAR 보존 정도
4. 공격 budget, query 비용과 end-to-end latency
5. seed와 실행 환경에 따른 재현성

## 3. 연구 트랙

| 트랙 | 상태 | 역할 |
|---|---|---|
| LFW-10 ResNet-50 classification | Legacy baseline | 초기 5종 공격·4종 방어 실험을 보존한다. Classification accuracy는 금융 인증 지표가 아니다. |
| ResNet feature verification | Bridge baseline | 학습된 classifier backbone으로 pair 생성, cosine score, EER calibration과 targeted PGD를 설명한다. |
| FaceNet VGGFace2 verification | 주요 verification 후보 | 커밋된 verification-defense artifact와 대응한다. 완전한 재현성을 인정하려면 누락된 batch attack 생성 provenance를 복원해야 한다. |
| Real-time face-auth reference prototype | 개발 트랙 | Session, challenge, gate, policy와 일회용 result token 동작을 정의한다. 연구 metric 계산과 분리한다. |
| Generative purification | 향후 확장 | Clean/adversarial verification baseline이 유효해진 뒤 수행하는 방어 목적 확장이다. |

## 4. 대상 사용자

- 실험을 구현하고 재현하는 팀원
- 공격·방어·탐지 증거를 평가하는 검토자
- 검증 완료된 run을 시연하는 운영자
- 한계를 연구하는 보안 연구자

운영 금융 서비스 고객은 대상 사용자가 아니다.

## 5. 포함 범위

- 버전이 지정된 dataset 및 pair manifest
- 역사 비교를 위한 classification baseline
- 얼굴 embedding과 verification threshold calibration
- Targeted white-box 및 black-box attack
- 전처리·학습·ensemble·temporal·detection defense
- Clean 성능 보존과 adaptive attack 평가
- 재현 가능한 run manifest, report와 test
- Session과 token 보안을 검증하는 제한된 reference flow

## 6. 별도 승인 전 제외 범위

- 운영 금융 배포 또는 보안 인증
- 실제 고객 생체 데이터
- 모바일 device attestation, KMS/HSM 또는 운영 계정 시스템
- 적절한 데이터셋 없이 수행하는 인구집단 공정성 주장
- 생성형 얼굴 사칭 또는 deepfake 제작
- Microservice, message broker 또는 분산 인프라
- 검증하지 않은 실험 결과를 공개 dashboard에 자동 반영하는 기능

## 7. 성공 조건

다음 조건을 모두 충족하면 연구 시스템으로서 완료된 것으로 본다.

- 보고된 모든 수치가 run ID와 분모로 추적된다.
- Calibration set과 test set이 분리된다.
- Attack 및 defense artifact가 버전이 있는 계약을 통과한다.
- 학습형 방어를 held-out data와 adaptive attack으로 평가한다.
- 모든 방어가 clean 성능 저하를 함께 보고한다.
- Clean checkout에서 CPU test와 문서화된 smoke experiment를 실행할 수 있다.
- 보안, 개인정보, license와 알려진 한계를 공개한다.

## 8. 규범 문서 우선순위

문서가 충돌하면 다음 순서를 적용한다.

1. `01_RESEARCH_REQUIREMENTS.md`
2. `04_DATA_AND_ARTIFACT_CONTRACT.md`
3. `05_FACE_VERIFICATION_SPEC.md`
4. `06_ATTACK_SPEC.md`, `07_DEFENSE_AND_DETECTION_SPEC.md`
5. `08_EXPERIMENT_PLAN.md`, `09_EVALUATION_METRICS.md`
6. 기존 handoff 및 진행 보고서

이슈 분해와 PR 운영에는 `15_ISSUE_AND_PR_WORKFLOW.md`를 적용한다.
