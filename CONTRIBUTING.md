# CONTRIBUTING — 26_HC160 협업 가이드

이 저장소의 문서, 이슈, PR은 기본적으로 **한국어로 작성**한다. 코드 식별자, 명령어,
파일명, 모델명, 지표명(FAR, FRR, ASR 등), 요구사항 ID(FR-XXX, EXP-XXX)는 원문
영어를 유지한다.

## 1. 작업 흐름

1. 이슈를 먼저 만든다.
2. 이슈의 범위와 완료 조건을 확인한 뒤 브랜치를 만든다.
3. 구현, 실험, 문서 수정을 함께 진행한다.
4. 검증 결과를 남긴 뒤 PR을 연다.
5. 리뷰 승인 전에는 `main`에 직접 push하지 않는다.

## 2. 브랜치 규칙

브랜치 이름은 작업 성격과 범위를 드러내야 한다.

```text
feat/FR-XXX-short-name
fix/short-bug-name
exp/EXP-XXX-short-name
docs/short-doc-name
chore/short-maintenance-name
```

기존 `codex/...` 브랜치는 보존할 수 있지만, 새 작업에는 위 형식을 우선 사용한다.

## 3. 커밋 규칙

커밋 메시지는 한국어로 작성하되, 타입과 scope는 관례를 따른다.

```text
feat(face-auth): 세션 기반 인증 게이트 추가
fix(verification): threshold 버전 누락 기록
docs: 실험 재현 절차 정리
exp: EXP-DET-001 feature squeezing 평가 추가
```

커밋 본문에는 필요한 경우 다음을 적는다.

- 왜 변경했는지
- 어떤 실험/테스트로 확인했는지
- 보안 주장 또는 한계가 바뀌었는지

자동 생성 도구 이름을 `Co-authored-by` trailer로 남기지 않는다.

## 4. 문서 작성 규칙

- 본문은 한국어로 쓴다.
- 영문 약어는 처음 등장할 때 의미를 짧게 풀어쓴다.
- 측정값은 표본 수, 데이터 split, seed, 모델/threshold 버전을 함께 적는다.
- 공격/방어 성능은 clean 성능 변화와 함께 적는다.
- 추론과 측정 사실을 구분한다.
- 원본 얼굴 이미지, 임베딩, 템플릿, 모델 가중치 경로가 공개 저장소에 들어가지 않게 한다.

문서를 바꿀 때는 관련 문서의 관계를 확인한다.

- 요구사항: `docs/01_RESEARCH_REQUIREMENTS.md`
- 데이터 계약: `docs/04_DATA_AND_ARTIFACT_CONTRACT.md`
- verification 기준: `docs/05_FACE_VERIFICATION_SPEC.md`
- 공격 기준: `docs/06_ATTACK_SPEC.md`
- 방어/탐지 기준: `docs/07_DEFENSE_AND_DETECTION_SPEC.md`
- 실험 계획: `docs/08_EXPERIMENT_PLAN.md`
- 재현 절차: `docs/12_REPRODUCIBILITY_GUIDE.md`, `docs/14_LOCAL_RUNBOOK.md`

## 5. 이슈 작성 규칙

이슈에는 다음을 반드시 적는다.

- 작업 배경
- 구현/실험 범위
- 하지 않을 일
- 완료 조건
- 관련 문서와 선행/후속 이슈
- 보안·윤리·데이터 취급 영향

실험 이슈는 추가로 다음을 적는다.

- 검증할 가설
- 데이터 split과 ground truth
- 지표와 통과 기준
- 재현 명령, config, seed, 모델/threshold/policy 버전
- 실패/오류/제외 표본 수

## 6. PR 머지 조건

PR은 아래 조건을 만족해야 머지할 수 있다.

- 관련 이슈가 연결되어 있다.
- 변경 범위가 이슈와 일치한다.
- 문서가 함께 업데이트되었다.
- 테스트 또는 실험 결과가 PR 본문에 적혀 있다.
- 보안 주장, 데이터 취급, 모델/threshold/policy 버전 영향이 설명되어 있다.
- raw face, embedding, template, checkpoint, 개인 식별 정보가 커밋되지 않았다.
- CI 또는 해당 로컬 검증 명령이 통과했다.

## 7. 금지 사항

- `main` 직접 push
- 근거 없는 보안 성능 주장
- validation/test split 혼용
- raw biometric data 또는 모델 가중치 커밋
- 실패/제외 표본을 숨긴 결과 보고
- 문서와 코드 계약이 어긋난 상태로 머지
