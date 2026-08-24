# CONTRIBUTING — GitHub 협업 진입점

이 저장소의 문서, 이슈, PR은 기본적으로 **한국어로 작성**한다. 코드 식별자, 명령어,
파일명, 모델명, 지표명(FAR, FRR, ASR 등), 요구사항 ID(DATA-XXX, VER-XXX,
ATK-XXX, EXP-XXX 등)는 원문 영어를 유지한다.

GitHub에서 이슈나 PR을 만들기 전에 아래 문서를 먼저 확인한다.

1. [`../CONTRIBUTING.md`](../CONTRIBUTING.md) — 일상 협업 규칙, 브랜치, 커밋, 문서 작성 규칙
2. [`../docs/15_ISSUE_AND_PR_WORKFLOW.md`](../docs/15_ISSUE_AND_PR_WORKFLOW.md) — 이슈 분해, 완료 조건, PR 머지 기준의 정본
3. [`.github/ISSUE_TEMPLATE`](./ISSUE_TEMPLATE) — 이슈 유형별 작성 양식
4. [`.github/PULL_REQUEST_TEMPLATE.md`](./PULL_REQUEST_TEMPLATE.md) — PR 작성 및 머지 전 체크리스트

요약 규칙:

- 이슈를 먼저 만들고 범위·완료 조건·제외 범위를 확정한다.
- 브랜치는 `component/REQUIREMENT-OR-WORK-ID-short-name` 형식을 따른다.
- 구현, 실험, 문서 변경은 같은 PR에서 추적 가능하게 묶는다.
- 실험 결과에는 표본 수, 실패/오류/제외 수, seed, config, 모델/threshold/policy 버전을 적는다.
- raw face, embedding, template, checkpoint, 개인정보는 커밋하지 않는다.
- 자동 생성 도구 이름을 `Co-authored-by` trailer로 남기지 않는다.
