# GETTING STARTED — 시작 가이드

## 1. 먼저 읽을 문서

1. `README.md`
2. `docs/README.md`
3. `CONTRIBUTING.md`
4. `00_PROJECT_OVERVIEW.md`
5. `01_RESEARCH_REQUIREMENTS.md`
6. `04_DATA_AND_ARTIFACT_CONTRACT.md`
7. `09_EVALUATION_METRICS.md`
8. `11_SECURITY_ETHICS_AND_LIMITATIONS.md`
9. `13_IMPLEMENTATION_STATUS.md`
10. `14_LOCAL_RUNBOOK.md`
11. 작업을 시작한다면 `15_ISSUE_AND_PR_WORKFLOW.md`

## 2. Lightweight validation

Research contract와 metric test는 standard library만으로 실행된다.

```bash
python -m unittest discover -s tests/research -v
```

이 명령은 GPU research pipeline이나 별도 face-auth application test를 설치·실행하지 않는다.

`requirements-face-auth.txt` 설치 후 전체 suite를 실행한다.

```bash
python -m unittest discover -s tests -v
```

문서화된 구현 snapshot은 Python 3.9에서 144개 test를 통과했다. 이는 특정 시점의
증거이므로 이후 revision은 현재 명령으로 다시 검증한다.

## 3. 트랙 선택

- Legacy classification: README와 기존 Colab attack pipeline을 따른다.
- Verification research: LFW data, versioned embedding checkpoint, pair manifest와 threshold artifact를 준비한다.
- Face-auth application prototype: 별도 문서와 dependency file을 따른다.

서로 다른 track의 checkpoint와 threshold를 섞지 않는다.

## 4. 실험 전 확인

- Dataset/model license 확인
- Input manifest validation
- Clean Git commit 기록
- 새 run ID와 seed 선택
- Configuration 고정
- 외부 artifact hash 검증
- Test data가 calibration 또는 tuning에 사용되지 않았는지 확인

## 5. 작업과 이슈 시작

새 기능이나 실험을 바로 구현하지 않는다. `docs/15_ISSUE_AND_PR_WORKFLOW.md`에 따라
구현 상태와 규범 문서를 읽고, 요구사항 ID·범위·제외 범위·완료 조건·검증 방법을 갖춘
이슈를 먼저 만든다.

## 6. 결과 공개 전 확인

- 전체 test 실행
- Numerator와 denominator 표시
- Clean-performance preservation 포함
- 장비 절대 경로와 민감 artifact 제거
- Run ID, hash와 limitation 인용
