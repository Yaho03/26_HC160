# Artifact schemas

이 디렉터리의 JSON Schema는 데이터셋 행과 snapshot, 검증 쌍·점수·임계값·clean 보고서, 공격·방어 결과, 실행 manifest와 산출물 참조의 구조적 계약이다.

`pad-evaluation-report.schema.json`은 PAD 실행 ID, Git 상태, manifest/model SHA-256, 선택된 영상별 SHA-256·바이트 크기, 임계값, 평가 결과와 APCER/BPCER/ACER 구조를 고정한다.

`authentication-decision.schema.json`은 evidence digest에 결합된 최종 인증 상태, 정책 버전, 게이트 결과, 챌린지 경계와 토큰 발급 여부를 고정한다. 사용자 ID, challenge nonce, 원본 프레임과 얼굴 템플릿은 포함하지 않는다. 생성된 `decision_id`는 run manifest의 `output_artifact_ids`와 `kind: decision` artifact reference로 등록할 수 있다.

`verification-score-export.schema.json`은 FaceNet 점수 JSONL의 해시와 model/preprocessing/dataset/pair manifest/Git provenance를 결합한다. Threshold와 clean report는 이 export metadata 해시를 다시 참조한다.

- 경로는 저장소 또는 실행 디렉터리 기준 상대 경로를 사용한다.
- 파일 내용은 가능한 경우 SHA-256으로 식별한다.
- Schema 검증은 구조를, `src/contracts/validation.py`는 분모·성공 판정 같은 의미 규칙을 검사한다.
- 기존 산출물에 새 스키마를 소급 적용하지 않는다. 새 실험부터 버전이 명시된 레코드를 생성한다.
