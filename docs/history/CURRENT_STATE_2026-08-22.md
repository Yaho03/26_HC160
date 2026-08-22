# HC160 연구 상태 동결 기록 (2026-08-22)

이 문서는 연구 기반 재구성을 시작하기 전, 기존 HC160 저장소의 재현 가능한 기준점을 기록한다. 기존 산출물을 수정하거나 재생성하지 않았으며, 아래 해시는 파일 내용의 SHA-256 값이다.

## 기준 커밋

- 저장소: `Yaho03/26_HC160`에서 이어진 로컬 이력
- 기준 커밋: `795b3bbce889c45a3a3cddac1049394785f783bc`
- 작업 브랜치: `codex/realtime-face-auth-v2`

## 주요 파일 해시

| 파일 | SHA-256 |
|---|---|
| `README.md` | `9f9125d417fdd4e3ad54ff220b0337cecbb1d3896e5cfe1dcf6e462ed5055484` |
| `environment.yml` | `4acccfcaa3c77d18972e88a78750079fca0e44db86409771b39c8cd873e92fc5` |
| `DB_SCHEMA.md` | `16d8f86deaad45de7d94a35db727dfeace6884a6d4f6dc1bfad9a87241a338d4` |
| `outputs/defenses/defense_summary.csv` | `b8df24ac46f1a82675247336553eead50d70741a1d74ed16442b9b2a8210b8a4` |
| `outputs/defenses/defense_report.md` | `4ca65f6b0f82afcf45821c4d16d0c10d8946ffac4b68e628f031132b91d7a0f9` |
| `outputs/verification_defense/attack_handoff_jpeg_index.csv` | `36873cd8a29b380bd0ba014440753e8c349812cb983c4ce019e7228353c86378` |
| `outputs/verification_defense/verification_defense_summary.csv` | `cbdde968db016e6133a1751e7a8df42d4fcdfaaed25d148aaad06fea26831bde` |
| `outputs/verification_defense/verification_defense_report.md` | `b97a7c3045aeba3ef9a10e143f540c8074feea5f0f1bb84d911ed4e1e2f0aabb` |
| `outputs/verification_defense/adv_training/training_history.json` | `92d2dd2c46bcfbc192ac837efa9c986ed9a5d70823721a82d26351741dd13441` |

## 해석 시 주의사항

- 기존 분류 실험과 얼굴 검증 실험은 서로 다른 평가 트랙이다. 정확도, 임계값, 공격 성공률을 직접 합산하거나 동일 지표처럼 비교하지 않는다.
- 기존 FaceNet 검증 방어 결과는 존재하지만, 대응하는 배치 공격 생성 과정과 원본 무손실 섭동의 완전한 provenance는 저장소만으로 복원되지 않는다.
- 기존 적대적 학습 결과에는 학습 표본과 평가 표본을 분리했다는 근거가 충분하지 않다.
- 기존 JPEG 산출물은 저장 과정의 손실 압축이 섭동에 영향을 줄 수 있으므로 원본 섭동 텐서의 증거로 취급하지 않는다.
- 외부 데이터셋, 모델 가중치, 로컬 절대 경로로만 참조된 산출물은 이 기록 시점에도 미해결 외부 의존성이다.
- 이 파일은 감사 기준점이며 최신 실험 결과 문서가 아니다. 이후 결과는 새 계약과 실행 manifest를 사용한다.
