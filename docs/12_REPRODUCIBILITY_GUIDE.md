# REPRODUCIBILITY GUIDE — 재현 가이드

| 항목 | 내용 |
|---|---|
| 문서명 | 실험 재현 가이드 |
| 버전 | v1.0 |
| 상태 | 확정 |
| 최종 수정일 | 2026-08-23 |

---

## 1. 재현 수준

| 수준 | 요구사항 |
|---|---|
| Unit | Standard-library contract와 metric test 통과 |
| Smoke | 작은 CPU dataset이 manifest, scoring과 reporting을 통과 |
| Baseline | 고정 dataset, checkpoint, threshold와 configuration으로 허용 오차 안에서 aggregate metric 재현 |
| Full | GPU attack/defense run에서 모든 artifact와 report table 재현 |

## 2. 필수 run metadata

- Git commit과 dirty-worktree flag
- Direct dependency lock hash
- Python, PyTorch, torchvision, CUDA/MPS와 OS version
- Device 이름과 deterministic-mode 설정
- Dataset, pair, checkpoint, preprocessing과 threshold artifact ID
- 전체 configuration과 hash
- Python, NumPy와 PyTorch seed
- 시작·종료 시간과 reproduction command
- Output artifact hash

## 3. 실행 환경 정책

`environment.yml`은 초기 환경 재현을 위해 유지한다. 새 setup file은 direct dependency만
기록하고 lightweight test, research ML, FaceNet과 camera extra를 구분한다. 장비별 Conda
`prefix`는 portable lock에 포함하지 않는다.

FaceNet exporter는 `requirements-face-auth.txt`의 `facenet-pytorch`를 사용한다.
VGGFace2 checkpoint는 외부 artifact이므로 file을 명시하고 license와 SHA-256을 확인한다.
보고 가능한 run에서 암시적 download에 의존하지 않는다.

## 4. Seed 정책

Python, NumPy, PyTorch CPU와 모든 CUDA device에 같은 정수를 설정한다. Deterministic
algorithm 활성화 여부를 기록한다. 이 mode가 성능을 낮추거나 지원하지 않는 operation을
거부할 수 있으므로 예외가 있으면 run manifest에 기록한다.

## 5. Artifact 정책

- Git: code, configuration, schema, metadata, summary table과 승인된 figure
- 외부 저장소: dataset, raw/derived face image, embedding과 checkpoint
- 외부 artifact마다 hash와 sensitivity가 있는 Git 추적 metadata row
- 완료된 run artifact는 append-only

## 6. Colab 정책

- 범위 없는 `git pull` 대신 특정 commit 또는 tag checkout
- 고정된 direct-dependency file 설치
- Hash 검증 후 artifact 복원
- 새 run directory에 output 작성
- Experiment Notebook에서 commit 또는 push 금지

## 7. 현재 baseline의 한계

과거 attack image와 checkpoint는 Git 외부에 있고 초기 run에서 hash와 정확한 package
version을 기록하지 않았다. 이 결과는 legacy evidence로 보존하지만 provenance를 복구하기
전에는 완전 재현 가능하다고 분류할 수 없다.
