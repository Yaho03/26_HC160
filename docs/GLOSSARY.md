# GLOSSARY — 용어집

| 용어 | 정의 |
|---|---|
| Enrollment | 주장된 identity의 기준 biometric representation을 만드는 과정 |
| Probe | Enrollment reference와 비교하는 얼굴 evidence |
| Verification | Probe가 주장된 identity와 일치하는지 판단하는 1:1 비교 |
| Identification/classification | 여러 identity/class 중 하나를 선택하는 작업. 금융 인증의 주 task가 아님 |
| Genuine pair | Enrollment와 probe가 같은 identity인 pair |
| Impostor pair | Enrollment와 probe가 다른 identity인 pair |
| FAR/FMR | Impostor attempt를 잘못 accept한 비율 |
| FRR/FNMR | Genuine attempt를 잘못 reject한 비율 |
| TAR | Genuine attempt를 올바르게 accept한 비율 |
| EER | FAR과 FRR이 거의 같아지는 operating point |
| TAR@FAR | 지정한 false-acceptance target에서 측정한 genuine acceptance |
| Targeted ASR | 공격 전 reject된 eligible impersonation attempt가 accept로 바뀐 비율 |
| White-box attack | 공격자가 model 세부 정보와 gradient를 사용할 수 있는 공격 |
| Black-box attack | 내부 gradient 없이 model query를 사용하는 공격 |
| Adaptive attack | Defended model 또는 defense transformation을 명시적으로 대상으로 삼는 공격 |
| Transfer attack | 한 model에서 만든 공격을 다른 model에서 평가하는 공격 |
| Defense success | 방어 전 성공한 공격이 방어 후 reject로 바뀐 상태 |
| Clean degradation | Defense로 인해 발생한 clean verification 성능 저하 |
| Detector | 의심 evidence를 생성하는 component. Authentication decision과 동일하지 않음 |
| Calibration split | Threshold와 operating parameter 선택에 사용하는 data |
| Test split | Parameter 고정 후 한 번만 사용하는 untouched data |
| Artifact | Hash, producer, sensitivity와 lineage를 가진 versioned file 또는 record |
| Run manifest | Code, configuration, input, output, environment와 command를 연결하는 immutable metadata |
| Identity drift | Purification 또는 transformation이 biometric identity representation을 바꾸는 현상 |
| PAD | Print, screen, mask 또는 replay media를 탐지하는 Presentation Attack Detection |
| Legacy result | 새 계약을 충족하지 않아도 원래 방법과 함께 보존하는 역사적 결과 |
