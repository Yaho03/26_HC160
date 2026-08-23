# SECURITY, ETHICS AND LIMITATIONS — 보안·윤리·한계

| 항목 | 내용 |
|---|---|
| 문서명 | 보안·윤리 및 제한 사항 |
| 버전 | v1.0 |
| 상태 | 확정 |
| 최종 수정일 | 2026-08-23 |

---

## 1. 연구 이용 범위

이 프로젝트는 승인된 방어 목적 보안 연구로 제한한다. Attack code는 프로젝트가 통제하는
model, data와 demo environment에서만 평가한다. 제3자 인증 시스템을 대상으로 사용해서는
안 된다.

## 2. 생체 데이터

Face image와 embedding은 민감 biometric artifact다.

- 데이터 최소화
- 커밋 metadata의 가명 identity
- 가능한 경우 암호화된 외부 저장소
- 역할별 접근 제한
- 명시적인 보관·삭제 일자
- Raw face와 embedding의 Git 커밋 금지
- 검토 없는 checkpoint 또는 생성 이미지 공개 금지

## 3. Dataset·model license

모든 dataset과 pretrained weight는 출처와 license 조건을 기록한다. Code license가 face
image나 제3자 weight 재배포 권한을 부여하지 않는다. Release review에서 각각 별도의
artifact로 검토한다.

## 4. 주장과 한계

- LFW 실험은 실제 금융 고객과 환경을 검증하지 않는다.
- 공개 pretrained model은 공개 benchmark와 identity가 겹칠 수 있다.
- 현재 dataset으로 인구집단 공정성을 주장할 수 없다.
- Python webcam prototype은 OS, driver, camera 또는 virtual-camera source를 보증할 수 없다.
- Simulation 기반 temporal 결과는 replay resistance의 증거가 아니다.
- Attack/defense 효과는 model, preprocessing, serialization과 threshold에 따라 달라진다.

## 5. 공정성

Demographic attribute가 없거나 불충분하면 이를 report에 밝힌다. 충분한 sample, 동의와
문서화된 방법 없이 어떤 subgroup도 안전하거나 취약하다고 표현하지 않는다.

## 6. 생성형 AI

기본 승인 범위는 방어 목적 purification이다. 생성형 impersonation, identity transfer와
deepfake 제작은 범위 밖이다. Purification은 attack 제거뿐 아니라 identity drift와 clean
false rejection을 측정해야 한다.

## 7. Release checklist

- Raw face image, embedding, local path, credential 또는 private URL 없음
- Dataset과 model license 필드 완성
- Result row가 immutable run ID로 해석됨
- 한계와 non-production 면책 문구 포함
- Attack artifact와 code에 승인된 연구 이용 맥락 존재
- 재생성한 result가 historical artifact를 조용히 대체하지 않음
