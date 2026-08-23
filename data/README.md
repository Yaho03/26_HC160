# Data workspace

원본 얼굴 이미지와 파생 데이터는 개인정보·라이선스·용량 문제로 Git에 커밋하지 않는다. 이 디렉터리에는 데이터 자체 대신 manifest와 획득·검증 절차만 둔다.

권장 로컬 구조:

```text
data/
  manifests/     # 버전 관리 가능한 비식별 메타데이터
  raw/           # 원본, 읽기 전용 취급
  processed/     # 결정적 전처리 결과
  splits/        # train/validation/test 및 pair 정의
```

모든 실험은 사용한 manifest, split, 전처리 설정의 해시를 실행 manifest에 기록해야 한다.
