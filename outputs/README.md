# Experiment outputs

기존 결과는 역사적 증거로 유지한다. 새 실험 결과는 실행 ID 단위의 디렉터리에 저장하고 `run_manifest.json`을 포함한다.

```text
outputs/<experiment>/<run_id>/
  run_manifest.json
  metrics.json
  records.jsonl
  artifacts/
```

요약 지표만으로 결과를 덮어쓰지 않는다. 표본별 판정, 사용한 설정과 입력 산출물 해시를 함께 보존한다. 민감한 얼굴 원본과 복구 가능한 생체정보는 출력에 포함하지 않는다.
