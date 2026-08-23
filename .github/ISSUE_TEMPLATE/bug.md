---
name: "Bug"
about: "재현 가능한 결함 보고"
title: "[BUG] "
labels: []
assignees: []
---

## 현상

<!-- 어떤 문제가 발생하는지 간결하게 적는다. -->

## 관련 계약 또는 기대 동작

<!-- 요구사항 ID, 규범 문서, schema 또는 기존 테스트를 연결한다. -->

- 요구사항 ID:
- 기준 문서/스키마:

## 재현 절차

1.
2.
3.

## 기대 동작


## 실제 동작


## 영향과 우선순위

**우선순위:** P0-critical / P1-high / P2-medium / P3-low

<!-- metric 유효성, split 누수, 민감 데이터, 데모·재현성에 미치는 영향을 적는다. -->

## 환경과 재현 정보

- Git commit:
- 실행 명령:
- config/seed:
- dataset/split:
- model/checkpoint:
- threshold/policy:
- CPU/GPU 및 주요 의존성:

## 로그와 증거

```text

```

## 완료 조건

- [ ] 최소 재현 사례가 자동 테스트 또는 fixture로 추가된다.
- [ ] 원인이 수정되고 기대 동작이 검증된다.
- [ ] 실패·오류 처리에 대한 회귀 테스트가 통과한다.
- [ ] 관련 문서·스키마·구현 상태가 업데이트된다.

## 관련 이슈

- blocked by:
- blocks:
