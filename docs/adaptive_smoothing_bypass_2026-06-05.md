# Adaptive PGD가 Gaussian Smoothing 방어를 우회하다 (공격팀 분석)

작성일: 2026-06-05
작성: 공격팀
실행: Kaggle 커널 `dohyunp/hc160-bypass-analysis`

## 실험 설계

같은 attack optimizer(`targeted_pgd_facenet_adaptive`)를 **동일 조건**으로 두 번 실행, 차이는 오직 "방어 인지 여부":

- **Plain PGD**: `--defense-transform none` (clean tensor에서 gradient)
- **Adaptive PGD**: `--defense-transform smoothing` (smoothing을 공격 루프 안에 넣어 통과하도록 최적화)

공통: steps=20, alpha=0.001, 100쌍(initial reject), PNG 저장, FaceNet vggface2, thr=0.4797.
평가: 생성된 adversarial PNG에 Gaussian smoothing(radius=3, 방어팀 설정)을 적용 후 target 대비 cosine로 accept 재판정.

- `raw_success`: 방어 미적용 시 target accept 수 (/100)
- `defended_accept`: smoothing 적용 후에도 accept 수 (/100)
- `bypass_rate` = defended_accept / raw_success
- `defense_success` = 1 - bypass_rate

## 결과

| eps | 공격 | raw_success | defended_accept | bypass | defense_success |
|---|---|---:|---:|---:|---:|
| 0.005 | Plain | 57 | 2 | 3.5% | 96.5% |
| 0.005 | Adaptive | 3 | 0 | 0.0% | 100.0% |
| 0.010 | Plain | 86 | 2 | 2.3% | 97.7% |
| 0.010 | Adaptive | 6 | 4 | 66.7% | 33.3% |
| 0.020 | Plain | 100 | 4 | 4.0% | 96.0% |
| 0.020 | **Adaptive** | 25 | **24** | **96.0%** | **4.0%** |

## 해석

1. **일반 PGD는 smoothing에 거의 다 막힘** — 우회율 ~2-4%. 방어팀 보고(smoothing defense success 78.8%)와 일관, 본 설정에선 더 강함(96%+).
2. **Adaptive PGD는 smoothing을 우회** — eps=0.020에서 방어 후에도 24/100 통과(우회율 96%), 동일 eps 일반 PGD(4/100)의 6배.
3. **Trade-off** — adaptive는 raw 성공률이 낮음(blur 견디게 만드느라 raw 희생). 운영 관점 결론: 방어 없으면 일반 PGD, smoothing 방어가 있으면 adaptive. eps↑에 따라 adaptive 우위 급증(0%→67%→96%).

## 결론

방어팀의 Gaussian smoothing 방어는 **adaptive attack 앞에서 무력화**(eps0.020 방어 성공 4%). 다층 방어가 단일 입력변환만으로는 부족함을 보임 → 생성형 정화(DAE/DiffPure) 같은 방어를 adaptive attack 기준으로 재평가할 필요.

## 산출물
- `outputs/analysis/smoothing_bypass_pgd_plain_s20.csv`
- `outputs/analysis/smoothing_bypass_pgd_adaptive_smoothing.csv`
- Drive: `hanium-aml/results/handoff/analysis/`
