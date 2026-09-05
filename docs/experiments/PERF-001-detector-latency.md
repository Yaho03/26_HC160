# PERF-001 — Detector Latency 실측

| 항목 | 내용 |
|---|---|
| 문서명 | Squeezing detector latency 측정 결과 |
| 요구사항 ID | `PERF-001` (`face_auth/BACKLOG.md`), `EXP-PERF-001` (`08_EXPERIMENT_PLAN.md`) |
| 상태 | 1차 측정 완료. 배치화 적용 후 재측정 완료. 예산 판정 불가 |
| 측정일 | 2026-09-05 |
| 기준 문서 | `07_DEFENSE_AND_DETECTION_SPEC.md` 7절, `09_EVALUATION_METRICS.md` 6절 |
| 도구 | `src/verification/defenses/latency_bench.py` |
| 단위 test | `tests/unit/test_latency_bench.py` (38건) |

---

## 1. 왜 측정했나

`07_DEFENSE_AND_DETECTION_SPEC.md` 7절의 잠정 통과 기준은 네 개다.

| 기준 | 상태 |
|---|---|
| conditional ASR 50% 이상 감소 | 측정 완료 (`probe_threshold`) |
| clean TAR 감소 2%p 이하 | 측정 완료 (`probe_threshold`) |
| 95% 신뢰구간과 모든 error 보고 | 측정 완료 |
| **명시한 reference hardware의 latency budget 충족** | **미측정** |

네 번째만 비어 있었다. `face_auth/BACKLOG.md`의 `PERF-001`도 같은 결손을
"명시 hardware의 FPS/drop/P95 report 없음"으로 기록하고 있다.

## 2. 측정 환경

**단일 기기 단일 환경이다.** 다른 기기의 수치로 재사용하지 않는다.

| 항목 | 값 |
|---|---|
| 기기 | MacBook Pro (`MacBookPro18,3`) |
| CPU | Apple M1 Pro, 10코어 (8 performance + 2 efficiency) |
| 메모리 | 16 GB |
| OS | macOS 26.6.2 |
| Python | 3.9.13, `~/opt/anaconda3/bin/python3` |
| 빌드 아키텍처 | **x86_64 (`sysctl.proc_translated = 1`, Rosetta 2 번역 실행)** |
| torch | 2.2.2 |
| torch 스레드 | 10 |
| 장치 | `cpu`, `mps` (cuda 없음) |
| 모델 | `facenet-vggface2-2.6.0` (`InceptionResnetV1`, eval) |

### 2.1 Rosetta 2 경고

이 환경의 Python은 x86_64 빌드이며 Apple Silicon에서 번역 실행된다. torch가
`Could not initialize NNPACK! Reason: Unsupported hardware`를 출력하므로 CPU
convolution 경로가 최적화 없이 돈다.

따라서 다음을 지킨다.

- **`cpu` 열을 "M1 Pro의 CPU 성능"으로 읽지 않는다.** native arm64 빌드에서는 더
  빠를 것이다. 얼마나 빠른지는 측정하지 않았으므로 수치를 추정하지 않는다.
- `mps` 열의 GPU 연산 자체는 번역 대상이 아니지만 호출을 발행하는 호스트 코드는
  번역된다. 5.2절이 지목하는 호출당 고정 비용에는 이 번역분이 섞여 있다.
- 같은 환경 안에서의 **상대 비교**(배치 vs 개별 호출, 고정 vs 랜덤화)는 유효하다.
  양쪽이 같은 조건에서 돌았기 때문이다.

## 3. 측정 방법

```bash
python -m src.verification.defenses.latency_bench \
  --device mps --repeats 50 --warmup 10 \
  --crop data/raw/lfw_data/.../German_Khan_0001.jpg \
  --out outputs/perf/PERF-001-latency-mps-run1.json
```

| 항목 | 값 |
|---|---|
| 표본 수 | 시나리오·장치·run마다 50 |
| 워밍업 | 10회 실행 후 폐기. 모델 로딩과 MPS 커널 컴파일을 제외하기 위해서다 |
| run 수 | 장치마다 2회. 별도 프로세스 |
| 입력 | LFW `German_Khan_0001.jpg`를 160×160으로 축소한 크롭 1장 |
| 백분위 | nearest-rank. 보간하지 않는다 |
| `max_frames` | 3 (`FeatureSqueezeConfig` 기본값) |

장치는 실행당 하나만 측정한다. `facenet_embed.get_model`이 모델을 전역 싱글톤으로
들고 있어 한 프로세스에서 장치를 바꾸면 두 번째 장치가 조용히 무시된다. CLI가
로드된 장치와 요청한 장치가 다르면 예외를 낸다.

### 3.1 단계 분해 방식

임베더를 감싸 forward 호출에 머문 시간만 따로 모은다. 운영 코드는 바꾸지 않는다.

- `forward` — 임베더 호출 내부에 머문 누적 시간
- `other` — 나머지. 변환 적용, PIL/numpy 변환, cosine 연산, 판정 로직
- `other`를 "변환 시간"이라고 부르지 않는다. 변환만의 비용은 별도의 변환 전용
  시나리오(5절)로 교차 확인한다.

### 3.2 포함하지 않은 것

카메라 캡처, MTCNN 얼굴 검출, quality/liveness/PAD 게이트, 정책 판정은 재지
않았다. squeezing detector 경로만 잰다.

## 4. 두 경로의 결과

단위 ms. `run1 / run2`.

### 4.1 face_auth 실시간 게이트 (`FeatureSqueezeInspector.evaluate`)

프레임당 변환 3종, 최근 3프레임. 실시간 인증 경로다.

| 장치 | 게이트 | 단계 | n | p50 | p95 |
|---|---|---|---|---|---|
| cpu | 1개 (`adversarial`) | total | 50 | 615.49 / 698.50 | 734.71 / 890.37 |
| cpu | 1개 | forward | 50 | 609.39 / 692.14 | 728.48 / 883.78 |
| cpu | 1개 | other | 50 | 6.19 / 6.32 | 6.45 / 6.77 |
| cpu | 2개 (+`adversarial_template`) | total | 50 | 635.79 / 632.01 | 738.43 / 732.99 |
| **mps** | **1개** | **total** | **50** | **616.43 / 621.95** | **643.18 / 648.95** |
| mps | 1개 | forward | 50 | 609.99 / 615.67 | 636.76 / 642.19 |
| mps | 1개 | other | 50 | 6.35 / 6.38 | 6.75 / 6.77 |
| mps | 2개 | total | 50 | 606.08 / 600.22 | 646.23 / 624.70 |

### 4.2 연구 트랙 계측 (`squeeze_probe.probe_crop`)

변환 14종 + 원본을 한 배치(15장)로 묶어 forward 1회.

| 장치 | 단계 | n | p50 | p95 |
|---|---|---|---|---|
| cpu | total | 50 | 439.02 / 450.80 | 570.84 / 560.61 |
| cpu | forward | 50 | 429.52 / 441.37 | 561.05 / 550.49 |
| cpu | other | 50 | 9.88 / 9.81 | 10.22 / 10.39 |
| **mps** | **total** | **50** | **84.59 / 84.80** | **94.67 / 90.06** |
| mps | forward | 50 | 74.95 / 74.80 | 84.87 / 80.10 |
| mps | other | 50 | 9.69 / 9.67 | 10.40 / 10.21 |

### 4.3 세 가지 관찰

**(a) 실시간 경로가 계측 경로보다 7배 느리다.** mps에서 게이트 616 ms, 연구 프로브
85 ms다. 게이트는 이미지를 9장(변환 3종 × 3프레임) 임베딩하고 프로브는 15장을
임베딩한다. 이미지가 적은 쪽이 더 느리다.

**(b) mps가 게이트를 빠르게 하지 못한다.** 연구 프로브는 cpu 439 ms → mps 85 ms로
5.2배 빨라지지만, 게이트는 615 ms → 616 ms로 사실상 변화가 없다. 가속기를 붙여도
실시간 경로가 그대로라는 뜻이다.

**(c) 게이트를 하나 더 붙여도 비용이 늘지 않는다.** `adversarial_template`을 추가한
2게이트 구성이 1게이트와 같다(mps 606 vs 616, cpu 636 vs 615). 변환과 임베딩을
프레임당 한 번만 계산해 두 게이트가 공유한다는 `07` 6.1절 설계가 실측으로 확인된다.

## 5. 병목은 변환이 아니라 forward다

### 5.1 변환만의 비용

| 시나리오 | 변환 수 | 장치 | n | p50 | p95 |
|---|---|---|---|---|---|
| `transform_only_gate` (게이트 프레임 1장분) | 3 | mps | 50 | 1.59 / 1.54 | 1.85 / 1.65 |
| `transform_only_research` (프로브 1장분) | 14 | mps | 50 | 8.86 / 8.90 | 9.29 / 9.58 |
| `transform_only_gate` | 3 | cpu | 50 | 1.61 / 1.53 | 1.79 / 1.88 |
| `transform_only_research` | 14 | cpu | 50 | 8.93 / 8.43 | 9.24 / 10.15 |

변환은 GPU를 쓰지 않으므로 장치와 무관하다. 값이 같은 것이 그 확인이다.

게이트 한 번의 변환 총량은 3프레임 × 1.6 ms ≈ 4.8 ms이고, 여기에 PIL/numpy 변환과
cosine 연산을 더한 `other`가 6.4 ms다. 게이트 total 616 ms의 **1.0%**다.

### 5.2 forward 격리

같은 이미지 수를 배치 1회와 개별 호출로 나눠 쟀다.

| 시나리오 | 호출 | 배치 | 장치 | n | p50 | p95 |
|---|---|---|---|---|---|---|
| `forward_only_single1` | 1 | 1 | mps | 50 | 67.28 / 65.65 | 74.27 / 73.96 |
| `forward_only_single_gate` (현재 게이트 경로) | 9 | 1 | mps | 50 | 602.85 / 587.16 | 680.97 / 608.20 |
| `forward_only_batch_gate` (같은 9장을 배치 1회로) | 1 | 9 | mps | 50 | 77.86 / 71.39 | 82.97 / 77.04 |
| `forward_only_single1` | 1 | 1 | cpu | 50 | 67.57 / 67.53 | 102.70 / 107.62 |
| `forward_only_single_gate` | 9 | 1 | cpu | 50 | 636.62 / 681.51 | 759.61 / 761.78 |
| `forward_only_batch_gate` | 1 | 9 | cpu | 50 | 339.06 / 290.02 | 389.31 / 389.07 |

**mps에서 배치 1장(67 ms)과 배치 9장(78 ms)의 차이가 11 ms다.** 이미지가 9배로
늘었는데 시간은 16% 늘었다. 이 크기에서 FaceNet forward는 연산이 아니라 호출당 고정
비용에 묶여 있다. 게이트가 느린 이유는 이 고정 비용을 9번 내기 때문이다.

원인은 `src/face_auth/inference/verifier.py`의 `FaceNetEmbedder.embed`가 이미지
목록을 받아 `get_embedding`을 한 장씩 호출하는 구조다. 반면 연구 트랙은
`facenet_embed.embed_batch`로 한 번에 forward한다. 두 경로의 7배 차이가 여기서 나온다.

### 5.3 계산되는 개선폭

이 문서는 개선을 구현하지 않았다. 아래는 측정된 두 값의 비율이며 예측이 아니다.

| 장치 | 개별 9회 | 배치 1회 | 비율 |
|---|---|---|---|
| mps | 602.85 ms | 77.86 ms | **7.7배** |
| cpu | 636.62 ms | 339.06 ms | 1.9배 |

mps에서 게이트 forward를 배치 1회로 묶으면 게이트 total이 616 ms에서 85 ms 수준
(연구 프로브와 같은 자리)으로 내려갈 여지가 있다. cpu에서는 2배 수준이다. 실제
개선폭은 배치화를 구현한 뒤 같은 도구로 재측정해야 확정된다.

## 6. 랜덤화의 latency 비용

고정 파라미터 변환 3종과 계열별 랜덤 추출 3종을 같은 횟수로 비교했다.

| 시나리오 | 장치 | n | p50 | p95 |
|---|---|---|---|---|
| `transform_only_fixed3` (blur0.8, jpeg_q75, median3) | mps | 50 | 1.45 / 1.39 | 1.78 / 1.56 |
| `transform_only_randomized3` (blur, jpeg, median 계열) | mps | 50 | 1.74 / 1.68 | 2.12 / 1.81 |
| `transform_only_sample_params3` (적용 없이 추출만) | mps | 50 | 0.02 / 0.02 | 0.02 / 0.02 |
| `transform_only_fixed3` | cpu | 50 | 1.44 / 1.39 | 1.72 / 1.42 |
| `transform_only_randomized3` | cpu | 50 | 1.75 / 1.69 | 2.10 / 1.98 |
| `transform_only_sample_params3` | cpu | 50 | 0.02 / 0.02 | 0.02 / 0.02 |

**차이는 p50 +0.30 ms, p95 +0.34 ms다.** 게이트 total 616 ms의 0.05%다.

그중 파라미터 추출 자체는 0.02 ms다. 나머지 약 0.28 ms는 랜덤화가 아니라 **파라미터
분포 차이**에서 온다. 랜덤 범위(`blur` 0.5~2.0, `jpeg` 30~75, `median` 3~5)가 고정
기준선(`blur` 0.8, `jpeg` 75, `median` 3)보다 평균적으로 무거운 연산을 뽑기 때문이다.
따라서 0.30 ms는 랜덤화 비용의 **상한**이다.

"파라미터가 매번 달라지면 캐시 효과가 사라진다"는 우려는 이 경로에서 성립하지
않는다. 고정 경로에도 변환 결과 캐시가 없다. `squeeze_probe`와 `feature_squeeze`
모두 매 호출에서 변환을 새로 계산한다. 사라질 캐시가 애초에 없다.

## 7. `07` 7절 latency 기준 판정

**판정할 수 없다. 기준값이 저장소에 정의돼 있지 않다.**

`07` 7절은 "명시한 reference hardware의 latency budget 충족"을 요구하지만, 그
budget의 수치를 정의한 문서가 없다. `09_EVALUATION_METRICS.md` 6절은 보고 형식
(p50/p95/표본 수/hardware 명시)만 규정하고 값을 정하지 않는다.
`face_auth/BACKLOG.md` `PERF-001`도 "FPS, drop과 P95 latency 보고"까지만 적는다.

이 문서는 **측정값을 제공하고 판정은 하지 않는다.** 예산 수치를 사후에 만들어
붙이면 `07` 7절의 "기준을 변경하려면 최종 test 결과를 보기 전에 기록한다"를 위반한다.

판정을 성립시키려면 다음이 필요하다.

1. **reference hardware 지정.** 이 M1 Pro/Rosetta 환경은 개발 노트북이지 승인된
   reference hardware가 아니다.
2. **budget 수치 확정.** 규범 문서에 값을 먼저 기록한다.
3. **적용 단위 명시.** detector 게이트 1회인지 인증 세션 전체인지. 아래 7.1이 그
   구분에 필요한 사실이다.

### 7.1 적용 단위에 대한 사실

`FullPipeline.evaluate`는 `adversarial_inspector.evaluate`를 **인증 판정당 1회**
호출한다. 프레임마다 호출하지 않는다. 따라서 측정값의 의미는 다음과 같다.

- 캡처 fps에는 영향을 주지 않는다. 캡처가 끝난 뒤 판정 단계에서 돈다.
- 인증 1회의 종단 지연에 mps 기준 **p50 0.62초, p95 0.65초**를 더한다.
- 이 값은 detector 게이트만의 비용이다. quality, liveness, PAD, continuity 게이트와
  MTCNN 검출 비용은 여기 포함되지 않는다. 세션 전체 지연은 더 크다.

### 7.2 `PERF-001`의 남은 결손

`PERF-001`의 완료 조건은 "FPS, drop과 P95 latency 보고"다. 이 문서는 셋 중 하나만
채운다.

| 항목 | 상태 |
|---|---|
| detector 게이트 P95 latency | 이 문서에서 측정 |
| 캡처 FPS | 미측정. `FrameSource` 카메라 세션 필요 |
| frame drop | 미측정. 같은 세션 필요 |

따라서 `PERF-001`은 완료가 아니라 **부분 구현**이다.

## 8. 한계

- **단일 기기 단일 환경이다.** MacBook Pro 18,3 하나에서만 쟀다. 다른 기기의
  수치로 쓰지 않는다.
- **Rosetta 2 번역 실행이다.** 2.1절 참조. `cpu` 열은 native arm64의 성능이 아니다.
- **run 2회다.** cpu에서 run 사이 p50이 615 ms와 699 ms로 13% 벌어졌다(`face_auth_gate_1`).
  mps는 616 ms와 622 ms로 안정적이었다. cpu 수치의 run 간 변동을 더 좁히려면 run을
  늘려야 한다.
- **같은 크롭을 반복 입력했다.** 프레임마다 내용이 달라지는 실제 세션의 분산을
  반영하지 않는다. JPEG 압축 시간은 내용에 따라 달라진다.
- **크롭 1장이다.** 얼굴 1장(LFW `German_Khan_0001`)만 썼다. 이미지별 변환 비용
  분산을 측정하지 않았다.
- **detector 게이트만 쟀다.** 캡처, MTCNN, 다른 게이트, 정책 판정은 포함하지 않는다.
- **개선을 구현하지 않았다.** 5.3절의 배치화 개선폭은 두 측정값의 비율이며, 실제로
  배치화한 코드를 측정한 값이 아니다.
- **탐지 성능과 무관하다.** 이 문서는 시간만 잰다. 배치화가 수치 결과를 바꾸지
  않는다는 것은 별도로 확인해야 한다.

## 9. 재현

```bash
# 장치마다 프로세스를 나눠 실행한다. 한 프로세스에서 장치를 바꾸면 무시된다.
for dev in cpu mps; do
  python -m src.verification.defenses.latency_bench \
    --device $dev --repeats 50 --warmup 10 \
    --crop <160px 얼굴 크롭 경로> \
    --out outputs/perf/PERF-001-latency-$dev.json
done

python -m unittest discover -s tests/unit -t .
```

`outputs/perf/`는 gitignore 대상이므로 JSON 산출물은 커밋하지 않는다. 수치는 이
문서에 남는다.

## 9. 배치화 적용 후 재측정 (2026-09-05)

5.2절이 지목한 원인을 고쳤다. `src/face_auth/inference/verifier.py`의
`FaceNetEmbedder.embed`가 이미지 목록을 한 배치로 forward한다.

측정 환경은 2절과 같다. Rosetta 2 번역 실행이며 절대 성능이 아니라 같은 환경 내
상대 비교만 유효하다.

### 9.1 embed 자체의 개선

같은 실행 안에서 변경 전 구현과 현재 구현을 번갈아 쟀다. 9장, mps, 30회 반복,
워밍업 10회다.

| 구현 | p50 | p95 |
|---|---|---|
| 개별 호출 9회 (변경 전) | 598.84 ms | 610.75 ms |
| 배치 1회 (현재) | 72.14 ms | 76.21 ms |

**8.30배다.** 5.3절이 별개로 잰 두 값의 비율로 계산한 7.7배보다 크다. 그 계산은
예측이 아니라 서로 다른 실행의 비율이었고, 이 값은 한 실행 안에서 같은 조건으로 잰
것이다.

### 9.2 게이트 전체의 개선

| 시나리오 | 변경 전 p50 | 변경 후 p50 | 배수 |
|---|---|---|---|
| `face_auth_gate_1` total | 616.43 | 224.19 | 2.75 |
| `face_auth_gate_1` forward | 609.99 | 218.05 | 2.80 |
| `face_auth_gate_2` total | 606.08 | 220.33 | 2.75 |

**게이트 전체는 2.75배에 그친다.** embed 하나가 8.3배 빨라졌는데 게이트가 그만큼
줄지 않은 이유는 게이트가 프레임마다 embed를 부르기 때문이다. `max_frames`가 3이므로
forward가 3회 남는다.

배치화는 프레임 안의 변환 3장을 묶었을 뿐 프레임 사이를 묶지 않았다. 프레임 3장의
변환 9장을 한 번에 묶으면 더 줄일 여지가 있으나, 그러려면 `FeatureSqueezeInspector`의
루프 구조를 바꿔야 한다. 이 문서의 범위 밖이며 별도 변경으로 판단한다.

### 9.3 벤치마크 시나리오 이름이 어긋났다

`forward_only_single_gate`는 `FaceNetEmbedder.embed`를 호출한다. 배치화 이후 이
시나리오는 더 이상 "개별 호출 9회"가 아니다. 재측정에서 `single_gate` 72.14 ms와
`batch_gate` 74.19 ms가 같아진 것이 그 결과다.

| 시나리오 | 변경 전 | 변경 후 | 의미 |
|---|---|---|---|
| `forward_only_single_gate` | 602.85 | 72.14 | 이름과 달리 이제 배치 1회다 |
| `forward_only_batch_gate` | 77.86 | 74.19 | 변화 없음 |

두 값이 같아진 것은 배치화가 적용됐다는 증거이지 성능 저하가 아니다. 시나리오 이름은
`latency_bench.py`가 소유하므로 여기서 바꾸지 않는다. 이름을 고치려면 그 파일을
수정해야 하며 별도 작업이다.

### 9.4 값이 바뀌지 않았다

배치화가 임베딩 값을 바꾸면 지금까지 산출한 모든 임계값이 무효가 된다.

| 항목 | 결과 |
|---|---|
| 배치와 순차의 최대 절대 차이 | 6.333e-08 |
| 허용치 | 1e-05 |
| 여유 | 허용치의 0.6% |

`tests/unit/test_batch_embed.py`가 이를 고정한다. dtype(float32), shape(512,), 순서,
L2 정규화를 각각 별도 테스트로 둔다. 값 동일성 테스트에 dtype 검증을 묻으면 float64로
바꿔도 값이 같아 통과한다.
