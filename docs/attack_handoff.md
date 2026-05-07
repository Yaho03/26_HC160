# 공격 파트 인수인계 문서

## 1. 목적

이 문서는 공격 담당 파트에서 구현한 코드, 실험 결과, Colab 실행 흐름, 방어 파트 연동 파일을 팀원이 이해하고 재현할 수 있도록 정리한 문서이다.

현재 Colab은 팀 공용 계정이 아니라 개인 계정 기준으로 사용한다. 따라서 Google Drive 경로는 각자 개인 Drive에 맞춰 세팅해야 한다.

## 2. 개인 Colab / Drive 사용 원칙

Colab 런타임은 개인 Google 계정에 연결된다.

따라서 아래 파일은 각자 개인 Google Drive에 올려두어야 한다.

```text
MyDrive/hanium-aml/archive.zip
```

`archive.zip`은 LFW dataset zip 파일이다.

각자 Colab에서 Drive mount 후 다음 경로로 접근한다.

```python
from google.colab import drive
drive.mount('/content/drive')
```

Colab에서 접근되는 Drive 경로:

```text
/content/drive/MyDrive/hanium-aml/archive.zip
```

주의:

- `/content` 아래 파일은 Colab 런타임이 종료되면 사라진다.
- 중요한 결과는 개인 Drive로 복사해야 한다.
- 팀원이 같은 결과를 보려면 GitHub에는 코드만 올리고, 데이터/결과 파일은 Drive 또는 별도 공유 방식으로 전달한다.

## 3. Colab 실행 기본 흐름

### 3.1 코드 가져오기

```python
!rm -rf /content/26_HC160
!git clone https://github.com/Yaho03/26_HC160.git /content/26_HC160
%cd /content/26_HC160
```

이미 clone한 상태에서 최신 코드만 받을 때:

```python
%cd /content/26_HC160
!git pull
```

### 3.2 의존성 설치

```python
!pip install -q adversarial-robustness-toolbox[pytorch] foolbox
!python setup_check.py
```

### 3.3 LFW zip 연결

각자 개인 Drive에 `MyDrive/hanium-aml/archive.zip`을 둔 뒤 실행한다.

```python
!mkdir -p data/raw/lfw_zip data/raw
!unzip -q /content/drive/MyDrive/hanium-aml/archive.zip -d data/raw/lfw_zip
!rm -f data/raw/lfw
!ln -s /content/26_HC160/data/raw/lfw_zip/lfw-deepfunneled/lfw-deepfunneled data/raw/lfw
!find data/raw/lfw -type f -name '*.jpg' | wc -l
```

정상 출력:

```text
13233
```

## 4. 모델 학습

LFW 상위 10명 identity classification 데이터셋 생성:

```python
!python -m src.datasets.prepare_lfw_identity_dataset
```

ResNet-50 학습:

```python
!python -m src.training.train_face_resnet50 --epochs 12 --batch-size 64 --num-workers 2
```

현재 기준 결과:

```text
test accuracy: 76.23%
attack evaluation subset clean accuracy: 91.00%
```

체크포인트 위치:

```text
checkpoints/face_resnet50_lfw10/best.pt
```

## 5. 공격 코드 목록

| 스크립트 | 공격 | 설명 |
|---|---|---|
| `src/attacks/targeted_fgsm_face.py` | targeted FGSM | 1-step white-box targeted attack |
| `src/attacks/targeted_pgd_face.py` | targeted PGD | iterative white-box targeted attack |
| `src/attacks/targeted_square_face.py` | targeted Square Attack | query 기반 black-box targeted attack |
| `src/attacks/targeted_jsma_face.py` | targeted multi-pixel JSMA | saliency 기반 multi-pixel JSMA 변형 |

## 6. 공격 실행 명령

### 6.1 FGSM

```python
!for eps in 0.005 0.010 0.030 0.050; do python -m src.attacks.targeted_fgsm_face --epsilon "$eps" --limit 100; done
```

### 6.2 PGD

```python
!python -m src.attacks.targeted_pgd_face --epsilon 0.005 --alpha 0.0005 --steps 10 --limit 100
!python -m src.attacks.targeted_pgd_face --epsilon 0.010 --alpha 0.001 --steps 10 --limit 100
!python -m src.attacks.targeted_pgd_face --epsilon 0.030 --alpha 0.003 --steps 10 --limit 100
!python -m src.attacks.targeted_pgd_face --epsilon 0.050 --alpha 0.005 --steps 10 --limit 100
```

### 6.3 Square Attack

```python
!for eps in 0.005 0.010 0.030 0.050; do python -m src.attacks.targeted_square_face --epsilon "$eps" --max-queries 300 --limit 100; done
```

### 6.4 Multi-pixel JSMA

```python
!python -m src.attacks.targeted_jsma_face --theta 0.05 --steps 20 --pixels-per-step 200 --max-pixel-ratio 0.05 --limit 100
```

## 7. 공격 결과 요약

공격 결과 summary 생성:

```python
!python -m src.reports.summarize_face_attack
!cat outputs/attacks/face_attack_summary.csv
```

현재 주요 결과:

| 공격 | 주요 설정 | 원본 정분류 샘플 기준 target ASR |
|---|---|---:|
| FGSM | eps=0.005 | 46.15% |
| PGD | eps=0.005~0.050 | 100.00% |
| Square | eps=0.050, queries=300 | 76.92% |
| multi-pixel JSMA | theta=0.05, steps=20, k=200 | 100.00% |

해석:

- FGSM은 빠르지만 targeted setting에서 성공률이 불안정하다.
- PGD는 가장 강한 white-box baseline이다.
- Square Attack은 black-box 조건에서 의미 있는 성공률을 보인다.
- JSMA 변형은 희소한 perturbation으로 target identity 전환이 가능하다.

## 8. 시각화 패널 생성

패널 형식:

```text
original | adversarial | perturbation
```

예시 실행:

```python
!python -m src.reports.make_face_attack_panels --metadata outputs/attacks/fgsm_face/metadata_targeted_eps0.005.csv --out-dir outputs/attack_panels/fgsm_eps0005
!python -m src.reports.make_face_attack_panels --metadata outputs/attacks/pgd_face/metadata_targeted_eps0.030_alpha0.003_steps10.csv --out-dir outputs/attack_panels/pgd_eps003
!python -m src.reports.make_face_attack_panels --metadata outputs/attacks/jsma_face/metadata_targeted_theta0.050_steps20_k200.csv --out-dir outputs/attack_panels/jsma_theta005
```

Colab에서 이미지 확인:

```python
from IPython.display import Image, display
display(Image('/content/26_HC160/outputs/attack_panels/jsma_theta005/panel_01.jpg'))
```

## 9. 방어 파트에 넘길 파일

공격별 metadata를 하나로 통합한다.

```python
!python -m src.reports.build_attack_index
```

생성 파일:

```text
outputs/attacks/attack_index.csv
```

현재 Colab 기준 생성 결과:

```text
Metadata files: 16
Indexed attacks: 1600
Rows by attack_family:
fgsm      400
jsma      100
pgd       700
square    400
```

방어 파트는 이 파일의 `adv_file` 컬럼을 입력 이미지로 사용하면 된다.

필수 유지 컬럼:

```text
sample_id
attack_family
attack
adv_file
true_label
target_label
success_on_clean
l0
l2
linf
time_sec
```

방어 결과 저장 시 `sample_id`를 반드시 유지해야 공격 결과와 join할 수 있다.

## 10. JPEG 방어 baseline 관련 메모

공격 담당이 방어를 맡는 것은 아니지만, `attack_index.csv`가 방어 입력으로 잘 동작하는지 검증하기 위해 JPEG baseline을 샘플로 작성했다.

관련 스크립트:

```text
src/defenses/defense_jpeg.py
src/defenses/summarize_defense.py
src/reports/create_defense_result_template.py
```

역할:

- 방어 담당이 어떤 식으로 `adv_file`을 읽으면 되는지 예시 제공
- `sample_id` 유지 방식 검증
- defense result CSV 형식 검증

이후 Gaussian, Bit-depth, ROI-first 등 실제 방어 확장은 방어 담당이 이어가는 것이 적절하다.

## 11. 개인 Drive에 결과 백업

각자 개인 Drive에 결과를 저장하려면 다음을 실행한다.

```python
!mkdir -p /content/drive/MyDrive/hanium-aml/results
!cp outputs/attacks/face_attack_summary.csv /content/drive/MyDrive/hanium-aml/results/
!cp outputs/attacks/attack_index.csv /content/drive/MyDrive/hanium-aml/results/
!cp -r outputs/attack_panels /content/drive/MyDrive/hanium-aml/results/
```

체크포인트도 저장하려면:

```python
!mkdir -p /content/drive/MyDrive/hanium-aml/results/checkpoints
!cp -r checkpoints/face_resnet50_lfw10 /content/drive/MyDrive/hanium-aml/results/checkpoints/
```

## 12. 현재 공격 파트 결론

현재 공격 파트는 수행계획서의 핵심 공격 항목 중 FGSM, Square, JSMA를 구현했고, 추가로 PGD를 강한 white-box baseline으로 구현했다.

정리:

```text
PGD ≈ multi-pixel JSMA > Square Attack > FGSM
```

- PGD: 가장 강력한 white-box iterative attack
- JSMA 변형: 프로젝트 독창성 포인트로 활용 가능
- Square: black-box attack 비교축 확보
- FGSM: 빠른 baseline

## 13. 다음 후보 작업

우선순위:

1. 공격 코드 공통 interface 정리
2. 공격별 대표 성공/실패 사례 선정
3. 방어 파트와 attack_index.csv 기반 연동 확인
4. ZOO 구현 여부 결정

ZOO는 계획서에 포함되어 있으나 query 비용과 구현 비용이 높으므로, 현재 결과를 먼저 안정화한 뒤 진행하는 것을 권장한다.
