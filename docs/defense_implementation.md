# 방어 기법 구현 정리

## 목적

`docs/defense_integration_format.md` 에 정의된 연동 포맷을 기반으로 방어 기법 3종을 구현한다.

입력은 공격 파트에서 생성한 아래 파일 하나만 사용한다.

```text
outputs/attacks/attack_index.csv
```

## 샘플 필터

```python
attack_index = pd.read_csv("outputs/attacks/attack_index.csv")
attack_inputs = attack_index[
    (attack_index["clean_correct"] == True) &
    (attack_index["success_on_clean"] == True)
]
```

## 방어 기법

| 방어 | 스크립트 | 파라미터 | 핵심 함수 |
|------|----------|----------|-----------|
| JPEG 압축 | `src/defenses/defense_jpeg.py` | `--quality 75` | `jpeg_recompress(image, quality)` |
| Gaussian Blur | `src/defenses/defense_smoothing.py` | `--radius 3` | `gaussian_smooth(image, radius)` |
| Bit-depth 축소 | `src/defenses/defense_bitdepth.py` | `--bits 4` | `reduce_bit_depth(image, bits)` |

### JPEG 압축

```python
def jpeg_recompress(image: Image.Image, quality: int) -> Image.Image:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=False)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")
```

### Gaussian Blur

```python
def gaussian_smooth(image: Image.Image, radius: float) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=radius)).convert("RGB")
```

### Bit-depth 축소

```python
def reduce_bit_depth(image: Image.Image, bits: int) -> Image.Image:
    tensor = transforms.ToTensor()(image.convert("RGB"))
    levels = 2 ** bits
    quantized = torch.round(tensor * (levels - 1)) / (levels - 1)
    return transforms.ToPILImage()(quantized.clamp(0, 1))
```

## 결과 저장 경로

```text
outputs/defenses/jpeg/jpeg_results_q75.csv
outputs/defenses/smoothing/smoothing_results_r3p0.csv
outputs/defenses/bitdepth/bitdepth_results_4bit.csv
```

결과 컬럼은 `docs/defense_integration_format.md` 의 권장 포맷을 따른다.  
`sample_id` 를 유지하여 공격 결과와 join 가능하다.

## 방어 성능 지표

```text
Defense Success Rate = attack_success_before=True 중 attack_success_after=False 비율
Recovery Rate        = attack_success_before=True 중 recovered=True 비율
Target Conf Drop     = target_conf_before - target_conf_after 평균
Defense Time         = defense_time_sec 평균
```

집계는 `src/defenses/summarize_defense.py` 로 수행한다.

```bash
python -m src.defenses.summarize_defense
```

## 실행

```bash
python -m src.defenses.defense_jpeg      --quality 75
python -m src.defenses.defense_smoothing --radius 3
python -m src.defenses.defense_bitdepth  --bits 4
python -m src.defenses.summarize_defense
```

Colab에서는 `notebooks/colab_defense_pipeline.ipynb` 를 사용한다.
