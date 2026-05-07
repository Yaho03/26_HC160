# hanium-aml-defense

Adversarial defense evaluation for the Hanium AML project.

Attack repo: [hanium-aml](https://github.com/no-carve-only-pizza/hanium-aml)

## Quick start

Open `notebooks/colab_defense_pipeline.ipynb` in Colab and run all cells.

### Google Drive에 올려둘 파일

```
내 드라이브/hanium-aml-defense/hanium_attack_outputs.zip
내 드라이브/hanium-aml-defense/best.pt
```

## 현재 구현된 방어 기법

- JPEG Compression (quality=75, 50)
- Spatial Smoothing (kernel=3, 5)
- Bit-depth Reduction (bits=4, 3)

## 실행

```bash
cd src
python defense_fgsm_only.py
```
