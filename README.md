# 26_HC160

Hanium AML project for targeted adversarial attacks and defense evaluation on face identity recognition.

## Attack pipeline

Attack-side Colab notebook:

```text
notebooks/colab_targeted_attack_pipeline.ipynb
```

Main attack scripts:

- `python -m src.attacks.targeted_fgsm_face`
- `python -m src.attacks.targeted_pgd_face`
- `python -m src.attacks.targeted_square_face`
- `python -m src.attacks.targeted_jsma_face`
- `python -m src.attacks.targeted_zoo_face`

Supporting modules:

- `src/common/`: shared model/device/attack utilities
- `src/datasets/`: LFW dataset preparation
- `src/training/`: ResNet-50 face identity training
- `src/reports/`: attack summary, index, representative panel, and plot generation

Attack handoff documents:

- `docs/attack_results_final_2026-05-02.md`
- `docs/attack_index_handoff.md`
- `docs/team_update_attack.md`

Defense modules should use `attack_index.csv` and read adversarial images from the `adv_file` column. Keep `sample_id` in every defense result row so attack and defense results can be joined later.

Final attack result files are not committed to Git because they contain generated images and model outputs. They are shared separately through Google Drive or zip handoff files.

Typical attack result workflow:

```bash
python -m src.datasets.prepare_lfw_identity_dataset
python -m src.training.train_face_resnet50 --epochs 12 --batch-size 64 --num-workers 2
python -m src.attacks.targeted_fgsm_face --epsilon 0.005 --limit 300
python -m src.attacks.targeted_pgd_face --epsilon 0.03 --alpha 0.003 --steps 10 --limit 300
python -m src.attacks.targeted_square_face --epsilon 0.05 --max-queries 300 --limit 300
python -m src.attacks.targeted_jsma_face --theta 0.05 --steps 20 --pixels-per-step 200 --limit 300
python -m src.attacks.targeted_zoo_face --epsilon 0.05 --max-queries 2000 --coords-per-iter 128 --learning-rate 0.02 --limit 300
python -m src.reports.summarize_face_attack
python -m src.reports.build_attack_index
```

## Defense pipeline

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

General defense utilities added by the attack side are under `src/defenses/`; the original FGSM-only defense script remains at `src/defense_fgsm_only.py` so the existing defense notebook keeps working.
