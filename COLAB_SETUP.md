# Colab Pro setup

Local zip confirmed:

```text
/Users/ehgus/projects/hanium-aml/archive.zip
```

Upload that file to Google Drive here:

```text
MyDrive/hanium-aml/archive.zip
```

In Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then clone the repo and unzip the dataset into Colab local storage:

```bash
git clone https://github.com/Yaho03/26_HC160.git /content/26_HC160
cd /content/26_HC160
mkdir -p data/raw/lfw_zip data/raw
unzip -q /content/drive/MyDrive/hanium-aml/archive.zip -d data/raw/lfw_zip
ln -s /content/26_HC160/data/raw/lfw_zip/lfw-deepfunneled/lfw-deepfunneled data/raw/lfw
find data/raw/lfw -type f -name "*.jpg" | wc -l
```

Expected image count:

```text
13233
```

Run the project steps:

```bash
python -m src.datasets.prepare_lfw_identity_dataset
python -m src.training.train_face_resnet50 --epochs 12 --batch-size 64 --num-workers 2
python -m src.attacks.targeted_fgsm_face --epsilon 0.03 --limit 100
```

FGSM epsilon sweep:

```bash
for eps in 0.005 0.010 0.030 0.050; do
  python -m src.attacks.targeted_fgsm_face --epsilon "$eps" --limit 100
done
```
