# Colab 전체 실행 가이드 (한 번에)

이 파일의 셀 3개를 순서대로 실행하면 끝.  
마지막에 `HC160_results_all.zip` 하나가 Drive에 저장됨.

---

## 셀 1 — 환경 세팅 (1~2분)

```bash
%%bash
set -e

cd /content
# repo clone (이미 있으면 pull)
if [ ! -d "26_HC160" ]; then
  git clone https://github.com/YOUR_GITHUB_ID/26_HC160.git
fi
cd 26_HC160
git pull

pip install -q facenet-pytorch diffusers accelerate

# Drive 마운트 확인
ls /content/drive/MyDrive/hanium-aml/ 2>/dev/null || echo "[경고] Drive 마운트 필요 — 아래 셀 실행 전 Drive 마운트하세요"
```

> **Drive 마운트**: 위 셀 실행 전 Colab 좌측 패널 → 파일 → Drive 마운트 (또는 `from google.colab import drive; drive.mount('/content/drive')`)

---

## 셀 2 — Drive에서 복원 + LFW 압축 해제 (~5분)

Drive에 모든 필수 파일 있음:
- `archive.zip` → LFW 이미지
- `results/verification_facenet/verification_metrics.json` → FaceNet threshold
- `results/verification/lfw_test_pairs.csv` → test pair 목록

> ⚠️ **실행 전 Drive 마운트 필수**: 좌측 폴더 아이콘 → Drive 마운트 클릭

```python
import os, shutil, subprocess, time, json, csv
from pathlib import Path

REPO = "/content/26_HC160"
DRIVE = "/content/drive/MyDrive/hanium-aml"
os.chdir(REPO)

def sh(cmd, label=""):
    if label: print(f"\n[{time.strftime('%H:%M:%S')}] {label}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    if out: print(out[-800:])
    return r.returncode

# ── 0. Drive 마운트 확인 ───────────────────────────────────────
if not Path(DRIVE).exists():
    raise SystemExit(
        "❌ Drive가 마운트되지 않았습니다.\n"
        "좌측 패널 폴더 아이콘 → 'Drive 마운트' 클릭 후 다시 실행하세요."
    )
print("✅ Drive 마운트 확인")

# ── 1. LFW: archive.zip 압축 해제 ─────────────────────────────
lfw_dir = Path(f"{REPO}/data/raw/lfw")
lfw_count = len(list(lfw_dir.rglob("*.jpg"))) if lfw_dir.exists() else 0

if lfw_count > 1000:
    print(f"✅ LFW 이미 있음 ({lfw_count}장) — 건너뜀")
else:
    archive = Path(f"{DRIVE}/archive.zip")
    if not archive.exists():
        raise SystemExit(f"❌ {DRIVE}/archive.zip 없음 — Drive 확인 필요")
    print("📦 archive.zip 압축 해제 중... (1~2분)")
    lfw_dir.parent.mkdir(parents=True, exist_ok=True)
    sh(f"unzip -q '{archive}' -d {REPO}/data/raw/")
    # 압축 해제 후 폴더명 정규화 (lfw-deepfunneled 등 → lfw)
    for candidate in [
        "lfw-deepfunneled/lfw-deepfunneled",
        "lfw-deepfunneled",
        "lfw",
    ]:
        p = Path(f"{REPO}/data/raw/{candidate}")
        if p.exists() and p.is_dir() and p != lfw_dir:
            shutil.move(str(p), str(lfw_dir))
            try: p.parent.rmdir()
            except: pass
            break
    lfw_count = len(list(lfw_dir.rglob("*.jpg")))
    print(f"✅ LFW 압축 해제 완료: {lfw_count}장")

# ── 2. verification_metrics.json 복원 ─────────────────────────
metrics_dst = Path(f"{REPO}/outputs/verification_facenet/verification_metrics.json")
metrics_src = Path(f"{DRIVE}/results/verification_facenet/verification_metrics.json")

if metrics_dst.exists():
    m = json.loads(metrics_dst.read_text())
    print(f"✅ verification_metrics.json 이미 있음 (threshold={m['eer_threshold']:.4f})")
elif metrics_src.exists():
    metrics_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(metrics_src, metrics_dst)
    m = json.loads(metrics_dst.read_text())
    print(f"✅ verification_metrics.json 복원 (threshold={m['eer_threshold']:.4f})")
else:
    raise SystemExit(f"❌ verification_metrics.json 없음 — Drive 경로: {metrics_src}")

# ── 3. lfw_test_pairs.csv 복원 ────────────────────────────────
pairs_dst = Path(f"{REPO}/outputs/verification/lfw_test_pairs.csv")
pairs_src = Path(f"{DRIVE}/results/verification/lfw_test_pairs.csv")

if pairs_dst.exists():
    with open(pairs_dst) as f:
        n = sum(1 for _ in f) - 1
    print(f"✅ lfw_test_pairs.csv 이미 있음 ({n}쌍)")
elif pairs_src.exists():
    pairs_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pairs_src, pairs_dst)
    with open(pairs_dst) as f:
        n = sum(1 for _ in f) - 1
    print(f"✅ lfw_test_pairs.csv 복원 ({n}쌍)")
else:
    raise SystemExit(f"❌ lfw_test_pairs.csv 없음 — Drive 경로: {pairs_src}")

# ── 4. 이전 FaceNet PGD 결과 복원 ─────────────────────────────
prev_src = Path(f"{DRIVE}/results/verification_attacks_facenet")
prev_dst = Path(f"{REPO}/outputs/verification_attacks_facenet")
if prev_src.exists():
    shutil.copytree(prev_src, prev_dst, dirs_exist_ok=True)
    print(f"✅ 이전 verification_attacks_facenet 복원 완료")

# ── 5. LFW 10-class 데이터셋 준비 (DAE 학습용) ────────────────
processed_dir = Path(f"{REPO}/data/processed/lfw_identity_10/test")
if processed_dir.exists() and len(list(processed_dir.rglob("*.jpg"))) > 100:
    print("✅ LFW processed 이미 있음 — 건너뜀")
else:
    sh("""python -m src.datasets.prepare_lfw_identity_dataset \
  --raw-dir data/raw/lfw \
  --out-dir data/processed/lfw_identity_10 \
  --num-identities 10 --seed 42""", "LFW 10-class 데이터셋 준비")

# ── 최종 확인 ─────────────────────────────────────────────────
print("\n" + "="*50)
all_ok = True
for label, path in [
    (f"LFW 이미지 ({lfw_count}장)", lfw_dir),
    ("lfw_test_pairs.csv", pairs_dst),
    ("verification_metrics.json", metrics_dst),
]:
    ok = path.exists()
    print(f"  {'✅' if ok else '❌'}  {label}")
    if not ok: all_ok = False

if all_ok:
    print("\n✅ 세팅 완료. 셀 3 실행하세요.")
else:
    raise SystemExit("❌ 일부 파일 없음 — 위 에러 메시지 확인")
```

---

## 셀 3 — 전체 실험 자동 실행 (3~4시간)

이 셀 하나가 모든 실험을 순서대로 돌리고 마지막에 zip으로 묶어 Drive에 올림.

```python
import subprocess, sys, time
from pathlib import Path

REPO = "/content/26_HC160"
DRIVE = "/content/drive/MyDrive/hanium-aml"
LOG_DIR = f"{REPO}/outputs/run_log"
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

def run(label, cmd, log_file=None):
    print(f"\n{'='*60}")
    print(f"[{time.strftime('%H:%M:%S')}] 시작: {label}")
    print(f"{'='*60}")
    lf = log_file or f"{LOG_DIR}/{label.replace(' ','_')}.log"
    with open(lf, "w") as f:
        result = subprocess.run(
            cmd, shell=True, cwd=REPO,
            stdout=f, stderr=subprocess.STDOUT
        )
    # 마지막 30줄 출력
    with open(lf) as f:
        lines = f.readlines()
    print("".join(lines[-30:]))
    if result.returncode != 0:
        print(f"[경고] {label} 비정상 종료 (returncode={result.returncode}) — 계속 진행")
    else:
        print(f"[완료] {label}")
    return result.returncode

# ──────────────────────────────────────────────────────────────────
# STEP 1: PGD PNG 재실행 (eps 0.005, 0.010)
# ──────────────────────────────────────────────────────────────────
for eps in ["0.005", "0.010"]:
    run(f"PGD-PNG eps={eps}", f"""
python -m src.verification.targeted_pgd_facenet_verification \
  --pairs outputs/verification/lfw_test_pairs.csv \
  --metrics outputs/verification_facenet/verification_metrics.json \
  --pretrained vggface2 \
  --epsilon {eps} --alpha 0.001 --steps 10 --limit 100 \
  --only-initial-rejects --image-format png \
  --out-dir outputs/verification_attacks_facenet/pgd_png
""")

run("PGD-PNG summarize", """
python -m src.verification.summarize_verification_attacks \
  --metadata-root outputs/verification_attacks_facenet/pgd_png \
  --out outputs/verification_attacks_facenet/pgd_png/summary.csv
""")

# ──────────────────────────────────────────────────────────────────
# STEP 2: FGSM sweep
# ──────────────────────────────────────────────────────────────────
for eps in ["0.005", "0.010", "0.020", "0.030"]:
    run(f"FGSM eps={eps}", f"""
python -m src.verification.targeted_fgsm_facenet_verification \
  --pairs outputs/verification/lfw_test_pairs.csv \
  --metrics outputs/verification_facenet/verification_metrics.json \
  --pretrained vggface2 \
  --epsilon {eps} --limit 100 --only-initial-rejects \
  --image-format png --out-dir outputs/verification_attacks_facenet/fgsm
""")

run("FGSM summarize", """
python -m src.verification.summarize_verification_attacks \
  --metadata-root outputs/verification_attacks_facenet/fgsm \
  --out outputs/verification_attacks_facenet/fgsm/summary.csv
""")

# ──────────────────────────────────────────────────────────────────
# STEP 3: Square (블랙박스)
# ──────────────────────────────────────────────────────────────────
for eps in ["0.010", "0.020", "0.030"]:
    run(f"Square eps={eps}", f"""
python -m src.verification.targeted_square_facenet_verification \
  --pairs outputs/verification/lfw_test_pairs.csv \
  --metrics outputs/verification_facenet/verification_metrics.json \
  --pretrained vggface2 \
  --epsilon {eps} --max-queries 300 --limit 100 \
  --only-initial-rejects --image-format png \
  --out-dir outputs/verification_attacks_facenet/square
""")

run("Square summarize", """
python -m src.verification.summarize_verification_attacks \
  --metadata-root outputs/verification_attacks_facenet/square \
  --out outputs/verification_attacks_facenet/square/summary.csv
""")

# ──────────────────────────────────────────────────────────────────
# STEP 4: Adaptive PGD (vs smoothing)
# ──────────────────────────────────────────────────────────────────
for eps in ["0.005", "0.010", "0.020"]:
    run(f"Adaptive-smoothing eps={eps}", f"""
python -m src.verification.targeted_pgd_facenet_adaptive \
  --pairs outputs/verification/lfw_test_pairs.csv \
  --metrics outputs/verification_facenet/verification_metrics.json \
  --pretrained vggface2 \
  --epsilon {eps} --alpha 0.001 --steps 20 --limit 100 \
  --only-initial-rejects --defense-transform smoothing \
  --smoothing-kernel 13 --smoothing-sigma 3.0 \
  --image-format png \
  --out-dir outputs/verification_attacks_facenet/pgd_adaptive_smoothing
""")

run("Adaptive summarize", """
python -m src.verification.summarize_verification_attacks \
  --metadata-root outputs/verification_attacks_facenet/pgd_adaptive_smoothing \
  --out outputs/verification_attacks_facenet/pgd_adaptive_smoothing/summary.csv
""")

# ──────────────────────────────────────────────────────────────────
# STEP 5: PGD adv training 데이터 500쌍
# ──────────────────────────────────────────────────────────────────
for eps in ["0.005", "0.010", "0.015", "0.020"]:
    run(f"PGD-adv-training eps={eps}", f"""
python -m src.verification.targeted_pgd_facenet_verification \
  --pairs outputs/verification/lfw_test_pairs.csv \
  --metrics outputs/verification_facenet/verification_metrics.json \
  --pretrained vggface2 \
  --epsilon {eps} --alpha 0.001 --steps 10 --limit 200 \
  --only-initial-rejects --image-format png \
  --out-dir outputs/verification_attacks_facenet/pgd_adv_training
""")

# ──────────────────────────────────────────────────────────────────
# STEP 6: Handoff 패키지 3개 빌드
# ──────────────────────────────────────────────────────────────────
run("Handoff PGD-PNG", """
python -m src.verification.build_verification_attack_handoff \
  --metadata-root outputs/verification_attacks_facenet/pgd_png \
  --verification-metrics outputs/verification_facenet/verification_metrics.json \
  --attack-summary outputs/verification_attacks_facenet/pgd_png/summary.csv \
  --epsilons 0.005,0.010 --successful-only \
  --out-dir outputs/handoff/facenet_pgd_png_package \
  --zip-out outputs/handoff/facenet_pgd_png_package.zip
""")

run("Handoff FGSM", """
python -m src.verification.build_verification_attack_handoff \
  --metadata-root outputs/verification_attacks_facenet/fgsm \
  --verification-metrics outputs/verification_facenet/verification_metrics.json \
  --attack-summary outputs/verification_attacks_facenet/fgsm/summary.csv \
  --epsilons ALL --successful-only \
  --out-dir outputs/handoff/facenet_fgsm_package \
  --zip-out outputs/handoff/facenet_fgsm_package.zip
""")

run("Handoff adv-training", """
python -m src.verification.build_verification_attack_handoff \
  --metadata-root outputs/verification_attacks_facenet/pgd_adv_training \
  --verification-metrics outputs/verification_facenet/verification_metrics.json \
  --attack-summary outputs/verification_attacks_facenet/pgd_png/summary.csv \
  --epsilons ALL --successful-only \
  --out-dir outputs/handoff/facenet_adv_training_package \
  --zip-out outputs/handoff/facenet_adv_training_package.zip
""")

# ──────────────────────────────────────────────────────────────────
# STEP 7: DAE 학습 (LFW 데이터 있을 때만)
# ──────────────────────────────────────────────────────────────────
lfw_images = list(Path(f"{REPO}/data/raw/lfw").rglob("*.jpg"))
if len(lfw_images) > 100:
    run("DAE train", """
python -m src.training.train_face_dae \
  --data-dir data/raw/lfw \
  --adv-dir outputs/handoff/facenet_pgd_png_package \
  --out-dir checkpoints/face_dae \
  --epochs 30 --batch-size 32 --base-ch 32
""")
    DAE_AVAILABLE = Path(f"{REPO}/checkpoints/face_dae/best.pt").exists()
else:
    print("[SKIP] DAE 학습: LFW 이미지 없음. DAE 관련 단계 건너뜀.")
    DAE_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────
# STEP 8: DAE 방어 평가
# ──────────────────────────────────────────────────────────────────
if DAE_AVAILABLE:
    run("DAE defense eval", """
python -m src.defenses.verification_defense_dae \
  --handoff-index outputs/handoff/facenet_pgd_png_package/attack_handoff_index.csv \
  --handoff-root outputs/handoff/facenet_pgd_png_package \
  --checkpoint checkpoints/face_dae/best.pt \
  --pretrained vggface2 \
  --out-dir outputs/defenses/verification/dae
""")
else:
    print("[SKIP] DAE 방어 평가: DAE 체크포인트 없음")

# ──────────────────────────────────────────────────────────────────
# STEP 9: DiffPure fallback 방어
# ──────────────────────────────────────────────────────────────────
run("DiffPure fallback", """
python -m src.defenses.verification_defense_diffpure \
  --handoff-index outputs/handoff/facenet_pgd_png_package/attack_handoff_index.csv \
  --handoff-root outputs/handoff/facenet_pgd_png_package \
  --use-fallback --fallback-noise-sigma 0.03 --fallback-denoise-sigma 1.5 \
  --pretrained vggface2 \
  --out-dir outputs/defenses/verification/diffpure_fallback
""")

# DiffPure 실제 (시간 있으면)
run("DiffPure real t=0.10", """
python -m src.defenses.verification_defense_diffpure \
  --handoff-index outputs/handoff/facenet_pgd_png_package/attack_handoff_index.csv \
  --handoff-root outputs/handoff/facenet_pgd_png_package \
  --model-id google/ddpm-celebahq-256 --t-diff 0.10 \
  --pretrained vggface2 \
  --out-dir outputs/defenses/verification/diffpure_t0p10
""")

# ──────────────────────────────────────────────────────────────────
# STEP 10: Adaptive vs DAE
# ──────────────────────────────────────────────────────────────────
if DAE_AVAILABLE:
    for eps in ["0.010", "0.020"]:
        run(f"Adaptive-DAE eps={eps}", f"""
python -m src.verification.targeted_pgd_facenet_adaptive \
  --pairs outputs/verification/lfw_test_pairs.csv \
  --metrics outputs/verification_facenet/verification_metrics.json \
  --pretrained vggface2 \
  --epsilon {eps} --alpha 0.001 --steps 20 --limit 100 \
  --only-initial-rejects --defense-transform dae \
  --dae-checkpoint checkpoints/face_dae/best.pt \
  --image-format png \
  --out-dir outputs/verification_attacks_facenet/pgd_adaptive_dae
""")
else:
    print("[SKIP] Adaptive vs DAE: DAE 체크포인트 없음")

# ──────────────────────────────────────────────────────────────────
# STEP 11: FRR 평가
# ──────────────────────────────────────────────────────────────────
frr_defenses = "smoothing jpeg bitdepth diffpure_fallback"
frr_dae_flag = ""
if DAE_AVAILABLE:
    frr_defenses += " dae"
    frr_dae_flag = "--dae-checkpoint checkpoints/face_dae/best.pt"

run("FRR evaluation", f"""
python -m src.verification.evaluate_defense_frr \
  --pairs outputs/verification/lfw_test_pairs.csv \
  --metrics outputs/verification_facenet/verification_metrics.json \
  --pretrained vggface2 \
  --defenses {frr_defenses} \
  {frr_dae_flag} \
  --limit 200 \
  --out-dir outputs/defenses/verification/frr_evaluation
""")

# ──────────────────────────────────────────────────────────────────
# STEP 12: 전체 방어 비교 요약
# ──────────────────────────────────────────────────────────────────
run("Defense summary", """
python -m src.defenses.summarize_verification_defenses \
  --defense-root outputs/defenses/verification \
  --out outputs/defenses/verification/defense_comparison_summary.csv
""")

# ──────────────────────────────────────────────────────────────────
# STEP 13: 비교 그래프
# ──────────────────────────────────────────────────────────────────
run("Plot graphs", """
pip install -q matplotlib pandas
python -m src.reports.plot_verification_attack_defense \
  --attack-summaries \
    outputs/verification_attacks_facenet/pgd_png/summary.csv \
    outputs/verification_attacks_facenet/fgsm/summary.csv \
    outputs/verification_attacks_facenet/square/summary.csv \
    outputs/verification_attacks_facenet/pgd_adaptive_smoothing/summary.csv \
  --out-dir outputs/figures/verification_attack_defense
""")

# ──────────────────────────────────────────────────────────────────
# STEP 14: 결과 전체를 zip 하나로 묶기
# ──────────────────────────────────────────────────────────────────
import subprocess, zipfile, os
from pathlib import Path

print("\n" + "="*60)
print("결과 묶기 시작")
zip_path = "/content/HC160_results_all.zip"

include_dirs = [
    "outputs/verification_attacks_facenet",
    "outputs/defenses/verification",
    "outputs/figures",
    "outputs/handoff/facenet_pgd_png_package/attack_handoff_index.csv",
    "outputs/handoff/facenet_fgsm_package/attack_handoff_index.csv",
    "checkpoints/face_dae/training_history.csv",
    "outputs/run_log",
]

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for target in include_dirs:
        p = Path(f"{REPO}/{target}")
        if p.is_file():
            zf.write(p, arcname=target)
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix in {".csv", ".json", ".md", ".txt", ".png", ".log"}:
                    zf.write(f, arcname=str(f.relative_to(REPO)))

size_mb = Path(zip_path).stat().st_size / 1024 / 1024
print(f"Zip 생성 완료: {zip_path} ({size_mb:.1f} MB)")

# Drive에도 저장
import shutil
drive_zip = f"{DRIVE}/results/HC160_results_all.zip"
Path(drive_zip).parent.mkdir(parents=True, exist_ok=True)
shutil.copy(zip_path, drive_zip)
print(f"Drive 저장 완료: {drive_zip}")
print("\n=== 모든 실험 완료 ===")
print("HC160_results_all.zip 을 다운로드해서 저한테 넘겨주세요!")
```

---

## 다 끝나면

1. Colab 좌측 패널 → 파일 → `/content/HC160_results_all.zip` 우클릭 → 다운로드
2. (또는) Drive에서 `hanium-aml/results/HC160_results_all.zip` 다운로드
3. 그 파일 하나를 저한테 주면 전체 분석해드림

---

## 주의사항

- **세션 유지**: 3~4시간이라 Colab Pro 없으면 끊길 수 있음  
  → 실행 전 `런타임 → GPU 변경 → T4` 확인  
  → 끊기면 셀 3만 다시 실행 (이미 완료된 단계는 빠르게 지나감)

- **LFW 데이터**: `data/raw/lfw/` 에 있어야 함  
  없으면 셀 2에서 경고 뜨고 DAE 학습이 실패함  
  → 미리 Drive `hanium-aml/data/raw/lfw/` 에 올려두거나  
  → 셀 2 실행 후 터미널에서 `kaggle datasets download -d jessicali9530/lfw-dataset` 로 받기

- **DiffPure 실제**: 모델 다운로드 ~1.5GB, 실행 ~1시간  
  시간 부족하면 STEP 9의 `DiffPure real` 부분 주석 처리해도 됨 (fallback으로 대체)
