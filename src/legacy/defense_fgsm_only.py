"""
FGSM 공격 이미지에 대한 방어 기법 평가
- JPEG Compression
- Spatial Smoothing (Gaussian Blur)
- Bit-depth Reduction

사용법:
    python defense_fgsm_only.py
    python defense_fgsm_only.py --limit 20   # 테스트용 20장만
"""

import io
import time
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import pandas as pd
import argparse
from pathlib import Path
from PIL import Image


# ============================================================
# 경로 설정
# ============================================================
REPO = Path(__file__).resolve().parent.parent
CKPT_PATH = REPO / "checkpoints" / "face_resnet50_lfw10" / "best.pt"
INDEX_PATH = REPO / "outputs" / "attacks" / "attack_index.csv"
OUT_DIR = REPO / "outputs" / "defenses"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORM = T.Compose([T.Resize((224, 224)), T.ToTensor()])


# ============================================================
# 모델 로드
# ============================================================
def load_model():
    assert CKPT_PATH.exists(), f"체크포인트 없음: {CKPT_PATH}"
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(2048, 10)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()

    print(f"[INFO] 모델 로드 완료 (device={DEVICE})")
    print(f"[INFO] 클래스: {ckpt['classes']}")
    return model, ckpt["classes"]


# ============================================================
# 추론
# ============================================================
def predict(model, img_tensor):
    with torch.no_grad():
        x = img_tensor.unsqueeze(0).to(DEVICE)
        probs = torch.softmax(model(x), dim=1)
        pred = probs.argmax(1).item()
    return pred, probs[0].cpu()


# ============================================================
# 방어 기법 3가지
# ============================================================
def defense_jpeg(img_tensor, quality=75):
    """JPEG 압축 - 고주파 노이즈(perturbation) 제거"""
    img = T.ToPILImage()(img_tensor)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return T.ToTensor()(Image.open(buf).convert("RGB").resize((224, 224)))


def defense_smoothing(img_tensor, kernel_size=3):
    """가우시안 블러 - 픽셀 단위 노이즈 평활화"""
    return T.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0))(img_tensor)


def defense_bitdepth(img_tensor, bits=4):
    """비트 깊이 축소 - 미세한 perturbation 양자화로 제거"""
    levels = 2 ** bits
    return torch.round(img_tensor * (levels - 1)) / (levels - 1)


# ============================================================
# 메인 실행
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="테스트용 샘플 수 제한")
    args = parser.parse_args()

    # 모델 로드
    model, classes = load_model()

    # attack_index.csv 로드 -> FGSM + 공격 성공 샘플만 필터
    assert INDEX_PATH.exists(), f"attack_index.csv 없음: {INDEX_PATH}"
    df = pd.read_csv(INDEX_PATH)

    fgsm_df = df[
        (df["attack_family"] == "fgsm") &
        (df["clean_correct"] == True) &
        (df["success_on_clean"] == True)
    ].copy()

    if args.limit:
        fgsm_df = fgsm_df.head(args.limit)

    print(f"\n[INFO] FGSM 공격 성공 샘플: {len(fgsm_df)}장")
    print(f"[INFO] 방어 기법 3가지 × {len(fgsm_df)}장 = 총 {3 * len(fgsm_df)}회 평가\n")

    # 방어 기법 목록
    defenses = {
        "jpeg_q75": lambda img: defense_jpeg(img, quality=75),
        "smoothing_k3": lambda img: defense_smoothing(img, kernel_size=3),
        "bitdepth_4bit": lambda img: defense_bitdepth(img, bits=4),
    }

    all_results = []

    for def_name, def_fn in defenses.items():
        print(f"--- {def_name} ---")
        success_count = 0
        recover_count = 0
        total = 0

        for i, (_, row) in enumerate(fgsm_df.iterrows()):
            adv_path = REPO / row["adv_file"]
            if not adv_path.exists():
                continue

            # 적대적 이미지 로드
            adv_img = TRANSFORM(Image.open(adv_path).convert("RGB"))

            # 방어 전 예측 (공격된 상태)
            pred_before, probs_before = predict(model, adv_img)

            # 방어 적용
            t0 = time.time()
            defended_img = def_fn(adv_img)
            dt = time.time() - t0

            # 방어 후 예측
            pred_after, probs_after = predict(model, defended_img)

            true_label = int(row["true_label"])
            target_label = int(row["target_label"])
            still_attacked = (pred_after == target_label)
            recovered = (pred_after == true_label)

            if not still_attacked:
                success_count += 1
            if recovered:
                recover_count += 1
            total += 1

            all_results.append({
                "sample_id": row["sample_id"],
                "attack_family": "fgsm",
                "defense": def_name,
                "true_label": true_label,
                "true_name": row["true_name"],
                "target_label": target_label,
                "target_name": row["target_name"],
                "pred_before_defense": pred_before,
                "pred_after_defense": pred_after,
                "attack_success_after_defense": still_attacked,
                "recovered": recovered,
                "target_conf_before": float(probs_before[target_label]),
                "target_conf_after": float(probs_after[target_label]),
                "defense_time_sec": round(dt, 6),
            })

        print(f"  Defense Success Rate: {success_count}/{total} ({100*success_count/total:.1f}%)")
        print(f"  Recovery Rate:        {recover_count}/{total} ({100*recover_count/total:.1f}%)")
        print()

    # 결과 저장
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(all_results)
    out_path = OUT_DIR / "fgsm_defense_results.csv"
    result_df.to_csv(out_path, index=False)
    print(f"[DONE] 결과 저장: {out_path}")
    print(f"[DONE] 총 {len(result_df)}행 (방어 3종 × FGSM 샘플 {total}장)")


if __name__ == "__main__":
    main()
