"""
attack_handoff_jpeg_index.csv 생성

공격팀이 전달한 attack_handoff_index.csv 에서
JPEG 파일 기준으로 재계산이 필요한 컬럼 2개를 업데이트한다.

변경 컬럼:
  similarity_after_attack  ← tensor 기준 → JPEG 파일 재로드 기준
  accepted_after_attack    ← 위 값 기준 재판정

나머지 컬럼은 원본 그대로 유지한다.

실행:
    python -m src.verification.defenses.build_jpeg_index \\
        --index <attack_handoff_index.csv> \\
        --pkg-root <패키지 루트> \\
        --out <저장 경로/attack_handoff_jpeg_index.csv>
"""

from __future__ import annotations

import argparse
import csv
import os

from PIL import Image
from tqdm import tqdm

from src.verification.defenses.facenet_embed import get_embedding, cosine_similarity


def build_jpeg_index(
    index_csv: str,
    pkg_root: str,
    out_path: str,
    device=None,
) -> None:
    rows = list(csv.DictReader(open(index_csv)))
    fieldnames = list(rows[0].keys())

    output_rows = []

    for row in tqdm(rows, desc="similarity 재계산", unit="샘플"):
        adv_path = os.path.join(pkg_root, row["adv_file"])
        enr_path = os.path.join(pkg_root, row["target_enroll_file"])
        threshold = float(row["threshold"])

        try:
            emb_adv = get_embedding(Image.open(adv_path).convert("RGB"), device)
            emb_enr = get_embedding(Image.open(enr_path).convert("RGB"), device)
            sim = cosine_similarity(emb_adv, emb_enr)
            accepted = sim >= threshold
        except Exception as e:
            print(f"[오류] {row['sample_id']}: {e}")
            sim = None
            accepted = None

        new_row = dict(row)
        new_row["similarity_after_attack"] = round(sim, 8) if sim is not None else "ERROR"
        new_row["accepted_after_attack"]   = accepted if accepted is not None else "ERROR"
        output_rows.append(new_row)

    # 결과 저장
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    # 집계 출력
    total    = len(output_rows)
    n_accept = sum(1 for r in output_rows if str(r["accepted_after_attack"]) == "True")
    n_reject = total - n_accept
    print(f"\n완료 — 전체 {total}개")
    print(f"  accepted_after_attack (JPEG 기준): {n_accept}개 ({n_accept/total*100:.1f}%)")
    print(f"  rejected (공격 실패):              {n_reject}개 ({n_reject/total*100:.1f}%)")
    print(f"  저장: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="attack_handoff_jpeg_index.csv 생성")
    parser.add_argument("--index",    required=True, help="원본 attack_handoff_index.csv 경로")
    parser.add_argument("--pkg-root", required=True, help="패키지 루트 (samples/ 상위)")
    parser.add_argument("--out",      default="outputs/verification_defense/attack_handoff_jpeg_index.csv",
                        help="저장 경로")
    args = parser.parse_args()

    build_jpeg_index(
        index_csv=args.index,
        pkg_root=args.pkg_root,
        out_path=args.out,
    )
