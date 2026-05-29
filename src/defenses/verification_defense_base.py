"""
Verification 방어 평가 공통 로직

각 verification_defense_*.py 에서 import해서 사용한다.
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Callable

from PIL import Image
from tqdm import tqdm

from src.defenses.facenet_embed import get_embedding, cosine_similarity


THRESHOLD = 0.47966246581077576  # EER 기준 (verification_metrics.json)


def run_verification_defense(
    defense_name: str,
    defense_fn: Callable,
    defense_params: dict,
    index_csv: str,
    pkg_root: str,
    out_dir: str,
    img_ext: str = "png",
    device=None,
) -> str:
    """
    단일 방어 기법을 전체 샘플에 적용하고 결과 CSV를 저장한다.

    Args:
        defense_name:   방어 이름 (출력 폴더명, CSV defense 컬럼값)
        defense_fn:     이미지 변환 함수 (PIL Image → PIL Image)
        defense_params: 방어 파라미터 dict (CSV defense_params 컬럼에 기록)
        index_csv:      attack_handoff_index.csv 경로
        pkg_root:       samples/ 폴더가 있는 패키지 루트
        out_dir:        결과 저장 루트 디렉터리
        img_ext:        방어 이미지 저장 확장자 (기본 png)
        device:         torch device (None이면 자동)

    Returns:
        저장된 결과 CSV 경로
    """
    defended_img_dir = Path(out_dir) / defense_name / "images"
    defended_img_dir.mkdir(parents=True, exist_ok=True)
    result_csv_path = Path(out_dir) / defense_name / f"verification_defense_{defense_name}.csv"

    rows = list(csv.DictReader(open(index_csv)))
    output_rows = []

    for row in tqdm(rows, desc=f"[{defense_name}]", unit="샘플"):
        sample_id = row["sample_id"]
        adv_path  = os.path.join(pkg_root, row["adv_file"])
        enr_path  = os.path.join(pkg_root, row["target_enroll_file"])
        threshold = float(row["threshold"])

        # ── similarity_after_attack 재계산 (JPEG 파일 기준) ──────────────────
        try:
            emb_adv = get_embedding(Image.open(adv_path).convert("RGB"), device)
            emb_enr = get_embedding(Image.open(enr_path).convert("RGB"), device)
            sim_after_attack = cosine_similarity(emb_adv, emb_enr)
        except Exception as e:
            output_rows.append(_error_row(row, defense_name, defense_params, str(e)))
            continue

        accepted_after_attack = sim_after_attack >= threshold

        # ── 방어 적용 ─────────────────────────────────────────────────────────
        t_start = time.time()
        try:
            adv_img      = Image.open(adv_path).convert("RGB")
            defended_img = defense_fn(adv_img)
        except Exception as e:
            output_rows.append(_error_row(row, defense_name, defense_params, str(e)))
            continue
        defense_time = time.time() - t_start

        # 방어 이미지 저장
        defended_path = defended_img_dir / f"{sample_id}_{defense_name}.{img_ext}"
        defended_img.save(str(defended_path))

        # ── similarity_after_defense 계산 ─────────────────────────────────────
        emb_defended = get_embedding(defended_img, device)
        sim_after_defense = cosine_similarity(emb_defended, emb_enr)

        # ── 판정 ──────────────────────────────────────────────────────────────
        accepted_after_defense        = sim_after_defense >= threshold
        attack_success_after_defense  = accepted_after_defense
        defense_success               = accepted_after_attack and not accepted_after_defense

        output_rows.append({
            "sample_id":                    sample_id,
            "defense":                      defense_name,
            "defense_params":               json.dumps(defense_params),
            "adv_file":                     row["adv_file"],
            "defended_file":                str(defended_path),
            "target_enroll_file":           row["target_enroll_file"],
            "threshold":                    threshold,
            "similarity_after_attack":      round(sim_after_attack, 8),
            "similarity_after_defense":     round(sim_after_defense, 8),
            "accepted_after_attack":        accepted_after_attack,
            "accepted_after_defense":       accepted_after_defense,
            "attack_success_after_defense": attack_success_after_defense,
            "defense_success":              defense_success,
            "defense_time_sec":             round(defense_time, 4),
        })

    # CSV 저장
    fieldnames = [
        "sample_id", "defense", "defense_params",
        "adv_file", "defended_file", "target_enroll_file", "threshold",
        "similarity_after_attack", "similarity_after_defense",
        "accepted_after_attack", "accepted_after_defense",
        "attack_success_after_defense", "defense_success",
        "defense_time_sec",
    ]
    with open(result_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    # 집계 출력
    total      = len(output_rows)
    n_atk      = sum(1 for r in output_rows if r.get("accepted_after_attack") is True)
    n_def_succ = sum(1 for r in output_rows if r.get("defense_success") is True)
    print(f"\n[{defense_name}] 완료 — 전체 {total}개")
    print(f"  accepted_after_attack (JPEG 재계산): {n_atk}/{total} ({n_atk/total*100:.1f}%)")
    print(f"  defense_success:                    {n_def_succ}/{total} ({n_def_succ/total*100:.1f}%)")
    print(f"  결과 저장: {result_csv_path}\n")

    return str(result_csv_path)


def _error_row(row: dict, defense: str, params: dict, error: str) -> dict:
    return {
        "sample_id":                    row["sample_id"],
        "defense":                      defense,
        "defense_params":               json.dumps(params),
        "adv_file":                     row.get("adv_file", ""),
        "defended_file":                "ERROR",
        "target_enroll_file":           row.get("target_enroll_file", ""),
        "threshold":                    row.get("threshold", ""),
        "similarity_after_attack":      "ERROR",
        "similarity_after_defense":     "ERROR",
        "accepted_after_attack":        "ERROR",
        "accepted_after_defense":       "ERROR",
        "attack_success_after_defense": "ERROR",
        "defense_success":              "ERROR",
        "defense_time_sec":             "ERROR",
        "error":                        error,
    }
