"""
Ensemble Voting Defense

ROI-first, Gaussian Smoothing, Randomized Smoothing 3종의 방어 결과를
voting하여 과반수(2/3 이상) reject이면 최종 거부한다.

단일 방어 우회 공격에 대한 robustness 향상:
  - 한 방어 기법을 우회해도 나머지 2종이 차단하면 reject
  - 3종 모두 accept해야만 최종 인증 통과

입력:
  - roi_first 결과 CSV
  - smoothing 결과 CSV (verification_defense_smoothing.py 출력)
  - randomized_smoothing 결과 CSV

실행:
    python -m src.verification.defenses.verification_defense_ensemble \\
        --roi-csv      outputs/verification_defense/roi_first/verification_defense_roi_first.csv \\
        --smoothing-csv outputs/verification_defense/smoothing/verification_defense_smoothing.csv \\
        --rs-csv       outputs/verification_defense/randomized_smoothing/verification_defense_randomized_smoothing.csv \\
        --out-dir      outputs/verification_defense
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


# ── Voting 로직 ───────────────────────────────────────────────────────────────

def majority_vote(votes: dict[str, bool | None]) -> bool:
    """
    과반수 accept이면 True, 과반수 reject이면 False 반환.
    유효 투표 수 기준 (None 제외).

    Args:
        votes: {"roi": True/False/None, "smoothing": ..., "randomized": ...}

    Returns:
        True  = 최종 accept (공격 성공)
        False = 최종 reject (공격 차단)
    """
    valid = {k: v for k, v in votes.items() if v is not None}
    if not valid:
        return False  # 모든 방어 실패 → 기본 reject
    accept_count = sum(1 for v in valid.values() if v)
    return accept_count > len(valid) / 2


# ── CSV 로드 ──────────────────────────────────────────────────────────────────

def load_csv_as_dict(path: str, key_col: str = "sample_id") -> dict[str, dict]:
    """CSV를 key_col 기준 dict로 로드한다."""
    result = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            result[row[key_col]] = row
    return result


def parse_bool(val) -> bool | None:
    """CSV 문자열 bool → Python bool. 파싱 실패 시 None."""
    if val is None or str(val).upper() in ("", "ERROR", "NONE"):
        return None
    return str(val).strip().lower() in ("true", "1", "yes")


# ── 앙상블 평가 ───────────────────────────────────────────────────────────────

def run_ensemble_defense(
    roi_csv: str,
    smoothing_csv: str,
    rs_csv: str,
    out_dir: str,
) -> str:
    roi_data   = load_csv_as_dict(roi_csv)
    smth_data  = load_csv_as_dict(smoothing_csv)
    rs_data    = load_csv_as_dict(rs_csv)

    all_ids = sorted(set(roi_data) | set(smth_data) | set(rs_data))
    out_rows = []

    for sid in all_ids:
        roi_row  = roi_data.get(sid, {})
        smth_row = smth_data.get(sid, {})
        rs_row   = rs_data.get(sid, {})

        # 각 방어 후 accept 여부
        roi_accept  = parse_bool(roi_row.get("accepted_after_defense"))
        smth_accept = parse_bool(smth_row.get("accepted_after_defense"))
        rs_accept   = parse_bool(rs_row.get("accepted_after_defense"))

        votes = {"roi": roi_accept, "smoothing": smth_accept, "randomized": rs_accept}
        ensemble_accepted = majority_vote(votes)

        # 방어 전 공격 성공 여부 (어느 CSV든 동일해야 하므로 roi 기준)
        accepted_after_attack = parse_bool(roi_row.get("accepted_after_attack"))
        defense_success = (
            accepted_after_attack is True and
            not ensemble_accepted
        )

        out_rows.append({
            "sample_id":               sid,
            "defense":                 "ensemble_voting",
            "roi_accepted":            roi_accept,
            "smoothing_accepted":      smth_accept,
            "randomized_accepted":     rs_accept,
            "ensemble_votes":          json.dumps({"roi": roi_accept, "smoothing": smth_accept, "randomized": rs_accept}),
            "ensemble_accepted":       ensemble_accepted,
            "accepted_after_attack":   accepted_after_attack,
            "attack_success_after_defense": ensemble_accepted,
            "defense_success":         defense_success,
        })

    # 저장
    out_path = Path(out_dir) / "ensemble" / "verification_defense_ensemble.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(out_rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    # 집계
    total   = len(out_rows)
    n_atk   = sum(1 for r in out_rows if r["accepted_after_attack"] is True)
    n_def   = sum(1 for r in out_rows if r["defense_success"] is True)
    n_still = sum(1 for r in out_rows if r["attack_success_after_defense"] is True)

    print(f"\n[ensemble_voting] 완료 — 전체 {total}개")
    print(f"  공격 성공 (방어 전):   {n_atk}개 ({n_atk/total*100:.1f}%)")
    print(f"  방어 성공:             {n_def}개 ({n_def/total*100:.1f}%)")
    print(f"  방어 후 공격 성공:     {n_still}개 ({n_still/total*100:.1f}%)")
    print(f"  저장: {out_path}")

    # 투표 패턴 분석
    from collections import Counter
    vote_patterns = Counter(
        (r["roi_accepted"], r["smoothing_accepted"], r["randomized_accepted"])
        for r in out_rows
    )
    print("\n  투표 패턴 (roi, smoothing, randomized):")
    for pattern, count in sorted(vote_patterns.items(), key=lambda x: -x[1]):
        print(f"    {pattern}: {count}개")

    return str(out_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble Voting Defense")
    parser.add_argument("--roi-csv",       required=True, help="roi_first 결과 CSV")
    parser.add_argument("--smoothing-csv", required=True, help="smoothing 결과 CSV")
    parser.add_argument("--rs-csv",        required=True, help="randomized_smoothing 결과 CSV")
    parser.add_argument("--out-dir",       default="outputs/verification_defense", help="결과 저장 디렉터리")
    args = parser.parse_args()

    run_ensemble_defense(
        roi_csv=args.roi_csv,
        smoothing_csv=args.smoothing_csv,
        rs_csv=args.rs_csv,
        out_dir=args.out_dir,
    )
