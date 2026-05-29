"""
Verification 방어 결과 집계

outputs/verification_defense/ 아래 3종 결과 CSV를 읽어
공격별 · 방어별 집계 테이블(verification_defense_summary.csv)을 생성한다.

실행:
    python -m src.defenses.verification_summarize \\
        [--results-dir outputs/verification_defense] \\
        [--index <attack_handoff_index.csv>]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFENSES = ["jpeg", "smoothing", "bitdepth"]

RESULT_FILES = {
    d: f"verification_defense_{d}.csv" for d in DEFENSES
}


def load_all_results(results_dir: str) -> list[dict]:
    """방어 3종 결과 CSV를 전부 읽어 하나의 리스트로 합친다."""
    all_rows = []
    for defense, fname in RESULT_FILES.items():
        path = Path(results_dir) / defense / fname
        if not path.exists():
            print(f"[경고] 파일 없음: {path}")
            continue
        rows = list(csv.DictReader(open(path)))
        all_rows.extend(rows)
    return all_rows


def compute_summary(rows: list[dict], index_csv: str | None = None) -> list[dict]:
    """
    공격 epsilon × 방어 조합별로 집계한다.

    집계 항목:
      - samples          : 전체 샘플 수
      - n_attack_success : 방어 전 공격 성공 수 (JPEG 재계산 기준)
      - attack_success_rate : 방어 전 공격 성공률
      - n_defense_success : defense_success 수
      - defense_success_rate : 방어 성공률
      - n_still_attack   : 방어 후에도 공격 성공 수
      - still_attack_rate : 방어 후 공격 성공률 (= ASR after defense)
      - avg_sim_after_attack   : 방어 전 similarity 평균
      - avg_sim_after_defense  : 방어 후 similarity 평균
      - avg_sim_drop     : similarity 평균 감소량
      - avg_defense_time_sec
    """
    # epsilon 정보 추가
    eps_map: dict[str, str] = {}
    if index_csv:
        for r in csv.DictReader(open(index_csv)):
            eps_map[r["sample_id"]] = r["epsilon"]

    summary_rows = []

    defenses = sorted(set(r["defense"] for r in rows))
    epsilons  = sorted(set(eps_map.get(r["sample_id"], "all") for r in rows))
    groups    = ["all"] + epsilons

    for defense in defenses:
        def_rows = [r for r in rows if r["defense"] == defense]

        for eps in groups:
            if eps == "all":
                subset = def_rows
            else:
                subset = [r for r in def_rows if eps_map.get(r["sample_id"], "") == eps]

            if not subset:
                continue

            n = len(subset)

            def _bool(v):
                return str(v).strip().lower() == "true"

            def _float(v):
                try:
                    return float(v)
                except Exception:
                    return None

            n_atk  = sum(1 for r in subset if _bool(r.get("accepted_after_attack", False)))
            n_def  = sum(1 for r in subset if _bool(r.get("defense_success", False)))
            n_still = sum(1 for r in subset if _bool(r.get("attack_success_after_defense", False)))

            sims_atk = [v for r in subset if (v := _float(r.get("similarity_after_attack"))) is not None]
            sims_def = [v for r in subset if (v := _float(r.get("similarity_after_defense"))) is not None]
            times    = [v for r in subset if (v := _float(r.get("defense_time_sec"))) is not None]

            avg_sim_atk  = round(sum(sims_atk) / len(sims_atk), 6) if sims_atk else None
            avg_sim_def  = round(sum(sims_def) / len(sims_def), 6) if sims_def else None
            avg_sim_drop = round(avg_sim_atk - avg_sim_def, 6) if (avg_sim_atk and avg_sim_def) else None
            avg_time     = round(sum(times) / len(times), 4) if times else None

            summary_rows.append({
                "defense":               defense,
                "epsilon":               eps,
                "samples":               n,
                "n_attack_success":      n_atk,
                "attack_success_rate":   round(n_atk / n, 4),
                "n_defense_success":     n_def,
                "defense_success_rate":  round(n_def / n, 4),
                "n_still_attack":        n_still,
                "still_attack_rate":     round(n_still / n, 4),
                "avg_sim_after_attack":  avg_sim_atk,
                "avg_sim_after_defense": avg_sim_def,
                "avg_sim_drop":          avg_sim_drop,
                "avg_defense_time_sec":  avg_time,
            })

    return summary_rows


def save_summary(summary_rows: list[dict], out_path: str) -> None:
    fieldnames = [
        "defense", "epsilon", "samples",
        "n_attack_success", "attack_success_rate",
        "n_defense_success", "defense_success_rate",
        "n_still_attack", "still_attack_rate",
        "avg_sim_after_attack", "avg_sim_after_defense", "avg_sim_drop",
        "avg_defense_time_sec",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"집계 저장: {out_path}")


def print_summary_table(summary_rows: list[dict]) -> None:
    """터미널에 집계 테이블 출력."""
    print()
    print(f"{'defense':<12} {'eps':>6} {'samples':>7} {'atk_rate':>9} {'def_succ':>9} {'ASR_after':>10} {'sim_drop':>9}")
    print("-" * 70)
    for r in summary_rows:
        print(
            f"{r['defense']:<12} {str(r['epsilon']):>6} {r['samples']:>7} "
            f"{r['attack_success_rate']:>9.1%} {r['defense_success_rate']:>9.1%} "
            f"{r['still_attack_rate']:>10.1%} {str(r['avg_sim_drop']):>9}"
        )
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verification 방어 결과 집계")
    parser.add_argument("--results-dir", default="outputs/verification_defense",
                        help="방어 결과 루트 디렉터리")
    parser.add_argument("--index", default=None,
                        help="attack_handoff_index.csv (epsilon 분류에 필요)")
    args = parser.parse_args()

    rows = load_all_results(args.results_dir)
    summary = compute_summary(rows, index_csv=args.index)
    print_summary_table(summary)

    out_path = str(Path(args.results_dir) / "verification_defense_summary.csv")
    save_summary(summary, out_path)
