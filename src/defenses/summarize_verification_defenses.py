"""방어 결과 CSV 여러 개를 읽어 통합 비교표를 생성.

지원 방어: jpeg, smoothing, bitdepth, dae, diffpure (어떤 포맷이든 defense 컬럼만 있으면 됨)

사용법:
  # 폴더 안 모든 verification_defense_*.csv 자동 탐색
  python -m src.defenses.summarize_verification_defenses \
    --defense-root outputs/defenses/verification \
    --out outputs/defenses/verification/defense_comparison_summary.csv

  # 파일 직접 지정
  python -m src.defenses.summarize_verification_defenses \
    --defense-csvs \
      outputs/defenses/verification/dae/verification_defense_dae.csv \
      outputs/defenses/verification/diffpure_t0p10/verification_defense_diffpure_t0.10.csv \
    --out outputs/defenses/verification/defense_comparison_summary.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_float(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        return default


def summarize_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    ok_rows = [r for r in all_rows if r.get("status", "ok") == "ok"]
    if not ok_rows:
        return []

    defense_name = ok_rows[0].get("defense", path.stem)
    defense_params = ok_rows[0].get("defense_params", "")

    results = []
    # epsilon별 + 전체 집계
    epsilon_groups: dict[str, list] = {}
    for r in ok_rows:
        eps_key = f"{parse_float(r.get('epsilon', 0)):.3f}"
        epsilon_groups.setdefault(eps_key, []).append(r)
    epsilon_groups["ALL"] = ok_rows

    for eps_label, rows in epsilon_groups.items():
        attacked = [r for r in rows if parse_bool(r.get("attack_success_before_defense", False))]
        n_attacked = len(attacked)
        n_total = len(rows)

        if n_attacked == 0:
            defense_success_rate = 0.0
            still_accepted_rate = 0.0
        else:
            n_defense_success = sum(parse_bool(r.get("defense_success", False)) for r in attacked)
            n_still_accepted = sum(parse_bool(r.get("accepted_after_defense", False)) for r in attacked)
            defense_success_rate = n_defense_success / n_attacked
            still_accepted_rate = n_still_accepted / n_attacked

        sims_attack = [parse_float(r.get("similarity_after_attack", 0)) for r in rows]
        sims_defense = [parse_float(r.get("similarity_after_defense", 0)) for r in rows]
        avg_sim_attack = sum(sims_attack) / max(len(sims_attack), 1)
        avg_sim_defense = sum(sims_defense) / max(len(sims_defense), 1)
        avg_sim_drop = avg_sim_attack - avg_sim_defense

        times = [parse_float(r.get("defense_time_sec", 0)) for r in rows]
        avg_time = sum(times) / max(len(times), 1)

        results.append({
            "defense": defense_name,
            "defense_params": defense_params,
            "source_file": str(path),
            "epsilon": eps_label,
            "n_total": n_total,
            "n_attacked": n_attacked,
            "defense_success_rate": defense_success_rate,
            "still_accepted_rate": still_accepted_rate,
            "avg_sim_after_attack": avg_sim_attack,
            "avg_sim_after_defense": avg_sim_defense,
            "avg_sim_drop": avg_sim_drop,
            "avg_defense_time_sec": avg_time,
        })

    return results


def find_defense_csvs(root: Path) -> list[Path]:
    pattern_names = ["verification_defense_", "defense_results"]
    found = []
    for p in sorted(root.rglob("*.csv")):
        if any(pat in p.name for pat in pattern_names) and "summary" not in p.name and "template" not in p.name:
            found.append(p)
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description="방어 결과 CSV 통합 비교표 생성.")
    parser.add_argument("--defense-root", type=Path, default=None,
                        help="방어 결과 CSV들이 있는 루트 폴더 (자동 탐색).")
    parser.add_argument("--defense-csvs", nargs="+", type=Path, default=None,
                        help="방어 결과 CSV 파일들 직접 지정 (--defense-root 대신 사용 가능).")
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/defenses/verification/defense_comparison_summary.csv"))
    args = parser.parse_args()

    if args.defense_csvs:
        paths = [p for p in args.defense_csvs if p.exists()]
    elif args.defense_root:
        paths = find_defense_csvs(args.defense_root)
    else:
        raise ValueError("--defense-root 또는 --defense-csvs 중 하나를 지정하세요.")

    if not paths:
        raise FileNotFoundError("방어 결과 CSV를 찾을 수 없습니다.")

    print(f"Found {len(paths)} defense CSV(s):")
    for p in paths:
        print(f"  {p}")

    all_rows: list[dict[str, object]] = []
    for path in paths:
        rows = summarize_csv(path)
        all_rows.extend(rows)
        if rows:
            overall = next((r for r in rows if r["epsilon"] == "ALL"), rows[0])
            print(f"  {overall['defense']:30s}  "
                  f"DSR={overall['defense_success_rate']:.1%}  "
                  f"ASR_after={overall['still_accepted_rate']:.1%}  "
                  f"sim_drop={overall['avg_sim_drop']:.4f}  "
                  f"time={overall['avg_defense_time_sec']:.3f}s")

    if not all_rows:
        raise ValueError("집계할 데이터가 없습니다.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    # 전체 기준(epsilon=ALL) 비교표 출력
    overall_rows = [r for r in all_rows if r["epsilon"] == "ALL"]
    overall_rows.sort(key=lambda r: -r["defense_success_rate"])

    print("\n=== 전체 비교 (epsilon=ALL) ===")
    print(f"{'Defense':<35} {'DSR':>8} {'ASR_after':>10} {'sim_drop':>10} {'time(s)':>8}")
    print("-" * 75)
    for r in overall_rows:
        print(
            f"{str(r['defense']):<35} "
            f"{r['defense_success_rate']:>8.1%} "
            f"{r['still_accepted_rate']:>10.1%} "
            f"{r['avg_sim_drop']:>10.4f} "
            f"{r['avg_defense_time_sec']:>8.3f}"
        )
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
