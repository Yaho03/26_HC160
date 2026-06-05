"""Verification attack vs defense 비교 시각화.

사용법:
  # 공격 결과만 (epsilon vs ASR 곡선)
  python -m src.reports.plot_verification_attack_defense \
    --attack-summaries \
      outputs/verification_attacks_facenet/pgd_png/verification_attack_summary_pgd_png.csv \
      outputs/verification_attacks_facenet/fgsm/verification_attack_summary_fgsm.csv \
      outputs/verification_attacks_facenet/pgd_adaptive/verification_attack_summary_adaptive.csv \
    --out-dir outputs/figures/verification_attack_defense

  # 방어 결과도 같이 (epsilon vs defense success)
  python -m src.reports.plot_verification_attack_defense \
    --attack-summaries ... \
    --defense-summary outputs/handoff/defense_results/verification_defense_summary.csv \
    --out-dir outputs/figures/verification_attack_defense
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ATTACK_LABELS = {
    "targeted_pgd_facenet_verification": "PGD",
    "targeted_fgsm_facenet_verification": "FGSM",
    "targeted_pgd_facenet_adaptive_smoothing": "Adaptive PGD\n(vs Smoothing)",
    "targeted_pgd_facenet_adaptive_jpeg": "Adaptive PGD\n(vs JPEG)",
    "targeted_pgd_facenet_adaptive_none": "PGD (no adapt)",
}

DEFENSE_LABELS = {
    "jpeg": "JPEG q=75",
    "smoothing": "Smoothing r=3",
    "bitdepth": "Bit-depth 4bit",
}

COLORS = {
    "targeted_pgd_facenet_verification": "#4062BB",
    "targeted_fgsm_facenet_verification": "#D1603D",
    "targeted_pgd_facenet_adaptive_smoothing": "#2E8B57",
    "targeted_pgd_facenet_adaptive_jpeg": "#9B59B6",
}

DEFENSE_COLORS = {
    "jpeg": "#E67E22",
    "smoothing": "#2980B9",
    "bitdepth": "#8E44AD",
}


def load_attack_summaries(paths: list[Path]) -> pd.DataFrame:
    dfs = []
    for path in paths:
        df = pd.read_csv(path)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    combined["epsilon"] = combined["epsilon"].astype(float)
    combined["asr"] = combined["target_accept_rate_after_attack"].astype(float) * 100
    combined["attack_label"] = combined["attack"].map(ATTACK_LABELS).fillna(combined["attack"])
    combined["color"] = combined["attack"].map(COLORS).fillna("#888888")
    return combined


def load_defense_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 방어팀 CSV 컬럼: defense, epsilon, defense_success_rate 등
    # 방어팀이 보내는 summary CSV 포맷에 맞게 컬럼명 정규화
    col_map = {
        "defense_success_rate": "defense_success_rate",
        "defense_success": "defense_success_rate",
    }
    for old, new in col_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    if "defense_success_rate" in df.columns:
        df["defense_success_pct"] = df["defense_success_rate"].astype(float) * 100
    if "epsilon" in df.columns:
        df["epsilon"] = df["epsilon"].astype(float)
    if "defense" in df.columns:
        df["defense_label"] = df["defense"].map(DEFENSE_LABELS).fillna(df["defense"])
    return df


def plot_asr_by_epsilon(df: pd.DataFrame, out_path: Path) -> None:
    """공격 종류별 epsilon vs ASR 곡선."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for attack, group in df.groupby("attack"):
        label = ATTACK_LABELS.get(attack, attack)
        color = COLORS.get(attack, "#888888")
        g = group.sort_values("epsilon")
        ax.plot(g["epsilon"], g["asr"], marker="o", label=label, color=color, linewidth=2)

    ax.set_title("FaceNet Verification Attack: Epsilon vs ASR", fontsize=14, pad=12)
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Target Accept Rate after Attack (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_defense_success_by_epsilon(df_defense: pd.DataFrame, out_path: Path) -> None:
    """방어 종류별 epsilon vs defense success 곡선."""
    if "epsilon" not in df_defense.columns or "defense_success_pct" not in df_defense.columns:
        print(f"  skip {out_path.name}: epsilon or defense_success_pct column missing")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for defense, group in df_defense.groupby("defense"):
        label = DEFENSE_LABELS.get(defense, defense)
        color = DEFENSE_COLORS.get(defense, "#888888")
        g = group.sort_values("epsilon")
        ax.plot(g["epsilon"], g["defense_success_pct"], marker="s", label=label,
                color=color, linewidth=2, linestyle="--")

    ax.set_title("Defense Success Rate by Epsilon", fontsize=14, pad=12)
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Defense Success Rate (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_attack_vs_defense(df_atk: pd.DataFrame, df_def: pd.DataFrame, out_path: Path) -> None:
    """공격 ASR vs 방어 성공률 — epsilon 기준 같은 축에 겹쳐 표시."""
    epsilons_atk = sorted(df_atk["epsilon"].unique())
    if len(epsilons_atk) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    # 공격 ASR (실선)
    for attack, group in df_atk.groupby("attack"):
        label = ATTACK_LABELS.get(attack, attack)
        color = COLORS.get(attack, "#888888")
        g = group.sort_values("epsilon")
        ax.plot(g["epsilon"], g["asr"], marker="o", label=f"ATK: {label}", color=color, linewidth=2)

    # 방어 성공률 (점선)
    if "epsilon" in df_def.columns and "defense_success_pct" in df_def.columns:
        for defense, group in df_def.groupby("defense"):
            label = DEFENSE_LABELS.get(defense, defense)
            color = DEFENSE_COLORS.get(defense, "#888888")
            g = group.sort_values("epsilon")
            ax.plot(g["epsilon"], g["defense_success_pct"], marker="s",
                    label=f"DEF: {label}", color=color, linewidth=2, linestyle="--")

    ax.set_title("Attack ASR vs Defense Success Rate by Epsilon", fontsize=14, pad=12)
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_attack_bar(df: pd.DataFrame, out_path: Path) -> None:
    """공격 종류별 최대 ASR 막대 차트 (epsilon은 최고 성공 epsilon 기준)."""
    best = df.loc[df.groupby("attack")["asr"].idxmax()].copy()
    best = best.sort_values("asr", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(best["attack_label"], best["asr"],
                  color=best["color"].tolist(), edgecolor="#202124", linewidth=0.8)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_title("Best ASR by Attack Type (Best Epsilon)", fontsize=14, pad=12)
    ax.set_ylabel("Target Accept Rate (%)")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {out_path}")


def print_table(df_atk: pd.DataFrame, df_def: pd.DataFrame | None) -> None:
    print("\n=== Attack Summary ===")
    cols = ["attack", "epsilon", "asr", "avg_l2", "avg_linf", "avg_time_sec"]
    available = [c for c in cols if c in df_atk.columns]
    print(df_atk[available].to_string(index=False))

    if df_def is not None:
        print("\n=== Defense Summary ===")
        def_cols = ["defense", "epsilon", "defense_success_pct"]
        available_def = [c for c in def_cols if c in df_def.columns]
        print(df_def[available_def].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verification attack vs defense 비교 시각화.")
    parser.add_argument(
        "--attack-summaries", nargs="+", type=Path, required=True,
        help="verification_attack_summary CSV 파일들 (여러 개 가능)."
    )
    parser.add_argument(
        "--defense-summary", type=Path, default=None,
        help="방어팀 verification_defense_summary.csv (선택)."
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/figures/verification_attack_defense"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df_atk = load_attack_summaries(args.attack_summaries)
    df_def = load_defense_summary(args.defense_summary) if args.defense_summary else None

    print_table(df_atk, df_def)

    plot_asr_by_epsilon(df_atk, args.out_dir / "asr_by_epsilon.png")
    plot_attack_bar(df_atk, args.out_dir / "attack_best_asr_bar.png")

    if df_def is not None:
        plot_defense_success_by_epsilon(df_def, args.out_dir / "defense_success_by_epsilon.png")
        plot_attack_vs_defense(df_atk, df_def, args.out_dir / "attack_vs_defense_by_epsilon.png")

    print(f"\nAll figures saved to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
