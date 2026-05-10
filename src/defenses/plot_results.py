"""Load defense result CSVs and generate summary stats, plots, and a markdown report.

Usage from Colab notebook:
    from src.defenses.plot_results import load_results, compute_summary, plot_all, generate_report

    df      = load_results(results_dir)
    summary = compute_summary(df)
    plot_all(df, summary, figures_dir)
    generate_report(summary, results_dir)
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RESULT_FILES: dict[str, tuple[str, str]] = {
    "jpeg":      ("jpeg",      "jpeg_results_q75.csv"),
    "smoothing": ("smoothing", "smoothing_results_r3p0.csv"),
    "bitdepth":  ("bitdepth",  "bitdepth_results_4bit.csv"),
}

ATTACK_ORDER = ["fgsm", "pgd", "square", "jsma", "zoo"]

DEFENSE_LABELS: dict[str, str] = {
    "jpeg":               "JPEG (q=75)",
    "gaussian_smoothing": "Gaussian (r=3)",
    "bit_depth":          "Bit-depth (4bit)",
}

COLORS = ["#4C72B0", "#55A868", "#C44E52"]


# ──────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────
def load_results(results_dir: str) -> pd.DataFrame:
    """방어 결과 CSV 3개를 합쳐 하나의 DataFrame으로 반환."""
    dfs = []
    for subdir, fname in RESULT_FILES.values():
        path = os.path.join(results_dir, subdir, fname)
        if not os.path.exists(path):
            print(f"건너뜀 (없음): {path}")
            continue
        tmp = pd.read_csv(path)
        if "status" in tmp.columns:
            tmp = tmp[tmp["status"] == "ok"]
        tmp["conf_drop"] = (
            tmp["target_conf_before_defense"].astype(float)
            - tmp["target_conf_after_defense"].astype(float)
        )
        dfs.append(tmp)

    if not dfs:
        raise FileNotFoundError(f"결과 CSV 없음: {results_dir}")

    df = pd.concat(dfs, ignore_index=True)
    for col in ["attack_success_before_defense", "attack_success_after_defense", "recovered"]:
        df[col] = df[col].astype(str).str.lower().isin(["true", "1", "yes"])
    print(f"로드 완료: {len(df):,}행")
    return df


# ──────────────────────────────────────────────
# 집계
# ──────────────────────────────────────────────
def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """공격 × 방어 조합별 집계 테이블 반환."""
    summary = (
        df.groupby(["attack_family", "defense"])
        .apply(lambda g: pd.Series({
            "n_samples":       len(g),
            "defense_rate_%":  round((~g["attack_success_after_defense"]).mean() * 100, 1),
            "recovery_rate_%": round(g["recovered"].mean() * 100, 1),
            "avg_conf_drop":   round(g["conf_drop"].mean(), 4),
            "avg_time_ms":     round(g["defense_time_sec"].astype(float).mean() * 1000, 2),
        }), include_groups=False)
        .reset_index()
    )
    summary["defense_label"] = summary["defense"].map(DEFENSE_LABELS).fillna(summary["defense"])
    summary["attack_family"] = pd.Categorical(
        summary["attack_family"], categories=ATTACK_ORDER, ordered=True
    )
    return summary.sort_values(["attack_family", "defense"]).reset_index(drop=True)


# ──────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────
def plot_heatmap(summary: pd.DataFrame, figures_dir: str) -> None:
    """방어 성공률 & 복원율 Heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle("Defense Evaluation — 5 Attacks × 3 Defenses", fontsize=13, fontweight="bold")

    for ax, metric, title, cmap in zip(
        axes,
        ["defense_rate_%", "recovery_rate_%"],
        ["Defense Success Rate (%)", "Recovery Rate (%)"],
        ["YlGn", "Blues"],
    ):
        pivot = summary.pivot(
            index="attack_family", columns="defense_label", values=metric
        ).reindex(ATTACK_ORDER)
        sns.heatmap(
            pivot, ax=ax, annot=True, fmt=".1f",
            cmap=cmap, vmin=0, vmax=100,
            linewidths=0.5, cbar_kws={"label": "%"},
        )
        ax.set_title(title)
        ax.set_xlabel("Defense")
        ax.set_ylabel("Attack")

    plt.tight_layout()
    out = os.path.join(figures_dir, "heatmap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"저장: {out}")


def plot_bar(summary: pd.DataFrame, figures_dir: str) -> None:
    """공격별 방어 성능 바차트."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Defense & Recovery Rate by Attack", fontsize=13, fontweight="bold")

    for ax, metric, ylabel in zip(
        axes,
        ["defense_rate_%", "recovery_rate_%"],
        ["Defense Success Rate (%)", "Recovery Rate (%)"],
    ):
        pivot = summary.pivot(
            index="attack_family", columns="defense_label", values=metric
        ).reindex(ATTACK_ORDER)
        pivot.plot(kind="bar", ax=ax, color=COLORS, edgecolor="white", width=0.7)
        ax.set_ylim(0, 110)
        ax.set_title(ylabel)
        ax.set_xlabel("Attack")
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.legend(title="Defense", loc="upper right", fontsize=8)

    plt.tight_layout()
    out = os.path.join(figures_dir, "bar_by_attack.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"저장: {out}")


def plot_boxplot(df: pd.DataFrame, figures_dir: str) -> None:
    """신뢰도 감소(conf_drop) Box plot."""
    df = df.copy()
    df["defense_label"] = df["defense"].map(DEFENSE_LABELS).fillna(df["defense"])
    df["attack_family"] = pd.Categorical(
        df["attack_family"], categories=ATTACK_ORDER, ordered=True
    )
    palette = dict(zip(DEFENSE_LABELS.values(), COLORS))

    fig, ax = plt.subplots(figsize=(13, 5))
    sns.boxplot(
        data=df, x="attack_family", y="conf_drop", hue="defense_label",
        ax=ax, palette=palette, width=0.6, linewidth=0.8,
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.3},
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Target Confidence Drop: before − after defense", fontsize=12, fontweight="bold")
    ax.set_xlabel("Attack")
    ax.set_ylabel("Confidence Drop")
    ax.legend(title="Defense", loc="upper right")

    plt.tight_layout()
    out = os.path.join(figures_dir, "boxplot_conf_drop.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"저장: {out}")


def plot_all(df: pd.DataFrame, summary: pd.DataFrame, figures_dir: str) -> None:
    """3종 그래프 한번에 생성."""
    os.makedirs(figures_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plot_heatmap(summary, figures_dir)
    plot_bar(summary, figures_dir)
    plot_boxplot(df, figures_dir)


# ──────────────────────────────────────────────
# 보고서
# ──────────────────────────────────────────────
def generate_report(summary: pd.DataFrame, results_dir: str) -> None:
    """마크다운 보고서 생성 및 저장."""
    try:
        from IPython.display import Markdown, display as ipy_display
        _display = ipy_display
        _Markdown = Markdown
    except ImportError:
        _display = print
        _Markdown = lambda x: x  # noqa: E731

    def pivot_md(metric: str, title: str) -> str:
        pivot = summary.pivot(
            index="attack_family", columns="defense_label", values=metric
        ).reindex(ATTACK_ORDER)
        cols = list(pivot.columns)
        lines = [f"### {title}\n"]
        lines.append("| 공격 \\ 방어 | " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * (len(cols) + 1))
        for atk, row in pivot.iterrows():
            lines.append(f"| **{atk}** | " + " | ".join(str(v) for v in row.values) + " |")
        return "\n".join(lines)

    report = f"""# Defense Evaluation Results

## 실험 조건

| 공격   | 파라미터 |
|--------|----------|
| FGSM   | ε = 0.005 |
| PGD    | ε = 0.03, α = 0.003, steps = 10 |
| SQUARE | ε = 0.05, max_queries = 300 |
| JSMA   | θ = 0.05, steps = 20, pixels_per_step = 200 |
| ZOO    | ε = 0.05, max_queries = 2000, coords_per_iter = 128, lr = 0.02 |

| 방어           | 파라미터 |
|----------------|----------|
| JPEG 압축      | quality = 75 |
| Gaussian blur  | radius = 3 |
| Bit-depth 축소 | bits = 4 |

{pivot_md('defense_rate_%', '방어 성공률 (%)')}

{pivot_md('recovery_rate_%', '복원율 (%)')}

{pivot_md('avg_conf_drop', '평균 신뢰도 감소')}
"""
    md_path = os.path.join(results_dir, "defense_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"저장: {md_path}")
    _display(_Markdown(report))
