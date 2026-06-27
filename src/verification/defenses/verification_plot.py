"""
Verification 방어 결과 시각화 + 보고서 생성

verification_defense_summary.csv 를 읽어 그래프 3종과 마크다운 보고서를 생성한다.

생성 파일:
  figures/vd_bar_defense_success.png   ← 방어 성공률 막대 차트 (epsilon별)
  figures/vd_bar_sim_drop.png          ← similarity 평균 감소량 막대 차트
  figures/vd_heatmap.png               ← defense_success_rate 히트맵
  verification_defense_report.md       ← 마크다운 보고서

실행:
    python -m src.verification.defenses.verification_plot \\
        [--results-dir outputs/verification_defense]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


DEFENSE_LABELS = {
    "jpeg":      "JPEG (q=75)",
    "smoothing": "Smoothing (r=3)",
    "bitdepth":  "Bit-depth (4bit)",
}
COLORS = ["#4C72B0", "#DD8452", "#55A868"]


# ── 데이터 로드 ───────────────────────────────────────────────────────────────

def load_summary(results_dir: str) -> list[dict]:
    path = Path(results_dir) / "verification_defense_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"집계 파일 없음: {path}\n먼저 verification_summarize.py 를 실행하세요.")
    return list(csv.DictReader(open(path)))


# ── 그래프 1: 방어 성공률 막대 차트 ──────────────────────────────────────────

def plot_defense_success(rows: list[dict], out_path: str) -> None:
    defenses = list(DEFENSE_LABELS.keys())
    epsilons = sorted(set(r["epsilon"] for r in rows if r["epsilon"] != "all"))

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(defenses))
    width = 0.25

    for i, eps in enumerate(epsilons):
        vals = []
        for d in defenses:
            hit = next((r for r in rows if r["defense"] == d and r["epsilon"] == eps), None)
            vals.append(float(hit["defense_success_rate"]) * 100 if hit else 0)
        bars = ax.bar(x + i * width, vals, width, label=f"eps={eps}", color=COLORS[i % len(COLORS)])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x + width * (len(epsilons) - 1) / 2)
    ax.set_xticklabels([DEFENSE_LABELS[d] for d in defenses], fontsize=11)
    ax.set_ylabel("Defense Success Rate (%)")
    ax.set_title("Verification Defense Success Rate by Epsilon")
    ax.set_ylim(0, 110)
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"저장: {out_path}")


# ── 그래프 2: similarity 감소량 막대 차트 ────────────────────────────────────

def plot_sim_drop(rows: list[dict], out_path: str) -> None:
    defenses = list(DEFENSE_LABELS.keys())
    epsilons = sorted(set(r["epsilon"] for r in rows if r["epsilon"] != "all"))

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(defenses))
    width = 0.25

    for i, eps in enumerate(epsilons):
        vals = []
        for d in defenses:
            hit = next((r for r in rows if r["defense"] == d and r["epsilon"] == eps), None)
            try:
                vals.append(float(hit["avg_sim_drop"]) if hit else 0)
            except Exception:
                vals.append(0)
        bars = ax.bar(x + i * width, vals, width, label=f"eps={eps}", color=COLORS[i % len(COLORS)])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x + width * (len(epsilons) - 1) / 2)
    ax.set_xticklabels([DEFENSE_LABELS[d] for d in defenses], fontsize=11)
    ax.set_ylabel("Avg Similarity Drop")
    ax.set_title("Average Cosine Similarity Drop After Defense")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"저장: {out_path}")


# ── 그래프 3: 히트맵 ──────────────────────────────────────────────────────────

def plot_heatmap(rows: list[dict], out_path: str) -> None:
    defenses = list(DEFENSE_LABELS.keys())
    epsilons = sorted(set(r["epsilon"] for r in rows if r["epsilon"] != "all")) + ["all"]

    data = np.zeros((len(defenses), len(epsilons)))
    for i, d in enumerate(defenses):
        for j, eps in enumerate(epsilons):
            hit = next((r for r in rows if r["defense"] == d and r["epsilon"] == eps), None)
            data[i, j] = float(hit["defense_success_rate"]) * 100 if hit else 0

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    plt.colorbar(im, ax=ax, label="Defense Success Rate (%)")

    ax.set_xticks(range(len(epsilons)))
    ax.set_xticklabels([f"eps={e}" if e != "all" else "ALL" for e in epsilons])
    ax.set_yticks(range(len(defenses)))
    ax.set_yticklabels([DEFENSE_LABELS[d] for d in defenses])
    ax.set_title("Defense Success Rate Heatmap")

    for i in range(len(defenses)):
        for j in range(len(epsilons)):
            ax.text(j, i, f"{data[i, j]:.1f}%", ha="center", va="center",
                    fontsize=11, color="black", fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"저장: {out_path}")


# ── 보고서 생성 ───────────────────────────────────────────────────────────────

def generate_report(rows: list[dict], out_path: str) -> None:
    lines = [
        "# Verification 기반 방어 평가 보고서\n",
        "## 실험 조건\n",
        "- 공격: targeted PGD (FaceNet verification)\n",
        "- 모델: InceptionResnetV1 (pretrained=vggface2)\n",
        "- Threshold: 0.47966 (EER 기준)\n",
        "- 샘플: 212개 (eps=0.005: 45개, eps=0.010: 167개)\n",
        "- **주의**: adv 이미지가 JPEG 저장으로 인해 eps=0.005 일부 perturbation 손상 → `similarity_after_attack` 재계산 적용\n",
        "\n---\n",
        "## 방어 성공률 (defense_success_rate)\n",
        "| 방어 | eps=0.005 | eps=0.010 | 전체 |",
        "|------|-----------|-----------|------|",
    ]

    defenses = list(DEFENSE_LABELS.keys())
    for d in defenses:
        row_005 = next((r for r in rows if r["defense"] == d and r["epsilon"] == "0.005"), None)
        row_010 = next((r for r in rows if r["defense"] == d and r["epsilon"] == "0.01"), None)
        row_all = next((r for r in rows if r["defense"] == d and r["epsilon"] == "all"), None)
        v005 = f"{float(row_005['defense_success_rate'])*100:.1f}%" if row_005 else "-"
        v010 = f"{float(row_010['defense_success_rate'])*100:.1f}%" if row_010 else "-"
        vall = f"{float(row_all['defense_success_rate'])*100:.1f}%" if row_all else "-"
        lines.append(f"| {DEFENSE_LABELS[d]} | {v005} | {v010} | {vall} |")

    lines += [
        "\n---\n",
        "## 방어 후 공격 성공률 (ASR after defense)\n",
        "| 방어 | eps=0.005 | eps=0.010 | 전체 |",
        "|------|-----------|-----------|------|",
    ]
    for d in defenses:
        row_005 = next((r for r in rows if r["defense"] == d and r["epsilon"] == "0.005"), None)
        row_010 = next((r for r in rows if r["defense"] == d and r["epsilon"] == "0.01"), None)
        row_all = next((r for r in rows if r["defense"] == d and r["epsilon"] == "all"), None)
        v005 = f"{float(row_005['still_attack_rate'])*100:.1f}%" if row_005 else "-"
        v010 = f"{float(row_010['still_attack_rate'])*100:.1f}%" if row_010 else "-"
        vall = f"{float(row_all['still_attack_rate'])*100:.1f}%" if row_all else "-"
        lines.append(f"| {DEFENSE_LABELS[d]} | {v005} | {v010} | {vall} |")

    lines += [
        "\n---\n",
        "## 평균 Similarity 변화\n",
        "| 방어 | 방어 전 (전체 평균) | 방어 후 (전체 평균) | 감소량 |",
        "|------|---------------------|---------------------|--------|",
    ]
    for d in defenses:
        row_all = next((r for r in rows if r["defense"] == d and r["epsilon"] == "all"), None)
        if row_all:
            lines.append(
                f"| {DEFENSE_LABELS[d]} | {float(row_all['avg_sim_after_attack']):.4f} "
                f"| {float(row_all['avg_sim_after_defense']):.4f} "
                f"| {float(row_all['avg_sim_drop']):.4f} |"
            )

    lines += [
        "\n---\n",
        "## 결론\n",
        "- **Smoothing**: 가장 높은 방어 성공률. Gaussian blur가 perturbation을 효과적으로 희석\n",
        "- **Bit-depth**: eps=0.005에서 상대적으로 나은 성능, eps=0.010에서 효과 제한적\n",
        "- **JPEG**: adv 이미지가 이미 JPEG로 저장되어 중복 압축 효과 없음 → 사실상 방어 불가\n",
        "\n> 자세한 시각화: `figures/` 폴더 참고\n",
    ]

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"저장: {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verification 방어 결과 시각화")
    parser.add_argument("--results-dir", default="outputs/verification_defense")
    args = parser.parse_args()

    rows = load_summary(args.results_dir)

    fig_dir = Path(args.results_dir) / "figures"
    fig_dir.mkdir(exist_ok=True)

    plot_defense_success(rows, str(fig_dir / "vd_bar_defense_success.png"))
    plot_sim_drop(rows,        str(fig_dir / "vd_bar_sim_drop.png"))
    plot_heatmap(rows,         str(fig_dir / "vd_heatmap.png"))
    generate_report(rows,      str(Path(args.results_dir) / "verification_defense_report.md"))
