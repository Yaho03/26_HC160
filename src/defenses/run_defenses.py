"""Run all 3 defense methods (JPEG / Gaussian / Bit-depth) against all attack families.

Usage from Colab notebook:
    import sys; sys.path.insert(0, REPO_DIR)
    from src.defenses.run_defenses import run_all_defenses
    run_all_defenses(
        attack_root = "/content/drive/MyDrive/hanium-aml-defense",
        index_path  = ".../outputs/attacks/attack_index.csv",
        ckpt_path   = ".../checkpoints/face_resnet50_lfw10/best.pt",
        results_dir = "/content/defense_results",
        repo_dir    = "/content/26_HC160",
    )
"""

from __future__ import annotations

import os
import subprocess
import sys

# 실험 조건 (README 기준)
DEFENSE_CONFIGS: list[dict] = [
    dict(
        module    = "src.defenses.defense_jpeg",
        args      = ["--quality", "75"],
        out_subdir= "jpeg",
        label     = "JPEG (quality=75)",
    ),
    dict(
        module    = "src.defenses.defense_smoothing",
        args      = ["--radius", "3"],
        out_subdir= "smoothing",
        label     = "Gaussian Smoothing (radius=3)",
    ),
    dict(
        module    = "src.defenses.defense_bitdepth",
        args      = ["--bits", "4"],
        out_subdir= "bitdepth",
        label     = "Bit-depth Reduction (4-bit)",
    ),
]


def run_all_defenses(
    attack_root: str,
    index_path: str,
    ckpt_path: str,
    results_dir: str,
    repo_dir: str,
) -> None:
    """
    Parameters
    ----------
    attack_root : CWD for subprocesses — adv_file 상대 경로가 여기 기준으로 해석됨
    index_path  : attack_index.csv 절대 경로
    ckpt_path   : best.pt 절대 경로
    results_dir : 방어 결과 CSV 저장 루트 디렉토리
    repo_dir    : 26_HC160 레포 경로 (PYTHONPATH에 추가됨)
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = repo_dir + os.pathsep + env.get("PYTHONPATH", "")
    os.makedirs(results_dir, exist_ok=True)

    for cfg in DEFENSE_CONFIGS:
        out_dir = os.path.join(results_dir, cfg["out_subdir"])
        cmd = [
            sys.executable, "-m", cfg["module"],
            "--attack-index", index_path,
            "--checkpoint",   ckpt_path,
            "--out-dir",      out_dir,
            *cfg["args"],
        ]
        print(f"\n{'='*55}")
        print(f"[{cfg['label']}]")
        print(f"{'='*55}")
        result = subprocess.run(cmd, cwd=attack_root, env=env)
        if result.returncode != 0:
            raise RuntimeError(
                f"{cfg['label']} 실행 실패 (returncode={result.returncode})"
            )

    print("\n[DONE] 방어 3종 완료")
