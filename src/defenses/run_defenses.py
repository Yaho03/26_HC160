"""Run all 3 defense methods (JPEG / Gaussian / Bit-depth) against all attack families.

Usage from Colab notebook:
    import sys; sys.path.insert(0, REPO_DIR)
    from src.defenses.run_defenses import run_all_defenses
    run_all_defenses(
        attack_root = "/content/attack_outputs",
        index_path  = ".../outputs/attacks/attack_index.csv",
        ckpt_path   = ".../checkpoints/face_resnet50_lfw10/best.pt",
        results_dir = "/content/defense_results",
        repo_dir    = "/content/26_HC160",
    )
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time

# 실험 조건 (README 기준)
DEFENSE_CONFIGS: list[dict] = [
    dict(
        module     = "src.defenses.defense_jpeg",
        args       = ["--quality", "75"],
        out_subdir = "jpeg",
        label      = "JPEG (quality=75)",
    ),
    dict(
        module     = "src.defenses.defense_smoothing",
        args       = ["--radius", "3"],
        out_subdir = "smoothing",
        label      = "Gaussian Smoothing (radius=3)",
    ),
    dict(
        module     = "src.defenses.defense_bitdepth",
        args       = ["--bits", "4"],
        out_subdir = "bitdepth",
        label      = "Bit-depth Reduction (4-bit)",
    ),
]

# tqdm 출력에서 처리된 샘플 수 파싱 (예: "100%|...| 47/47 ...")
_TQDM_RE = re.compile(r"(\d+)/(\d+)")


def _run_with_progress(cmd: list, cwd: str, env: dict, label: str) -> None:
    """subprocess를 실행하면서 tqdm 출력을 파싱해 진행바를 표시한다."""
    try:
        from tqdm.auto import tqdm as _tqdm
    except ImportError:
        subprocess.run(cmd, cwd=cwd, env=env, check=True)
        return

    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    pbar = _tqdm(total=None, desc=f"  {label}", unit="sample",
                 dynamic_ncols=True, leave=True)
    try:
        for line in proc.stdout:
            line = line.rstrip()
            # tqdm 진행 출력은 화면에 그대로 내보냄
            if line:
                print(line, flush=True)
            # "n/total" 패턴으로 진행바 업데이트
            m = _TQDM_RE.search(line)
            if m:
                n, total = int(m.group(1)), int(m.group(2))
                if pbar.total != total:
                    pbar.total = total
                    pbar.refresh()
                pbar.n = n
                pbar.refresh()
    finally:
        pbar.close()

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"{label} 실행 실패 (returncode={proc.returncode})")


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
    try:
        from tqdm.auto import tqdm as _tqdm
        outer = _tqdm(DEFENSE_CONFIGS, desc="전체 방어 진행", unit="defense",
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    except ImportError:
        outer = DEFENSE_CONFIGS

    env = os.environ.copy()
    env["PYTHONPATH"] = repo_dir + os.pathsep + env.get("PYTHONPATH", "")
    os.makedirs(results_dir, exist_ok=True)

    for cfg in outer:
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

        t0 = time.time()
        _run_with_progress(cmd, cwd=attack_root, env=env, label=cfg["label"])
        elapsed = time.time() - t0
        print(f"  완료  ({elapsed:.1f}s)")

    print("\n[DONE] 방어 3종 완료")
