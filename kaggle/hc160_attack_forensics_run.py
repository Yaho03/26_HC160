"""HC160 Kaggle run: attack sweeps + handoff packages.

This script replaces the old Colab phase-3 cells for the attack owner.
It uses attached Kaggle datasets:

- dohyunp/hc160-verification-data
- dohyunp/hc160-meta
- dohyunp/hc160-src

Outputs are copied to /kaggle/working:

- handoff/*.zip
- summaries/*_summary.csv
- run_log/*.log
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch


REPO = Path("/kaggle/tmp/26_HC160")
INPUT = Path("/kaggle/input")
SRC = INPUT / "hc160-src"
DATA = INPUT / "hc160-verification-data"
META = INPUT / "hc160-meta"
OUT = Path("/kaggle/working")
PY = sys.executable


def existing_roots(*paths: Path) -> list[Path]:
    return [path for path in paths if path.exists()]


def input_search_roots() -> list[Path]:
    return existing_roots(META, DATA, SRC, INPUT / "datasets", INPUT)


def sh(cmd: str, cwd: Path | None = None, check: bool = False) -> int:
    print(f"\n$ {cmd}", flush=True)
    rc = subprocess.run(cmd, shell=True, cwd=cwd).returncode
    if check and rc != 0:
        raise RuntimeError(f"command failed with rc={rc}: {cmd}")
    return rc


def find_repo_root() -> Path:
    search_roots = existing_roots(SRC, INPUT / "datasets", INPUT)
    for root in search_roots:
        if (root / "src" / "verification").exists():
            return root
        for path in root.rglob("src/verification"):
            return path.parent.parent

    unpacked = Path("/kaggle/tmp/hc160_src_unpacked")
    zip_files = []
    for root in search_roots:
        for zip_file in root.rglob("*.zip"):
            try:
                with zipfile.ZipFile(zip_file) as zf:
                    if any(name.startswith("src/verification/") for name in zf.namelist()):
                        zip_files.append(zip_file)
            except zipfile.BadZipFile:
                continue
    if zip_files:
        if unpacked.exists():
            shutil.rmtree(unpacked)
        unpacked.mkdir(parents=True, exist_ok=True)
        for zip_file in zip_files:
            print(f"extracting source archive: {zip_file}", flush=True)
            with zipfile.ZipFile(zip_file) as zf:
                zf.extractall(unpacked)
        for root in search_roots:
            for item in root.iterdir():
                if item.is_file() and item.suffix != ".zip":
                    shutil.copy(item, unpacked / item.name)
        if (unpacked / "src" / "verification").exists():
            return unpacked
    raise FileNotFoundError("src/verification not found in hc160-src dataset")


def find_one(name: str, roots: list[Path]) -> Path:
    for root in roots:
        direct = root / name
        if direct.exists():
            return direct
        hits = list(root.rglob(name))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"{name} not found under {roots}")


def stage_repo() -> None:
    print("=== Kaggle input layout ===", flush=True)
    for root in sorted(Path("/kaggle/input").glob("*")):
        print(f"  {root}", flush=True)

    src_root = find_repo_root()
    print(f"repo source root: {src_root}", flush=True)
    if REPO.exists():
        shutil.rmtree(REPO)
    REPO.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_root, REPO)
    sh(f"{PY} -m pip install -q numpy==1.26.4", check=True)
    sh(
        f"{PY} -m pip install -q torch==2.2.2 torchvision==0.17.2 "
        f"--index-url https://download.pytorch.org/whl/cu121",
        check=True,
    )
    sh(f"{PY} -m pip install -q numpy==1.26.4", check=True)
    sh(f"{PY} -m pip install -q --no-deps facenet-pytorch", check=True)
    sh(
        f"{PY} - <<'PY'\n"
        "import numpy as np\n"
        "import torch\n"
        "print('numpy:', np.__version__)\n"
        "print('torch:', torch.__version__)\n"
        "print('cuda:', torch.version.cuda)\n"
        "print('gpu_count:', torch.cuda.device_count())\n"
        "if torch.cuda.is_available():\n"
        "    print('gpu0:', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))\n"
        "PY",
        check=True,
    )


def restore_data() -> None:
    (REPO / "outputs/verification").mkdir(parents=True, exist_ok=True)
    (REPO / "outputs/verification_facenet").mkdir(parents=True, exist_ok=True)
    roots = input_search_roots()

    shutil.copy(
        find_one("lfw_test_pairs.csv", roots),
        REPO / "outputs/verification/lfw_test_pairs.csv",
    )
    shutil.copy(
        find_one("verification_metrics.json", roots),
        REPO / "outputs/verification_facenet/verification_metrics.json",
    )

    sample = next(Path("/kaggle/input").rglob("*_0001.jpg"), None)
    if sample is None:
        sample = next(Path("/kaggle/input").rglob("*.jpg"), None)
    if sample is None:
        raise FileNotFoundError("No LFW jpg found under /kaggle/input")
    raw_src = sample.parent.parent
    print(f"LFW raw root: {raw_src}", flush=True)

    processed = REPO / "data/processed/lfw_identity_10/test"
    if not (processed.exists() and len(list(processed.rglob("*.jpg"))) > 100):
        sh(
            f"{PY} -m src.datasets.prepare_lfw_identity_dataset "
            f"--raw-dir {raw_src} "
            f"--out-dir data/processed/lfw_identity_10 "
            f"--num-identities 10 --seed 42",
            cwd=REPO,
        )


def run_job(label: str, cmd: str, gpu: int, log_dir: Path) -> int:
    log = log_dir / f"{label}.log"
    t0 = time.time()
    with log.open("w") as f:
        rc = subprocess.run(
            f"CUDA_VISIBLE_DEVICES={gpu} {cmd}",
            shell=True,
            cwd=REPO,
            stdout=f,
            stderr=subprocess.STDOUT,
        ).returncode
    status = "OK " if rc == 0 else "FAIL"
    print(f"[{status}] gpu{gpu} {label} ({time.time() - t0:.0f}s, rc={rc})", flush=True)
    if rc != 0 and log.exists():
        print(f"--- tail {log.name} ---", flush=True)
        print("\n".join(log.read_text(errors="replace").splitlines()[-80:]), flush=True)
    return rc


def run_attacks() -> None:
    pairs = "outputs/verification/lfw_test_pairs.csv"
    metrics = "outputs/verification_facenet/verification_metrics.json"
    log_dir = REPO / "outputs/run_log"
    log_dir.mkdir(parents=True, exist_ok=True)

    common = (
        f"--pairs {pairs} --metrics {metrics} --pretrained vggface2 "
        f"--only-initial-rejects --image-format png"
    )
    jobs: list[tuple[str, str]] = []

    for eps in ["0.005", "0.010"]:
        jobs.append((
            f"pgd_png_eps{eps}",
            f"{PY} -m src.verification.targeted_pgd_facenet_verification {common} "
            f"--epsilon {eps} --alpha 0.001 --steps 10 --limit 100 "
            f"--out-dir outputs/verification_attacks_facenet/pgd_png",
        ))

    for eps in ["0.005", "0.010", "0.020", "0.030"]:
        jobs.append((
            f"fgsm_eps{eps}",
            f"{PY} -m src.verification.targeted_fgsm_facenet_verification {common} "
            f"--epsilon {eps} --limit 100 "
            f"--out-dir outputs/verification_attacks_facenet/fgsm",
        ))

    for eps in ["0.010", "0.020", "0.030"]:
        jobs.append((
            f"square_eps{eps}",
            f"{PY} -m src.verification.targeted_square_facenet_verification {common} "
            f"--epsilon {eps} --max-queries 300 --limit 100 "
            f"--out-dir outputs/verification_attacks_facenet/square",
        ))

    for eps in ["0.005", "0.010", "0.020"]:
        jobs.append((
            f"adaptive_smoothing_eps{eps}",
            f"{PY} -m src.verification.targeted_pgd_facenet_adaptive {common} "
            f"--epsilon {eps} --alpha 0.001 --steps 20 --limit 100 "
            f"--defense-transform smoothing --smoothing-kernel 13 --smoothing-sigma 3.0 "
            f"--out-dir outputs/verification_attacks_facenet/pgd_adaptive_smoothing",
        ))

    for eps in ["0.005", "0.010", "0.015", "0.020"]:
        jobs.append((
            f"pgd_adv_training_eps{eps}",
            f"{PY} -m src.verification.targeted_pgd_facenet_verification {common} "
            f"--epsilon {eps} --alpha 0.001 --steps 10 --limit 200 "
            f"--out-dir outputs/verification_attacks_facenet/pgd_adv_training",
        ))

    n_gpu = max(1, torch.cuda.device_count())
    print(f"=== GPUs detected: {torch.cuda.device_count()} -> {n_gpu} worker(s) ===", flush=True)
    buckets = [[] for _ in range(n_gpu)]
    for idx, job in enumerate(jobs):
        buckets[idx % n_gpu].append(job)

    start = time.time()
    def run_bucket(gpu: int) -> list[tuple[str, int]]:
        return [(label, run_job(label, cmd, gpu, log_dir)) for label, cmd in buckets[gpu]]

    with ThreadPoolExecutor(max_workers=n_gpu) as executor:
        results = list(executor.map(run_bucket, range(n_gpu)))
    failures = [(label, rc) for bucket_result in results for label, rc in bucket_result if rc != 0]
    if failures:
        raise RuntimeError(f"attack jobs failed: {failures}")
    print(f"=== attacks finished in {(time.time() - start) / 60:.1f} min ===", flush=True)


def summarize_and_package() -> None:
    metrics = "outputs/verification_facenet/verification_metrics.json"
    attack_root = Path("outputs/verification_attacks_facenet")
    subdirs = ["pgd_png", "fgsm", "square", "pgd_adaptive_smoothing", "pgd_adv_training"]
    for subdir in subdirs:
        root = attack_root / subdir
        if root.exists():
            sh(
                f"{PY} -m src.verification.summarize_verification_attacks "
                f"--metadata-root {root} --out {root}/summary.csv",
                cwd=REPO,
            )

    handoff_cmds = [
        (
            "pgd_png",
            "outputs/handoff/facenet_pgd_png_package",
            "outputs/handoff/facenet_pgd_png_package.zip",
            "--epsilons 0.005,0.010",
        ),
        (
            "fgsm",
            "outputs/handoff/facenet_fgsm_package",
            "outputs/handoff/facenet_fgsm_package.zip",
            "--epsilons ALL",
        ),
        (
            "pgd_adv_training",
            "outputs/handoff/facenet_adv_training_package",
            "outputs/handoff/facenet_adv_training_package.zip",
            "--epsilons ALL",
        ),
    ]
    for subdir, out_dir, zip_out, eps_arg in handoff_cmds:
        sh(
            f"{PY} -m src.verification.build_verification_attack_handoff "
            f"--metadata-root outputs/verification_attacks_facenet/{subdir} "
            f"--verification-metrics {metrics} "
            f"--attack-summary outputs/verification_attacks_facenet/{subdir}/summary.csv "
            f"{eps_arg} --successful-only "
            f"--out-dir {out_dir} --zip-out {zip_out}",
            cwd=REPO,
        )


def copy_outputs() -> None:
    for dirname in ["handoff", "summaries", "run_log"]:
        (OUT / dirname).mkdir(parents=True, exist_ok=True)

    for zip_path in (REPO / "outputs/handoff").glob("*.zip"):
        shutil.copy(zip_path, OUT / "handoff" / zip_path.name)

    for summary in (REPO / "outputs/verification_attacks_facenet").glob("*/summary.csv"):
        shutil.copy(summary, OUT / "summaries" / f"{summary.parent.name}_summary.csv")

    if (REPO / "outputs/run_log").exists():
        shutil.copytree(REPO / "outputs/run_log", OUT / "run_log", dirs_exist_ok=True)

    print("=== /kaggle/working deliverables ===", flush=True)
    sh("find /kaggle/working -maxdepth 2 -type f | sort")


def main() -> None:
    stage_repo()
    restore_data()
    run_attacks()
    summarize_and_package()
    copy_outputs()


if __name__ == "__main__":
    main()
