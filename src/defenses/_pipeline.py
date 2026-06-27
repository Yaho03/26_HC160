"""Shared runner for the preprocessing-style defense scripts.

Each defense_*.py used to repeat the same ~150 lines: load attack_index.csv,
filter rows, load the classifier, loop over rows applying a transform and
re-running inference, write a result CSV, then print summary stats. This
module factors that out so each defense script only needs to supply the
per-image transform and its own CLI parameters.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from tqdm import tqdm

from src.common.device import get_device


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def load_filtered_attack_index(
    attack_index_path: Path,
    only_success_on_clean: bool,
    attack_family: str,
    limit: int,
    error_label: str,
) -> pd.DataFrame:
    attack_index = pd.read_csv(attack_index_path)
    if only_success_on_clean:
        attack_index = attack_index[
            bool_series(attack_index["clean_correct"]) & bool_series(attack_index["success_on_clean"])
        ]
    if attack_family:
        attack_index = attack_index[attack_index["attack_family"] == attack_family]
    if limit > 0:
        attack_index = attack_index.head(limit)
    if attack_index.empty:
        raise ValueError(f"No attack rows selected for {error_label}")
    return attack_index


def run_defense_pipeline(
    *,
    attack_index_path: Path,
    checkpoint_path: Path,
    out_dir: Path,
    defense_name: str,
    defense_params: dict[str, object],
    transform: Callable[[Image.Image], Image.Image],
    image_out_dir: Path,
    image_filename: Callable[[object], str],
    result_path: Path,
    only_success_on_clean: bool,
    attack_family: str,
    limit: int,
    progress_desc: str,
    save_format: str | None = None,
    save_kwargs: dict[str, object] | None = None,
) -> Path:
    """Run a single preprocessing defense over the filtered attack index and
    write a result CSV. Returns the path to that CSV."""
    device = get_device()
    ckpt = torch.load(checkpoint_path, map_location=device)
    classes = ckpt["classes"]
    model = build_model(len(classes)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    attack_index = load_filtered_attack_index(
        attack_index_path, only_success_on_clean, attack_family, limit, defense_name
    )

    image_out_dir.mkdir(parents=True, exist_ok=True)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    to_tensor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    def model_input(pixel_tensor: torch.Tensor) -> torch.Tensor:
        return (pixel_tensor - mean) / std

    params_json = json.dumps(defense_params)
    save_kwargs = save_kwargs or {}
    rows: list[dict[str, object]] = []

    for _, attack in tqdm(attack_index.iterrows(), total=len(attack_index), desc=progress_desc):
        adv_path = Path(str(attack["adv_file"]))
        if not adv_path.exists():
            rows.append({
                "sample_id": attack["sample_id"],
                "attack_family": attack["attack_family"],
                "attack": attack["attack"],
                "defense": defense_name,
                "defense_params": params_json,
                "input_adv_file": str(adv_path),
                "defended_file": "",
                "pred_before_defense": attack.get("pred_after", ""),
                "pred_after_defense": "",
                "pred_after_defense_name": "",
                "target_label": attack["target_label"],
                "true_label": attack["true_label"],
                "attack_success_before_defense": attack["success_on_clean"],
                "attack_success_after_defense": "",
                "recovered": "",
                "target_conf_before_defense": attack.get("target_conf_after", ""),
                "target_conf_after_defense": "",
                "true_conf_after_defense": "",
                "defense_time_sec": "",
                "status": "missing_adv_file",
            })
            continue

        start = time.perf_counter()
        image = Image.open(adv_path).convert("RGB")
        defended = transform(image)
        defended_path = image_out_dir / image_filename(attack["sample_id"])
        if save_format:
            defended.save(defended_path, format=save_format, **save_kwargs)
        else:
            defended.save(defended_path)

        tensor = to_tensor(defended).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = F.softmax(model(model_input(tensor)), dim=1)
            pred_after_defense = int(probs.argmax(dim=1).item())
            target_label = int(attack["target_label"])
            true_label = int(attack["true_label"])
            target_conf_after = float(probs[0, target_label].cpu())
            true_conf_after = float(probs[0, true_label].cpu())
        elapsed = time.perf_counter() - start

        attack_success_after = pred_after_defense == target_label
        recovered = pred_after_defense == true_label

        rows.append({
            "sample_id": attack["sample_id"],
            "attack_family": attack["attack_family"],
            "attack": attack["attack"],
            "defense": defense_name,
            "defense_params": params_json,
            "input_adv_file": str(adv_path),
            "defended_file": str(defended_path),
            "pred_before_defense": attack.get("pred_after", ""),
            "pred_after_defense": pred_after_defense,
            "pred_after_defense_name": classes[pred_after_defense],
            "target_label": target_label,
            "true_label": true_label,
            "attack_success_before_defense": attack["success_on_clean"],
            "attack_success_after_defense": attack_success_after,
            "recovered": recovered,
            "target_conf_before_defense": attack.get("target_conf_after", ""),
            "target_conf_after_defense": target_conf_after,
            "true_conf_after_defense": true_conf_after,
            "defense_time_sec": elapsed,
            "status": "ok",
        })

    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    _print_summary(rows, result_path)
    return result_path


def _print_summary(rows: list[dict[str, object]], result_path: Path) -> None:
    ok_rows = [row for row in rows if row["status"] == "ok"]
    if ok_rows:
        before_success = [row for row in ok_rows if str(row["attack_success_before_defense"]).lower() in {"true", "1"}]
        after_success_count = sum(bool(row["attack_success_after_defense"]) for row in before_success)
        recovered_count = sum(bool(row["recovered"]) for row in before_success)
        defense_success_rate = 1 - (after_success_count / len(before_success)) if before_success else 0.0
        recovery_rate = recovered_count / len(before_success) if before_success else 0.0
        avg_target_drop = sum(
            float(row["target_conf_before_defense"]) - float(row["target_conf_after_defense"])
            for row in ok_rows
        ) / len(ok_rows)
        avg_time = sum(float(row["defense_time_sec"]) for row in ok_rows) / len(ok_rows)
        print(f"Rows: {len(rows)} ok={len(ok_rows)}")
        print(f"Defense success rate: {defense_success_rate:.2%}")
        print(f"Recovery rate: {recovery_rate:.2%}")
        print(f"Avg target confidence drop: {avg_target_drop:.4f}")
        print(f"Avg defense time: {avg_time:.4f}s")
    else:
        print(f"Rows: {len(rows)} ok=0")
    print(f"Saved: {result_path.resolve()}")
