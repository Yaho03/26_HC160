"""Attack-team analysis: how well do adversarial samples survive a defense transform?

Red-team view of the defense. For every adversarial image produced by a
verification attack, optionally apply an input-transform defense (Gaussian
smoothing, matching the defense team's r=3 setting), recompute the FaceNet
similarity against the target enrollment, and check whether it still gets
accepted (>= threshold).

Reported per metadata file (i.e. per epsilon):
  raw_success      - adv accepted as target with NO defense (file-reloaded truth)
  defended_accept  - adv still accepted as target AFTER the defense
  bypass_rate      - defended_accept / raw_success  (attack robustness)
  defense_success  - 1 - bypass_rate                (defense effectiveness)

This lets us compare plain PGD vs adaptive PGD against the same smoothing
defense the defense team reported (~78.8% defense success on plain PGD).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image

from src.verification.facenet_utils import (
    build_facenet_model,
    cosine_score,
    facenet_embedding,
    facenet_pixel_transform,
    load_facenet_image,
)
from src.defenses.defense_smoothing import gaussian_smooth


def load_threshold(metrics_path: Path | None, override: float | None) -> float:
    if override is not None:
        return override
    if metrics_path and metrics_path.exists():
        data = json.loads(metrics_path.read_text())
        return float(data["eer_threshold"])
    raise SystemExit("threshold unknown: pass --threshold or a valid --metrics file")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metadata-root", type=Path, required=True,
                   help="dir containing metadata_*.csv from an attack run")
    p.add_argument("--metrics", type=Path,
                   default=Path("outputs/verification_facenet/verification_metrics.json"))
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--pretrained", default="vggface2")
    p.add_argument("--defense", default="smoothing", choices=["none", "smoothing"])
    p.add_argument("--radius", type=float, default=3.0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    threshold = load_threshold(args.metrics, args.threshold)
    model, device = build_facenet_model(args.pretrained)
    to_pixel = facenet_pixel_transform()

    def embed_pil(img: Image.Image) -> torch.Tensor:
        px = to_pixel(img.convert("RGB")).unsqueeze(0).to(device)
        return facenet_embedding(model, px)

    rows_out = []
    meta_files = sorted(args.metadata_root.glob("metadata_*.csv"))
    if not meta_files:
        raise SystemExit(f"no metadata_*.csv under {args.metadata_root}")

    for mf in meta_files:
        with mf.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        raw_success = 0
        defended_accept = 0
        sim_raw_sum = 0.0
        sim_def_sum = 0.0
        n_eval = 0
        for r in rows:
            adv_path = Path(r["adv_file"])
            tgt_path = Path(r["target_enroll_file"])
            if not adv_path.exists() or not tgt_path.exists():
                continue
            with torch.no_grad():
                tgt_emb = facenet_embedding(model, load_facenet_image(tgt_path, to_pixel, device))
                adv_img = Image.open(adv_path).convert("RGB")
                sim_raw = cosine_score(embed_pil(adv_img), tgt_emb)
                if args.defense == "smoothing":
                    def_img = gaussian_smooth(adv_img, args.radius)
                else:
                    def_img = adv_img
                sim_def = cosine_score(embed_pil(def_img), tgt_emb)
            n_eval += 1
            sim_raw_sum += sim_raw
            sim_def_sum += sim_def
            if sim_raw >= threshold:
                raw_success += 1
                if sim_def >= threshold:
                    defended_accept += 1
        bypass = defended_accept / raw_success if raw_success else 0.0
        rows_out.append({
            "metadata_file": str(mf),
            "defense": args.defense,
            "radius": args.radius if args.defense == "smoothing" else "",
            "threshold": round(threshold, 6),
            "evaluated": n_eval,
            "raw_success": raw_success,
            "defended_accept": defended_accept,
            "bypass_rate": round(bypass, 4),
            "defense_success": round(1 - bypass, 4),
            "avg_sim_raw": round(sim_raw_sum / n_eval, 6) if n_eval else "",
            "avg_sim_defended": round(sim_def_sum / n_eval, 6) if n_eval else "",
        })
        print(f"{mf.name}: raw_success={raw_success} defended_accept={defended_accept} "
              f"bypass={bypass:.1%} defense_success={1-bypass:.1%}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
