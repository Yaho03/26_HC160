"""Build FaceAuth attack-forensics session logs from verification attack artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import hashlib
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


DEFAULT_HANDOFF_INDEX = Path(
    "tmp_verification_defense_latest/verification_defense/attack_handoff_jpeg_index.csv"
)
DEFAULT_DEFENSE_FILES = [
    Path("tmp_verification_defense_latest/verification_defense/jpeg/verification_defense_jpeg.csv"),
    Path("tmp_verification_defense_latest/verification_defense/bitdepth/verification_defense_bitdepth.csv"),
    Path("tmp_verification_defense_latest/verification_defense/smoothing/verification_defense_smoothing.csv"),
]
DEFAULT_OUT_DIR = Path("outputs/forensics")

RULES = [
    {
        "id": "FA-R001",
        "name": "Threshold Margin Spike",
        "severity": "high",
        "description": "Attack similarity is comfortably above the verification threshold.",
    },
    {
        "id": "FA-R002",
        "name": "Borderline Repeated Attempts",
        "severity": "medium",
        "description": "Attack similarity lands near the threshold, which needs repeated-attempt monitoring.",
    },
    {
        "id": "FA-R003",
        "name": "High Query Black-box Pattern",
        "severity": "medium",
        "description": "High query count suggests black-box probing such as Square/ZOO.",
    },
    {
        "id": "FA-R004",
        "name": "Strong Defense Bypass",
        "severity": "critical",
        "description": "A strong defense such as smoothing still accepts the adversarial attempt.",
    },
    {
        "id": "FA-R005",
        "name": "Multi-target Source",
        "severity": "medium",
        "description": "The same source identity appears against multiple target identities.",
    },
    {
        "id": "FA-R006",
        "name": "High-risk Target Account",
        "severity": "medium",
        "description": "The same target identity receives many accepted attack attempts.",
    },
    {
        "id": "FA-R007",
        "name": "Look-alike Hard Negative",
        "severity": "medium",
        "description": "The clean source-target similarity is already high for a negative pair.",
    },
    {
        "id": "FA-R008",
        "name": "Low-norm Successful Attack",
        "severity": "high",
        "description": "The attack succeeds with a small perturbation norm.",
    },
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: Any, default: float | None = None) -> float | None:
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def required_float(row: dict[str, str], field: str) -> float:
    value = parse_float(row.get(field))
    if value is None:
        sample_id = row.get("sample_id", "<unknown>")
        raise ValueError(f"{sample_id}: required numeric field is missing or invalid: {field}")
    return value


def attack_family(attack: str) -> str:
    attack_lower = attack.lower()
    if "adaptive" in attack_lower:
        return "adaptive"
    if "fgsm" in attack_lower:
        return "fgsm"
    if "square" in attack_lower:
        return "square"
    if "zoo" in attack_lower:
        return "zoo"
    if "pgd" in attack_lower:
        return "pgd"
    return attack_lower or "unknown"


def stable_sample_id(row: dict[str, str]) -> str:
    base = "|".join([
        row.get("pair_id", ""),
        row.get("attack", ""),
        row.get("epsilon", ""),
        row.get("steps", ""),
        row.get("adv_file", ""),
    ])
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]
    return f"vf_{digest}"


def normalize_attack_row(row: dict[str, str]) -> dict[str, str]:
    """Normalize metadata CSV rows and handoff-index rows to one forensics shape."""
    normalized = dict(row)
    normalized.setdefault("sample_id", "")
    if not normalized["sample_id"]:
        normalized["sample_id"] = stable_sample_id(normalized)

    if "similarity_after_attack" not in normalized and "similarity_after" in normalized:
        normalized["similarity_after_attack"] = normalized["similarity_after"]
    if "attack_success_before_defense" not in normalized:
        normalized["attack_success_before_defense"] = normalized.get("attack_success", "")
    if "accepted_after_attack" not in normalized and "accepted_after" in normalized:
        normalized["accepted_after_attack"] = normalized["accepted_after"]
    if "target_enroll_file" not in normalized and "target_file" in normalized:
        normalized["target_enroll_file"] = normalized["target_file"]
    if "source_name" not in normalized and "source_label" in normalized:
        normalized["source_name"] = normalized["source_label"]
    if "target_name" not in normalized and "target_label" in normalized:
        normalized["target_name"] = normalized["target_label"]
    return normalized


def read_attack_rows(handoff_index: Path, metadata_roots: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if handoff_index.exists():
        rows.extend(read_csv(handoff_index))

    for root in metadata_roots:
        if not root.exists():
            continue
        if root.is_file():
            paths = [root]
        else:
            paths = sorted(root.rglob("metadata_*.csv"))
        for path in paths:
            for row in read_csv(path):
                row["source_metadata"] = str(path)
                rows.append(row)

    return [normalize_attack_row(row) for row in rows]


def risk_level(score: int) -> str:
    if score >= 80:
        return "critical"
    if score >= 60:
        return "high"
    if score >= 30:
        return "medium"
    return "low"


def load_defense_rows(paths: list[Path]) -> dict[str, list[dict[str, str]]]:
    by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for path in paths:
        for row in read_csv(path):
            row["source_defense_file"] = str(path)
            by_sample[row["sample_id"]].append(row)
    return dict(by_sample)


def strongest_defense_state(defense_rows: list[dict[str, str]]) -> dict[str, Any]:
    if not defense_rows:
        return {
            "defense": "",
            "bypassed_defenses": "",
            "accepted_after_defense": False,
            "defense_bypassed": False,
            "strong_defense_bypassed": False,
            "similarity_after_defense": None,
            "defense_success": False,
            "defense_time_sec": None,
        }

    bypassed = [row for row in defense_rows if parse_bool(row.get("accepted_after_defense"))]
    bypassed_names = sorted({row.get("defense", "") for row in bypassed if row.get("defense")})
    strong_defense_bypassed = any(name in {"smoothing", "dae", "diffpure"} for name in bypassed_names)
    def defense_similarity(row: dict[str, str]) -> float:
        value = parse_float(row.get("similarity_after_defense"))
        return value if value is not None else -1.0

    selected = bypassed[0] if bypassed else max(defense_rows, key=defense_similarity)
    return {
        "defense": selected.get("defense", ""),
        "bypassed_defenses": ";".join(bypassed_names),
        "accepted_after_defense": parse_bool(selected.get("accepted_after_defense")),
        "defense_bypassed": bool(bypassed),
        "strong_defense_bypassed": strong_defense_bypassed,
        "similarity_after_defense": parse_float(selected.get("similarity_after_defense")),
        "defense_success": parse_bool(selected.get("defense_success")),
        "defense_time_sec": parse_float(selected.get("defense_time_sec")),
    }


def build_context(rows: list[dict[str, str]]) -> dict[str, Any]:
    source_targets: dict[str, set[str]] = defaultdict(set)
    target_accept_counts: Counter[str] = Counter()
    pair_attempt_counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        source = row.get("source_name", "")
        target = row.get("target_name", "")
        if source and target:
            source_targets[source].add(target)
            pair_attempt_counts[(source, target)] += 1
        if target and parse_bool(row.get("accepted_after_attack")):
            target_accept_counts[target] += 1

    high_risk_targets = {
        target for target, count in target_accept_counts.items() if count >= 10
    }
    multi_target_sources = {
        source for source, targets in source_targets.items() if len(targets) >= 2
    }
    return {
        "high_risk_targets": high_risk_targets,
        "multi_target_sources": multi_target_sources,
        "target_accept_counts": target_accept_counts,
        "pair_attempt_counts": pair_attempt_counts,
    }


def rule_findings(
    row: dict[str, str],
    defense_state: dict[str, Any],
    context: dict[str, Any],
    threshold_margin: float,
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    accepted = parse_bool(row.get("accepted_after_attack"))
    similarity_before = parse_float(row.get("similarity_before"))
    l2 = parse_float(row.get("l2"))
    linf = parse_float(row.get("linf"))
    queries = parse_float(row.get("queries_used") or row.get("max_queries"))
    source = row.get("source_name", "")
    target = row.get("target_name", "")
    repeated_attempts = context["pair_attempt_counts"].get((source, target), 0)

    def add(rule_id: str, reason: str) -> None:
        findings.append({"rule_id": rule_id, "reason": reason})

    if accepted and threshold_margin >= 0.05:
        add("FA-R001", f"accepted=true, threshold_margin={threshold_margin:.6f} >= 0.050000")
    if repeated_attempts >= 2 and -0.03 <= threshold_margin <= 0.03:
        add(
            "FA-R002",
            f"source-target attempts={repeated_attempts}, threshold_margin={threshold_margin:.6f} within ±0.030000",
        )
    if queries is not None and queries >= 100:
        add("FA-R003", f"queries_used={queries:.0f} >= 100")
    if defense_state["strong_defense_bypassed"]:
        add("FA-R004", f"strong defenses bypassed={defense_state['bypassed_defenses']}")
    if source and source in context["multi_target_sources"]:
        add("FA-R005", f"source={source} targeted multiple identities")
    if target and target in context["high_risk_targets"]:
        count = context["target_accept_counts"][target]
        add("FA-R006", f"target={target} accepted attack attempts={count} >= 10")
    if similarity_before is not None and similarity_before >= 0.30:
        add("FA-R007", f"negative-pair similarity_before={similarity_before:.6f} >= 0.300000")
    low_l2 = l2 is not None and l2 <= 1.25
    low_linf = linf is not None and linf <= 0.0051
    if accepted and (low_l2 or low_linf):
        add("FA-R008", f"accepted=true, l2={l2}, linf={linf}; low-norm boundary met")
    return findings


def rule_hits(
    row: dict[str, str],
    defense_state: dict[str, Any],
    context: dict[str, Any],
    threshold_margin: float,
) -> list[str]:
    return [finding["rule_id"] for finding in rule_findings(row, defense_state, context, threshold_margin)]


def score_from_hits(
    hits: list[str],
    accepted: bool,
    defense_bypassed: bool,
    strong_defense_bypassed: bool,
) -> int:
    score = 0
    if accepted:
        score += 30
    if defense_bypassed:
        score += 10
    if strong_defense_bypassed:
        score += 20
    weights = {
        "FA-R001": 15,
        "FA-R002": 8,
        "FA-R003": 10,
        "FA-R004": 15,
        "FA-R005": 10,
        "FA-R006": 10,
        "FA-R007": 5,
        "FA-R008": 10,
    }
    score += sum(weights.get(hit, 0) for hit in hits)
    return min(score, 100)


def build_sessions(
    handoff_rows: list[dict[str, str]],
    defense_by_sample: dict[str, list[dict[str, str]]],
    start_time: datetime,
) -> list[dict[str, Any]]:
    context = build_context(handoff_rows)
    sessions = []
    for idx, row in enumerate(handoff_rows, start=1):
        sample_id = row["sample_id"]
        threshold = required_float(row, "threshold")
        similarity_after = required_float(row, "similarity_after_attack")
        threshold_margin = similarity_after - threshold
        defense_state = strongest_defense_state(defense_by_sample.get(sample_id, []))
        findings = rule_findings(row, defense_state, context, threshold_margin)
        hits = [finding["rule_id"] for finding in findings]
        accepted = parse_bool(row.get("accepted_after_attack"))
        risk_score = score_from_hits(
            hits,
            accepted,
            defense_state["defense_bypassed"],
            defense_state["strong_defense_bypassed"],
        )
        family = attack_family(row.get("attack", ""))
        timestamp = start_time + timedelta(seconds=idx * 37)

        sessions.append({
            "session_id": f"faceauth_sess_{idx:06d}",
            "attempt_id": sample_id,
            "timestamp": timestamp.isoformat(),
            "account_id": f"acct_{row.get('target_name', 'unknown')}",
            "source_identity": row.get("source_name", ""),
            "target_identity": row.get("target_name", ""),
            "pair_id": row.get("pair_id", ""),
            "attack": row.get("attack", ""),
            "attack_family": family,
            "is_adaptive": family == "adaptive",
            "epsilon": row.get("epsilon", ""),
            "alpha": row.get("alpha", ""),
            "steps": row.get("steps", ""),
            "queries_used": row.get("queries_used", ""),
            "similarity_before": row.get("similarity_before", ""),
            "similarity_after_attack": row.get("similarity_after_attack", ""),
            "similarity_gain": row.get("similarity_gain", ""),
            "threshold": row.get("threshold", ""),
            "threshold_margin": round(threshold_margin, 6),
            "accepted_before": parse_bool(row.get("accepted_before")),
            "accepted_after_attack": accepted,
            "attack_success_before_defense": parse_bool(row.get("attack_success_before_defense")),
            "defense": defense_state["defense"],
            "bypassed_defenses": defense_state["bypassed_defenses"],
            "similarity_after_defense": "" if defense_state["similarity_after_defense"] is None else round(defense_state["similarity_after_defense"], 6),
            "accepted_after_defense": defense_state["accepted_after_defense"],
            "defense_bypassed": defense_state["defense_bypassed"],
            "strong_defense_bypassed": defense_state["strong_defense_bypassed"],
            "defense_success": defense_state["defense_success"],
            "defense_time_sec": "" if defense_state["defense_time_sec"] is None else defense_state["defense_time_sec"],
            "l2": row.get("l2", ""),
            "linf": row.get("linf", ""),
            "time_sec": row.get("time_sec", ""),
            "risk_score": risk_score,
            "risk_level": risk_level(risk_score),
            "rule_hits": ";".join(hits),
            "rule_reasons": json.dumps(findings, ensure_ascii=False, sort_keys=True),
            "source_file": row.get("source_file", ""),
            "target_enroll_file": row.get("target_enroll_file", ""),
            "adv_file": row.get("adv_file", ""),
            "perturbation_file": row.get("perturbation_file", ""),
        })
    return sessions


def build_summary(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = len(sessions)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sessions:
        by_family[str(row["attack_family"])].append(row)

    rows = []
    for family, group in sorted(by_family.items()):
        rows.append({
            "attack_family": family,
            "sessions": len(group),
            "accepted_after_attack": sum(bool(row["accepted_after_attack"]) for row in group),
            "attack_accept_rate": round(sum(bool(row["accepted_after_attack"]) for row in group) / len(group), 6),
            "defense_bypassed": sum(bool(row["defense_bypassed"]) for row in group),
            "defense_bypass_rate": round(sum(bool(row["defense_bypassed"]) for row in group) / len(group), 6),
            "critical_or_high": sum(row["risk_level"] in {"critical", "high"} for row in group),
            "avg_risk_score": round(sum(int(row["risk_score"]) for row in group) / len(group), 3),
        })

    rows.insert(0, {
        "attack_family": "ALL",
        "sessions": total,
        "accepted_after_attack": sum(bool(row["accepted_after_attack"]) for row in sessions),
        "attack_accept_rate": round(sum(bool(row["accepted_after_attack"]) for row in sessions) / total, 6),
        "defense_bypassed": sum(bool(row["defense_bypassed"]) for row in sessions),
        "defense_bypass_rate": round(sum(bool(row["defense_bypassed"]) for row in sessions) / total, 6),
        "critical_or_high": sum(row["risk_level"] in {"critical", "high"} for row in sessions),
        "avg_risk_score": round(sum(int(row["risk_score"]) for row in sessions) / total, 3),
    })
    return rows


def write_rules(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "domain": "financial_faceauth_attack_forensics",
        "rules": RULES,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-index", type=Path, default=DEFAULT_HANDOFF_INDEX)
    parser.add_argument(
        "--metadata-roots",
        type=Path,
        nargs="*",
        default=[],
        help="Optional verification attack metadata roots/files. Use for Kaggle attack sweeps before defense results exist.",
    )
    parser.add_argument("--defense-files", type=Path, nargs="*", default=DEFAULT_DEFENSE_FILES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--start-time", default="2026-06-28T09:00:00+09:00")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    handoff_rows = read_attack_rows(args.handoff_index, args.metadata_roots)
    if not handoff_rows:
        raise FileNotFoundError(
            f"No attack rows found. Checked handoff={args.handoff_index} metadata_roots={args.metadata_roots}"
        )

    defense_by_sample = load_defense_rows(args.defense_files)
    start_time = datetime.fromisoformat(args.start_time)
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)

    sessions = build_sessions(handoff_rows, defense_by_sample, start_time)
    summary = build_summary(sessions)

    write_csv(sessions, args.out_dir / "attack_sessions.csv")
    write_csv(summary, args.out_dir / "attack_risk_summary.csv")
    write_rules(args.out_dir / "attack_detection_rules.json")

    print(f"Sessions: {len(sessions)}")
    print(f"Saved: {args.out_dir / 'attack_sessions.csv'}")
    print(f"Saved: {args.out_dir / 'attack_risk_summary.csv'}")
    print(f"Saved: {args.out_dir / 'attack_detection_rules.json'}")


if __name__ == "__main__":
    main()
