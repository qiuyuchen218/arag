#!/usr/bin/env python3
"""Prepare Stage 2.7 generalization audit artifacts without running b1."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


ONLINE_FORBIDDEN_KEYS = {
    "gold_answer", "answer", "answers", "aliases", "answer_aliases", "supporting_facts",
    "supporting_paragraphs", "decomposition", "correct", "correctness", "llm_accuracy",
    "contain_accuracy", "failure_label", "failure_type_label", "status",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_hash(path: Path) -> str:
    return _sha256_bytes(path.read_bytes()) if path.exists() else ""


def _json_hash(value: Any) -> str:
    return _sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8"))


def _run(cmd: List[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=Path(__file__).resolve().parents[1], text=True).strip()
    except Exception:
        return ""


def _source_tree_hash(repo: Path) -> str:
    files = []
    for root in ["src", "scripts", "tests", "configs"]:
        base = repo / root
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                files.append((str(path.relative_to(repo)), _file_hash(path)))
    return _json_hash(files)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _strip_online_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in row.items() if k not in ONLINE_FORBIDDEN_KEYS}


def _select_cohorts(
    all_rows: List[Dict[str, Any]],
    wrong_rows: List[Dict[str, Any]],
    wrong_size: int,
    correct_size: int,
    seed: int,
    exclude_ids: set,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    wrong_ids = {r.get("qid") or r.get("id") for r in wrong_rows}
    wrong_pool = [r for r in wrong_rows if (r.get("qid") or r.get("id")) not in exclude_ids]
    correct_pool = [
        r for r in all_rows
        if (r.get("qid") or r.get("id")) not in wrong_ids
        and (r.get("qid") or r.get("id")) not in exclude_ids
        and _looks_correct(r)
    ]
    return (
        rng.sample(wrong_pool, min(wrong_size, len(wrong_pool))),
        rng.sample(correct_pool, min(correct_size, len(correct_pool))),
    )


def _looks_correct(row: Dict[str, Any]) -> bool:
    if row.get("llm_accuracy") is True or row.get("contain_accuracy") is True:
        return True
    gold = str(row.get("gold_answer", "") or "").strip().lower()
    pred = str(row.get("pred_answer", "") or "").strip().lower()
    return bool(gold and gold in pred)


def _cohort_manifest(rows: List[Dict[str, Any]], label: str, input_path: Path, seed: int) -> Dict[str, Any]:
    return {
        "cohort_label_for_offline_eval": label,
        "seed": seed,
        "sample_count": len(rows),
        "sample_ids": [r.get("qid") or r.get("id") for r in rows],
        "input_file": str(input_path),
        "input_file_hash": _file_hash(input_path),
        "online_fields_removed": sorted(ONLINE_FORBIDDEN_KEYS),
        "gold_available_to_online_pipeline": False,
    }


def _manual_audit_rows(wrong: List[Dict[str, Any]], correct: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    for label, pool in [("historical_wrong", wrong), ("historical_correct", correct)]:
        for row in rng.sample(pool, min(5, len(pool))):
            rows.append({
                "sample_id": row.get("qid") or row.get("id"),
                "cohort": label,
                "question": row.get("question"),
                "root_top1_acceptable": None,
                "root_top3_contains_acceptable": None,
                "rollback_acceptable": None,
                "failure_type_acceptable": None,
                "repair_plan_acceptable": None,
                "correct_sample_false_trigger": None,
                "notes": "",
            })
    return rows


def _readiness(metrics: Dict[str, Any] = None, api_blocked: bool = False) -> Dict[str, Any]:
    metrics = metrics or {}
    hard_ok = (
        metrics.get("evidence_isolation_valid_rate") == 1.0
        and metrics.get("temporal_edge_valid_rate") == 1.0
        and metrics.get("posthoc_root_rate") == 0.0
        and metrics.get("invalid_inheritance_rate", 0.0) == 0.0
        and metrics.get("gold_leakage_rate", 0.0) == 0.0
    )
    ready = bool(hard_ok and not api_blocked and metrics.get("sample_count", 0) >= 20)
    return {
        "repair_readiness": "ready_for_limited_b1_pilot" if ready else "not_ready",
        "api_blocked": api_blocked,
        "repair_dry_run": True,
        "true_b1_executed": False,
        "gate_summary": {
            "hard_gates_passed": hard_ok,
            "requires_blind_api_eval": api_blocked,
            "minimum_blind_sample_count_met": metrics.get("sample_count", 0) >= 20,
        },
        "next_allowed_stage_if_ready": "limited_b1_pilot_5_to_10_wrong_samples_max_one_branch",
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare Stage 2.7 audit artifacts.")
    parser.add_argument("--all-predictions", default="results/baseline_qwen3max/musique/predictions.jsonl")
    parser.add_argument("--wrong-cases", default="results/baseline_qwen3max/musique/wrong_cases.jsonl")
    parser.add_argument("--output-dir", default="results/stage27_generalization_audit")
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--wrong-size", type=int, default=20)
    parser.add_argument("--correct-size", type=int, default=20)
    parser.add_argument("--exclude-sample-id", action="append", default=[
        "musique_2hop__511454_120259",
        "musique_4hop2__71753_648517_70784_79935",
    ])
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    out = repo / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    all_path = repo / args.all_predictions
    wrong_path = repo / args.wrong_cases
    all_rows = _read_jsonl(all_path)
    wrong_rows = _read_jsonl(wrong_path)
    wrong, correct = _select_cohorts(
        all_rows,
        wrong_rows,
        args.wrong_size,
        args.correct_size,
        args.seed,
        set(args.exclude_sample_id or []),
    )
    wrong_input = out / "blind_wrong_input.jsonl"
    correct_input = out / "blind_correct_input.jsonl"
    _write_jsonl(wrong_input, [_strip_online_fields(r) for r in wrong])
    _write_jsonl(correct_input, [_strip_online_fields(r) for r in correct])

    wrong_manifest = _cohort_manifest(wrong, "historical_wrong", wrong_input, args.seed)
    correct_manifest = _cohort_manifest(correct, "historical_correct", correct_input, args.seed)
    (out / "blind_wrong_manifest.json").write_text(json.dumps(wrong_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "blind_correct_manifest.json").write_text(json.dumps(correct_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_jsonl(out / "manual_audit_template.jsonl", _manual_audit_rows(wrong, correct, args.seed))

    config_path = repo / "configs/example.yaml"
    manifest = {
        "experiment_id": f"stage27_generalization_audit_seed{args.seed}",
        "created_at": _utc_now(),
        "git_status_short": _run(["git", "status", "--short"]),
        "git_diff_stat": _run(["git", "diff", "--stat"]),
        "commit_sha": _run(["git", "rev-parse", "HEAD"]),
        "source_tree_hash": _source_tree_hash(repo),
        "config_hash": _file_hash(config_path),
        "input_file_hash": {
            "all_predictions": _file_hash(all_path),
            "wrong_cases": _file_hash(wrong_path),
            "blind_wrong_input": _file_hash(wrong_input),
            "blind_correct_input": _file_hash(correct_input),
        },
        "seed": args.seed,
        "sample_selection_strategy": "fixed-seed historical wrong plus historical correct excluding development qids",
        "sample_ids": {
            "blind_wrong": wrong_manifest["sample_ids"],
            "blind_correct": correct_manifest["sample_ids"],
        },
        "cohort_labels_for_offline_eval": ["historical_wrong", "historical_correct"],
        "external_api_enabled": False,
        "repair_dry_run": True,
        "gold_available_to_online_pipeline": False,
        "model_configuration_hash": _file_hash(config_path),
        "corpus_index_version": "recorded_at_runtime_by_tools",
        "verifier_prompt_hash": "recorded_at_runtime_by_verifier",
        "question_planner_version": "heuristic_question_decomposer:v1",
        "query_intent_parser_version": "heuristic_query_intent:v1",
        "hypothesis_extractor_version": "online_hypothesis_tracker:v1",
        "blame_algorithm_version": "estimated_cognitive_blame:v2.7",
    }
    (out / "experiment_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    readiness = _readiness(api_blocked=True)
    (out / "repair_readiness.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    report = (
        "# Stage 2.7 Generalization Audit\n\n"
        "Offline audit artifacts were generated without external API calls.\n\n"
        "- repair.dry_run: true\n"
        "- true b1 executed: false\n"
        "- external API blind evaluation: blocked pending explicit approval\n"
        f"- blind wrong input: `{wrong_input}`\n"
        f"- blind correct input: `{correct_input}`\n"
    )
    (out / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "output_dir": str(out),
        "wrong_count": len(wrong),
        "correct_count": len(correct),
        "repair_readiness": readiness["repair_readiness"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
