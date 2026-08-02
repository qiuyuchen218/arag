#!/usr/bin/env python3
"""Diagnosis-only aggregate checks for ARAG Stage 2.7 audit outputs."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean, variance
from typing import Any, Dict, List


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_trace(row: Dict[str, Any]) -> Dict[str, Any]:
    trace_path = row.get("trace_path")
    if not trace_path:
        return {}
    path = Path(trace_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _rate(values: List[bool]) -> Dict[str, Any]:
    numerator = sum(1 for v in values if v)
    denominator = len(values)
    return {
        "value": (numerator / denominator) if denominator else None,
        "numerator": numerator,
        "denominator": denominator,
        "reason_if_undefined": None if denominator else "empty_denominator",
    }


def _metric_for_row(row: Dict[str, Any], with_gold_eval: bool) -> Dict[str, Any]:
    trace = _load_trace(row)
    metadata = trace.get("metadata", {}) or {}
    repair_plan = row.get("repair_plan") or metadata.get("repair_plan") or {}
    blame = row.get("blame_results") or metadata.get("blame_results") or []
    matrix = row.get("candidate_constraint_matrix") or metadata.get("candidate_constraint_matrix") or {}
    query_intents = row.get("query_intents") or metadata.get("query_intents") or []
    claims = row.get("claim_assessments") or metadata.get("claim_assessments") or []
    root = repair_plan.get("root_cause_node")
    root_node = next((n for n in trace.get("nodes", []) if n.get("id") == root), {})
    scores = [float(b.get("blame", 0.0) or 0.0) for b in blame]
    top_scores = sorted(scores, reverse=True)
    root_to_failure_flags = [
        bool(b.get("root_to_failure_contains_execution")) for b in blame
        if b.get("node_id") == root or b.get("full_causal_path")
    ]
    matrix_candidates = matrix.get("candidates", []) or []
    constraint_results = [
        cr
        for cand in matrix_candidates
        for cr in (cand.get("constraint_results") or {}).values()
        if isinstance(cr, dict)
    ]
    query_schema_valid = [_query_intent_schema_valid(qi) for qi in query_intents]
    entity_sep = [_entity_predicate_separated(qi) for qi in query_intents if qi.get("candidate_entities")]
    repair_trigger = bool(row.get("repair_eligible", metadata.get("repair_eligible", False)))
    diagnostic_status = repair_plan.get("diagnostic_status", "NO_REPAIR_NEEDED")
    leakage = _gold_leakage_audit(row, metadata)
    result = {
        "trace_valid": bool(metadata.get("trace_valid", row.get("trace_valid", False))),
        "evidence_isolation_valid": all(c.get("evidence_isolation_valid", True) for c in claims),
        "temporal_edge_valid": not any("causal_edge_temporal_invalid" in e for e in metadata.get("validation_errors", [])),
        "rollback_valid": all(b.get("rollback_valid", True) for b in blame if b.get("rollback_checkpoint")),
        "query_intent_parse_success": any((qi.get("parser_confidence", 0) or 0) >= 0.5 for qi in query_intents),
        "plan_valid": bool(row.get("question_plan") or metadata.get("question_plan")),
        "nonempty_candidate_matrix": bool(matrix.get("candidates")) or bool(query_intents),
        "duplicate_hypothesis": _has_duplicate_hypotheses(row, metadata),
        "posthoc_root": bool(root_node.get("metadata", {}).get("posthoc_summary")),
        "full_causal_path_valid": (
            all(bool(b.get("full_causal_path_valid", b.get("causal_path_valid", True))) for b in blame if b.get("node_id") == root or b.get("full_causal_path"))
            and (all(root_to_failure_flags) if root_to_failure_flags else False)
        ),
        "root_to_failure_contains_execution": all(root_to_failure_flags) if root_to_failure_flags else False,
        "repair_trigger": repair_trigger,
        "abstention": str((row.get("answer_assessment") or {}).get("completeness_status", "")).upper() in {"INCOMPLETE", "MISSING", "TARGET_UNRESOLVED"},
        "verifier_uncertain": any(c.get("status") == "VERIFIER_UNCERTAIN" for c in claims),
        "plan_uncertain": (
            "PLAN_UNCERTAIN" in row.get("failure_types", [])
            or str((row.get("answer_assessment") or {}).get("completeness_status", "")).upper() == "PLAN_UNCERTAIN"
        ),
        "average_root_candidates": len(blame),
        "blame_score_variance": variance(scores) if len(scores) > 1 else 0.0,
        "root_margin_top1_top2": (top_scores[0] - top_scores[1]) if len(top_scores) > 1 else (top_scores[0] if top_scores else 0.0),
        "num_causal_root_candidates": sum(1 for b in blame if b.get("causal_path_valid") or b.get("full_causal_path_valid")),
        "num_nonzero_blame_nodes": sum(1 for s in scores if s > 0),
        "fraction_equal_blame_candidates": _fraction_equal(scores),
        "diagnostic_uncertainty": mean([
            float((b.get("dimension_breakdown") or {}).get("diagnostic_uncertainty", 0.0) or 0.0)
            for b in blame
        ]) if blame else 0.0,
        "root_selection_confidence": float((blame[0].get("diagnostic_confidence", 0.0) if blame else 0.0) or 0.0),
        "query_intent_schema_valid": all(query_schema_valid) if query_schema_valid else False,
        "query_intent_high_confidence": any((qi.get("parser_confidence", 0) or 0) >= 0.75 for qi in query_intents),
        "query_intent_unknown": any(qi.get("query_mode") == "unknown" for qi in query_intents),
        "query_intent_llm_fallback": any(qi.get("parser_mode") == "llm" for qi in query_intents),
        "entity_predicate_separation": all(entity_sep) if entity_sep else True,
        "given_entity_misclassified": _given_entity_misclassified(row, metadata),
        "candidate_canonicalization_collision": _canonicalization_collision(row, metadata),
        "candidate_constraint_nonunknown": any(cr.get("status") not in {"UNKNOWN", "unknown", None} for cr in constraint_results),
        "candidate_constraint_supported": any(cr.get("status") in {"SATISFIED", "satisfied"} for cr in constraint_results),
        "candidate_constraint_contradicted": any(cr.get("status") in {"CONTRADICTED", "contradicted"} for cr in constraint_results),
        "commitment_allowed": any(c.get("commitment_allowed") for c in matrix_candidates),
        "premature_commitment": any(e.get("is_premature") for e in row.get("commitment_events", metadata.get("commitment_events", [])) or []),
        "local_verified_claim": any(c.get("status") == "VERIFIED" for c in claims),
        "dependency_blocked_claim": any(c.get("reasoning_status") == "DEPENDENCY_BLOCKED" or c.get("status") == "DEPENDENCY_BROKEN" for c in claims),
        "invalidated_claim": any(c.get("reasoning_status") == "INVALIDATED" for c in claims),
        "inherited_proof_bundle": bool(repair_plan.get("inherited_minimal_evidence_sets")),
        "invalid_inheritance": _has_invalid_inheritance(repair_plan),
        "average_repair_cost": mean([float(b.get("repair_cost", 0) or 0) for b in blame]) if blame else 0.0,
        "failure_types": row.get("failure_types", metadata.get("failure_types", [])),
        "diagnostic_status": diagnostic_status,
        "online_input_contains_gold": leakage["online_input_contains_gold"],
        "verifier_input_contains_gold": leakage["verifier_input_contains_gold"],
        "planner_input_contains_gold": leakage["planner_input_contains_gold"],
        "repair_input_contains_gold": leakage["repair_input_contains_gold"],
    }
    if with_gold_eval:
        result["answer_exact_match"] = str(row.get("pred_answer", "")).strip() == str(row.get("gold_answer", "")).strip()
    return result


def _fraction_equal(scores: List[float]) -> float:
    if len(scores) <= 1:
        return 0.0
    counts = Counter(round(s, 10) for s in scores)
    return max(counts.values()) / len(scores)


def _query_intent_schema_valid(qi: Dict[str, Any]) -> bool:
    required = [
        "query_intent_id", "raw_query", "normalized_query", "query_mode",
        "epistemic_action", "commitment_level", "parser_confidence",
    ]
    return all(k in qi for k in required)


def _entity_predicate_separated(qi: Dict[str, Any]) -> bool:
    predicate = str(qi.get("predicate") or "").lower()
    if not predicate:
        return True
    terms = {t for t in predicate.replace("_", " ").split() if t}
    for entity in qi.get("candidate_entities", []) or []:
        entity_terms = set(str(entity).lower().replace("_", " ").split())
        if terms & entity_terms:
            return False
    return True


def _given_entity_misclassified(row: Dict[str, Any], metadata: Dict[str, Any]) -> bool:
    plan = row.get("question_plan") or metadata.get("question_plan") or {}
    given = {
        str(v.get("value", "")).lower()
        for v in plan.get("variables", []) or []
        if v.get("binding_status") == "GIVEN_BINDING" and v.get("value")
    }
    hyps = row.get("hypothesis_assessments") or metadata.get("hypothesis_assessments") or []
    return any(str(h.get("canonical_entity") or h.get("candidate_entity", "")).lower() in given for h in hyps)


def _canonicalization_collision(row: Dict[str, Any], metadata: Dict[str, Any]) -> bool:
    hyps = row.get("hypothesis_assessments") or metadata.get("hypothesis_assessments") or []
    aliases_by_canon: Dict[str, set] = {}
    for hyp in hyps:
        canon = str(hyp.get("canonical_entity") or hyp.get("candidate_entity", "")).lower()
        if not canon:
            continue
        aliases_by_canon.setdefault(canon, set()).update(str(a).lower() for a in hyp.get("aliases", []) or [])
    return any(len(aliases) > 3 for aliases in aliases_by_canon.values())


def _has_invalid_inheritance(repair_plan: Dict[str, Any]) -> bool:
    bundles = repair_plan.get("inherited_minimal_evidence_sets", []) or []
    for bundle in bundles:
        if bundle.get("depends_on_invalidated_node"):
            return True
        if bundle.get("valid_at_rollback") is False:
            return True
        if not bundle.get("evidence_span_ids"):
            return True
    return False


def _gold_leakage_audit(row: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, bool]:
    gold = str(row.get("gold_answer", "") or row.get("answer", "") or "").strip().lower()
    if not gold:
        return {
            "online_input_contains_gold": False,
            "verifier_input_contains_gold": False,
            "planner_input_contains_gold": False,
            "repair_input_contains_gold": False,
        }

    def contains(value: Any) -> bool:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True).lower()
        return gold in text

    online_payload = {
        "question_plan": row.get("question_plan") or metadata.get("question_plan"),
        "query_intents": row.get("query_intents") or metadata.get("query_intents"),
        "hypothesis_assessments": row.get("hypothesis_assessments") or metadata.get("hypothesis_assessments"),
        "candidate_constraint_matrix": row.get("candidate_constraint_matrix") or metadata.get("candidate_constraint_matrix"),
    }
    verifier_payload = row.get("claim_assessments") or metadata.get("claim_assessments") or []
    repair_payload = row.get("repair_plan") or metadata.get("repair_plan") or {}
    return {
        "online_input_contains_gold": contains(online_payload),
        "verifier_input_contains_gold": contains(verifier_payload),
        "planner_input_contains_gold": contains(online_payload.get("question_plan")),
        "repair_input_contains_gold": contains(repair_payload),
    }


def _has_duplicate_hypotheses(row: Dict[str, Any], metadata: Dict[str, Any]) -> bool:
    hyps = row.get("hypothesis_assessments") or metadata.get("hypothesis_assessments") or []
    keys = [
        (h.get("branch_id", "b0"), h.get("target_variable"), str(h.get("canonical_entity") or h.get("candidate_entity", "")).lower())
        for h in hyps
        if h.get("canonical_entity") or h.get("candidate_entity")
    ]
    return len(keys) != len(set(keys))


def main():
    parser = argparse.ArgumentParser(description="Evaluate ARAG diagnosis traces without feeding gold into diagnosis.")
    parser.add_argument("--input", required=True, help="Input predictions.jsonl")
    parser.add_argument("--output", required=True, help="Output metrics JSON")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--diagnosis-only", action="store_true", default=True)
    parser.add_argument("--with-gold-eval", action="store_true")
    args = parser.parse_args()

    rows = _read_jsonl(Path(args.input))
    rng = random.Random(args.seed)
    if args.sample_size and args.sample_size < len(rows):
        rows = rng.sample(rows, args.sample_size)

    per_row = [_metric_for_row(row, args.with_gold_eval) for row in rows]
    failure_counter = Counter(ft for r in per_row for ft in r["failure_types"])
    diagnostic_counter = Counter(r["diagnostic_status"] for r in per_row)
    metrics = {
        "sample_count": len(per_row),
        "trace_valid_rate": _rate([r["trace_valid"] for r in per_row]),
        "evidence_isolation_valid_rate": _rate([r["evidence_isolation_valid"] for r in per_row]),
        "temporal_edge_valid_rate": _rate([r["temporal_edge_valid"] for r in per_row]),
        "rollback_valid_rate": _rate([r["rollback_valid"] for r in per_row]),
        "full_causal_path_valid_rate": _rate([r["full_causal_path_valid"] for r in per_row]),
        "root_to_failure_contains_execution_rate": _rate([r["root_to_failure_contains_execution"] for r in per_row]),
        "query_intent_parse_success_rate": _rate([r["query_intent_parse_success"] for r in per_row]),
        "query_intent_schema_valid_rate": _rate([r["query_intent_schema_valid"] for r in per_row]),
        "query_intent_high_confidence_rate": _rate([r["query_intent_high_confidence"] for r in per_row]),
        "query_intent_unknown_rate": _rate([r["query_intent_unknown"] for r in per_row]),
        "query_intent_llm_fallback_rate": _rate([r["query_intent_llm_fallback"] for r in per_row]),
        "entity_predicate_separation_rate": _rate([r["entity_predicate_separation"] for r in per_row]),
        "given_entity_misclassified_rate": _rate([r["given_entity_misclassified"] for r in per_row]),
        "plan_valid_rate": _rate([r["plan_valid"] for r in per_row]),
        "nonempty_candidate_matrix_rate": _rate([r["nonempty_candidate_matrix"] for r in per_row]),
        "candidate_constraint_nonunknown_rate": _rate([r["candidate_constraint_nonunknown"] for r in per_row]),
        "candidate_constraint_supported_rate": _rate([r["candidate_constraint_supported"] for r in per_row]),
        "candidate_constraint_contradicted_rate": _rate([r["candidate_constraint_contradicted"] for r in per_row]),
        "commitment_allowed_rate": _rate([r["commitment_allowed"] for r in per_row]),
        "premature_commitment_rate": _rate([r["premature_commitment"] for r in per_row]),
        "duplicate_hypothesis_rate": _rate([r["duplicate_hypothesis"] for r in per_row]),
        "canonicalization_collision_rate": _rate([r["candidate_canonicalization_collision"] for r in per_row]),
        "posthoc_root_rate": _rate([r["posthoc_root"] for r in per_row]),
        "repair_trigger_rate": _rate([r["repair_trigger"] for r in per_row]),
        "abstention_rate": _rate([r["abstention"] for r in per_row]),
        "insufficient_diagnostic_evidence_rate": _rate([r["diagnostic_status"] == "INSUFFICIENT_DIAGNOSTIC_EVIDENCE" for r in per_row]),
        "verifier_uncertain_rate": _rate([r["verifier_uncertain"] for r in per_row]),
        "plan_uncertain_rate": _rate([r["plan_uncertain"] for r in per_row]),
        "average_root_candidates": mean([r["average_root_candidates"] for r in per_row]) if per_row else 0.0,
        "blame_score_variance": mean([r["blame_score_variance"] for r in per_row]) if per_row else 0.0,
        "average_root_margin_top1_top2": mean([r["root_margin_top1_top2"] for r in per_row]) if per_row else 0.0,
        "average_num_causal_root_candidates": mean([r["num_causal_root_candidates"] for r in per_row]) if per_row else 0.0,
        "average_num_nonzero_blame_nodes": mean([r["num_nonzero_blame_nodes"] for r in per_row]) if per_row else 0.0,
        "fraction_equal_blame_candidates": mean([r["fraction_equal_blame_candidates"] for r in per_row]) if per_row else 0.0,
        "average_diagnostic_uncertainty": mean([r["diagnostic_uncertainty"] for r in per_row]) if per_row else 0.0,
        "average_root_selection_confidence": mean([r["root_selection_confidence"] for r in per_row]) if per_row else 0.0,
        "local_verified_claim_rate": _rate([r["local_verified_claim"] for r in per_row]),
        "dependency_blocked_claim_rate": _rate([r["dependency_blocked_claim"] for r in per_row]),
        "invalidated_claim_rate": _rate([r["invalidated_claim"] for r in per_row]),
        "inherited_proof_bundle_rate": _rate([r["inherited_proof_bundle"] for r in per_row]),
        "invalid_inheritance_rate": _rate([r["invalid_inheritance"] for r in per_row]),
        "gold_leakage_rate": _rate([
            r["online_input_contains_gold"]
            or r["verifier_input_contains_gold"]
            or r["planner_input_contains_gold"]
            or r["repair_input_contains_gold"]
            for r in per_row
        ]),
        "online_input_contains_gold_rate": _rate([r["online_input_contains_gold"] for r in per_row]),
        "verifier_input_contains_gold_rate": _rate([r["verifier_input_contains_gold"] for r in per_row]),
        "planner_input_contains_gold_rate": _rate([r["planner_input_contains_gold"] for r in per_row]),
        "repair_input_contains_gold_rate": _rate([r["repair_input_contains_gold"] for r in per_row]),
        "average_repair_cost": mean([r["average_repair_cost"] for r in per_row]) if per_row else 0.0,
        "failure_type_distribution": dict(failure_counter),
        "diagnostic_status_distribution": dict(diagnostic_counter),
    }
    if args.with_gold_eval:
        metrics["answer_exact_match_rate"] = _rate([r.get("answer_exact_match", False) for r in per_row])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
