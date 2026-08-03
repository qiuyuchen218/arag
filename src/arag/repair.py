"""Estimated blame and append-only branch repair scaffolding for ARAG v2."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from arag.core.schemas import clean_json, stable_hash, utc_now


@dataclass
class BranchState:
    branch_id: str
    parent_branch_id: Optional[str]
    forked_from_node_id: str
    rollback_checkpoint_id: str
    reason: str
    inherited_claim_ids: List[str] = field(default_factory=list)
    inherited_evidence_ids: List[str] = field(default_factory=list)
    invalidated_node_ids: List[str] = field(default_factory=list)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    inherited_nodes: List[str] = field(default_factory=list)
    rejected_hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    active_unresolved_subgoals: List[str] = field(default_factory=list)
    repair_instruction: str = ""
    branch_budget: int = 0
    status: str = "planned"
    cost: float = 0.0
    created_at: str = field(default_factory=utc_now)


class BranchManager:
    """Append-only branch registry; old branch records are never mutated."""

    def __init__(self):
        self.branches: List[BranchState] = [BranchState("b0", None, "", "", "initial", status="completed")]

    def fork(
        self,
        parent_branch_id: str,
        forked_from_node_id: str,
        rollback_checkpoint_id: str,
        reason: str,
        inherited_claim_ids: List[str] = None,
        inherited_evidence_ids: List[str] = None,
        invalidated_node_ids: List[str] = None,
        constraints: List[Dict[str, Any]] = None,
        inherited_nodes: List[str] = None,
        rejected_hypotheses: List[Dict[str, Any]] = None,
        active_unresolved_subgoals: List[str] = None,
        repair_instruction: str = "",
        branch_budget: int = 0,
    ) -> BranchState:
        if parent_branch_id not in {b.branch_id for b in self.branches}:
            raise ValueError(f"Unknown parent branch: {parent_branch_id}")
        branch_id = f"b{len(self.branches)}"
        state = BranchState(
            branch_id,
            parent_branch_id,
            forked_from_node_id,
            rollback_checkpoint_id,
            reason,
            inherited_claim_ids or [],
            inherited_evidence_ids or [],
            invalidated_node_ids or [],
            constraints or [],
            inherited_nodes or [],
            rejected_hypotheses or [],
            active_unresolved_subgoals or [],
            repair_instruction,
            branch_budget,
        )
        self.branches.append(state)
        return state

    def mark_completed(self, branch_id: str, cost: float = 0.0):
        for branch in self.branches:
            if branch.branch_id == branch_id:
                branch.status = "completed"
                branch.cost = cost
                return branch
        raise ValueError(f"Unknown branch: {branch_id}")

    def select(self, branch_id: str):
        target = None
        for branch in self.branches:
            if branch.branch_id == branch_id:
                target = branch
        if target is None:
            raise ValueError(f"Unknown branch: {branch_id}")
        if target.status != "completed":
            raise ValueError("Only completed branches can be selected")
        for branch in self.branches:
            if branch.branch_id == branch_id:
                branch.status = "selected"
        return target

    def to_dict(self) -> Dict[str, Any]:
        return clean_json({"branches": [asdict(b) for b in self.branches]})


class BlameEngine:
    """Rule-based estimated blame over v2 defect dimensions.

    Results are `estimated` unless a CounterfactualRunner later records an
    actual intervention and support delta.
    """

    ROUTES = {
        "1-E": ["llm_call", "plan_query", "retriever_call"],
        "C": ["llm_call", "claim"],
        "1-P": ["read_call", "context_snapshot"],
        "1-H": ["subgoal", "claim"],
        "1-R": ["plan_query", "retriever_call"],
        "1-D": ["retriever_call"],
        "U": ["plan_query", "retriever_call", "read_call"],
    }

    def score(self, root_bad_claim: Dict[str, Any], trace: Dict[str, Any]) -> List[Dict[str, Any]]:
        defect = root_bad_claim.get("defect_vector", {})
        candidates = []
        for node in trace.get("nodes", []):
            ntype = node.get("type")
            responsibility = 0.0
            dimensions = {}
            for dim, node_types in self.ROUTES.items():
                if ntype in node_types:
                    val = float(defect.get(dim, 0.0) or 0.0)
                    dimensions[dim] = val
                    responsibility += val
            if responsibility <= 0:
                continue
            repairability = 0.0 if ntype == "question" else 1.0
            cost = {"plan_query": 1, "retriever_call": 2, "read_call": 1, "context_snapshot": 1, "llm_call": 3}.get(ntype, 2)
            expected_gain = min(0.5, responsibility / 7)
            blame = responsibility * repairability * expected_gain / (cost + 1e-6)
            candidates.append(clean_json({
                "node_id": node["id"],
                "node_type": ntype,
                "blame": blame,
                "blame_type": "estimated",
                "dimension_breakdown": dimensions,
                "repairability": repairability,
                "expected_support_gain": expected_gain,
                "repair_cost": cost,
                "suggested_action": self._action(ntype, dimensions),
            }))
        candidates.sort(key=lambda x: (-x["blame"], trace_node_step(trace, x["node_id"])))
        return candidates

    def score_cognitive(
        self,
        failure_types: List[str],
        unresolved_subgoals: List[Dict[str, Any]],
        hypotheses: List[Dict[str, Any]],
        trace: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        candidates = []
        node_by_id = {n["id"]: n for n in trace.get("nodes", [])}
        failed_nodes = [sg.get("subgoal_id") for sg in unresolved_subgoals if sg.get("subgoal_id") in node_by_id]
        failed_nodes += [
            n["id"] for n in trace.get("nodes", [])
            if n.get("type") == "claim" and n.get("status") in {"UNSUPPORTED", "CONTRADICTED", "INVALID_PROVENANCE", "DEPENDENCY_BROKEN", "UNCERTAIN"}
        ]
        failed_nodes += [
            n["id"] for n in trace.get("nodes", [])
            if n.get("type") == "coverage_assessment" and n.get("status") in {"UNTESTED", "STALLED", "EXHAUSTED"}
        ]
        if failure_types:
            failed_nodes += [n["id"] for n in trace.get("nodes", []) if n.get("type") == "answer"]
        ancestor_paths = _causal_ancestor_paths(trace, failed_nodes)
        ancestor_ids = set(ancestor_paths)
        for node in trace.get("nodes", []):
            if node.get("type") != "epistemic_state_event" or node.get("id") not in ancestor_ids:
                continue
            md = node.get("metadata", {}) or {}
            new_state = md.get("new_state")
            if new_state not in {"COMMITTED", "USED_AS_PREMISE"}:
                continue
            support = float(md.get("support_score_at_event", 0.0) or 0.0)
            missing = md.get("missing_constraint_ids", []) or []
            unsupported = support < 0.7
            if not unsupported and not missing:
                continue
            decision_id = md.get("generated_by_decision_id")
            decision = node_by_id.get(decision_id, {})
            if decision.get("type") != "decision_record":
                continue
            if not md.get("authoritative") or decision.get("metadata", {}).get("authoritative") is not True:
                continue
            action_role = (md.get("action_role") or decision.get("metadata", {}).get("action_role") or "").upper()
            if action_role in {"EXPLORE", "TEST", "VERIFY", "DISAMBIGUATE"}:
                continue
            downstream = _affected_downstream(trace, node["id"], set(failed_nodes))
            execution_downstream = _downstream_execution_nodes(trace, node["id"])
            if not execution_downstream:
                continue
            rollback = _rollback_before(trace, node["id"])
            rollback_valid = bool(rollback and trace_node_event_seq(trace, rollback) < trace_node_event_seq(trace, node["id"]))
            epistemic_force = 1.0 if new_state == "USED_AS_PREMISE" else 0.8
            constraint_gap = len(missing) / max(len(missing) + len(md.get("available_evidence_ids", []) or []), 1)
            downstream_influence = min(1.0, len(execution_downstream) / 3)
            action_criticality = 1.0 if action_role in {"USE_AS_PREMISE", "COMMIT"} else 0.5
            defect = (1 - support) * epistemic_force * max(constraint_gap, 0.5 if unsupported else 0.0) * downstream_influence * action_criticality
            candidates.append({
                "node_id": node["id"],
                "node_type": "epistemic_state_event",
                "root_bad_type": "unsupported_epistemic_promotion",
                "failure_type": "UNSUPPORTED_COMMITMENT" if new_state == "COMMITTED" else "PREMATURE_BINDING",
                "proposition_id": md.get("proposition_id"),
                "decision_id": decision_id,
                "new_state": new_state,
                "event_seq": node.get("event_seq"),
                "blame": defect,
                "blame_type": "estimated",
                "dimension_breakdown": {
                    "unsupportedness": 1 - support,
                    "epistemic_force": epistemic_force,
                    "constraint_gap": constraint_gap,
                    "downstream_influence": downstream_influence,
                    "action_criticality": action_criticality,
                },
                "diagnostic_confidence": 0.8 if md.get("authoritative") else 0.45,
                "repairability": 1.0 if rollback_valid else 0.0,
                "expected_support_gain": min(0.7, defect),
                "repair_cost": 1,
                "suggested_action": "reopen_proposition_before_using_as_premise",
                "root_generator_llm": decision.get("metadata", {}).get("generated_by_llm_call_id"),
                "rollback_checkpoint": rollback,
                "rollback_valid": rollback_valid,
                "affected_downstream_nodes": downstream,
                "downstream_execution_nodes": execution_downstream,
                "causal_path_to_failure": ancestor_paths.get(node["id"], [node["id"]]),
                "causal_path_valid": rollback_valid and bool(execution_downstream),
                "full_causal_path_valid": rollback_valid and bool(execution_downstream),
                "root_to_failure_contains_execution": bool(execution_downstream),
            })
        for node in trace.get("nodes", []):
            if node.get("type") != "coverage_assessment":
                continue
            if node.get("status") not in {"UNTESTED", "STALLED", "EXHAUSTED"}:
                continue
            md = node.get("metadata", {}) or {}
            if node.get("id") not in ancestor_ids and node.get("id") not in failed_nodes:
                continue
            query_ids = md.get("plan_query_ids", []) or []
            decision_ids = md.get("query_decision_ids", []) or []
            rollback = _rollback_before(trace, node["id"])
            rollback_valid = bool(rollback and trace_node_event_seq(trace, rollback) < trace_node_event_seq(trace, node["id"]))
            state = node.get("status")
            if state == "UNTESTED":
                failure_type = "RELATION_DEPENDENCY_MISSING"
                root_bad_type = "retrieval_coverage_untested"
                suggested = "target_missing_required_relation"
                confidence = 0.68
            elif state == "STALLED":
                failure_type = "QUERY_STRATEGY_STALLED"
                root_bad_type = "retrieval_strategy_stalled"
                suggested = "rewrite_query_with_new_entities_or_strategy"
                confidence = 0.72
            else:
                failure_type = "RETRIEVAL_COVERAGE_FAILURE"
                root_bad_type = "retrieval_coverage_exhausted"
                suggested = "increase_retrieval_budget_or_switch_strategy"
                confidence = 0.62
            if any(f in failure_types for f in {"PREMATURE_TERMINATION", "ANSWER_MISSING", "DEPENDENCY_BROKEN"}) and state == "ACTIVE":
                failure_type = "PREMATURE_TERMINATION"
            path = ancestor_paths.get(node["id"], [node["id"]])
            candidates.append({
                "node_id": node["id"],
                "node_type": "coverage_assessment",
                "root_bad_type": root_bad_type,
                "failure_type": failure_type,
                "root_type": failure_type,
                "subgoal_id": md.get("subgoal_id"),
                "relation_id": md.get("relation_id"),
                "coverage_state": state,
                "unresolved_reason": md.get("unresolved_reason"),
                "event_seq": node.get("event_seq"),
                "blame": 0.45 if query_ids else 0.35,
                "blame_type": "estimated",
                "dimension_breakdown": {
                    "coverage_gap": 1.0,
                    "query_diversity": md.get("query_diversity", 0.0),
                    "query_count": len(query_ids),
                    "read_count": md.get("read_call_count", 0),
                },
                "diagnostic_confidence": confidence,
                "repairability": 1.0 if rollback_valid else 0.5,
                "expected_support_gain": 0.45,
                "repair_cost": 2,
                "suggested_action": suggested,
                "root_generator_llm": _nearest_prior_llm(trace, node["id"]),
                "rollback_checkpoint": rollback,
                "rollback_valid": rollback_valid,
                "affected_downstream_nodes": [md.get("subgoal_id")] if md.get("subgoal_id") else [],
                "downstream_execution_nodes": decision_ids + query_ids,
                "causal_path_to_failure": path,
                "causal_path_valid": rollback_valid and bool(query_ids or decision_ids),
                "full_causal_path_valid": rollback_valid and bool(query_ids or decision_ids),
                "root_to_failure_contains_execution": bool(query_ids or decision_ids),
                "root_role": "earliest_execution_error",
            })
        if not candidates:
            return clean_json([{
                "node_id": None,
                "node_type": "none",
                "root_bad_type": "observability_gap",
                "failure_type": "OBSERVABILITY_GAP",
                "diagnosis_state": "NO_AUTHORITATIVE_EPISTEMIC_DECISION",
                "blame": 0.0,
                "blame_type": "none",
                "diagnostic_confidence": 0.0,
                "repairability": 0.0,
                "expected_support_gain": 0.0,
                "repair_cost": 0,
                "suggested_action": "collect_explicit_epistemic_context_or_coverage_events_before_root_identification",
                "causal_path_valid": False,
                "full_causal_path_valid": False,
                "root_to_failure_contains_execution": False,
            }])

        for hyp in hypotheses:
            missing = hyp.get("missing_constraints", []) or []
            node_id = hyp.get("commitment_event_id") or hyp.get("hypothesis_id")
            if hyp.get("posthoc_summary"):
                continue
            committed = hyp.get("commitment_state") == "COMMITTED" or bool(hyp.get("commitment_event_id"))
            if missing and committed and "PREMATURE_ENTITY_COMMITMENT" in failure_types and node_id in ancestor_ids:
                if node_id in node_by_id:
                    affected = _affected_downstream(trace, node_id, set(failed_nodes))
                    suffix_path = ancestor_paths.get(node_id, [])
                    local_defect = min(1.0, len(missing) / max(len(hyp.get("constraint_results", {}) or missing), 1))
                    temporal_priority = 1.0 / max(trace_node_step(trace, node_id), 1)
                    repair_cost = 1 + 0.1 * len(missing)
                    contribution = 1.0 if affected else 0.0
                    compatibility = 1.0
                    repairability = 1.0
                    expected_gain = min(0.7, 0.25 + 0.1 * len(missing))
                    uncertainty = 0.35 if hyp.get("extractor_confidence", 1.0) < 0.5 else 0.1
                    counterfactual_gain = expected_gain if affected else 0.0
                    blame = (
                        contribution
                        * compatibility
                        * local_defect
                        * repairability
                        * counterfactual_gain
                        * (0.2 + 0.8 * temporal_priority)
                        * (1 - uncertainty)
                        / (1 + repair_cost)
                    )
                    rollback, root_llm, rollback_valid = _rollback_for_decision(trace, node_id)
                    path_to_root = _decision_path_to_root(trace, node_id, rollback, root_llm)
                    root_to_failure = _root_to_failure_path(trace, node_id, set(failed_nodes))
                    full_path = _merge_paths(path_to_root, root_to_failure or suffix_path)
                    root_to_failure_contains_execution = _path_has_execution_node_after_root(trace, root_to_failure, node_id)
                    root_to_failure_reaches_failure = bool(root_to_failure and root_to_failure[-1] in set(failed_nodes))
                    causal_valid = (
                        _path_has_execution_node(trace, full_path)
                        and bool(path_to_root)
                        and bool(root_to_failure)
                        and root_to_failure_contains_execution
                        and root_to_failure_reaches_failure
                    )
                    candidates.append({
                        "node_id": node_id,
                        "hypothesis_id": hyp.get("hypothesis_id"),
                        "node_type": node_by_id[node_id].get("type"),
                        "root_bad_type": "commitment_event" if node_by_id[node_id].get("type") == "commitment_event" else "hypothesis",
                        "failure_type": "PREMATURE_ENTITY_COMMITMENT",
                        "blame": blame,
                        "blame_type": "estimated",
                        "dimension_breakdown": {
                            "causal_contribution": contribution,
                            "critical_mass": contribution,
                            "failure_compatibility": compatibility,
                            "local_defect": local_defect,
                            "constraint_defect": local_defect,
                            "repairability": repairability,
                            "counterfactual_gain": counterfactual_gain,
                            "temporal_priority": temporal_priority,
                            "diagnostic_uncertainty": uncertainty,
                        },
                        "diagnostic_confidence": 1 - uncertainty,
                        "repairability": repairability,
                        "expected_support_gain": expected_gain,
                        "repair_cost": repair_cost,
                        "suggested_action": "resolve_missing_subgoal_constraints_before_entity_commitment",
                        "root_generator_llm": root_llm,
                        "rollback_checkpoint": rollback,
                        "rollback_valid": rollback_valid,
                        "affected_downstream_nodes": affected,
                        "causal_path_to_root": path_to_root,
                        "causal_path_root_to_failure": root_to_failure,
                        "full_causal_path": full_path,
                        "causal_path_to_failure": full_path,
                        "causal_path_valid": causal_valid,
                        "full_causal_path_valid": causal_valid,
                        "root_to_failure_path_nonempty": bool(root_to_failure),
                        "root_to_failure_contains_execution": root_to_failure_contains_execution,
                        "root_to_failure_reaches_failure_frontier": root_to_failure_reaches_failure,
                    })
        for sg in unresolved_subgoals:
            node_id = sg.get("subgoal_id")
            if node_id in node_by_id and node_id in set(failed_nodes):
                affected = _affected_downstream(trace, node_id, set(failed_nodes))
                candidates.append({
                    "node_id": node_id,
                    "node_type": "subgoal",
                    "root_bad_type": "subgoal",
                    "failure_type": "ANSWER_MISSING" if "ANSWER_MISSING" in failure_types else "DEPENDENCY_BROKEN",
                    "blame": 0.05,
                    "blame_type": "estimated",
                    "dimension_breakdown": {"unresolved_required_subgoal": 1.0},
                    "repairability": 1.0,
                    "expected_support_gain": 0.35,
                    "repair_cost": 1,
                    "suggested_action": "solve_required_subgoal",
                    "rollback_checkpoint": _rollback_before(trace, node_id),
                    "affected_downstream_nodes": affected,
                    "causal_path_to_failure": ancestor_paths.get(node_id, [node_id]),
                })
        candidates.sort(key=lambda x: (
            not x.get("causal_path_valid", False),
            -x.get("diagnostic_confidence", 0.0),
            -x.get("expected_support_gain", 0.0),
            trace_node_step(trace, x.get("node_id")),
            x.get("repair_cost", 99),
        ))
        return clean_json(candidates)

    @staticmethod
    def _action(node_type: str, dimensions: Dict[str, float]) -> str:
        if node_type == "plan_query":
            return "rewrite_query"
        if node_type == "retriever_call":
            return "increase_top_k_or_switch_to_hybrid"
        if node_type == "read_call":
            return "read_missed_candidate"
        if node_type == "context_snapshot":
            return "inject_delivered_evidence"
        if node_type == "llm_call":
            return "regenerate_claim_with_same_evidence"
        return "resolve_missing_subgoal"


def trace_node_step(trace: Dict[str, Any], node_id: str) -> int:
    for node in trace.get("nodes", []):
        if node.get("id") == node_id:
            md = node.get("metadata", {}) or {}
            return int(
                node.get("step_index")
                or md.get("step_index")
                or md.get("first_proposed_at")
                or md.get("last_updated_at")
                or 10**9
            )
    return 10**9


def trace_node_event_seq(trace: Dict[str, Any], node_id: str) -> int:
    for node in trace.get("nodes", []):
        if node.get("id") == node_id:
            return int(node.get("event_seq") or trace_node_step(trace, node_id))
    return 10**9


def _is_valid_causal_edge(trace: Dict[str, Any], edge: Dict[str, Any]) -> bool:
    if not edge.get("metadata", {}).get("causal"):
        return False
    if edge.get("metadata", {}).get("inferred"):
        return False
    by_id = {n["id"]: n for n in trace.get("nodes", [])}
    src = by_id.get(edge.get("source"), {})
    tgt = by_id.get(edge.get("target"), {})
    if not src or not tgt:
        return False
    if src.get("branch_id", "b0") != tgt.get("branch_id", "b0"):
        return False
    return trace_node_event_seq(trace, src.get("id")) < trace_node_event_seq(trace, tgt.get("id"))


def _causal_ancestor_paths(trace: Dict[str, Any], failed_nodes: List[str]) -> Dict[str, List[str]]:
    upstream: Dict[str, List[str]] = {}
    for edge in trace.get("edges", []):
        src, tgt = edge.get("source"), edge.get("target")
        if _is_valid_causal_edge(trace, edge):
            upstream.setdefault(tgt, []).append(src)
    paths: Dict[str, List[str]] = {}
    queue = [(fid, [fid]) for fid in failed_nodes if fid]
    seen = set()
    while queue:
        node, path = queue.pop(0)
        if node in seen:
            continue
        seen.add(node)
        paths.setdefault(node, path)
        for parent in upstream.get(node, []):
            if parent not in seen:
                paths[parent] = [parent] + path
                queue.append((parent, [parent] + path))
    return paths


def _affected_downstream(trace: Dict[str, Any], node_id: str, failed_nodes: set) -> List[str]:
    downstream: Dict[str, List[str]] = {}
    for edge in trace.get("edges", []):
        if not _is_valid_causal_edge(trace, edge):
            continue
        src, tgt = edge.get("source"), edge.get("target")
        downstream.setdefault(src, []).append(tgt)
    queue = [node_id]
    seen = set()
    affected = []
    while queue:
        cur = queue.pop(0)
        if cur in seen:
            continue
        seen.add(cur)
        if cur in failed_nodes and cur != node_id:
            affected.append(cur)
        queue.extend(downstream.get(cur, []))
    return affected


def _downstream_execution_nodes(trace: Dict[str, Any], node_id: str) -> List[str]:
    by_id = {n["id"]: n for n in trace.get("nodes", [])}
    downstream: Dict[str, List[str]] = {}
    for edge in trace.get("edges", []):
        if not _is_valid_causal_edge(trace, edge):
            continue
        downstream.setdefault(edge.get("source"), []).append(edge.get("target"))
    queue = list(downstream.get(node_id, []))
    seen = set()
    found = []
    while queue:
        cur = queue.pop(0)
        if cur in seen:
            continue
        seen.add(cur)
        if by_id.get(cur, {}).get("type") in {"decision_record", "plan_query", "retriever_call", "read_call", "llm_call", "answer"}:
            found.append(cur)
        queue.extend(downstream.get(cur, []))
    return found


def _rollback_before(trace: Dict[str, Any], node_id: str) -> Optional[str]:
    step = trace_node_step(trace, node_id)
    candidates = [
        n for n in trace.get("nodes", [])
        if n.get("type") in {"context_snapshot", "llm_call", "plan_query"} and trace_node_step(trace, n.get("id")) < step
    ]
    if not candidates:
        return None
    snapshots = [n for n in candidates if n.get("type") == "context_snapshot"]
    pool = snapshots or candidates
    return max(pool, key=lambda n: trace_node_step(trace, n.get("id"))).get("id")


def _nearest_prior_llm(trace: Dict[str, Any], node_id: str) -> Optional[str]:
    step = trace_node_step(trace, node_id)
    candidates = [
        n for n in trace.get("nodes", [])
        if n.get("type") == "llm_call" and trace_node_step(trace, n.get("id")) < step
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda n: trace_node_step(trace, n.get("id"))).get("id")


def _rollback_for_decision(trace: Dict[str, Any], node_id: str) -> tuple[Optional[str], Optional[str], bool]:
    by_id = {n["id"]: n for n in trace.get("nodes", [])}
    root = by_id.get(node_id, {})
    md = root.get("metadata", {})
    root_llm = md.get("generated_by_llm_call_id")
    source = md.get("source_event_id") or md.get("proposed_by")
    if not root_llm and source in by_id and by_id[source].get("type") == "plan_query":
        for edge in trace.get("edges", []):
            if edge.get("type") == "invokes" and edge.get("target") == source:
                root_llm = edge.get("source")
                break
    rollback = None
    if root_llm:
        for edge in trace.get("edges", []):
            if edge.get("type") == "consumed_by" and edge.get("target") == root_llm:
                rollback = edge.get("source")
                break
    if not rollback:
        rollback = _rollback_before(trace, node_id)
    r_step = trace_node_step(trace, node_id)
    rb_step = trace_node_step(trace, rollback) if rollback else 10**9
    consumed = True
    if root_llm:
        consumed = any(
            e.get("type") == "consumed_by" and e.get("source") == rollback and e.get("target") == root_llm
            for e in trace.get("edges", [])
        )
    valid = bool(rollback and rb_step < r_step and consumed)
    return rollback, root_llm, valid


def _path_has_execution_node(trace: Dict[str, Any], path: List[str]) -> bool:
    by_id = {n["id"]: n for n in trace.get("nodes", [])}
    return any(
        by_id.get(pid, {}).get("type") in {"context_snapshot", "llm_call", "plan_query", "retriever_call", "read_call", "tool_result"}
        for pid in path
    )


def _decision_path_to_failure(
    trace: Dict[str, Any],
    node_id: str,
    suffix: List[str],
    rollback: Optional[str],
    root_llm: Optional[str],
) -> List[str]:
    by_id = {n["id"]: n for n in trace.get("nodes", [])}
    md = by_id.get(node_id, {}).get("metadata", {}) or {}
    source = md.get("source_event_id") or md.get("proposed_by")
    prefix = [p for p in [rollback, root_llm, source, node_id] if p]
    path = []
    for item in prefix + list(suffix or []):
        if item not in path:
            path.append(item)
    return path


def _decision_path_to_root(
    trace: Dict[str, Any],
    node_id: str,
    rollback: Optional[str],
    root_llm: Optional[str],
) -> List[str]:
    by_id = {n["id"]: n for n in trace.get("nodes", [])}
    md = by_id.get(node_id, {}).get("metadata", {}) or {}
    source = md.get("source_event_id") or md.get("proposed_by")
    return _merge_paths([p for p in [rollback, root_llm, source, node_id] if p])


def _root_to_failure_path(trace: Dict[str, Any], root_id: str, failed_nodes: set) -> List[str]:
    by_id = {n["id"]: n for n in trace.get("nodes", [])}
    root_step = trace_node_step(trace, root_id)
    downstream: Dict[str, List[str]] = {}
    for edge in trace.get("edges", []):
        if not _is_valid_causal_edge(trace, edge):
            continue
        src, tgt, et = edge.get("source"), edge.get("target"), edge.get("type")
        if trace_node_step(trace, tgt) >= root_step or tgt in failed_nodes:
            downstream.setdefault(src, []).append(tgt)
    queue = [(root_id, [root_id])]
    seen = set()
    while queue:
        cur, path = queue.pop(0)
        if cur in seen:
            continue
        seen.add(cur)
        if cur in failed_nodes and cur != root_id:
            return path
        for child in downstream.get(cur, []):
            if child not in seen:
                queue.append((child, path + [child]))
    return []


def _merge_paths(*paths: List[str]) -> List[str]:
    merged: List[str] = []
    for path in paths:
        for item in path or []:
            if item and item not in merged:
                merged.append(item)
    return merged


def _path_has_execution_node_after_root(trace: Dict[str, Any], path: List[str], root_id: str) -> bool:
    by_id = {n["id"]: n for n in trace.get("nodes", [])}
    after_root = False
    for pid in path or []:
        if pid == root_id:
            after_root = True
            continue
        if after_root and by_id.get(pid, {}).get("type") in {"plan_query", "retriever_call", "read_call", "tool_result", "context_snapshot", "llm_call"}:
            return True
    return False


def rejected_hypothesis(content: str, failure_type: str, source_event: str, reason: str, scope: str = "branch", reconsider_if: str = "new supporting evidence is delivered") -> Dict[str, Any]:
    return {
        "hypothesis_id": f"hyp_{stable_hash(content, failure_type, source_event, reason)}",
        "content": content,
        "status": "rejected_under_current_evidence",
        "failure_type": failure_type,
        "source_event": source_event,
        "reason": reason,
        "scope": scope,
        "reconsider_if": reconsider_if,
    }
