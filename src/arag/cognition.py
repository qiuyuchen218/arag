"""Cognitive graph helpers for question plans, hypotheses, and adequacy.

These components operate only on observable text: the user question, public
assistant messages, tool calls, and final answers. They do not request or store
hidden chain-of-thought.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from arag.core.schemas import clean_json, stable_hash, utc_now


@dataclass
class Subgoal:
    subgoal_id: str
    content: str
    required: bool = True
    status: str = "open"
    expected_output_type: str = "free_text"
    required_constraints: List[str] = field(default_factory=list)
    satisfied_constraints: List[str] = field(default_factory=list)
    generated_by: Optional[str] = None
    branch_id: str = "b0"
    dependencies: List[str] = field(default_factory=list)
    resolved_by_claim_ids: List[str] = field(default_factory=list)


@dataclass
class QuestionPlan:
    question_id: str
    original_question: str
    expected_answer_type: str
    required_answer_slots: List[str]
    subgoals: List[Subgoal]
    constraints: List[str] = field(default_factory=list)
    variables: List[Dict[str, Any]] = field(default_factory=list)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    answer_spec: Dict[str, Any] = field(default_factory=dict)
    generated_by: str = "heuristic_question_decomposer"
    analyzer_model: str = "heuristic"
    analyzer_version: str = "v1"
    prompt_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return clean_json(asdict(self))

    def relation_by_id(self) -> Dict[str, Dict[str, Any]]:
        return {r.get("relation_id"): r for r in self.relations if r.get("relation_id")}

    def subgoal_by_id(self) -> Dict[str, Subgoal]:
        return {s.subgoal_id: s for s in self.subgoals}


class FakeQuestionDecomposer:
    def decompose(self, question: str, question_id: str = None) -> QuestionPlan:
        return heuristic_question_plan(question, question_id)


def heuristic_question_plan(question: str, question_id: str = None) -> QuestionPlan:
    q = (question or "").strip()
    q_lower = q.lower()
    if q_lower.startswith("when") or " when " in q_lower:
        answer_type, slots = "temporal", ["date"]
    elif q_lower.startswith("who"):
        answer_type, slots = "person", ["person"]
    elif q_lower.startswith("where"):
        answer_type, slots = "location", ["location"]
    elif q_lower.startswith("how many"):
        answer_type, slots = "numeric", ["number"]
    else:
        answer_type, slots = "free_text", ["answer"]

    variables, relations, constraints = _question_relations(q, question_id or q)
    subgoals = []
    prev_required = None
    for relation in relations:
        sg_id = f"subgoal_{stable_hash(question_id or q, relation['relation_id'])}"
        deps = [prev_required] if prev_required else []
        subgoals.append(Subgoal(
            subgoal_id=sg_id,
            content=f"Resolve relation: {relation['subject_variable']} --{relation['predicate']}--> {relation['object_variable']}",
            expected_output_type=relation.get("expected_output_type", "relation"),
            required_constraints=[relation["relation_id"]],
            dependencies=deps,
        ))
        relation["subgoal_id"] = sg_id
        relation["dependencies"] = deps
        prev_required = sg_id
    if "date" in slots:
        answer_relation_id = f"rel_{stable_hash(question_id or q, 'answer_date')}"
        subgoals.append(Subgoal(
            subgoal_id=f"subgoal_{stable_hash(question_id or q, 'answer_date')}",
            content="Find the date that answers the question.",
            expected_output_type="temporal",
            required_constraints=["date_slot_filled"],
            dependencies=[subgoals[-1].subgoal_id],
        ))
        relations.append({
            "relation_id": answer_relation_id,
            "subject_variable": relations[-1]["object_variable"] if relations else "target_entity",
            "predicate": "answer_date",
            "object_variable": "answer",
            "required": True,
            "dependencies": [relations[-1]["relation_id"]] if relations else [],
            "status": "open",
            "supporting_claim_ids": [],
            "supporting_evidence_ids": [],
            "subgoal_id": subgoals[-1].subgoal_id,
            "expected_output_type": "temporal",
        })
    return QuestionPlan(
        question_id=str(question_id or stable_hash(q)),
        original_question=q,
        expected_answer_type=answer_type,
        required_answer_slots=slots,
        subgoals=subgoals,
        constraints=constraints,
        variables=variables,
        relations=relations,
        answer_spec={"answer_type": answer_type, "required_slots": slots},
        prompt_hash=stable_hash(q, answer_type, slots),
    )


def _question_relations(question: str, seed: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    q = question or ""
    variables = [{"variable_id": "answer", "role": "answer", "binding_status": "UNKNOWN_VARIABLE"}]
    relations: List[Dict[str, Any]] = []
    constraints: List[str] = []

    def add_relation(subject: str, predicate: str, obj: str, expected: str = "relation"):
        rid = f"rel_{stable_hash(seed, subject, predicate, obj)}"
        relations.append({
            "relation_id": rid,
            "subject_variable": subject,
            "predicate": predicate,
            "object_variable": obj,
            "required": True,
            "dependencies": [],
            "identity_constraint": expected != "temporal",
            "answer_constraint": expected == "temporal",
            "status": "open",
            "supporting_claim_ids": [],
            "supporting_evidence_ids": [],
            "expected_output_type": expected,
        })
        constraints.append(rid)

    possessive = re.search(r"\b([A-Z][A-Za-z0-9_-]+(?:\s+[A-Z][A-Za-z0-9_-]+)*)'s\s+([a-z_ -]+?)\s+(?:abolished|created|founded|established)", q)
    if possessive:
        person = possessive.group(1).strip()
        relation_name = "_".join(possessive.group(2).strip().split())
        variables.extend([
            {"variable_id": "person", "role": "given_entity", "value": person},
            {"variable_id": "target_entity", "role": relation_name},
        ])
        variables[-2]["binding_status"] = "GIVEN_BINDING"
        variables[-1]["binding_status"] = "UNKNOWN_VARIABLE"
        add_relation("person", relation_name, "target_entity", "entity")
        add_relation("target_entity", "abolished_date" if "abolished" in q.lower() else "created_date", "answer", "temporal")
        return variables, relations, constraints

    clauses = [
        ("region_immediately_north", "region where Israel is located"),
        ("battle_location", "location of the Battle of Qurah and Umm al Maradim"),
        ("creation_date", "target region"),
    ]
    if "immediately north" in q.lower() and "battle" in q.lower():
        variables.extend([
            {"variable_id": "base_region", "role": "given_constraint", "description": "region where Israel is located"},
            {"variable_id": "battle_location", "role": "given_constraint", "description": "location of the Battle of Qurah and Umm al Maradim"},
            {"variable_id": "target_region", "role": "answer_subject"},
        ])
        for v in variables[-3:]:
            v["binding_status"] = "UNKNOWN_VARIABLE"
        add_relation("base_region", "located_region_of", "Israel", "location")
        add_relation("battle_location", "location_of_battle", "Battle of Qurah and Umm al Maradim", "location")
        add_relation("target_region", "immediately_north_of", "base_region", "location")
        add_relation("target_region", "contains_or_matches_battle_location_constraint", "battle_location", "location")
        add_relation("target_region", "created_date", "answer", "temporal")
        return variables, relations, constraints

    variables.append({"variable_id": "target_entity", "role": "answer_subject", "binding_status": "UNKNOWN_VARIABLE"})
    add_relation("question", "identify_target_entity", "target_entity", "entity")
    return variables, relations, constraints


PREDICATE_TERMS = {
    "abolished", "abolish", "created", "create", "founded", "found", "established", "establish",
    "date", "when", "location", "birthplace", "born", "created_date", "abolished_date",
    "established_date", "founded_date", "history", "country", "region", "city", "county",
}


@dataclass
class QueryIntent:
    query_intent_id: str
    branch_id: str
    step_index: int
    source_plan_query_id: Optional[str]
    generated_by_llm_call_id: Optional[str]
    raw_query: str
    normalized_query: str
    target_subgoal_id: Optional[str]
    target_relation_id: Optional[str]
    predicate: Optional[str]
    known_bindings: Dict[str, str] = field(default_factory=dict)
    unknown_variable: Optional[str] = None
    candidate_entities: List[str] = field(default_factory=list)
    query_mode: str = "unknown"
    epistemic_action: str = "explore"
    commitment_level: str = "exploratory"
    parser_mode: str = "heuristic"
    parser_confidence: float = 0.0
    parse_warnings: List[str] = field(default_factory=list)


@dataclass
class HypothesisState:
    hypothesis_id: str
    branch_id: str
    target_variable: str
    target_subgoal_id: Optional[str]
    canonical_entity: str
    aliases: List[str] = field(default_factory=list)
    first_proposed_at: int = 0
    last_updated_at: int = 0
    proposed_by: Optional[str] = None
    status: str = "PROPOSED"
    commitment_state: str = "UNCOMMITTED"
    constraint_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    support_evidence_ids: List[str] = field(default_factory=list)
    contradiction_evidence_ids: List[str] = field(default_factory=list)
    motivated_query_ids: List[str] = field(default_factory=list)
    update_history: List[Dict[str, Any]] = field(default_factory=list)
    extractor_mode: str = "heuristic"
    extractor_confidence: float = 0.0
    posthoc_summary: bool = False


@dataclass
class CommitmentEvent:
    commitment_event_id: str
    hypothesis_id: str
    branch_id: str
    step_index: int
    source_event_id: Optional[str]
    generated_by_llm_call_id: Optional[str]
    target_variable: str
    target_subgoal_id: Optional[str]
    candidate_entity: str
    commitment_type: str
    satisfied_identity_constraints: List[str] = field(default_factory=list)
    missing_identity_constraints: List[str] = field(default_factory=list)
    contradicted_identity_constraints: List[str] = field(default_factory=list)
    commitment_strength: float = 0.0
    is_premature: bool = False
    reason: str = ""
    motivated_downstream_query_ids: List[str] = field(default_factory=list)


def canonicalize_entity(text: str) -> Optional[str]:
    text = re.sub(r"[{}]+", " ", str(text or ""))
    text = re.sub(r"\b(?:chunk|doc|document|span|id|ids|read chunks?)\s*[:#]?\s*\d+\b", " ", text, flags=re.I)
    text = re.sub(r"\b\d+\b", " ", text)
    pieces = []
    for token in re.split(r"\s+", text.strip()):
        clean = re.sub(r"^[^\w]+|[^\w]+$", "", token)
        if not clean:
            continue
        if clean.lower() in PREDICATE_TERMS:
            continue
        pieces.append(clean)
    entity = " ".join(pieces).strip(" -:,;.")
    if len(entity) < 3 or entity.lower() in PREDICATE_TERMS:
        return None
    return entity


def parse_query_intent(
    question_plan: QuestionPlan,
    raw_query: str,
    tool_name: str = None,
    arguments: Dict[str, Any] = None,
    source_plan_query_id: str = None,
    generated_by_llm_call_id: str = None,
    branch_id: str = "b0",
    step_index: int = 0,
) -> QueryIntent:
    raw = str(raw_query or "")
    normalized = re.sub(r"\s+", " ", raw).strip()
    lower = normalized.lower()
    warnings: List[str] = []
    relation = _best_relation_for_query(question_plan, lower)
    text_predicate = _predicate_from_text(lower)
    predicate = text_predicate or (relation.get("predicate") if relation else None)
    if text_predicate and relation and relation.get("predicate") == "identify_target_entity":
        relation = None
    candidate_entities = _candidate_entities_from_query(normalized, predicate)
    if tool_name == "read_chunk":
        mode, action, level = "unknown", "explore", "exploratory"
        candidate_entities = []
        warnings.append("read_intent_no_entity_proposal")
    elif predicate in {"abolished_date", "created_date", "answer_date", "established", "created", "abolished"}:
        mode, action = "answer_lookup", "test"
        level = "tentative" if candidate_entities else "exploratory"
    elif candidate_entities and predicate:
        mode, action, level = "relation_lookup", "test", "tentative"
    elif candidate_entities:
        mode, action, level = "entity_discovery", "propose", "exploratory"
    else:
        mode, action, level = "broad_exploration", "explore", "exploratory"
    confidence = 0.8 if relation and (candidate_entities or tool_name == "read_chunk") else (0.55 if predicate or candidate_entities else 0.25)
    return QueryIntent(
        query_intent_id=f"qi_{stable_hash(branch_id, source_plan_query_id, generated_by_llm_call_id, normalized, step_index)}",
        branch_id=branch_id,
        step_index=step_index,
        source_plan_query_id=source_plan_query_id,
        generated_by_llm_call_id=generated_by_llm_call_id,
        raw_query=raw,
        normalized_query=normalized,
        target_subgoal_id=relation.get("subgoal_id") if relation else (question_plan.subgoals[0].subgoal_id if question_plan.subgoals else None),
        target_relation_id=relation.get("relation_id") if relation else None,
        predicate=predicate,
        known_bindings=_known_bindings(question_plan),
        unknown_variable=relation.get("object_variable") if relation else None,
        candidate_entities=candidate_entities,
        query_mode=mode,
        epistemic_action=action,
        commitment_level=level,
        parser_confidence=confidence,
        parse_warnings=warnings,
    )


def _known_bindings(question_plan: QuestionPlan) -> Dict[str, str]:
    return {
        str(v.get("variable_id")): str(v.get("value"))
        for v in question_plan.variables
        if v.get("binding_status") == "GIVEN_BINDING" and v.get("value")
    }


def _best_relation_for_query(question_plan: QuestionPlan, lower_query: str) -> Optional[Dict[str, Any]]:
    best = None
    best_score = 0
    for rel in question_plan.relations:
        pred = str(rel.get("predicate", ""))
        terms = [t for t in re.split(r"[_\s]+", pred.lower()) if t]
        score = sum(1 for t in terms if t in lower_query)
        if score > best_score:
            best, best_score = rel, score
    return best


def _predicate_from_text(lower_query: str) -> Optional[str]:
    if "birthplace" in lower_query:
        return "birthplace"
    if "abolished" in lower_query:
        return "abolished_date"
    if any(term in lower_query for term in ["created", "creation date", "founded", "established"]):
        return "created_date"
    if "location" in lower_query:
        return "location"
    return None


def _candidate_entities_from_query(query: str, predicate: str = None) -> List[str]:
    text = re.sub(r"['\"]", " ", query or "")
    text = re.sub(r"\b(?:when|where|who|what|was|were|is|are|the|of|in|for|or|and|as|a|an|to)\b", " ", text, flags=re.I)
    if predicate:
        for term in re.split(r"[_\s]+", predicate):
            text = re.sub(rf"\b{re.escape(term)}\b", " ", text, flags=re.I)
    text = re.sub(r"\b(?:created|founded|established|abolished|birthplace|location|date|history|country|region|city|county)\b", " ", text, flags=re.I)
    candidates = []
    for match in re.finditer(r"\b[A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*){0,5}\b", text):
        entity = canonicalize_entity(match.group(0))
        if entity and entity not in candidates:
            candidates.append(entity)
    return candidates


class OnlineHypothesisTracker:
    def __init__(self, question_plan: QuestionPlan):
        self.question_plan = question_plan
        self.states: Dict[tuple, HypothesisState] = {}
        self.commitments: List[CommitmentEvent] = []

    def observe_query(self, intent: QueryIntent, visible_evidence_ids: Optional[List[str]] = None):
        for entity in intent.candidate_entities:
            self._upsert(
                entity,
                intent.unknown_variable or "target_entity",
                intent.target_subgoal_id,
                intent.step_index,
                intent.source_plan_query_id,
                "query_candidate",
                intent.parser_confidence,
                visible_evidence_ids or [],
                motivated_query_id=intent.source_plan_query_id,
            )
            if self._is_commitment_intent(intent, entity):
                self._commit(intent, entity, "answer_relation_query")

    def observe_text(
        self,
        text: str,
        source_event_id: str,
        step_index: int,
        visible_evidence_ids: Optional[List[str]] = None,
        posthoc_summary: bool = False,
    ):
        for hyp in ObservableStateExtractor().extract(
            self.question_plan,
            text,
            source_event_id=source_event_id,
            branch_id="b0",
            step_index=step_index,
            visible_evidence_ids=visible_evidence_ids or [],
        ):
            entity = hyp.candidate_entity
            if not entity:
                continue
            state = self._upsert(
                entity,
                hyp.normalized_binding.get("target_variable", "target_entity"),
                hyp.target_subgoal_id,
                step_index,
                source_event_id,
                "assistant_candidate",
                0.55,
                visible_evidence_ids or [],
                posthoc_summary=posthoc_summary,
            )
            if posthoc_summary:
                state.posthoc_summary = True

    def _upsert(
        self,
        entity: str,
        target_variable: str,
        target_subgoal_id: Optional[str],
        step_index: int,
        source_event_id: Optional[str],
        action: str,
        confidence: float,
        evidence_ids: List[str],
        motivated_query_id: str = None,
        posthoc_summary: bool = False,
    ) -> HypothesisState:
        canonical = canonicalize_entity(entity) or entity
        key = ("b0", target_variable, canonical.lower())
        previous = self.states.get(key)
        if previous is None:
            previous_status = None
            state = HypothesisState(
                hypothesis_id=f"hyp_{stable_hash('b0', target_variable, canonical.lower())}",
                branch_id="b0",
                target_variable=target_variable,
                target_subgoal_id=target_subgoal_id,
                canonical_entity=canonical,
                aliases=[entity],
                first_proposed_at=step_index,
                last_updated_at=step_index,
                proposed_by=source_event_id,
                constraint_results=_candidate_constraint_results(canonical, self.question_plan, evidence_ids),
                support_evidence_ids=list(dict.fromkeys(evidence_ids)),
                extractor_confidence=confidence,
                posthoc_summary=posthoc_summary,
            )
            self.states[key] = state
        else:
            previous_status = previous.status
            state = previous
            state.last_updated_at = max(state.last_updated_at, step_index)
            if entity not in state.aliases:
                state.aliases.append(entity)
            state.support_evidence_ids = list(dict.fromkeys(state.support_evidence_ids + list(evidence_ids)))
            _merge_constraint_results(state.constraint_results, _candidate_constraint_results(canonical, self.question_plan, evidence_ids))
        if evidence_ids:
            state.status = "PARTIALLY_SUPPORTED"
        if motivated_query_id and motivated_query_id not in state.motivated_query_ids:
            state.motivated_query_ids.append(motivated_query_id)
        state.update_history.append({
            "step_index": step_index,
            "source_event_id": source_event_id,
            "action": action,
            "previous_status": previous_status,
            "new_status": state.status,
            "new_evidence_ids": list(evidence_ids),
            "reason": "observable candidate mention or query intent",
        })
        return state

    def _is_commitment_intent(self, intent: QueryIntent, entity: str) -> bool:
        return (
            intent.commitment_level == "committed"
            or intent.epistemic_action == "commit"
            or (intent.query_mode == "answer_lookup" and bool(entity))
        )

    def _commit(self, intent: QueryIntent, entity: str, commitment_type: str):
        canonical = canonicalize_entity(entity) or entity
        key = ("b0", intent.unknown_variable or "target_entity", canonical.lower())
        state = self.states.get(key)
        if not state:
            return
        matrix = build_candidate_constraint_matrix(self.question_plan, list(self.states.values()))
        entry = next((c for c in matrix["candidates"] if c.get("hypothesis_id") == state.hypothesis_id), {})
        missing = [
            cid for cid, cr in (entry.get("identity_constraints") or {}).items()
            if cr.get("status") != "SATISFIED"
        ]
        contradicted = [
            cid for cid, cr in (entry.get("identity_constraints") or {}).items()
            if cr.get("status") == "CONTRADICTED"
        ]
        satisfied = [
            cid for cid, cr in (entry.get("identity_constraints") or {}).items()
            if cr.get("status") == "SATISFIED"
        ]
        state.commitment_state = "COMMITTED"
        event = CommitmentEvent(
            commitment_event_id=f"commit_{stable_hash(state.hypothesis_id, intent.source_plan_query_id, intent.step_index)}",
            hypothesis_id=state.hypothesis_id,
            branch_id=state.branch_id,
            step_index=intent.step_index,
            source_event_id=intent.source_plan_query_id,
            generated_by_llm_call_id=intent.generated_by_llm_call_id,
            target_variable=state.target_variable,
            target_subgoal_id=state.target_subgoal_id,
            candidate_entity=state.canonical_entity,
            commitment_type=commitment_type,
            satisfied_identity_constraints=satisfied,
            missing_identity_constraints=missing,
            contradicted_identity_constraints=contradicted,
            commitment_strength=0.8 if not missing and not contradicted else 0.55,
            is_premature=bool(missing or contradicted),
            reason="candidate used as subject for answer-relation lookup",
            motivated_downstream_query_ids=[intent.source_plan_query_id] if intent.source_plan_query_id else [],
        )
        self.commitments.append(event)

    def hypothesis_dicts(self) -> List[Dict[str, Any]]:
        return [clean_json(asdict(s)) for s in self.states.values()]

    def commitment_dicts(self) -> List[Dict[str, Any]]:
        return [clean_json(asdict(c)) for c in self.commitments]


def _merge_constraint_results(base: Dict[str, Dict[str, Any]], incoming: Dict[str, Dict[str, Any]]):
    for cid, result in incoming.items():
        if cid not in base or base[cid].get("status") == "UNKNOWN":
            base[cid] = result


@dataclass
class Hypothesis:
    hypothesis_id: str
    content: str
    normalized_binding: Dict[str, Any] = field(default_factory=dict)
    hypothesis_type: str = "entity_binding"
    generated_by: Optional[str] = None
    source_event_id: Optional[str] = None
    branch_id: str = "b0"
    step_index: int = 0
    target_subgoal_id: Optional[str] = None
    satisfied_constraints: List[str] = field(default_factory=list)
    missing_constraints: List[str] = field(default_factory=list)
    cited_evidence_ids: List[str] = field(default_factory=list)
    available_evidence_ids: List[str] = field(default_factory=list)
    selected_supporting_evidence_ids: List[str] = field(default_factory=list)
    candidate_entity: Optional[str] = None
    constraint_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    commitment_strength: float = 0.0
    status: str = "proposed"


class ObservableStateExtractor:
    """Heuristic extractor for public hypothesis-like statements."""

    ENTITY_PATTERNS = [
        re.compile(r"\b(?:may be|might be|would be|is|was|suggests?)\s+([A-Z][A-Za-z][A-Za-z\s'-]{1,60})"),
        re.compile(r"\bsearch(?:ing)?(?: for)?\s+\"?([A-Z][A-Za-z][A-Za-z\s'-]{1,60})\"?"),
    ]

    def extract(
        self,
        question_plan: QuestionPlan,
        text: str,
        source_event_id: str = None,
        branch_id: str = "b0",
        step_index: int = 0,
        visible_evidence_ids: List[str] = None,
    ) -> List[Hypothesis]:
        hypotheses: List[Hypothesis] = []
        target = question_plan.subgoals[0].subgoal_id if question_plan.subgoals else None
        question_entities = {str(v.get("value", "")).lower() for v in question_plan.variables if v.get("role") == "given_entity"}
        for pattern in self.ENTITY_PATTERNS:
            for match in pattern.finditer(text or ""):
                entity = " ".join(match.group(1).split())
                if len(entity) < 3 or entity.lower().startswith(("the ", "a ")):
                    continue
                content = f"The target entity may be {entity}"
                satisfied = ["entity_mentioned"]
                if entity.lower() in question_entities:
                    missing = []
                    commitment = 0.1
                else:
                    missing = list(question_plan.constraints or ["all_question_constraints_satisfied"])
                    commitment = 0.6 if any(w in (text or "").lower() for w in ["created", "established", "abolished", "target"]) else 0.3
                hypotheses.append(Hypothesis(
                    hypothesis_id=f"hyp_{stable_hash(branch_id, source_event_id, entity, step_index)}",
                    content=content,
                    normalized_binding={"entity": entity},
                    hypothesis_type="entity_binding",
                    generated_by=source_event_id,
                    source_event_id=source_event_id,
                    branch_id=branch_id,
                    step_index=step_index,
                    target_subgoal_id=target,
                    satisfied_constraints=satisfied,
                    missing_constraints=missing,
                    cited_evidence_ids=[],
                    available_evidence_ids=list(visible_evidence_ids or []),
                    candidate_entity=entity,
                    constraint_results=_candidate_constraint_results(entity, question_plan),
                    commitment_strength=commitment,
                    status="partially_supported" if visible_evidence_ids and missing else "proposed",
                ))
        dedup = {}
        for h in hypotheses:
            dedup.setdefault(h.hypothesis_id, h)
        return list(dedup.values())


def _candidate_constraint_results(entity: str, question_plan: QuestionPlan, evidence_ids: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    results = {}
    evidence_ids = list(evidence_ids or [])
    for cid in question_plan.constraints:
        status = "SATISFIED" if evidence_ids else "UNKNOWN"
        results[cid] = {
            "constraint_id": cid,
            "status": status,
            "support_score": 0.5 if evidence_ids else 0.0,
            "supporting_claim_ids": [],
            "supporting_evidence_ids": evidence_ids[:3],
            "contradiction_evidence_ids": [],
            "verifier_mode": "heuristic_observable",
            "reason": "candidate has provenance-visible evidence" if evidence_ids else "no relation-specific evidence selected",
        }
    return results


def build_candidate_constraint_matrix(question_plan: QuestionPlan, hypotheses: List[Any]) -> Dict[str, Any]:
    relations = question_plan.relations
    identity_ids = [r["relation_id"] for r in relations if r.get("identity_constraint")]
    answer_ids = [r["relation_id"] for r in relations if r.get("answer_constraint")]
    candidates = []
    for hyp in hypotheses:
        hd = asdict(hyp) if hasattr(hyp, "__dataclass_fields__") else dict(hyp)
        entity = hd.get("canonical_entity") or hd.get("candidate_entity") or hd.get("normalized_binding", {}).get("entity")
        if not entity:
            continue
        cr = hd.get("constraint_results") or {}
        identity = {cid: cr.get(cid, {"constraint_id": cid, "status": "UNKNOWN", "supporting_evidence_ids": []}) for cid in identity_ids}
        answer = {cid: cr.get(cid, {"constraint_id": cid, "status": "UNKNOWN", "supporting_evidence_ids": []}) for cid in answer_ids}
        statuses = [v.get("status") for v in identity.values()]
        satisfied = sum(1 for s in statuses if s == "SATISFIED")
        contradicted = sum(1 for s in statuses if s == "CONTRADICTED")
        unknown = sum(1 for s in statuses if s in {"UNKNOWN", "VERIFIER_UNCERTAIN", None})
        coverage = satisfied / max(len(identity_ids), 1)
        candidates.append({
            "candidate_entity": entity,
            "hypothesis_id": hd.get("hypothesis_id"),
            "identity_constraints": identity,
            "answer_constraints": answer,
            "constraint_results": cr,
            "identity_coverage": coverage,
            "contradiction_count": contradicted,
            "unknown_count": unknown,
            "commitment_allowed": bool(identity_ids) and satisfied == len(identity_ids) and contradicted == 0,
            "confidence": min(1.0, coverage * float(hd.get("extractor_confidence", hd.get("commitment_strength", 0.5)) or 0.5)),
        })
    return clean_json({
        "target_variable": "target_entity",
        "required_identity_constraint_ids": identity_ids,
        "answer_constraint_ids": answer_ids,
        "candidates": candidates,
        "commitment_gate": "all_required_identity_constraints_satisfied_no_contradictions",
        "matrix_confidence": max([c["confidence"] for c in candidates] or [0.0]),
    })


DATE_RE = re.compile(
    r"\b(?:\d{3,4}(?:\s*(?:BC|BCE|AD|CE))?|\d{1,2}\s+[A-Z][a-z]+\s+\d{3,4}|[A-Z][a-z]+\s+\d{1,2},?\s+\d{3,4})\b"
)


def classify_claim(content: str) -> tuple[str, float]:
    text = (content or "").strip()
    lower = text.lower()
    if not text or re.fullmatch(r"\d+[\.)]?", text):
        return "meta_claim", 0.0
    if any(p in lower for p in ["i searched", "i found", "let me", "i have not found", "cannot find", "search results show"]):
        if DATE_RE.search(text):
            return "abstention_claim", 0.2
        return "process_claim", 0.0
    if any(p in lower for p in ["cannot be answered", "not available", "no information", "not present in the provided"]):
        return "abstention_claim", 0.4
    if DATE_RE.search(text):
        return "answer_claim", 1.0
    return "factual_claim", 0.7


@dataclass
class AnswerAssessment:
    expected_answer_type: str
    required_slots: List[str]
    extracted_answer_slots: Dict[str, Any]
    slot_coverage: float
    directness: float
    abstained: bool
    critical_answer_claim_id: Optional[str]
    completeness_status: str
    explanation: str
    critical_answer_claim_missing: bool = False
    candidate_answers: List[Dict[str, Any]] = field(default_factory=list)
    selected_answer: Optional[Dict[str, Any]] = None
    ambiguity: str = "none"
    target_binding_valid: bool = False
    support_valid: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return clean_json(asdict(self))


def assess_answer(question_plan: QuestionPlan, answer: str, claim_assessments: List[Dict[str, Any]]) -> AnswerAssessment:
    slots: Dict[str, Any] = {}
    candidate_answers: List[Dict[str, Any]] = []
    answer_text = answer or ""
    if "date" in question_plan.required_answer_slots:
        for match in DATE_RE.finditer(answer_text):
            candidate_answers.append({
                "value": match.group(0),
                "normalized_value": _normalize_date_like(match.group(0)),
                "answer_type": "temporal",
                "target_entity": None,
                "relation": "answer_date",
                "source_claim_id": None,
                "support_status": "UNKNOWN",
                "directly_asserted": _is_direct_answer_position(answer_text, match.start()),
                "hedged": _is_hedged(answer_text, match.start()),
                "position": match.start(),
            })
        if candidate_answers:
            normalized = sorted({c["normalized_value"] for c in candidate_answers})
            if len(normalized) == 1:
                slots["date"] = candidate_answers[0]["value"]
    abstained = bool(re.search(r"\b(cannot answer|cannot be answered|not available|no information|not found|unable to find|cannot confirm|unclear|not clear)\b", answer_text, re.I))
    coverage = len([s for s in question_plan.required_answer_slots if s in slots]) / max(len(question_plan.required_answer_slots), 1)
    critical = None
    support_valid = False
    for assessment in claim_assessments:
        claim = assessment.get("claim", {})
        if claim.get("claim_type") == "answer_claim" and claim.get("criticality", 0) > 0:
            if critical is None:
                critical = claim.get("claim_id")
            if assessment.get("status") == "VERIFIED":
                support_valid = True
    normalized_values = {c["normalized_value"] for c in candidate_answers}
    ambiguity = "multiple_distinct_answers" if len(normalized_values) > 1 else "none"
    selected = None
    if candidate_answers and ambiguity == "none":
        direct = [c for c in candidate_answers if c["directly_asserted"] and not c["hedged"]]
        selected = direct[0] if direct else candidate_answers[0]
    target_binding_valid = all(sg.status == "resolved" for sg in question_plan.subgoals if sg.required and sg.expected_output_type != "temporal")
    complete = (
        coverage >= 1.0
        and critical is not None
        and support_valid
        and target_binding_valid
        and ambiguity == "none"
        and selected is not None
        and not abstained
    )
    if ambiguity != "none":
        status = "AMBIGUOUS"
    elif complete:
        status = "COMPLETE"
    elif critical is not None and not support_valid:
        status = "UNSUPPORTED"
    else:
        status = "INCOMPLETE"
    return AnswerAssessment(
        expected_answer_type=question_plan.expected_answer_type,
        required_slots=question_plan.required_answer_slots,
        extracted_answer_slots=slots,
        slot_coverage=coverage,
        directness=1.0 if complete else (0.3 if candidate_answers else 0.0),
        abstained=abstained,
        critical_answer_claim_id=critical,
        completeness_status=status,
        explanation="answer passed slot, binding, support, and ambiguity gates" if complete else "answer failed slot, binding, support, ambiguity, or abstention gate",
        critical_answer_claim_missing=critical is None,
        candidate_answers=candidate_answers,
        selected_answer=selected,
        ambiguity=ambiguity,
        target_binding_valid=target_binding_valid,
        support_valid=support_valid,
    )


def _normalize_date_like(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _is_direct_answer_position(text: str, pos: int) -> bool:
    prefix = text[max(0, pos - 80):pos].lower()
    return any(p in prefix for p in ["answer is", "was created in", "was abolished in", "created in", "abolished in", "established in"])


def _is_hedged(text: str, pos: int) -> bool:
    window = text[max(0, pos - 120):pos + 120].lower()
    return any(p in window for p in ["cannot confirm", "unclear", "not clear", "may be", "might be", "not enough"])


def infer_failure_types(answer_assessment: Dict[str, Any], hypothesis_assessments: List[Dict[str, Any]], subgoals: List[Dict[str, Any]]) -> List[str]:
    failures = []
    status = answer_assessment.get("completeness_status")
    if status == "INCOMPLETE":
        failures.append("ANSWER_MISSING")
    elif status == "UNSUPPORTED":
        failures.append("ANSWER_UNSUPPORTED")
    elif status == "AMBIGUOUS":
        failures.append("ANSWER_AMBIGUOUS")
    elif status == "TARGET_UNRESOLVED":
        failures.append("INVALID_ENTITY_BINDING")
    if any(h.get("commitment_state") == "COMMITTED" and h.get("missing_constraints") for h in hypothesis_assessments):
        failures.append("PREMATURE_ENTITY_COMMITMENT")
    if any(s.get("required") and str(s.get("status")).upper() not in {"RESOLVED"} for s in subgoals):
        failures.append("DEPENDENCY_BROKEN")
    return list(dict.fromkeys(failures))


def build_shadow_repair_plan(
    failure_types: List[str],
    root_bad_hypotheses: List[str],
    unresolved_subgoals: List[str],
    blame_results: List[Dict[str, Any]],
    question_plan: Dict[str, Any],
    candidate_matrix: Dict[str, Any] = None,
    commitment_events: List[Dict[str, Any]] = None,
    claim_assessments: List[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    root = blame_results[0] if blame_results else {}
    required_constraints = question_plan.get("constraints", [])
    failed_subgoal = (unresolved_subgoals or root_bad_hypotheses or [None])[0]
    matrix = candidate_matrix or {
        "candidates": [],
        "commitment_gate": "all_required_constraints_satisfied",
        "required_constraint_ids": required_constraints,
    }
    root_is_decision = root.get("node_type") in {"commitment_event", "query_intent", "plan_query", "hypothesis"}
    root_temporally_repairable = bool(root.get("root_generator_llm")) and bool(root.get("rollback_valid"))
    diagnostic_status = (
        "REPAIRABLE_ROOT_FOUND"
        if root_is_decision and root_temporally_repairable and root.get("causal_path_valid", bool(root.get("causal_path_to_failure")))
        else "INSUFFICIENT_DIAGNOSTIC_EVIDENCE"
    )
    inherited = []
    evsets = []
    for assessment in claim_assessments or []:
        claim = assessment.get("claim", {})
        if assessment.get("status") == "VERIFIED" and assessment.get("evidence_isolation_valid", True):
            inherited.append(claim.get("claim_id"))
            evset = assessment.get("best_evidence_set") or {}
            if evset.get("evidence_set_id"):
                evsets.append({
                    "claim_id": claim.get("claim_id"),
                    "claim_text": claim.get("content"),
                    "EvidenceSet_id": evset.get("evidence_set_id"),
                    "evidence_span_ids": evset.get("evidence_span_ids", []),
                    "source_branch": claim.get("branch_id", "b0"),
                    "support_vector": assessment.get("support_vector", {}),
                    "valid_at_rollback": True,
                    "depends_on_invalidated_node": False,
                })
    actions = [
        {"action_id": "act_reopen", "action_type": "reopen_subgoal", "target_subgoal_id": failed_subgoal, "target_relation_id": None, "known_bindings": {}, "unknown_variable": None, "candidate_entity": None, "preconditions": [], "success_condition": "subgoal has a verified resolver", "failure_condition": "no legal evidence found", "budget": 1},
        {"action_id": "act_retrieve_relation", "action_type": "retrieve_relation", "target_subgoal_id": failed_subgoal, "target_relation_id": "next_required_relation", "known_bindings": {}, "unknown_variable": "target_variable", "candidate_entity": None, "preconditions": ["rollback context restored"], "success_condition": "relation-specific EvidenceSet found", "failure_condition": "retrieval miss", "budget": 2},
        {"action_id": "act_verify_constraint", "action_type": "verify_constraint", "target_subgoal_id": failed_subgoal, "target_relation_id": "next_required_relation", "known_bindings": {}, "unknown_variable": "target_variable", "candidate_entity": "candidate_entity", "preconditions": ["candidate evidence selected"], "success_condition": "identity constraint SATISFIED", "failure_condition": "CONTRADICTED or UNKNOWN", "budget": 1},
        {"action_id": "act_commit", "action_type": "commit_entity", "target_subgoal_id": failed_subgoal, "target_relation_id": None, "known_bindings": {}, "unknown_variable": "target_variable", "candidate_entity": "candidate_entity", "preconditions": ["commitment_allowed=true"], "success_condition": "candidate is committed without missing identity constraints", "failure_condition": "commitment gate fails", "budget": 1},
        {"action_id": "act_retrieve_answer", "action_type": "retrieve_answer", "target_subgoal_id": None, "target_relation_id": "answer_relation", "known_bindings": {}, "unknown_variable": "answer", "candidate_entity": "committed_entity", "preconditions": ["target committed"], "success_condition": "answer EvidenceSet found", "failure_condition": "answer missing", "budget": 2},
        {"action_id": "act_verify_answer", "action_type": "verify_answer", "target_subgoal_id": None, "target_relation_id": "answer_relation", "known_bindings": {}, "unknown_variable": "answer", "candidate_entity": "committed_entity", "preconditions": ["answer evidence selected"], "success_condition": "answer claim VERIFIED with isolated EvidenceSet", "failure_condition": "unsupported or ambiguous answer", "budget": 1},
    ]
    return clean_json({
        "repair_plan_id": f"repair_{stable_hash(failure_types, root_bad_hypotheses, unresolved_subgoals)}",
        "mode": "shadow_dry_run",
        "diagnostic_status": diagnostic_status,
        "diagnostic_confidence": root.get("diagnostic_confidence", 0.0),
        "created_at": utc_now(),
        "failed_claim_or_subgoal": failed_subgoal,
        "failed_subgoal": failed_subgoal,
        "root_cause_node": root.get("node_id"),
        "root_generator_llm": root.get("root_generator_llm"),
        "alternative_root_candidates": blame_results[1:4],
        "causal_path_to_failure": root.get("causal_path_to_failure", []),
        "rollback_checkpoint": root.get("rollback_checkpoint") or root.get("node_id"),
        "failure_types": failure_types,
        "verified_inherited_claims": inherited,
        "inherited_minimal_evidence_sets": evsets,
        "invalidated_nodes": root.get("affected_downstream_nodes", []),
        "candidate_constraint_matrix": matrix,
        "rejected_hypothesis": {
            "content": "A partially constrained entity binding should not be treated as the final target entity.",
            "status": "rejected_under_current_evidence",
            "failure_type": "PREMATURE_ENTITY_COMMITMENT" if "PREMATURE_ENTITY_COMMITMENT" in failure_types else (failure_types[0] if failure_types else "UNKNOWN"),
            "reason": "missing required constraints remain unresolved",
            "scope": "current_branch",
            "reconsider_if": "new evidence satisfies the missing constraints",
        },
        "rejected_bindings": root_bad_hypotheses,
        "preserved_bindings": [],
        "open_subgoals": unresolved_subgoals,
        "required_constraints": required_constraints,
        "forbidden_query_patterns": ["single-hop query that ignores required constraints"],
        "rejected_commitments": [e for e in commitment_events or [] if e.get("is_premature")],
        "recommended_next_actions": actions,
        "query_templates": ["{known_entity} {relation_predicate} {unknown_variable}", "{committed_entity} {answer_relation}"],
        "success_criteria": [
            "all required subgoals resolved",
            "candidate commitment_allowed=true",
            "answer has exactly one normalized candidate",
            "answer claim VERIFIED with isolated EvidenceSet",
        ],
        "suggested_next_query": "Use the action sequence; do not commit an entity until required constraints pass.",
        "expected_support_gain": root.get("expected_support_gain", 0.0),
        "estimated_cost": root.get("repair_cost", 1),
    })
