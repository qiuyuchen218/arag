import json

from arag.cognition import (
    FakeQuestionDecomposer,
    OnlineHypothesisTracker,
    build_candidate_constraint_matrix,
    parse_query_intent,
)
from arag.core.schemas import EvidenceSpan
from arag.repair import BlameEngine
from arag.verification import Claim, ClaimSupportScorer, FakeVerificationBackend


def test_entity_renaming_invariance_for_query_intent_shape():
    plan_a = FakeQuestionDecomposer().decompose("When was Entity_A's birthplace abolished?", "qa")
    plan_b = FakeQuestionDecomposer().decompose("When was RandomName_X's birthplace abolished?", "qb")
    intent_a = parse_query_intent(plan_a, "Entity_A birthplace")
    intent_b = parse_query_intent(plan_b, "RandomName_X birthplace")
    assert intent_a.query_mode == intent_b.query_mode
    assert intent_a.predicate == intent_b.predicate
    assert len(plan_a.relations) == len(plan_b.relations)
    assert len(plan_a.subgoals) == len(plan_b.subgoals)


def test_query_paraphrases_map_to_equivalent_created_predicate():
    plan = FakeQuestionDecomposer().decompose("When was Region_A created?", "q")
    predicates = {
        parse_query_intent(plan, "When was Region_A established?").predicate,
        parse_query_intent(plan, "What year was Region_A founded?").predicate,
        parse_query_intent(plan, "Give the creation date of Region_A.").predicate,
    }
    assert predicates == {"created_date"}


def test_predicate_stripping_keeps_canonical_entity_small():
    plan = FakeQuestionDecomposer().decompose("When was the target region created?", "q")
    intent = parse_query_intent(plan, "Region_A created or established as a country")
    assert intent.candidate_entities == ["Region_A"]
    assert intent.predicate == "created_date"
    assert all("created" not in e.lower() and "established" not in e.lower() for e in intent.candidate_entities)


def test_exploration_vs_commitment_event():
    plan = FakeQuestionDecomposer().decompose("When was the target region created?", "q")
    tracker = OnlineHypothesisTracker(plan)
    tracker.observe_query(parse_query_intent(plan, "possible locations related to Event_A"))
    assert not tracker.commitment_dicts()
    tracker.observe_query(parse_query_intent(plan, "when was Candidate_A created", source_plan_query_id="pq_1", generated_by_llm_call_id="llm_1", step_index=3))
    assert tracker.commitment_dicts()[0]["is_premature"] is True


def test_candidate_matrix_has_unknown_without_evidence_and_supported_with_evidence():
    plan = FakeQuestionDecomposer().decompose("When was the target region created?", "q")
    tracker = OnlineHypothesisTracker(plan)
    tracker.observe_query(parse_query_intent(plan, "Region_A history"), ["span_a"])
    tracker.observe_query(parse_query_intent(plan, "Region_B history"))
    matrix = build_candidate_constraint_matrix(plan, tracker.hypothesis_dicts())
    by_entity = {c["candidate_entity"]: c for c in matrix["candidates"]}
    assert any(cr["status"] == "SATISFIED" for cr in by_entity["Region_A"]["constraint_results"].values())
    assert any(cr["status"] == "UNKNOWN" for cr in by_entity["Region_B"]["constraint_results"].values())


def test_duplicate_and_distractor_evidence_do_not_inflate_minimal_set():
    support = EvidenceSpan.from_text("Entity_A was created in 1901.", "c", "doc", "1", 0)
    duplicate = EvidenceSpan(**{**support.__dict__})
    distractor = EvidenceSpan.from_text("Unrelated_B was created in 2002.", "c", "doc", "2", 0)
    claim = Claim("claim_a", "Entity_A was created in 1901.")
    scorer = ClaimSupportScorer(FakeVerificationBackend(authoritative_for_test=True))
    assessment = scorer.score(claim, [support, duplicate, distractor], [support.span_id, duplicate.span_id, distractor.span_id])
    assert len(assessment["best_evidence_set"]["evidence_span_ids"]) <= 3
    assert assessment["support_vector"]["D"] <= 1.0


def test_posthoc_hypothesis_cannot_be_cognitive_root():
    trace = {
        "nodes": [
            {"id": "ctx_1", "type": "context_snapshot", "step_index": 1},
            {"id": "llm_1", "type": "llm_call", "step_index": 2},
            {"id": "hyp_post", "type": "hypothesis", "metadata": {"posthoc_summary": True, "first_proposed_at": 9}},
            {"id": "subgoal_x", "type": "subgoal"},
        ],
        "edges": [
            {"source": "hyp_post", "target": "subgoal_x", "type": "proposed_for"},
        ],
    }
    blame = BlameEngine().score_cognitive(
        ["PREMATURE_ENTITY_COMMITMENT"],
        [{"subgoal_id": "subgoal_x", "required": True}],
        [{"hypothesis_id": "hyp_post", "posthoc_summary": True, "commitment_state": "COMMITTED", "missing_constraints": ["rel"]}],
        trace,
    )
    assert all(b["node_id"] != "hyp_post" for b in blame)


def test_ambiguous_parser_low_confidence_no_commitment():
    plan = FakeQuestionDecomposer().decompose("When was the target created?", "q")
    intent = parse_query_intent(plan, "created date location")
    tracker = OnlineHypothesisTracker(plan)
    tracker.observe_query(intent)
    assert intent.parser_confidence < 0.6
    assert intent.candidate_entities == []
    assert not tracker.commitment_dicts()
