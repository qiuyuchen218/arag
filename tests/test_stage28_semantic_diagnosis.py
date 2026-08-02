from arag.cognition import (
    canonicalize_entity,
    extract_structured_propositions,
    ground_relations,
    heuristic_question_plan,
    parse_query_intent,
    assess_answer,
)
from arag.core.schemas import EvidenceSpan
from arag.verification import Claim, ClaimSupportScorer, FakeVerificationBackend, VerificationResult


def test_possessive_question_plan_has_semantic_relation_dag():
    plan = heuristic_question_plan("When was Lady Godiva's birthplace abolished?", "q1")
    assert plan.schema_valid is True
    assert plan.semantic_valid is True
    predicates = [r["predicate"] for r in plan.relations]
    assert predicates == ["birthplace", "abolished_date"]
    assert plan.dependency_edges
    assert plan.relations[1]["dependencies"] == [plan.relations[0]["relation_id"]]


def test_generic_fallback_is_not_semantically_valid():
    plan = heuristic_question_plan("Explain the thing?", "q2")
    assert plan.schema_valid is True
    assert plan.semantic_valid is False
    assert plan.planner_confidence < 0.5
    assert plan.relations[0]["relation_type"] == "FALLBACK_PLACEHOLDER"


def test_query_intent_aligns_to_relation_and_marks_open_relation_invalid():
    plan = heuristic_question_plan("When was Lady Godiva's birthplace abolished?", "q3")
    qi = parse_query_intent(plan, "Lady Godiva birthplace", tool_name="keyword_search")
    assert qi.aligned_relation_id == plan.relations[0]["relation_id"]
    assert qi.semantic_valid is True
    assert qi.alignment_score > 0

    fallback = heuristic_question_plan("Explain the thing?", "q4")
    bad_qi = parse_query_intent(fallback, "random broad search", tool_name="keyword_search")
    assert bad_qi.semantic_valid is False
    assert "aligned_relation_is_fallback_placeholder" in bad_qi.parse_warnings or bad_qi.aligned_relation_id is None


def test_canonicalization_preserves_internal_digits_and_type_words():
    assert canonicalize_entity("Entity 2 County Club University") == "Entity 2 County Club University"
    assert canonicalize_entity("When") is None


def test_structured_propositions_ground_temporal_relation():
    plan = heuristic_question_plan("When was Appleford created?", "q5")
    span = EvidenceSpan(
        span_id="span_a",
        artifact_id="artifact_a",
        doc_id="1",
        chunk_id="1",
        sentence_id=1,
        text="Appleford was created in 1930.",
        start_offset=0,
        end_offset=30,
        content_hash="h",
    )
    claim = Claim("claim_a", "Appleford was created in 1930.", aligned_relation_ids=[plan.relations[0]["relation_id"]], resolves_subgoal_ids=[plan.relations[0]["subgoal_id"]])
    assessment = {
        "claim": claim.__dict__,
        "evidence_status": "VERIFIED",
        "status": "VERIFIED",
        "evidence_set_span_ids": ["span_a"],
    }
    props = extract_structured_propositions(plan, [span], [assessment], "1930")
    groundings = ground_relations(plan, props)
    assert any(g["status"] == "SATISFIED" and g["relation_id"] == plan.relations[0]["relation_id"] for g in groundings)


def test_claim_evidence_status_not_overwritten_by_dependency_block():
    claim = Claim("claim_b", "Appleford was created in 1930.", aligned_relation_ids=["rel_created"], dependencies=["sg_parent"])
    span = EvidenceSpan(
        span_id="span_b",
        artifact_id="artifact_b",
        doc_id="1",
        chunk_id="1",
        sentence_id=1,
        text="Appleford was created in 1930.",
        start_offset=0,
        end_offset=30,
        content_hash="h",
    )
    backend = FakeVerificationBackend(
        VerificationResult(
            0.95,
            0.0,
            0.05,
            verifier_mode="real_uncalibrated",
            authoritative=True,
            evidence_entailment=0.95,
            evidence_relevance=1.0,
            evidence_contradiction=0.0,
            insufficient_evidence=0.05,
        )
    )
    result = ClaimSupportScorer(backend).score(
        claim,
        [span],
        ["span_b"],
        parent_supports=[0.0],
        dependency_ids=["sg_parent"],
        blocked_dependency_ids=["sg_parent"],
        plan_semantic_valid=True,
    )
    assert result["evidence_status"] == "VERIFIED"
    assert result["reasoning_status"] == "DEPENDENCY_BLOCKED"
    assert result["overall_status"] == "DEPENDENCY_BLOCKED"


def test_plan_uncertain_propagates_to_answer_assessment():
    plan = heuristic_question_plan("Explain the thing?", "q6")
    assessment = assess_answer(plan, "It was created in 1930.", [], [], [])
    assert assessment.completeness_status == "PLAN_UNCERTAIN"
    assert assessment.support_status == "PLAN_UNCERTAIN"
