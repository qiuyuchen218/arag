import pytest

from arag.core.schemas import EvidenceSpan
from arag.repair import BranchManager
from arag.utils.trace_graph import TraceGraph
from arag.verification import (
    Claim,
    ClaimSupportScorer,
    FakeVerificationBackend,
    VerificationResult,
    _as_probability,
    failure_frontier,
)


def test_correct_authorized_proof_does_not_need_repair():
    span = EvidenceSpan.from_text("The answer is June 1982.", "c", "doc", "1", 0)
    claim = Claim("claim_ok", "The answer is June 1982.", generated_by="ans_001", step_index=4)
    result = VerificationResult(0.95, 0.01, 0.04, verifier_mode="real_uncalibrated", authoritative=True)
    assessment = ClaimSupportScorer(FakeVerificationBackend(result)).score(claim, [span], [span.span_id])
    assert assessment["status"] == "VERIFIED"
    assert failure_frontier([assessment]) == []


def test_future_evidence_is_rejected_by_validator():
    trace = TraceGraph("s", "d")
    trace.add_question("q")
    llm = trace.add_llm_call("m", 1)
    answer = trace.add_answer(llm, "a", 1, "final_answer")
    span = EvidenceSpan.from_text("future", "c", "doc", "1", 0)
    trace.add_evidence_span({**span.__dict__, "step_index": 8})
    assessment = ClaimSupportScorer(FakeVerificationBackend(authoritative_for_test=True)).score(
        Claim("claim_future", "future", generated_by=answer),
        [span],
        delivered_span_ids=[],
    )
    trace.add_claim_assessment(assessment, answer)
    assert assessment["status"] == "INVALID_PROVENANCE"


def test_retrieved_but_not_delivered_cannot_support_claim():
    span = EvidenceSpan.from_text("retrieved only", "c", "doc", "1", 0)
    assessment = ClaimSupportScorer(FakeVerificationBackend(authoritative_for_test=True)).score(
        Claim("claim_retrieved", "retrieved only"),
        [span],
        delivered_span_ids=[],
    )
    assert assessment["provenance_gate"]["delivered_in_context"] is False
    assert assessment["status"] == "INVALID_PROVENANCE"


def test_planned_branch_cannot_be_selected_until_completed():
    manager = BranchManager()
    branch = manager.fork("b0", "pq_001", "ctx_001", "repair")
    with pytest.raises(ValueError):
        manager.select(branch.branch_id)
    manager.mark_completed(branch.branch_id, cost=1.0)
    selected = manager.select(branch.branch_id)
    assert selected.status == "selected"


def test_claim_and_evidence_set_are_graph_nodes():
    trace = TraceGraph("s", "d")
    q = trace.add_question("q")
    llm = trace.add_llm_call("m", 1)
    ans = trace.add_answer(llm, "The answer.", 1, "final_answer")
    span = EvidenceSpan.from_text("The answer.", "c", "doc", "1", 0)
    trace.add_evidence_span(span.__dict__)
    assessment = ClaimSupportScorer(FakeVerificationBackend()).score(
        Claim("claim_graph", "The answer.", generated_by=ans),
        [span],
        [span.span_id],
    )
    trace.add_claim_assessment(assessment, ans)
    graph = trace.to_dict()
    assert q == "q_001"
    assert any(n["type"] == "claim" and n["id"] == "claim_graph" for n in graph["nodes"])
    assert any(n["type"] == "evidence_set" for n in graph["nodes"])
    assert any(e["type"] == "jointly_supports" for e in graph["edges"])


def test_verifier_probability_parser_accepts_qualitative_labels():
    assert _as_probability("low", default=1.0) == 0.15
    assert _as_probability("high", default=0.0) == 0.85
    assert _as_probability("0.72", default=0.0) == 0.72


def test_verifier_only_sees_selected_evidence_set_spans():
    spans = [
        EvidenceSpan.from_text("Paris is in France.", "c", "d", "1", 0),
        EvidenceSpan.from_text("Berlin is in Germany.", "c", "d", "2", 0),
        EvidenceSpan.from_text("Madrid is in Spain.", "c", "d", "3", 0),
        EvidenceSpan.from_text("Rome is in Italy.", "c", "d", "4", 0),
    ]
    claim = Claim("claim_iso", "Paris is in France.")
    assessment = ClaimSupportScorer(FakeVerificationBackend(authoritative_for_test=True)).score(
        claim,
        spans,
        [s.span_id for s in spans],
    )
    assert set(assessment["verifier_input_span_ids"]) == set(assessment["best_evidence_set"]["evidence_span_ids"])
    assert len(assessment["verifier_input_span_ids"]) <= 3
    assert assessment["evidence_isolation_valid"] is True


def test_verifier_external_span_reference_invalidates_support():
    span = EvidenceSpan.from_text("Paris is in France.", "c", "d", "1", 0)
    result = VerificationResult(
        0.95,
        0.0,
        0.05,
        relevance=1.0,
        explanation="supported by span_deadbeef",
        authoritative=True,
        referenced_span_ids=["span_deadbeef"],
    )
    assessment = ClaimSupportScorer(FakeVerificationBackend(result)).score(
        Claim("claim_leak", "Paris is in France."),
        [span],
        [span.span_id],
    )
    assert assessment["status"] == "INVALID_PROVENANCE"
    assert assessment["support_vector"]["P"] == 0.0
    assert assessment["evidence_isolation_valid"] is False


def test_world_knowledge_without_evidence_is_not_verified():
    span = EvidenceSpan.from_text("The document discusses capitals generally.", "c", "d", "1", 0)
    result = VerificationResult(
        0.1,
        0.0,
        0.9,
        relevance=0.1,
        explanation="true by world knowledge, not by evidence",
        authoritative=True,
        world_knowledge_plausibility=0.99,
        evidence_entailment=0.1,
        evidence_relevance=0.1,
    )
    assessment = ClaimSupportScorer(FakeVerificationBackend(result)).score(
        Claim("claim_world", "Paris is the capital of France."),
        [span],
        [span.span_id],
    )
    assert assessment["status"] == "UNSUPPORTED"


def test_high_entailment_low_relevance_is_not_verified_or_minimal():
    span = EvidenceSpan.from_text("Paris is in France.", "c", "d", "1", 0)
    result = VerificationResult(0.95, 0.0, 0.05, relevance=0.1, authoritative=True)
    assessment = ClaimSupportScorer(FakeVerificationBackend(result)).score(
        Claim("claim_low_r", "Paris is in France."),
        [span],
        [span.span_id],
    )
    assert assessment["status"] != "VERIFIED"
    assert assessment["best_evidence_set"]["minimal_sufficient"] is False
