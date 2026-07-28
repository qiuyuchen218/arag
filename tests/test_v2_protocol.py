import json

from arag.core.context import AgentContext
from arag.core.schemas import EvidenceSpan, ToolResult
from arag.repair import BlameEngine, BranchManager, rejected_hypothesis
from arag.tools.keyword_search import KeywordSearchTool
from arag.tools.read_chunk import ReadChunkTool
from arag.verification import (
    Claim,
    ClaimSupportScorer,
    FakeVerificationBackend,
    VerificationResult,
    entropy_u,
    failure_frontier,
)


def write_chunks(path):
    path.write_text(json.dumps([
        {"id": "1", "doc_id": "a", "text": "Alpha beta gamma. Alpha supports claim."},
        {"id": "2", "doc_id": "a", "text": "Alpha beta gamma repeated nearby."},
        {"id": "3", "doc_id": "b", "text": "Delta evidence contradicts nothing."},
    ]), encoding="utf-8")


def test_tool_result_serialization():
    result = ToolResult("c1", "keyword_search", "success", "hello", [{"x": float("nan")}])
    data = result.to_dict()
    assert data["results"][0]["x"] is None
    json.dumps(data, allow_nan=False)


def test_keyword_bm25_phrase_span_and_doc_collapse(tmp_path):
    chunks = tmp_path / "chunks.json"
    write_chunks(chunks)
    context = AgentContext()
    rendered, log = KeywordSearchTool(str(chunks)).execute(context, keywords=['"Alpha beta" OR Delta'], top_k=2)
    payload = log["tool_result"]
    assert "Chunk ID" in rendered
    assert payload["diagnostics"]["candidate_count"] >= 2
    assert len(payload["results"]) == 2
    assert payload["results"][0]["candidate_rank"] == 1
    assert payload["results"][0]["matched_spans"][0]["span_id"].startswith("span_")


def test_read_cache_still_delivers_content_to_new_context(tmp_path):
    chunks = tmp_path / "chunks.json"
    write_chunks(chunks)
    tool = ReadChunkTool(str(chunks))
    context = AgentContext()
    first, first_log = tool.execute(context, chunk_ids=["1"])
    second, second_log = tool.execute(context, chunk_ids=["1"])
    assert "Alpha beta gamma" in first
    assert "Alpha beta gamma" in second
    assert second_log["tool_result"]["results"][0]["from_cache"] is True
    assert second_log["tool_result"]["diagnostics"]["returned_span_ids"]


def test_provenance_gate_invalid_without_delivery():
    span = EvidenceSpan.from_text("Paris is in France.", "c", "d", "1", 0)
    claim = Claim("claim1", "Paris is in France.")
    scorer = ClaimSupportScorer(FakeVerificationBackend(authoritative_for_test=True))
    assessment = scorer.score(claim, [span], delivered_span_ids=[])
    assert assessment["status"] == "INVALID_PROVENANCE"
    assert assessment["provenance_gate"]["G_prov"] == 0.0


def test_default_fake_verifier_is_unassessed_not_authoritative():
    span = EvidenceSpan.from_text("Paris is in France.", "c", "d", "1", 0)
    assessment = ClaimSupportScorer(FakeVerificationBackend()).score(
        Claim("claim_fake", "Paris is in France."),
        [span],
        [span.span_id],
    )
    assert assessment["status"] == "UNASSESSED"
    assert assessment["diagnostic_status"] == "FAKE_SUPPORTED"
    assert assessment["authoritative"] is False
    assert assessment["repair_eligible"] is False


def test_contradiction_lowers_support_and_entropy_is_normalized():
    span = EvidenceSpan.from_text("Paris is in France.", "c", "d", "1", 0)
    claim = Claim("claim1", "Paris is in Germany.")
    backend = FakeVerificationBackend(VerificationResult(0.1, 0.8, 0.1, authoritative=True), authoritative_for_test=True)
    assessment = ClaimSupportScorer(backend).score(claim, [span], [span.span_id])
    assert assessment["status"] == "CONTRADICTED"
    assert 0 <= entropy_u(0.1, 0.8, 0.1) <= 1


def test_failure_frontier_and_estimated_blame():
    assessments = [
        {"claim": {"claim_id": "a", "dependencies": []}, "status": "UNSUPPORTED", "authoritative": True, "defect_vector": {"1-R": 1.0}},
        {"claim": {"claim_id": "b", "dependencies": ["a"]}, "status": "UNSUPPORTED", "authoritative": True, "defect_vector": {"1-R": 1.0}},
    ]
    assert failure_frontier(assessments) == ["a"]
    trace = {"nodes": [{"id": "pq_001", "type": "plan_query", "step_index": 2}]}
    blame = BlameEngine().score(assessments[0], trace)
    assert blame[0]["blame_type"] == "estimated"
    assert blame[0]["suggested_action"] == "rewrite_query"


def test_cognitive_blame_requires_causal_ancestor_and_has_downstream():
    trace = {
        "nodes": [
            {"id": "ctx_001", "type": "context_snapshot", "step_index": 1},
            {"id": "llm_001", "type": "llm_call", "step_index": 2},
            {"id": "pq_004", "type": "plan_query", "step_index": 3},
            {"id": "hyp_k", "type": "hypothesis", "step_index": None, "metadata": {"first_proposed_at": 3}},
            {"id": "commit_k", "type": "commitment_event", "step_index": None, "metadata": {"step_index": 4, "generated_by_llm_call_id": "llm_001", "source_event_id": "pq_004"}},
            {"id": "subgoal_x", "type": "subgoal", "step_index": None},
        ],
        "edges": [
            {"source": "ctx_001", "target": "llm_001", "type": "consumed_by"},
            {"source": "llm_001", "target": "pq_004", "type": "invokes"},
            {"source": "pq_004", "target": "hyp_k", "type": "proposes"},
            {"source": "hyp_k", "target": "commit_k", "type": "generates"},
            {"source": "commit_k", "target": "pq_004", "type": "motivates"},
            {"source": "hyp_k", "target": "subgoal_x", "type": "proposed_for"},
            {"source": "commit_k", "target": "subgoal_x", "type": "proposed_for"},
        ],
    }
    hypotheses = [{
        "hypothesis_id": "hyp_k",
        "commitment_event_id": "commit_k",
        "commitment_state": "COMMITTED",
        "missing_constraints": ["rel_a"],
        "constraint_results": {"rel_a": {"status": "unknown"}},
    }]
    unresolved = [{"subgoal_id": "subgoal_x", "required": True}]
    blame = BlameEngine().score_cognitive(["PREMATURE_ENTITY_COMMITMENT"], unresolved, hypotheses, trace)
    assert blame
    assert blame[0]["affected_downstream_nodes"]
    assert blame[0]["rollback_checkpoint"] == "ctx_001"


def test_branch_fork_and_rejected_hypothesis():
    manager = BranchManager()
    hyp = rejected_hypothesis("bad route", "low_support", "pq_001", "not supported")
    branch = manager.fork("b0", "pq_001", "llm_001", "repair", ["c1"], ["s1"], ["pq_001"], [hyp])
    assert branch.branch_id == "b1"
    assert branch.constraints[0]["reconsider_if"]
