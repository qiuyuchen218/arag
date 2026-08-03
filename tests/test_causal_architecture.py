import pytest

from arag.cognition import (
    FakeQuestionDecomposer,
    build_shadow_repair_plan,
    extract_structured_propositions,
    ground_relations,
)
from arag.core.schemas import EvidenceSpan
from arag.agent.base import BaseAgent
from arag.repair import BlameEngine
from arag.tools.keyword_search import KeywordSearchTool
from arag.tools.read_chunk import ReadChunkTool
from arag.tools.semantic_search import SemanticSearchTool
from arag.utils.trace_graph import TraceGraph
from arag.verification import SimpleClaimExtractor


def test_decision_record_orders_tool_query_causally():
    trace = TraceGraph("s", "d")
    trace.add_question("question")
    ctx = trace.add_context_snapshot(None, {"snapshot_kind": "input_context"})
    llm = trace.add_llm_call(model="mock", loop=1)
    dec = trace.add_decision_record(llm, {
        "decision_type": "query_selection",
        "selected_tool": "keyword_search",
        "query_or_action": "alpha beta",
    }, input_context_node=ctx)
    pq = trace.add_plan_query(llm, "alpha beta", "keyword_search", {"keywords": ["alpha", "beta"]}, 1, decision_id=dec)
    trace.add_retriever_call(pq, "keyword_search", "alpha beta", {"keywords": ["alpha", "beta"]}, 1)
    trace.add_answer(llm, "Done.", 1, "final_answer")

    trace.validate()
    by_id = {n["id"]: n for n in trace.nodes}
    assert by_id[dec]["event_seq"] < by_id[pq]["event_seq"]
    assert any(e["source"] == dec and e["target"] == pq and e["metadata"]["causal"] for e in trace.edges)


def test_causal_edge_time_reversal_is_rejected():
    trace = TraceGraph("s", "d")
    trace.add_question("question")
    llm = trace.add_llm_call(model="mock", loop=1)
    dec = trace.add_decision_record(llm, {"decision_type": "query_selection", "query_or_action": "q"})
    trace.add_edge(dec, llm, "motivates", {"causal": True})
    trace.add_answer(llm, "Done.", 1, "final_answer")

    with pytest.raises(ValueError):
        trace.validate()
    assert any("CAUSAL_EDGE_TIME_REVERSAL" in e for e in trace.metadata["validation_errors"])


def test_markdown_list_fragments_are_not_critical_claims():
    claims = SimpleClaimExtractor().extract("Given that:\n1. Alpha was created in 1901.\nHowever:\n2.")
    by_text = {c.content: c for c in claims}
    assert by_text["Given that:"].claim_type == "incomplete_fragment"
    assert by_text["However:"].claim_type == "incomplete_fragment"
    assert all(c.criticality == 0.0 for c in claims if c.claim_type == "incomplete_fragment")


def test_date_without_relation_subject_predicate_does_not_satisfy_relation():
    plan = FakeQuestionDecomposer().decompose("When was Appleford created?", "q")
    span = EvidenceSpan.from_text("A different place was opened in 1930.", "c", "doc", "chunk", 0)
    props = extract_structured_propositions(plan, [span], [], "")
    groundings = ground_relations(plan, props)
    assert all(g["status"] != "SATISFIED" for g in groundings)


def test_repair_plan_hard_gates_unconfirmed_causal_root():
    plan = FakeQuestionDecomposer().decompose("When was the target region created?", "q")
    repair = build_shadow_repair_plan(
        ["PREMATURE_ENTITY_COMMITMENT"],
        ["hyp_a"],
        [plan.subgoals[0].subgoal_id],
        [{
            "node_id": "commit_001",
            "node_type": "commitment_event",
            "causal_path_valid": False,
            "root_to_failure_contains_execution": False,
            "rollback_valid": True,
            "root_generator_llm": "llm_001",
        }],
        plan.to_dict(),
    )
    assert repair["root_cause_node"] is None
    assert repair["candidate_root_node"] == "commit_001"
    assert repair["root_selection_status"] == "CANDIDATE_ROOT_UNCONFIRMED"


def test_blame_ignores_inferred_posthoc_edges():
    trace = {
        "nodes": [
            {"id": "llm_001", "type": "llm_call", "step_index": 1, "event_seq": 1, "branch_id": "b0"},
            {"id": "pq_001", "type": "plan_query", "step_index": 2, "event_seq": 2, "branch_id": "b0"},
            {"id": "commit_001", "type": "commitment_event", "metadata": {"step_index": 3}, "branch_id": "b0"},
            {"id": "subgoal_x", "type": "subgoal", "metadata": {}, "branch_id": "b0"},
        ],
        "edges": [
            {"source": "pq_001", "target": "commit_001", "type": "expresses", "metadata": {"causal": False, "inferred": True}},
            {"source": "commit_001", "target": "pq_001", "type": "motivates", "metadata": {"causal": False, "inferred": True}},
        ],
    }
    blame = BlameEngine().score_cognitive(
        ["PREMATURE_ENTITY_COMMITMENT"],
        [{"subgoal_id": "subgoal_x", "required": True}],
        [{"hypothesis_id": "hyp", "commitment_event_id": "commit_001", "commitment_state": "COMMITTED", "missing_constraints": ["rel"]}],
        trace,
    )
    assert not any(b.get("node_id") == "commit_001" and b.get("causal_path_valid") for b in blame)


def _promotion_trace(state="COMMITTED", support=0.0, missing=None, action_role="COMMIT"):
    trace = TraceGraph("s", "d")
    trace.add_question("question")
    ctx = trace.add_context_snapshot(None, {"snapshot_kind": "input_context"})
    llm = trace.add_llm_call(model="mock", loop=1)
    prop = trace.add_proposition_node({
        "subject": "Alpha",
        "predicate": "created_date",
        "object": "1901",
        "relation_id": "rel_created",
    })
    planned_decision = "dec_001"
    estate = trace.add_epistemic_state_event(
        prop,
        state,
        generated_by_decision_id=planned_decision,
        available_evidence_ids=[],
        support_score_at_event=support,
        missing_constraint_ids=missing if missing is not None else ["rel_created"],
        authoritative=True,
        action_role=action_role,
    )
    dec = trace.add_decision_record(llm, {
        "decision_id": planned_decision,
        "decision_type": "binding_commitment",
        "action_role": action_role,
        "query_or_action": "use proposition",
        "authoritative": True,
    }, input_context_node=ctx)
    trace.link_state_event_to_decision(estate, dec)
    pq = trace.add_plan_query(llm, "Alpha created date", "keyword_search", {"keywords": ["Alpha", "created"]}, 1, decision_id=dec)
    trace.add_retriever_call(pq, "keyword_search", "Alpha created date", {"keywords": ["Alpha", "created"]}, 1)
    trace.add_answer(llm, "Alpha was created in 1901.", 1, "final_answer")
    trace.validate()
    return trace.to_dict(), estate


def test_low_support_test_query_is_not_error_root():
    graph, estate = _promotion_trace(state="UNDER_TEST", support=0.0, missing=["rel_created"], action_role="TEST")
    blame = BlameEngine().score_cognitive(["ANSWER_UNSUPPORTED"], [], [], graph)
    assert all(b.get("node_id") != estate for b in blame)


def test_low_support_commitment_driving_query_is_root():
    graph, estate = _promotion_trace(state="COMMITTED", support=0.0, missing=["rel_created"], action_role="COMMIT")
    blame = BlameEngine().score_cognitive(["ANSWER_UNSUPPORTED"], [], [], graph)
    assert blame[0]["node_id"] == estate
    assert blame[0]["failure_type"] == "UNSUPPORTED_COMMITMENT"


def test_supported_commitment_is_not_error_root():
    graph, estate = _promotion_trace(state="COMMITTED", support=1.0, missing=[], action_role="COMMIT")
    blame = BlameEngine().score_cognitive(["ANSWER_UNSUPPORTED"], [], [], graph)
    assert all(b.get("node_id") != estate for b in blame)


def test_future_evidence_does_not_change_historical_support():
    graph, estate = _promotion_trace(state="COMMITTED", support=0.0, missing=["rel_created"], action_role="COMMIT")
    event = next(n for n in graph["nodes"] if n["id"] == estate)
    assert event["metadata"]["support_score_at_event"] == 0.0
    assert event["metadata"]["available_evidence_ids"] == []


def test_repeated_state_changes_are_append_only_and_first_bad_use_wins():
    trace = TraceGraph("s", "d")
    trace.add_question("question")
    ctx = trace.add_context_snapshot(None, {})
    llm = trace.add_llm_call(model="mock", loop=1)
    prop = trace.add_proposition_node({"subject": "A", "predicate": "p", "object": "B", "relation_id": "rel"})
    first = trace.add_epistemic_state_event(prop, "USED_AS_PREMISE", "dec_first", [], 0.0, ["rel"], True, action_role="USE_AS_PREMISE")
    dec1 = trace.add_decision_record(llm, {"decision_id": "dec_first", "decision_type": "binding_commitment", "action_role": "USE_AS_PREMISE", "query_or_action": "first", "authoritative": True}, input_context_node=ctx)
    trace.link_state_event_to_decision(first, dec1)
    pq1 = trace.add_plan_query(llm, "first downstream", "keyword_search", {"keywords": ["first"]}, 1, decision_id=dec1)
    trace.add_retriever_call(pq1, "keyword_search", "first downstream", {"keywords": ["first"]}, 1)
    second = trace.add_epistemic_state_event(prop, "COMMITTED", "dec_second", [], 0.0, ["rel"], True, action_role="COMMIT")
    dec2 = trace.add_decision_record(llm, {"decision_id": "dec_second", "decision_type": "binding_commitment", "action_role": "COMMIT", "query_or_action": "second", "authoritative": True}, input_context_node=ctx)
    trace.link_state_event_to_decision(second, dec2)
    trace.add_plan_query(llm, "second downstream", "keyword_search", {"keywords": ["second"]}, 1, decision_id=dec2)
    trace.add_answer(llm, "bad", 1, "final_answer")
    trace.validate()
    graph = trace.to_dict()
    states = [n for n in graph["nodes"] if n["type"] == "epistemic_state_event"]
    assert [s["id"] for s in states] == [first, second]
    blame = BlameEngine().score_cognitive(["ANSWER_UNSUPPORTED"], [], [], graph)
    assert blame[0]["node_id"] == first


def test_proposition_registry_reuses_normalized_identity():
    trace = TraceGraph("s", "d")
    p1 = trace.add_proposition_node({"subject": "Lady  Godiva", "predicate": "birthplace", "object": "Coventry", "relation_id": "rel_a"})
    p2 = trace.add_proposition_node({"subject": "lady godiva", "predicate": "birthplace", "object": "coventry", "relation_id": "rel_b"})
    assert p1 == p2
    node = next(n for n in trace.nodes if n["id"] == p1)
    assert node["metadata"]["relation_ids"] == ["rel_a", "rel_b"]


def test_no_authoritative_epistemic_decision_blocks_root_selection():
    trace = {
        "nodes": [
            {"id": "answer_1", "type": "answer", "step_index": 4, "event_seq": 4, "branch_id": "b0"},
        ],
        "edges": [],
    }
    blame = BlameEngine().score_cognitive(["ANSWER_UNSUPPORTED"], [], [], trace)
    assert blame[0]["failure_type"] == "OBSERVABILITY_GAP"
    assert blame[0]["diagnosis_state"] == "NO_AUTHORITATIVE_EPISTEMIC_DECISION"


def test_retrieval_tool_schemas_require_nonempty_epistemic_context():
    for cls in [KeywordSearchTool, SemanticSearchTool, ReadChunkTool]:
        tool = cls.__new__(cls)
        params = tool.get_schema()["function"]["parameters"]
        assert "epistemic_context" in params["required"]
        ctx_schema = params["properties"]["epistemic_context"]
        assert set(["action_role", "active_subgoal_ids", "purpose", "propositions"]).issubset(ctx_schema["required"])
        assert ctx_schema["properties"]["propositions"]["minItems"] == 1


def test_retrieval_coverage_failure_can_be_root_without_premise_error():
    trace = TraceGraph("s", "When was X abolished?")
    q = trace.add_question("When was X abolished?")
    sg = trace.add_subgoal_node({"subgoal_id": "sg_abolish", "content": "Find abolished date", "required": True}, q)
    llm = trace.add_llm_call(model="mock", loop=1)
    dec = trace.add_decision_record(llm, {"decision_type": "query_selection", "query_or_action": "X history", "authoritative": True})
    pq = trace.add_plan_query(llm, "X history", "keyword_search", {"keywords": ["X", "history"]}, 1, decision_id=dec)
    episode = trace.add_retrieval_episode({
        "subgoal_id": sg,
        "relation_id": "rel_abolish",
        "required_relation": "abolished_date",
        "query_decision_ids": [dec],
        "plan_query_ids": [pq],
        "coverage_state": "EXHAUSTED",
        "unresolved_reason": "terminated_before_required_relation_grounded",
    })
    cov = trace.add_coverage_assessment(episode, {
        "subgoal_id": sg,
        "relation_id": "rel_abolish",
        "required_relation": "abolished_date",
        "coverage_state": "EXHAUSTED",
        "plan_query_ids": [pq],
        "query_decision_ids": [dec],
    })
    trace.add_answer(llm, "Cannot answer.", 2, "final_answer")
    blame = BlameEngine().score_cognitive(["ANSWER_MISSING", "DEPENDENCY_BROKEN"], [{"subgoal_id": sg, "required": True}], [], trace.to_dict())
    assert blame[0]["node_id"] == cov
    assert blame[0]["failure_type"] == "RETRIEVAL_COVERAGE_FAILURE"


def test_operational_detector_marks_downstream_entity_binding_as_premise():
    agent = BaseAgent.__new__(BaseAgent)
    premises = agent._detect_operational_premise_use(
        "semantic_search",
        {"query": "Syria creation date"},
        {
            "targets": [{
                "subject": "Syria",
                "predicate": "creation_date",
                "object": "unknown",
                "relation_id": "rel_created",
                "stance": "HYPOTHESIS",
            }]
        },
        "EXPLORE",
    )
    assert premises
    assert premises[0]["predicate"] == "binding"
    assert premises[0]["object"] == "Syria"
    assert premises[0]["operational_detector"] == "downstream_relation_input_slot"


def test_operational_detector_does_not_mark_plain_coventry_test_as_premise():
    agent = BaseAgent.__new__(BaseAgent)
    premises = agent._detect_operational_premise_use(
        "keyword_search",
        {"keywords": ["Godiva", "Coventry"]},
        {
            "targets": [{
                "subject": "Lady Godiva",
                "predicate": "associated_with",
                "object": "Coventry",
                "relation_id": "rel_assoc",
                "stance": "HYPOTHESIS",
            }]
        },
        "TEST",
    )
    assert premises == []
