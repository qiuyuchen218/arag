from arag.cognition import (
    FakeQuestionDecomposer,
    OnlineHypothesisTracker,
    ObservableStateExtractor,
    assess_answer,
    build_candidate_constraint_matrix,
    build_shadow_repair_plan,
    classify_claim,
    infer_failure_types,
    parse_query_intent,
)


def test_when_question_without_date_is_answer_incomplete():
    plan = FakeQuestionDecomposer().decompose("When was the place abolished?", "q1")
    assessment = assess_answer(plan, "The documents do not contain this information.", [])
    assert assessment.completeness_status == "INCOMPLETE"
    assert assessment.slot_coverage == 0
    assert assessment.critical_answer_claim_missing is True


def test_process_and_list_text_are_not_critical_claims():
    assert classify_claim("1.")[1] == 0.0
    assert classify_claim("I searched many times.")[0] == "process_claim"
    assert classify_claim("The answer is 918.")[0] == "answer_claim"


def test_partial_entity_hypothesis_marks_missing_constraints():
    plan = FakeQuestionDecomposer().decompose("When was the target region created?", "q2")
    hyps = ObservableStateExtractor().extract(
        plan,
        "The target region may be Syria.",
        source_event_id="llm_001",
        visible_evidence_ids=["span_1"],
    )
    assert hyps
    assert hyps[0].status == "partially_supported"
    assert hyps[0].missing_constraints


def test_failure_types_and_shadow_repair_plan():
    plan = FakeQuestionDecomposer().decompose("When was the target region created?", "q3")
    answer = {"completeness_status": "INCOMPLETE"}
    hyps = [{"hypothesis_id": "hyp_1", "missing_constraints": ["second_constraint"], "commitment_state": "COMMITTED"}]
    subgoals = [sg.__dict__ for sg in plan.subgoals]
    failures = infer_failure_types(answer, hyps, subgoals)
    assert "ANSWER_MISSING" in failures
    assert "PREMATURE_ENTITY_COMMITMENT" in failures
    repair = build_shadow_repair_plan(
        failures,
        ["hyp_1"],
        [subgoals[0]["subgoal_id"]],
        [{"node_id": "hyp_1", "expected_support_gain": 0.4, "repair_cost": 1}],
        plan.to_dict(),
    )
    assert repair["mode"] == "shadow_dry_run"
    assert repair["rejected_hypothesis"]["status"] == "rejected_under_current_evidence"
    assert repair["required_constraints"]
    assert repair["recommended_next_actions"]


def test_multiple_distinct_dates_are_ambiguous_not_complete():
    plan = FakeQuestionDecomposer().decompose("When was the target region created?", "q4")
    assessment = assess_answer(
        plan,
        "The candidate dates are 1923, 1943, and 1946.",
        [{"claim": {"claim_id": "c1", "claim_type": "answer_claim", "criticality": 1.0}, "status": "VERIFIED"}],
    )
    assert assessment.completeness_status == "AMBIGUOUS"
    assert assessment.ambiguity == "multiple_distinct_answers"


def test_given_entity_mention_is_not_premature_commitment():
    plan = FakeQuestionDecomposer().decompose("When was Lady Godiva's birthplace abolished?", "q5")
    hyps = ObservableStateExtractor().extract(
        plan,
        "Lady Godiva was mentioned in the question.",
        source_event_id="pq_001",
    )
    assert not any(h.candidate_entity == "Lady Godiva" and h.missing_constraints for h in hyps)


def test_query_intent_splits_entity_from_predicate():
    plan = FakeQuestionDecomposer().decompose("When was Entity_A's birthplace abolished?", "q6")
    intent = parse_query_intent(plan, "Entity_A birthplace")
    assert intent.predicate == "birthplace"
    assert intent.known_bindings["person"] == "Entity_A"
    assert all("birthplace" not in ent.lower() for ent in intent.candidate_entities)

    created = parse_query_intent(plan, "when was Region_B established")
    assert created.candidate_entities == ["Region_B"]
    assert "established" not in created.candidate_entities[0].lower()

    read = parse_query_intent(plan, "read chunks: 123", tool_name="read_chunk")
    assert read.candidate_entities == []
    assert read.parse_warnings == ["read_intent_no_entity_proposal"]


def test_online_hypothesis_state_dedupes_and_exploration_is_uncommitted():
    plan = FakeQuestionDecomposer().decompose("When was the target region created?", "q7")
    tracker = OnlineHypothesisTracker(plan)
    first = parse_query_intent(plan, "Region_A history")
    second = parse_query_intent(plan, "Region_A background")
    tracker.observe_query(first, ["span_a"])
    tracker.observe_query(second, ["span_b"])
    states = tracker.hypothesis_dicts()
    assert len(states) == 1
    assert states[0]["canonical_entity"] == "Region_A"
    assert states[0]["commitment_state"] == "UNCOMMITTED"
    assert len(states[0]["update_history"]) == 2


def test_answer_lookup_query_can_create_premature_commitment_event():
    plan = FakeQuestionDecomposer().decompose("When was the target region created?", "q8")
    tracker = OnlineHypothesisTracker(plan)
    intent = parse_query_intent(
        plan,
        "Region_B created date",
        source_plan_query_id="pq_001",
        generated_by_llm_call_id="llm_001",
        step_index=5,
    )
    tracker.observe_query(intent)
    events = tracker.commitment_dicts()
    assert events
    assert events[0]["candidate_entity"] == "Region_B"
    assert events[0]["is_premature"] is True


def test_candidate_constraint_matrix_contains_all_candidates():
    plan = FakeQuestionDecomposer().decompose("When was the target region created?", "q9")
    tracker = OnlineHypothesisTracker(plan)
    tracker.observe_query(parse_query_intent(plan, "Region_A created"), ["span_a"])
    tracker.observe_query(parse_query_intent(plan, "Region_B created"))
    matrix = build_candidate_constraint_matrix(plan, tracker.hypothesis_dicts())
    assert {c["candidate_entity"] for c in matrix["candidates"]} == {"Region_A", "Region_B"}
    by_entity = {c["candidate_entity"]: c for c in matrix["candidates"]}
    assert by_entity["Region_A"]["identity_coverage"] > by_entity["Region_B"]["identity_coverage"]
    assert by_entity["Region_B"]["unknown_count"] > 0
