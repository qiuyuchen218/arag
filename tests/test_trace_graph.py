import json

from arag.utils.claim_extractor import extract_claims, extract_intermediate_claims
from arag.utils.trace_error import normalize_error
from arag.utils.trace_graph import TraceGraph


def _assert_valid(trace: TraceGraph):
    graph = trace.to_dict()
    trace.validate()
    node_ids = {node["id"] for node in graph["nodes"]}
    assert graph["trace_schema_version"] == "2.1"
    assert all({"id", "type", "content", "metadata", "timestamp", "status"} <= set(n)
               for n in graph["nodes"])
    assert all({"source", "target", "type", "metadata"} <= set(e) for e in graph["edges"])
    assert all(e["source"] in node_ids and e["target"] in node_ids for e in graph["edges"])
    json.dumps(graph, allow_nan=False)
    return graph


def test_proxy_failure_graph():
    trace = TraceGraph("sample", "questions")
    trace.add_question("question")
    llm = trace.add_llm_call(model="qwen", loop=1, status="failed")
    trace.add_error(llm, "ProxyError: 127.0.0.1 connection refused", "llm_call", 1, "llm_api_error")
    trace.add_answer(None, "", 1, "llm_api_error", failed=True)
    graph = _assert_valid(trace)
    assert not [n for n in graph["nodes"] if n["type"] in {"llm_analysis", "claim"}]
    assert graph["metadata"]["normalized_termination_reason"] == "infrastructure_error"


def test_execution_graph_success_and_chunk_dedupe():
    trace = TraceGraph("sample", "questions")
    q = trace.add_question("Who and when?")
    llm = trace.add_llm_call(
        model="mock",
        loop=1,
        message={"content": "", "tool_calls": [{"id": "tc1"}]},
        response={"input_tokens": 10, "output_tokens": 4, "cost": 0.1},
    )
    pq = trace.add_plan_query(llm, "person signing date", "semantic_search", {"query": "person signing date"}, 1, "tc1")
    ret = trace.add_retriever_call(pq, "semantic_search", "person signing date", {"query": "person signing date"}, 1, "tc1")
    c1 = trace.add_retrieval_result(
        ret,
        "Person joined the club.",
        {"chunk_id": "1", "doc_id": "doc1", "loop": 1},
        {"loop": 1, "tool_call_id": "tc1", "tool_name": "semantic_search", "rank": 1, "raw_score": .9},
    )
    c1_again = trace.add_retrieval_result(
        ret,
        "Person joined the club.",
        {"chunk_id": "1", "doc_id": "doc1", "loop": 1},
        {"loop": 1, "tool_call_id": "tc1", "tool_name": "semantic_search", "rank": 2, "raw_score": .7},
    )
    assert q == "q_001"
    assert c1 == c1_again
    answer = trace.add_answer(llm, "The answer.", 1, "final_answer")
    graph = _assert_valid(trace)
    assert [n["type"] for n in graph["nodes"]].count("retrieved_chunk") == 1
    assert {e["type"] for e in graph["edges"]} >= {"next", "invokes", "executes", "retrieves", "generates"}
    assert len([e for e in graph["edges"] if e["type"] == "retrieves"]) == 2
    assert graph["metadata"]["retrieval_occurrence_count"] == 2
    assert graph["metadata"]["unique_chunk_count"] == 1
    assert answer.startswith("ans_")


def test_read_chunk_updates_artifact_without_new_timeline_step():
    trace = TraceGraph("read", "questions")
    trace.add_question("question")
    llm = trace.add_llm_call(model="mock", loop=1)
    pq = trace.add_plan_query(llm, "read chunk: 7", "read_chunk", {"chunk_id": 7}, 1, "tc-read")
    ret = trace.add_retriever_call(pq, "read_chunk", "read chunk: 7", {"chunk_id": 7}, 1, "tc-read", operation="read")
    chunk = trace.add_retrieval_result(
        ret,
        "Full chunk text",
        {"chunk_id": "7", "doc_id": "doc7", "loop": 1, "source": "read_chunk"},
        {"loop": 1, "tool_call_id": "tc-read", "tool_name": "read_chunk", "rank": 1, "operation": "read"},
        edge_type="reads",
    )
    trace.add_answer(llm, "Done.", 1, "final_answer")
    graph = _assert_valid(trace)
    node = next(n for n in graph["nodes"] if n["id"] == chunk)
    assert node["step_index"] is None
    assert node["metadata"]["full_content_available"] is True
    assert graph["metadata"]["read_occurrence_count"] == 1


def test_empty_retrieval_is_runtime_warning_not_evidence_judgment():
    trace = TraceGraph("empty", "questions")
    trace.add_question("question")
    llm = trace.add_llm_call(model="mock", loop=1)
    pq = trace.add_plan_query(llm, "no hits", "semantic_search", {"query": "no hits"}, 1, "tc1")
    ret = trace.add_retriever_call(pq, "semantic_search", "no hits", {"query": "no hits"}, 1, "tc1")
    trace.add_error(ret, "empty retrieval", "retrieval", 1, "no_retrieval", fatal=False)
    trace.add_answer(llm, "An answer", 1, "final_answer")
    graph = _assert_valid(trace)
    assert "evidence_missing" not in graph["metadata"]
    assert not [n for n in graph["nodes"] if n["type"] == "claim"]
    assert graph["metadata"]["empty_retrieval_count"] == 1


def test_helpers_are_dataset_agnostic():
    assert extract_claims("Error: bad") == []
    assert len(extract_claims("One; Two.")) == 2
    assert extract_intermediate_claims("I found a useful fact. Let me search again.")[0]["content"] == "I found a useful fact."
    assert normalize_error("Authentication failed", "error")["error_subtype"] == "authentication_error"
