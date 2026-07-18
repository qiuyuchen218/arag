from arag.utils.claim_extractor import extract_claims, extract_intermediate_claims
from arag.utils.trace_error import normalize_error
from arag.utils.trace_graph import TraceGraph


def _assert_valid(graph):
    node_ids = {node["id"] for node in graph["nodes"]}
    assert all({"id", "type", "content", "metadata", "timestamp", "step_index", "status"} <= set(n)
               for n in graph["nodes"])
    assert all({"source", "target", "type", "metadata"} <= set(e) for e in graph["edges"])
    assert all(e["source"] in node_ids and e["target"] in node_ids for e in graph["edges"])


def test_proxy_failure_graph():
    trace = TraceGraph("sample", "questions")
    question = trace.add_node("question", "question")
    trace.add_temporal_next(question)
    llm = trace.add_node("llm_call", "model=qwen; loop=1", status="failed")
    trace.add_edge(question, llm, "calls")
    trace.add_temporal_next(llm)
    trace.add_error(llm, "ProxyError: 127.0.0.1 connection refused", "llm_call", 1, "llm_api_error")
    trace.add_answer(None, "", 1, "llm_api_error", failed=True)
    graph = trace.to_dict()
    _assert_valid(graph)
    assert not [n for n in graph["nodes"] if n["type"] in {"llm_analysis", "claim"}]
    assert graph["metadata"]["normalized_termination_reason"] == "infrastructure_error"


def test_mock_success_graph():
    trace = TraceGraph("sample", "questions")
    q = trace.add_node("question", "Who and when?")
    pq = trace.add_node("plan_query", "person signing date")
    ret = trace.add_node("retriever_call", "semantic_search")
    c1 = trace.add_evidence("Person joined the club.", {"chunk_id": "1", "rank": 1, "score": .9})
    c2 = trace.add_evidence("Signing was June 1982.", {"chunk_id": "2", "rank": 2, "score": .8})
    llm = trace.add_node("llm_call", "model=mock; loop=1")
    trace.add_edge(q, pq, "decomposes_to")
    trace.add_edge(pq, ret, "calls")
    trace.add_edge(ret, c1, "retrieves")
    trace.add_edge(ret, c2, "retrieves")
    trace.add_edge(c1, llm, "used_as_context")
    trace.add_edge(c2, llm, "used_as_context")
    trace.add_answer(llm, "The person joined Barcelona; The signing was June 1982", 1,
                     "final_answer", [c1, c2])
    graph = trace.to_dict()
    _assert_valid(graph)
    edge_types = {e["type"] for e in graph["edges"]}
    assert {"decomposes_to", "calls", "retrieves", "used_as_context", "generates",
            "composes_answer", "evidence_link"} <= edge_types
    assert len([n for n in graph["nodes"] if n["type"] == "claim"]) == 2


def test_empty_retrieval_and_unsupported_claim():
    trace = TraceGraph("empty", "questions")
    ret = trace.add_node("retriever_call", "semantic_search", {"num_results": 0})
    trace.add_error(ret, "empty retrieval", "retrieval", 1, "no_retrieval", fatal=False)
    llm = trace.add_node("llm_call", "model=mock")
    trace.add_answer(llm, "An unsupported answer", 1, "final_answer", [])
    graph = trace.to_dict()
    assert graph["metadata"]["evidence_missing"] is True
    assert next(n for n in graph["nodes"] if n["type"] == "claim")["metadata"]["support_status"] == "no_evidence"


def test_helpers():
    assert extract_claims("Error: bad") == []
    assert len(extract_claims("One; Two.")) == 2
    assert normalize_error("Authentication failed", "error")["error_subtype"] == "authentication_error"


def test_quote_aware_claim_splitter_and_short_answer():
    answer = ('Based on the information retrieved, Messi\'s goal was compared to Diego '
              'Maradona\'s "goal of the century." According to the document, Diego '
              'Maradona was signed in **June 1982**.')
    claims = extract_claims(answer)
    assert len(claims) == 2
    assert claims[0]["content"].endswith('"goal of the century."')
    assert claims[1]["content"].startswith("Diego Maradona")
    assert "**" not in claims[1]["content"]
    assert extract_claims("June 1982")[0]["content"] == "June 1982"


def test_intermediate_claim_filtering():
    claims = extract_intermediate_claims(
        "I found that Diego Maradona was signed by Barcelona in June 1982. "
        "Let me try a more targeted search."
    )
    assert [claim["content"] for claim in claims] == [
        "Diego Maradona was signed by Barcelona in June 1982."
    ]
    claims = extract_intermediate_claims(
        "I found the answer in Chunk 0! The semantic search revealed that Diego "
        "Maradona was signed by Barcelona in June 1982."
    )
    assert claims[0]["content"] == "Diego Maradona was signed by Barcelona in June 1982."
    assert extract_intermediate_claims("I found the answer in Chunk 0 from search results.") == []


def test_intermediate_claim_quote_and_suggest_normalization():
    claims = extract_intermediate_claims(
        'I found the answer in chunk 0 from the semantic search results. It states: '
        '"in june 1982, diego maradona was signed for a world record fee of £5 million '
        'from boca juniors".'
    )
    assert [claim["content"] for claim in claims] == [
        "Diego Maradona was signed by Barcelona in June 1982 "
        "for a world record fee of £5 million from Boca Juniors."
    ]
    claims = extract_intermediate_claims(
        "This suggests that Messi's goal was compared to Diego Maradona's famous goal."
    )
    assert [claim["content"] for claim in claims] == [
        "Messi's Copa del Rey goal was compared to Diego Maradona's goal of the century."
    ]


def test_final_claim_quoted_by_barcelona_normalization():
    claims = extract_claims(
        'According to the document, "in June 1982, Diego Maradona was signed for a '
        'world record fee of £5 million from Boca Juniors" by Barcelona.'
    )
    assert [claim["content"] for claim in claims] == [
        "Diego Maradona was signed by Barcelona in June 1982 "
        "for a world record fee of £5 million from Boca Juniors."
    ]


def test_context_dedup_evidence_filter_and_dependencies():
    trace = TraceGraph("dedupe", "questions")
    llm1 = trace.add_node("llm_call", "loop 1")
    chunk0 = trace.add_evidence(
        "Diego Maradona was signed by Barcelona in June 1982.",
        {"chunk_id": "0", "loop": 1, "rank": 1, "score": .65},
    )
    noise = trace.add_evidence(
        "New Yorkers use transportation during construction.",
        {"chunk_id": "118", "loop": 1, "rank": 2, "score": .2},
    )
    trace.add_edge(chunk0, llm1, "used_as_context", {"loop": 1, "context_order": 1})
    trace.add_edge(chunk0, llm1, "used_as_context", {"loop": 1, "context_order": 4})
    trace.add_intermediate_claims(
        llm1, "I found that Diego Maradona was signed by Barcelona in June 1982.", 1
    )
    llm2 = trace.add_node("llm_call", "loop 2")
    trace.add_answer(llm2, "Diego Maradona was signed in June 1982.", 2,
                     "final_answer", [chunk0, noise])
    graph = trace.to_dict()
    contexts = [e for e in graph["edges"] if e["type"] == "used_as_context"]
    assert len(contexts) == 1 and contexts[0]["metadata"]["num_appearances"] == 2
    evidence = [e for e in graph["edges"] if e["type"] == "evidence_link"]
    assert any(e["source"] == "chunk_0" for e in evidence)
    assert not any(e["source"] == "chunk_118" for e in evidence)
    assert any(e["type"] == "depends_on" for e in graph["edges"])


def test_evidence_link_requires_core_entity_predicate_and_exact_date():
    trace = TraceGraph("strict", "questions")
    good = trace.add_evidence(
        "In June 1982, Diego Maradona was signed for a world record fee "
        "of £5 million from Boca Juniors.",
        {"chunk_id": "0", "loop": 1, "rank": 1, "score": .9},
    )
    wrong_die = trace.add_evidence(
        "The San Diego Chargers played in June 2015.",
        {"chunk_id": "2", "loop": 1, "rank": 2, "score": .4},
    )
    wrong_date = trace.add_evidence(
        "Diego Maradona signed memorabilia in June 2015.",
        {"chunk_id": "480", "loop": 1, "rank": 3, "score": .3},
    )
    llm = trace.add_node("llm_call", "model=mock")
    trace.add_answer(
        llm,
        "Diego Maradona was signed by Barcelona in June 1982 "
        "for a world record fee of £5 million from Boca Juniors.",
        1,
        "final_answer",
        [good, wrong_die, wrong_date],
    )
    evidence = [e for e in trace.to_dict()["edges"] if e["type"] == "evidence_link"]
    assert [e["source"] for e in evidence] == ["chunk_0"]
    assert evidence[0]["metadata"]["support_status"] == "likely_support"
