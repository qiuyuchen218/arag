import json

from arag.agent.base import BaseAgent
from arag.tools.base import BaseTool
from arag.tools.registry import ToolRegistry
from arag.utils.trace_graph import TraceGraph


class MockLLM:
    model = "mock"
    temperature = 0.0
    max_tokens = 128

    def __init__(self):
        self.calls = 0

    def chat(self, messages, tools=None, temperature=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "semantic_search",
                            "arguments": json.dumps({"query": "alpha"}),
                        },
                    }],
                },
                "input_tokens": 11,
                "output_tokens": 3,
                "cost": 0.01,
            }
        return {
            "message": {"role": "assistant", "content": "Final answer.", "tool_calls": []},
            "input_tokens": 15,
            "output_tokens": 4,
            "cost": 0.02,
        }


class MockSearchTool(BaseTool):
    @property
    def name(self):
        return "semantic_search"

    def get_schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "mock search",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }

    def execute(self, context, **kwargs):
        result = {
            "chunk_id": "c1",
            "doc_id": "d1",
            "score": 0.42,
            "matched_sentences": ["Alpha sentence."],
        }
        context.add_search_event(self.name, kwargs.get("query"), [result])
        return "Chunk c1: Alpha sentence.", {"retrieved_tokens": 4}


def test_base_agent_records_execution_graph_for_tool_call():
    tools = ToolRegistry()
    tools.register(MockSearchTool())
    agent = BaseAgent(MockLLM(), tools, max_loops=3)
    trace = TraceGraph("sample", "mock")

    result = agent.run("Question?", trace_logger=trace)
    graph = trace.to_dict()
    trace.validate()

    assert result["answer"] == "Final answer."
    assert [n["type"] for n in graph["nodes"]].count("llm_call") == 2
    assert [n["type"] for n in graph["nodes"]].count("plan_query") == 1
    assert [n["type"] for n in graph["nodes"]].count("retriever_call") == 1
    assert [n["type"] for n in graph["nodes"]].count("retrieved_chunk") == 1
    assert {e["type"] for e in graph["edges"]} >= {"next", "invokes", "executes", "retrieves", "generates"}
    assert not [n for n in graph["nodes"] if n["type"] == "claim"]
    retrieve = next(e for e in graph["edges"] if e["type"] == "retrieves")
    assert retrieve["metadata"]["tool_call_id"] == "call_1"
    assert retrieve["metadata"]["raw_score"] == 0.42
