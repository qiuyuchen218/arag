"""Base agent implementation for ARAG."""

import json
from typing import Any, Dict, List

import tiktoken

from arag.core.context import AgentContext
from arag.core.llm import LLMClient
from arag.tools.registry import ToolRegistry


class BaseAgent:
    """Base agent with tool calling capabilities."""

    def __init__(
        self,
        llm_client: LLMClient,
        tools: ToolRegistry,
        system_prompt: str = None,
        max_loops: int = 10,
        max_token_budget: int = 128000,
        verbose: bool = False,
    ):
        self.llm = llm_client
        self.tools = tools
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.max_loops = max_loops
        self.max_token_budget = max_token_budget
        self.verbose = verbose
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")

    @staticmethod
    def _preview(value: Any, max_chars: int = 1000) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            try:
                value = json.dumps(value, ensure_ascii=False)
            except TypeError:
                value = str(value)
        if len(value) <= max_chars:
            return value
        return value[:max_chars] + "...[truncated]"

    @staticmethod
    def _query_from_tool_args(func_name: str, func_args: Dict[str, Any]) -> str:
        if "query" in func_args:
            return str(func_args.get("query", ""))
        if "keywords" in func_args:
            return " ".join(str(x) for x in func_args.get("keywords", []))
        if "chunk_ids" in func_args:
            return "read chunks: " + ", ".join(str(x) for x in func_args.get("chunk_ids", []))
        if "chunk_id" in func_args:
            return "read chunk: " + str(func_args.get("chunk_id"))
        return f"{func_name}: {json.dumps(func_args, ensure_ascii=False)}"

    def _calculate_message_tokens(self, messages: List[Dict[str, Any]]) -> int:
        total = len(self.tokenizer.encode(self.system_prompt))
        for msg in messages:
            content = msg.get("content", "")
            if content:
                total += len(self.tokenizer.encode(str(content)))
        return total

    def _build_run_result(
        self,
        answer: str,
        trajectory: List[Dict[str, Any]],
        total_cost: float,
        loop_count: int,
        context: AgentContext,
        message_trace: List[Dict[str, Any]],
        messages: List[Dict[str, Any]],
        termination_reason: str,
        **extra: Any,
    ) -> Dict[str, Any]:
        return {
            "answer": answer,
            "trajectory": trajectory,
            "total_cost": total_cost,
            "loops": loop_count,
            "message_trace": message_trace,
            "final_messages": messages,
            "termination_reason": termination_reason,
            **context.get_summary(),
            **extra,
        }

    def _trace_question(self, trace_logger: Any, query: str) -> str:
        if trace_logger is None:
            return None
        question_node = trace_logger.latest_node_id("question")
        if question_node is None:
            question_node = trace_logger.add_node(
                "question",
                content=query,
                metadata={"role": "user_input"},
            )
            trace_logger.add_temporal_next(question_node)
        return question_node

    def _trace_llm_analysis(
        self,
        trace_logger: Any,
        question_node: str,
        evidence_nodes: List[str],
        message: Dict[str, Any],
        response: Dict[str, Any],
        loop_count: int,
        forced: bool = False,
        reason: str = None,
    ) -> str:
        if trace_logger is None:
            return None

        analysis_node = trace_logger.add_node(
            "llm_call",
            content=f"model={getattr(self.llm, 'model', None)}; loop={loop_count}",
            metadata={
                "loop": loop_count,
                "model": getattr(self.llm, "model", None),
                "temperature": getattr(self.llm, "temperature", None),
                "max_tokens": getattr(self.llm, "max_tokens", None),
                "input_tokens": response.get("input_tokens", 0) if response else 0,
                "output_tokens": response.get("output_tokens", 0) if response else 0,
                "cost": response.get("cost", 0) if response else 0,
                "tool_calls": message.get("tool_calls", []),
                "forced": forced,
                "reason": reason,
                "prompt_preview": "",
                "response_preview": self._preview(message.get("content", "")),
            },
        )

        if evidence_nodes:
            for evidence_node in evidence_nodes:
                trace_logger.add_edge(evidence_node, analysis_node, "used_as_context", {
                    "loop": loop_count, "context_order": evidence_nodes.index(evidence_node) + 1,
                    "timestamp": trace_logger._utc_now(),
                })
        elif question_node:
            trace_logger.add_edge(question_node, analysis_node, "calls", {
                "loop": loop_count, "call_order": loop_count, "timestamp": trace_logger._utc_now(),
            })

        trace_logger.add_temporal_next(analysis_node)
        trace_logger.add_intermediate_claims(analysis_node, message.get("content", ""), loop_count)
        return analysis_node

    def _trace_tool_evidence(
        self,
        trace_logger: Any,
        analysis_node: str,
        func_name: str,
        func_args: Dict[str, Any],
        tool_result: str,
        tool_log: Dict[str, Any],
        context: AgentContext,
        search_history_start: int,
        read_chunk_ids_start: set,
        loop_count: int,
        tool_call_id: str = None,
    ) -> List[str]:
        if trace_logger is None:
            return []

        query_text = self._query_from_tool_args(func_name, func_args)
        query_node = trace_logger.add_node(
            "plan_query",
            content=query_text,
            metadata={
                "loop": loop_count,
                "source": "tool_call_argument",
                "implicit": True,
                "tool_name": func_name,
                "arguments": func_args,
            },
        )
        question_node = trace_logger.latest_node_id("question")
        trace_logger.add_edge(question_node, query_node, "decomposes_to", {
            "loop": loop_count, "method": "implicit_from_tool_call", "timestamp": trace_logger._utc_now(),
        })
        trace_logger.add_edge(analysis_node, query_node, "proposes_query", {
            "loop": loop_count, "tool_name": func_name, "tool_call_id": tool_call_id,
            "method": "tool_call_argument", "timestamp": trace_logger._utc_now(),
        })
        trace_logger.add_temporal_next(query_node)

        tool_failed = bool(tool_log.get("error"))
        num_results = sum(len(event.get("results", [])) for event in context.search_history[search_history_start:])
        tool_node = trace_logger.add_node(
            "retriever_call",
            content=f"{func_name}({query_text})",
            metadata={
                "loop": loop_count,
                "tool_name": func_name,
                "arguments": func_args,
                "tool_log": tool_log,
                "tool_output_preview": self._preview(tool_result, 1500),
                "query": query_text,
                "top_k": func_args.get("top_k") or func_args.get("k"),
                "num_results": num_results,
            },
            status="failed" if tool_failed else "success",
        )
        trace_logger.add_edge(query_node, tool_node, "calls", {
            "loop": loop_count, "call_order": len([n for n in trace_logger.nodes if n["type"] == "retriever_call"]),
            "timestamp": trace_logger._utc_now(),
        })
        trace_logger.add_temporal_next(tool_node)

        if tool_failed:
            trace_logger.add_error(tool_node, tool_log.get("error"), "retrieval", loop_count,
                                   "retrieval_error")

        evidence_nodes = []

        for event in context.search_history[search_history_start:]:
            for rank, result in enumerate(event.get("results", []), start=1):
                retrieval_score = result.get("score", result.get("similarity"))
                matched = result.get("matched_sentences", [])
                if matched and isinstance(matched[0], dict):
                    content = " ".join(str(item.get("sentence", "")) for item in matched)
                elif isinstance(matched, list):
                    content = " ".join(str(item) for item in matched)
                else:
                    content = self._preview(result)

                chunk_id = result.get("chunk_id")
                evidence_node = trace_logger.add_evidence(
                    content=content,
                    metadata={
                        "loop": loop_count,
                        "source": "search_result",
                        "tool_name": event.get("tool_name"),
                        "rank": rank,
                        "chunk_id": chunk_id,
                        "doc_id": result.get("doc_id") or chunk_id,
                        "score": retrieval_score,
                        "query": event.get("query"),
                        "result_metadata": result,
                    },
                    dedupe_key=f"chunk:{chunk_id}" if chunk_id is not None else None,
                )
                trace_logger.add_edge(tool_node, evidence_node, "retrieves", {
                    "rank": rank, "score": retrieval_score, "retrieval_round": loop_count,
                    "tool_name": event.get("tool_name"), "query": event.get("query"),
                    "matched_sentences": matched,
                    "timestamp": trace_logger._utc_now(),
                })
                evidence_nodes.append(evidence_node)

        requested_chunk_ids = []
        if "chunk_ids" in func_args:
            requested_chunk_ids = [str(x) for x in func_args.get("chunk_ids", [])]
        elif "chunk_id" in func_args:
            requested_chunk_ids = [str(func_args.get("chunk_id"))]

        new_read_ids = set(context.read_chunk_ids) - read_chunk_ids_start
        readable_ids = sorted(new_read_ids.union(set(requested_chunk_ids)))

        for rank, chunk_id in enumerate(readable_ids, start=1):
            chunk = context.read_chunks.get(str(chunk_id), {})
            content = chunk.get("content")
            if content is None:
                content = self._preview(tool_result, 1500)

            evidence_node = trace_logger.add_evidence(
                content=content,
                metadata={
                    "loop": loop_count,
                    "source": "read_chunk",
                    "tool_name": func_name,
                    "rank": rank,
                    "chunk_id": str(chunk_id),
                    "doc_id": str(chunk_id),
                    "tokens": chunk.get("tokens", 0),
                    "already_read": str(chunk_id) not in new_read_ids,
                    **chunk.get("metadata", {}),
                },
                dedupe_key=f"chunk:{chunk_id}",
            )
            trace_logger.add_edge(tool_node, evidence_node, "retrieves", {
                "rank": rank, "score": None, "retrieval_round": loop_count,
                "tool_name": func_name, "query": query_text, "matched_sentences": [],
                "timestamp": trace_logger._utc_now(),
            })
            evidence_nodes.append(evidence_node)

        retrieval_tools = {"keyword_search", "semantic_search", "hybrid_search", "read_chunk", "read_chunks"}
        if not tool_failed and func_name in retrieval_tools and not evidence_nodes:
            trace_logger.add_error(tool_node, "empty retrieval", "retrieval", loop_count,
                                   "no_retrieval", fatal=False)

        return evidence_nodes

    def _trace_answer(
        self,
        trace_logger: Any,
        analysis_node: str,
        evidence_nodes: List[str],
        final_answer: str,
        loop_count: int,
        reason: str = None,
    ) -> str:
        if trace_logger is None:
            return None

        return trace_logger.add_answer(analysis_node, final_answer, loop_count,
                                       reason or "final_answer", evidence_nodes)

    def _force_final_answer(
        self,
        messages: List[Dict[str, Any]],
        context: AgentContext,
        total_cost: float,
        reason: str,
        trace_logger: Any = None,
        question_node: str = None,
        evidence_nodes: List[str] = None,
        loop_count: int = 0,
    ) -> tuple:
        force_prompt = (
            "You have reached the limit. "
            "You MUST now provide a final answer based on the information you have gathered so far. "
            "Do NOT call any more tools. Synthesize the available information and respond directly."
        )
        messages.append({"role": "user", "content": force_prompt})

        call_error = None
        try:
            response = self.llm.chat(messages=messages, tools=None, temperature=0.0)
            total_cost += response["cost"]
            forced_message = response["message"]
            final_answer = forced_message.get("content", "")
        except Exception as e:
            response = {}
            final_answer = ""
            forced_message = {"role": "assistant", "content": ""}
            call_error = e

        if call_error is None:
            analysis_node = self._trace_llm_analysis(
                trace_logger, question_node, evidence_nodes or [], forced_message, response,
                loop_count, forced=True, reason=reason,
            )
            self._trace_answer(trace_logger, analysis_node, evidence_nodes or [], final_answer,
                               loop_count, reason=reason)
        elif trace_logger:
            llm_node = trace_logger.add_node(
                "llm_call", f"model={getattr(self.llm, 'model', None)}; loop={loop_count}",
                {"loop": loop_count, "model": getattr(self.llm, "model", None), "forced": True,
                 "reason": reason, "input_tokens": 0, "output_tokens": 0, "cost": 0,
                 "tool_calls": [], "prompt_preview": self._preview(messages), "response_preview": ""},
                status="failed",
            )
            for index, evidence_node in enumerate(evidence_nodes or [], start=1):
                trace_logger.add_edge(evidence_node, llm_node, "used_as_context", {
                    "loop": loop_count, "context_order": index, "timestamp": trace_logger._utc_now()})
            if not evidence_nodes and question_node:
                trace_logger.add_edge(question_node, llm_node, "calls", {
                    "loop": loop_count, "call_order": loop_count, "timestamp": trace_logger._utc_now()})
            trace_logger.add_temporal_next(llm_node)
            trace_logger.add_error(llm_node, call_error, "llm_call", loop_count, reason)
            trace_logger.add_answer(None, "", loop_count, reason, failed=True, raw_error=str(call_error))

        if self.verbose:
            print(f"Forced answer: {final_answer[:200]}...")
            print(f"Total cost: ${total_cost:.6f}")

        return final_answer, total_cost, forced_message

    def run(self, query: str, trace_logger: Any = None) -> Dict[str, Any]:
        context = AgentContext()
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]

        question_node = self._trace_question(trace_logger, query)

        trajectory = []
        message_trace = []
        evidence_nodes = []
        total_cost = 0.0
        loop_count = 0
        tool_schemas = self.tools.get_all_schemas()

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Question: {query}")
            print(f"{'='*60}\n")

        for loop_idx in range(self.max_loops):
            loop_count = loop_idx + 1

            current_tokens = self._calculate_message_tokens(messages)
            if current_tokens > self.max_token_budget:
                if self.verbose:
                    print(f"Token budget exceeded ({current_tokens} > {self.max_token_budget}), forcing answer...")

                final_answer, total_cost, forced_message = self._force_final_answer(
                    messages,
                    context,
                    total_cost,
                    "token_budget_exceeded",
                    trace_logger=trace_logger,
                    question_node=question_node,
                    evidence_nodes=evidence_nodes,
                    loop_count=loop_count,
                )
                messages.append(forced_message)
                message_trace.append({
                    "loop": loop_count,
                    "role": "assistant",
                    "content": forced_message.get("content", ""),
                    "tool_calls": forced_message.get("tool_calls", []),
                    "forced": True,
                    "reason": "token_budget_exceeded",
                })

                return self._build_run_result(
                    final_answer,
                    trajectory,
                    total_cost,
                    loop_count,
                    context,
                    message_trace,
                    messages,
                    "token_budget_exceeded",
                    token_budget_exceeded=True,
                )

            if self.verbose:
                print(f"Loop {loop_count}/{self.max_loops} (Tokens: {current_tokens}/{self.max_token_budget})")

            try:
                response = self.llm.chat(messages=messages, tools=tool_schemas)
            except Exception as e:
                if self.verbose:
                    print(f"LLM error: {e}")
                llm_node = trace_logger.add_node("llm_call",
                    content=f"model={getattr(self.llm, 'model', None)}; loop={loop_count}",
                    metadata={"loop": loop_count, "model": getattr(self.llm, "model", None),
                              "temperature": getattr(self.llm, "temperature", None),
                              "max_tokens": getattr(self.llm, "max_tokens", None), "input_tokens": 0,
                              "output_tokens": 0, "cost": 0, "tool_calls": [],
                              "prompt_preview": self._preview(messages), "response_preview": ""},
                    status="failed") if trace_logger else None
                if trace_logger:
                    trace_logger.add_edge(question_node, llm_node, "calls", {
                        "loop": loop_count, "call_order": loop_count, "timestamp": trace_logger._utc_now()})
                    trace_logger.add_temporal_next(llm_node)
                    error_node = trace_logger.add_error(llm_node, e, "llm_call", loop_count, "llm_api_error")
                    answer_node = trace_logger.add_answer(None, "", loop_count, "llm_api_error",
                                                          failed=True, raw_error=str(e))
                    trace_logger.add_edge(answer_node, error_node, "failed_with", {
                        "stage": "answer", "fatal": True, "timestamp": trace_logger._utc_now()})
                return self._build_run_result(
                    "", trajectory, total_cost, loop_count, context, message_trace, messages,
                    "llm_api_error", raw_error=str(e),
                )

            total_cost += response["cost"]
            message = response["message"]
            messages.append(message)

            analysis_node = self._trace_llm_analysis(
                trace_logger,
                question_node,
                evidence_nodes,
                message,
                response,
                loop_count,
            )

            message_trace.append({
                "loop": loop_count,
                "role": "assistant",
                "content": message.get("content", ""),
                "tool_calls": message.get("tool_calls", []),
                "input_tokens": response.get("input_tokens", 0),
                "output_tokens": response.get("output_tokens", 0),
                "cost": response.get("cost", 0),
                "forced": False,
            })

            if self.verbose and message.get("content"):
                print(f"Assistant: {message['content'][:200]}...")

            tool_calls = message.get("tool_calls")
            if not tool_calls:
                final_answer = message.get("content", "")
                self._trace_answer(
                    trace_logger,
                    analysis_node,
                    evidence_nodes,
                    final_answer,
                    loop_count,
                    reason="final_answer",
                )
                return self._build_run_result(
                    final_answer,
                    trajectory,
                    total_cost,
                    loop_count,
                    context,
                    message_trace,
                    messages,
                    "final_answer",
                )

            for tc in tool_calls:
                func_name = tc["function"]["name"]
                try:
                    func_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    func_args = {}

                if self.verbose:
                    print(f"Tool: {func_name}")
                    print(f"  Args: {func_args}")

                search_history_start = len(context.search_history)
                read_chunk_ids_start = set(context.read_chunk_ids)

                try:
                    tool_result, tool_log = self.tools.execute(func_name, context, **func_args)
                except Exception as e:
                    tool_result = f"Error executing tool: {str(e)}"
                    tool_log = {"retrieved_tokens": 0, "error": str(e)}

                new_evidence_nodes = self._trace_tool_evidence(
                    trace_logger,
                    analysis_node,
                    func_name,
                    func_args,
                    tool_result,
                    tool_log,
                    context,
                    search_history_start,
                    read_chunk_ids_start,
                    loop_count,
                    tc.get("id"),
                )
                evidence_nodes.extend(new_evidence_nodes)

                if self.verbose:
                    output_preview = tool_result[:300] + "..." if len(tool_result) > 300 else tool_result
                    print(f"  Result: {output_preview}")
                    if tool_log.get("retrieved_tokens", 0) > 0:
                        print(f"  Tokens: {tool_log['retrieved_tokens']}")
                    print()

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result,
                })
                message_trace.append({
                    "loop": loop_count,
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "tool_name": func_name,
                    "arguments": func_args,
                    "content": tool_result,
                    "tool_log": tool_log,
                    "retrieved_tokens": tool_log.get("retrieved_tokens", 0),
                    "error": tool_log.get("error"),
                })

                traj_entry = {
                    "loop": loop_count,
                    "tool_name": func_name,
                    "arguments": func_args,
                    "tool_result": tool_result,
                    **tool_log,
                }
                trajectory.append(traj_entry)

        if self.verbose:
            print(f"Max loops reached ({self.max_loops}), forcing answer...")

        final_answer, total_cost, forced_message = self._force_final_answer(
            messages,
            context,
            total_cost,
            "max_loops_exceeded",
            trace_logger=trace_logger,
            question_node=question_node,
            evidence_nodes=evidence_nodes,
            loop_count=loop_count,
        )
        messages.append(forced_message)
        message_trace.append({
            "loop": loop_count,
            "role": "assistant",
            "content": forced_message.get("content", ""),
            "tool_calls": forced_message.get("tool_calls", []),
            "forced": True,
            "reason": "max_loops_exceeded",
        })

        return self._build_run_result(
            final_answer,
            trajectory,
            total_cost,
            loop_count,
            context,
            message_trace,
            messages,
            "max_loops_exceeded",
            max_loops_exceeded=True,
        )
