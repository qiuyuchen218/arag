"""Base agent implementation for ARAG."""

import json
from typing import Any, Dict, List, Optional

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
        """Build a result object with both legacy fields and full observability."""
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

    def _force_final_answer(
        self,
        messages: List[Dict[str, Any]],
        context: AgentContext,
        total_cost: float,
        reason: str
    ) -> tuple:
        """Force the model to give a final answer when limits are reached."""
        force_prompt = (
            "You have reached the limit. "
            "You MUST now provide a final answer based on the information you have gathered so far. "
            "Do NOT call any more tools. Synthesize the available information and respond directly."
        )

        messages.append({"role": "user", "content": force_prompt})

        try:
            response = self.llm.chat(messages=messages, tools=None, temperature=0.0)
            total_cost += response["cost"]
            forced_message = response["message"]
            final_answer = forced_message.get("content", "")

            if self.verbose:
                print(f"Forced answer: {final_answer[:200]}...")
                print(f"Total cost: ${total_cost:.6f}")
        except Exception as e:
            if self.verbose:
                print(f"Error getting forced answer: {e}")
            final_answer = f"Error: {reason} and failed to generate final answer."
            forced_message = {"role": "assistant", "content": final_answer}

        return final_answer, total_cost, forced_message

    def run(self, query: str) -> Dict[str, Any]:
        context = AgentContext()
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]

        trajectory = []
        message_trace = []
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
                    messages, context, total_cost, "Token budget exceeded"
                )
                messages.append(forced_message)
                message_trace.append({
                    "loop": loop_count,
                    "role": "assistant",
                    "content": forced_message.get("content", ""),
                    "tool_calls": forced_message.get("tool_calls", []),
                    "forced": True,
                    "reason": "Token budget exceeded",
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
                break

            total_cost += response["cost"]
            message = response["message"]
            messages.append(message)
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

                try:
                    tool_result, tool_log = self.tools.execute(func_name, context, **func_args)
                except Exception as e:
                    tool_result = f"Error executing tool: {str(e)}"
                    tool_log = {"retrieved_tokens": 0, "error": str(e)}

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
                    **tool_log
                }
                trajectory.append(traj_entry)

        if self.verbose:
            print(f"Max loops reached ({self.max_loops}), forcing answer...")

        final_answer, total_cost, forced_message = self._force_final_answer(
            messages, context, total_cost, "Maximum loops exceeded"
        )
        messages.append(forced_message)
        message_trace.append({
            "loop": loop_count,
            "role": "assistant",
            "content": forced_message.get("content", ""),
            "tool_calls": forced_message.get("tool_calls", []),
            "forced": True,
            "reason": "Maximum loops exceeded",
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