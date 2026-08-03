"""Base agent implementation for ARAG."""

import json
import re
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
        try:
            return trace_logger.add_question(query, {"role": "user_input"})
        except Exception:
            return None

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
        try:
            analysis_node = trace_logger.add_llm_call(
                model=getattr(self.llm, "model", None),
                loop=loop_count,
                message=message,
                response=response,
                forced=forced,
                reason=reason,
                metadata={
                    "temperature": getattr(self.llm, "temperature", None),
                    "max_tokens": getattr(self.llm, "max_tokens", None),
                },
            )
            return analysis_node
        except Exception:
            return None

    def _trace_tool_evidence(
        self,
        trace_logger: Any,
        analysis_node: str,
        decision_node: str,
        func_name: str,
        func_args: Dict[str, Any],
        tool_result: str,
        tool_log: Dict[str, Any],
        context: AgentContext,
        search_history_start: int,
        read_chunk_ids_start: set,
        loop_count: int,
        tool_call_id: str = None,
        raw_arguments: str = None,
        arguments_parse_error: str = None,
    ) -> List[str]:
        if trace_logger is None:
            return []

        query_text = self._query_from_tool_args(func_name, func_args)
        call_order = len([n for n in getattr(trace_logger, "nodes", []) if n["type"] == "plan_query"]) + 1
        try:
            query_node = trace_logger.add_plan_query(
                analysis_node, query_text, func_name, func_args, loop_count,
                tool_call_id=tool_call_id, call_order=call_order,
                raw_arguments=raw_arguments, arguments_parse_error=arguments_parse_error,
                decision_id=decision_node,
            )
        except Exception:
            return []

        tool_failed = bool(tool_log.get("error"))
        num_results = sum(len(event.get("results", [])) for event in context.search_history[search_history_start:])
        operation = "read" if func_name in {"read_chunk", "read_chunks"} else "search"
        tool_node = trace_logger.add_retriever_call(
            query_node, func_name, query_text, func_args, loop_count,
            tool_call_id=tool_call_id, call_order=call_order, operation=operation,
            status="failed" if tool_failed else "success",
            metadata={
                "tool_log": tool_log,
                "tool_output_preview": self._preview(tool_result, 1500),
                "top_k": func_args.get("top_k") or func_args.get("k"),
                "num_results": num_results,
            },
        )
        structured_tool_result = tool_log.get("tool_result") if isinstance(tool_log, dict) else None
        if isinstance(structured_tool_result, dict):
            try:
                tool_result_node = trace_logger.add_tool_result_node(tool_node, structured_tool_result)
                for item in structured_tool_result.get("results", []) or []:
                    if isinstance(item, dict):
                        for span in item.get("matched_spans", []) or []:
                            if isinstance(span, dict):
                                trace_logger.add_evidence_span({
                                    **span,
                                    "source_tool_call": tool_node,
                                    "source_branch": item.get("branch_id") or "b0",
                                }, tool_result_node)
                for receipt in structured_tool_result.get("results", []) or []:
                    if isinstance(receipt, dict) and receipt.get("returned_span_ids"):
                        # Read receipts identify delivered spans, but the receipt
                        # text can be a whole chunk. Sentence-level EvidenceSpan
                        # artifacts are reconstructed later from read_chunk
                        # metadata; registering the full receipt text once per
                        # span would duplicate evidence artifacts.
                        continue
            except Exception:
                pass

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
                evidence_node = trace_logger.add_retrieval_result(
                    tool_node,
                    content,
                    occurrence={
                        "loop": loop_count,
                        "tool_call_id": tool_call_id,
                        "tool_name": event.get("tool_name"),
                        "query": event.get("query"),
                        "rank": rank,
                        "raw_score": retrieval_score,
                        "operation": "search",
                        "result_metadata": result,
                    },
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

            evidence_node = trace_logger.add_retrieval_result(
                tool_node,
                content,
                occurrence={
                    "loop": loop_count,
                    "tool_call_id": tool_call_id,
                    "tool_name": func_name,
                    "query": query_text,
                    "rank": rank,
                    "raw_score": None,
                    "operation": "read",
                    "result_metadata": chunk.get("metadata", {}),
                },
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
                edge_type="reads",
            )
            evidence_nodes.append(evidence_node)

        retrieval_tools = {"keyword_search", "semantic_search", "hybrid_search", "read_chunk", "read_chunks"}
        if not tool_failed and func_name in retrieval_tools and not evidence_nodes:
            trace_logger.add_error(tool_node, "empty retrieval", "retrieval", loop_count,
                                   "no_retrieval", fatal=False)

        return evidence_nodes

    def _trace_decision_record(
        self,
        trace_logger: Any,
        analysis_node: str,
        input_context_node: str,
        func_name: str,
        func_args: Dict[str, Any],
        epistemic_context: Dict[str, Any],
        loop_count: int,
        call_order: int,
        tool_call_id: str = None,
        delivered_span_ids: List[str] = None,
    ) -> str:
        if trace_logger is None:
            return None
        query_text = self._query_from_tool_args(func_name, func_args)
        operation = "read" if func_name in {"read_chunk", "read_chunks"} else "query_selection"
        explicit = bool((epistemic_context or {}).get("explicit"))
        action_role = str((epistemic_context or {}).get("action_role") or ("TEST" if operation == "query_selection" else "VERIFY")).upper()
        decision = {
            "decision_id": f"dec_{loop_count:03d}_{call_order:03d}",
            "decision_type": "query_selection" if action_role not in {"COMMIT", "USE_AS_PREMISE"} else "binding_commitment",
            "active_subgoal_ids": list((epistemic_context or {}).get("active_subgoal_ids") or []),
            "target_relation_ids": [
                p.get("relation_id") for p in self._epistemic_props(epistemic_context or {}, "targets")
                if isinstance(p, dict) and p.get("relation_id")
            ],
            "premise_relation_ids": [
                p.get("relation_id") for p in self._epistemic_props(epistemic_context or {}, "premises")
                if isinstance(p, dict) and p.get("relation_id")
            ],
            "proposed_bindings": list(self._epistemic_props(epistemic_context or {}, "targets")),
            "premises": list(self._epistemic_props(epistemic_context or {}, "premises")),
            "selected_tool": func_name,
            "query_or_action": query_text,
            "public_reason_code": "explicit_epistemic_context" if explicit else "tool_call_selected_by_public_assistant_message",
            "confidence": 0.7 if explicit else 0.35,
            "tool_call_id": tool_call_id,
            "loop": loop_count,
            "call_order": call_order,
            "action_role": action_role,
            "extractor_mode": "explicit" if explicit else "inferred",
            "authoritative": bool(explicit),
        }
        try:
            pre_state_ids = []
            operational_premises = self._detect_operational_premise_use(
                func_name,
                func_args,
                epistemic_context or {},
                action_role,
            )
            if operational_premises:
                epistemic_context = {
                    **(epistemic_context or {}),
                    "premises": list(self._epistemic_props(epistemic_context or {}, "premises")) + operational_premises,
                    "operational_role_mismatch": action_role in {"EXPLORE", "TEST", "VERIFY", "DISAMBIGUATE"},
                }
                decision["operational_premise_use_detected"] = True
                decision["role_mismatch"] = epistemic_context["operational_role_mismatch"]
                decision["premises"] = list(self._epistemic_props(epistemic_context or {}, "premises"))
            if action_role in {"COMMIT", "USE_AS_PREMISE"} or operational_premises:
                pre_state_ids = self._trace_epistemic_context(
                    trace_logger,
                    decision["decision_id"],
                    epistemic_context or {},
                    action_role,
                    list(dict.fromkeys(delivered_span_ids or [])),
                    authoritative=bool(explicit),
                )
            decision_id = trace_logger.add_decision_record(
                analysis_node,
                decision,
                input_context_node=input_context_node,
                visible_evidence_ids=delivered_span_ids or [],
            )
            for state_id in pre_state_ids:
                trace_logger.link_state_event_to_decision(state_id, decision_id)
            if action_role not in {"COMMIT", "USE_AS_PREMISE"}:
                self._trace_epistemic_context(trace_logger, decision_id, epistemic_context or {}, action_role, list(dict.fromkeys(delivered_span_ids or [])), authoritative=bool(explicit))
            return decision_id
        except Exception:
            return None

    @staticmethod
    def _epistemic_props(epistemic_context: Dict[str, Any], kind: str = "all") -> List[Dict[str, Any]]:
        if not isinstance(epistemic_context, dict):
            return []
        props: List[Dict[str, Any]] = []
        if kind in {"all", "premises"}:
            props.extend([p for p in epistemic_context.get("premises", []) or [] if isinstance(p, dict)])
        if kind in {"all", "targets"}:
            targets = epistemic_context.get("targets")
            if targets is None:
                targets = epistemic_context.get("propositions", [])
            props.extend([p for p in targets or [] if isinstance(p, dict)])
        return props

    def _detect_operational_premise_use(
        self,
        func_name: str,
        func_args: Dict[str, Any],
        epistemic_context: Dict[str, Any],
        action_role: str,
    ) -> List[Dict[str, Any]]:
        """Detect when a concrete entity is used as input to a downstream relation query."""
        if func_name in {"read_chunk", "read_chunks"}:
            return []
        query = self._query_from_tool_args(func_name, func_args).lower()
        downstream_markers = {
            "created": "created_date",
            "creation": "created_date",
            "founded": "founded_date",
            "established": "established_date",
            "abolished": "abolished_date",
            "abolition": "abolished_date",
            "dissolved": "abolished_date",
            "date": "answer_date",
        }
        detected_predicate = next((pred for marker, pred in downstream_markers.items() if marker in query), None)
        if not detected_predicate:
            return []
        existing = self._epistemic_props(epistemic_context, "premises")
        seen = {(str(p.get("subject", "")).lower(), str(p.get("predicate", "")).lower(), str(p.get("object", "")).lower()) for p in existing}
        operational: List[Dict[str, Any]] = []
        for prop in self._epistemic_props(epistemic_context, "targets"):
            subject = str(prop.get("subject") or "").strip()
            obj = str(prop.get("object") or "").strip()
            pred = str(prop.get("predicate") or "").lower()
            concrete_subject = subject and subject.lower() not in {"unknown", "target_entity", "answer"}
            target_unknown = obj.lower() in {"", "unknown", "unknown location", "answer"}
            relation_query_for_subject = concrete_subject and subject.lower() in query and (target_unknown or pred in downstream_markers.values())
            if not relation_query_for_subject:
                continue
            premise = {
                "subject": "target_entity",
                "predicate": "binding",
                "object": subject,
                "relation_id": prop.get("relation_id"),
                "stance": "COMMITTED",
                "supporting_evidence_ids": prop.get("supporting_evidence_ids", []) or [],
                "missing_constraint_ids": prop.get("missing_constraint_ids", []) or [prop.get("relation_id") or "required_relation_grounding"],
                "operational_detector": "downstream_relation_input_slot",
                "downstream_predicate": detected_predicate,
                "source_action_role": action_role,
            }
            key = (premise["subject"].lower(), premise["predicate"].lower(), premise["object"].lower())
            if key not in seen:
                operational.append(premise)
                seen.add(key)
        return operational

    def _trace_epistemic_context(
        self,
        trace_logger: Any,
        decision_id: str,
        epistemic_context: Dict[str, Any],
        action_role: str,
        visible_span_ids: List[str],
        authoritative: bool,
    ) -> List[str]:
        if not trace_logger or not decision_id:
            return []
        state_ids = []
        role_state = {
            "EXPLORE": "HYPOTHESIS",
            "TEST": "UNDER_TEST",
            "VERIFY": "UNDER_TEST",
            "DISAMBIGUATE": "UNDER_TEST",
            "COMMIT": "COMMITTED",
            "USE_AS_PREMISE": "USED_AS_PREMISE",
        }.get(action_role, "HYPOTHESIS")
        premise_keys = {id(p) for p in self._epistemic_props(epistemic_context or {}, "premises")}
        for prop in self._epistemic_props(epistemic_context or {}, "all"):
            if not isinstance(prop, dict):
                continue
            is_premise = id(prop) in premise_keys
            proposition_id = trace_logger.add_proposition_node(prop)
            support_ids = list(dict.fromkeys(prop.get("supporting_evidence_ids") or []))
            available_ids = list(dict.fromkeys(visible_span_ids + support_ids))
            missing = list(prop.get("missing_constraint_ids") or [])
            stance = str(prop.get("stance") or role_state).upper()
            new_state = "USED_AS_PREMISE" if is_premise else ("COMMITTED" if stance == "COMMITTED" else ("USED_AS_PREMISE" if action_role == "USE_AS_PREMISE" else role_state))
            support_score = min(1.0, len(support_ids) / max(len(missing) + len(support_ids), 1))
            state_ids.append(trace_logger.add_epistemic_state_event(
                proposition_id,
                new_state,
                generated_by_decision_id=decision_id,
                available_evidence_ids=available_ids,
                support_score_at_event=support_score,
                missing_constraint_ids=missing,
                authoritative=authoritative,
                extractor_mode=prop.get("operational_detector") or ("explicit" if authoritative else "inferred"),
                action_role="USE_AS_PREMISE" if is_premise else action_role,
            ))
        return state_ids

    @staticmethod
    def _assess_span_against_proposition(prop: Dict[str, Any], spans: List[Dict[str, Any]]) -> Dict[str, Any]:
        text = " ".join(str(s.get("text", "")) for s in spans if isinstance(s, dict)).lower()
        subject = str(prop.get("subject") or "").lower()
        predicate = str(prop.get("predicate") or "").lower().replace("_", " ")
        obj = str(prop.get("object") or "").lower()
        def toks(value: str) -> List[str]:
            return [t for t in re.findall(r"[a-z0-9]+", value or "") if len(t) > 2 and t not in {"the", "and", "unknown", "answer"}]
        pred_synonyms = {
            "located in": {"located", "location", "in", "near", "off", "coast", "island", "islands", "region"},
            "location": {"located", "location", "in", "near", "off", "coast", "island", "islands", "region"},
            "birthplace": {"born", "birth", "birthplace", "native"},
            "created date": {"created", "creation", "founded", "established", "date"},
            "abolished date": {"abolished", "abolition", "dissolved", "date"},
            "established date": {"established", "founded", "created", "date"},
        }
        subject_tokens = toks(subject)
        object_tokens = toks(obj)
        pred_terms = set(toks(predicate))
        for key, vals in pred_synonyms.items():
            if key in predicate:
                pred_terms.update(vals)
        subject_match = bool(subject and (subject in text or (subject_tokens and sum(1 for t in subject_tokens if t in text) >= max(1, min(2, len(subject_tokens))))))
        predicate_match = bool(pred_terms and any(t in text for t in pred_terms))
        object_match = bool(obj and (obj in text or (object_tokens and sum(1 for t in object_tokens if t in text) >= max(1, min(2, len(object_tokens))))))
        if obj in {"", "?", "unknown", "unknown location"}:
            object_match = False
        provenance_valid = bool(spans)
        negation_scope = any(marker in text for marker in ["not ", "no evidence", "cannot confirm", "unclear", "alleged", "possibly", "may be"])
        entailed = subject_match and predicate_match and object_match and provenance_valid and not negation_scope
        contradicted = bool(subject_match and object_match and any(marker in text for marker in ["not ", "never", "incorrect", "false"]))
        return {
            "status": "CONTRADICTED" if contradicted else ("SUPPORTED" if entailed else "INSUFFICIENT"),
            "subject_match": subject_match,
            "predicate_match": predicate_match,
            "object_match": object_match,
            "provenance_valid": provenance_valid,
            "entailment": 1.0 if entailed else 0.0,
            "contradiction": 1.0 if contradicted else 0.0,
            "insufficient": 0.0 if entailed else 1.0,
            "assessment_mode": "relation_specific_heuristic",
        }

    def _trace_evidence_assessments(
        self,
        trace_logger: Any,
        decision_id: str,
        epistemic_context: Dict[str, Any],
        delivered_span_ids: List[str],
        tool_result_payload: Dict[str, Any],
    ):
        if not trace_logger or not decision_id or not isinstance(epistemic_context, dict):
            return
        span_by_id = {}
        if isinstance(tool_result_payload, dict):
            for item in tool_result_payload.get("results", []) or []:
                if not isinstance(item, dict):
                    continue
                for span in item.get("matched_spans", []) or []:
                    if isinstance(span, dict) and span.get("span_id"):
                        span_by_id[span["span_id"]] = span
        spans = [span_by_id[sid] for sid in delivered_span_ids if sid in span_by_id]
        for prop in self._epistemic_props(epistemic_context, "targets"):
            if not isinstance(prop, dict):
                continue
            proposition_id = trace_logger.add_proposition_node(prop)
            assessment = self._assess_span_against_proposition(prop, spans)
            trace_logger.add_evidence_assessment(proposition_id, decision_id, delivered_span_ids, assessment)
            if assessment["status"] in {"SUPPORTED", "CONTRADICTED"}:
                trace_logger.add_epistemic_state_event(
                    proposition_id,
                    "SUPPORTED" if assessment["status"] == "SUPPORTED" else "REJECTED",
                    generated_by_decision_id=decision_id,
                    available_evidence_ids=delivered_span_ids,
                    support_score_at_event=float(assessment.get("entailment", 0.0) or 0.0),
                    missing_constraint_ids=[] if assessment["status"] == "SUPPORTED" else ["contradicted_by_evidence"],
                    authoritative=bool((epistemic_context or {}).get("explicit")),
                    extractor_mode="relation_specific_evidence_assessment",
                    action_role="VERIFY",
                )

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
        try:
            visible_span_ids = []
            premise_ids = []
            for state in getattr(trace_logger, "nodes", []):
                if state.get("type") != "epistemic_state_event":
                    continue
                md = state.get("metadata", {}) or {}
                if md.get("new_state") in {"COMMITTED", "USED_AS_PREMISE"} and md.get("authoritative"):
                    pid = md.get("proposition_id")
                    if pid:
                        premise_ids.append(pid)
            premise_ids = list(dict.fromkeys(premise_ids))
            answer_lower = (final_answer or "").lower()
            abstained = any(p in answer_lower for p in ["cannot answer", "not found", "no information", "unable to find", "cannot confirm"])
            decision_id = trace_logger.add_decision_record(analysis_node, {
                "decision_type": "abstention_decision" if abstained else "answer_decision",
                "active_subgoal_ids": [],
                "premise_proposition_ids": premise_ids,
                "available_evidence_ids": visible_span_ids,
                "conclusion": final_answer,
                "action_role": "USE_AS_PREMISE" if premise_ids else "COMMIT",
                "query_or_action": "final_answer",
                "public_reason_code": "final_public_answer",
                "confidence": 0.5,
                "authoritative": True,
                "extractor_mode": "online_final_decision",
                "termination_reason": reason or ("abstention" if abstained else "submit_answer"),
            })
            if premise_ids:
                for node in getattr(trace_logger, "nodes", []):
                    if node.get("type") == "epistemic_state_event" and node.get("metadata", {}).get("new_state") in {"COMMITTED", "USED_AS_PREMISE"}:
                        trace_logger.link_state_event_to_decision(node["id"], decision_id)
            else:
                trace_logger.metadata.setdefault("root_identification_blocking_reasons", [])
                if "ANSWER_DECISION_WITHOUT_EXPLICIT_PREMISES" not in trace_logger.metadata["root_identification_blocking_reasons"]:
                    trace_logger.metadata["root_identification_blocking_reasons"].append("ANSWER_DECISION_WITHOUT_EXPLICIT_PREMISES")
        except Exception:
            pass
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
            try:
                llm_node = trace_logger.add_llm_call(
                    model=getattr(self.llm, "model", None),
                    loop=loop_count,
                    message=forced_message,
                    response={},
                    forced=True,
                    reason=reason,
                    status="failed",
                )
                trace_logger.add_context_snapshot(llm_node, {
                    "message_count": len(messages),
                    "visible_chunk_ids": list(dict.fromkeys(evidence_nodes or [])),
                    "semantics": "available_to_model_not_proven_used",
                })
                trace_logger.add_error(llm_node, call_error, "llm_call", loop_count, reason)
                trace_logger.add_answer(None, "", loop_count, reason, failed=True, raw_error=str(call_error))
            except Exception:
                pass

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

            input_context_node = None
            if trace_logger:
                try:
                    delivered_span_ids = []
                    for delivery in context.context_deliveries:
                        delivered_span_ids.extend(delivery.get("span_ids", []) or [])
                    input_context_node = trace_logger.add_context_snapshot(None, {
                        "snapshot_kind": "input_context",
                        "branch_id": context.branch_id,
                        "loop": loop_count,
                        "message_count": len(messages),
                        "message_indices": list(range(len(messages))),
                        "visible_tool_call_ids": [
                            msg.get("tool_call_id") for msg in messages if msg.get("role") == "tool"
                        ],
                        "visible_chunk_ids": list(dict.fromkeys(evidence_nodes or [])),
                        "delivered_span_ids": list(dict.fromkeys(delivered_span_ids)),
                        "semantics": "input_context_visible_to_llm_not_proven_used",
                    })
                    trace_logger.link_event_next(input_context_node)
                except Exception:
                    input_context_node = None

            try:
                response = self.llm.chat(messages=messages, tools=tool_schemas)
            except Exception as e:
                if self.verbose:
                    print(f"LLM error: {e}")
                if trace_logger:
                    try:
                        llm_node = trace_logger.add_llm_call(
                            model=getattr(self.llm, "model", None),
                            loop=loop_count,
                            message={"role": "assistant", "content": "", "tool_calls": []},
                            response={},
                            status="failed",
                            metadata={
                                "temperature": getattr(self.llm, "temperature", None),
                                "max_tokens": getattr(self.llm, "max_tokens", None),
                            },
                        )
                        trace_logger.add_error(llm_node, e, "llm_call", loop_count, "llm_api_error")
                        trace_logger.add_answer(None, "", loop_count, "llm_api_error",
                                                failed=True, raw_error=str(e))
                    except Exception:
                        pass
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
            if trace_logger and input_context_node and analysis_node:
                try:
                    trace_logger._node(analysis_node)["metadata"]["input_context_snapshot_id"] = input_context_node
                    trace_logger.add_edge(input_context_node, analysis_node, "consumed_by", {
                        "semantics": "input_context_visible_to_llm_not_proven_used",
                        "timestamp": trace_logger.timestamp(),
                    })
                except Exception:
                    pass

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
                raw_arguments = tc["function"].get("arguments", "")
                arguments_parse_error = None
                try:
                    func_args = json.loads(raw_arguments)
                except json.JSONDecodeError as exc:
                    func_args = {}
                    arguments_parse_error = str(exc)
                epistemic_context = func_args.pop("epistemic_context", None) if isinstance(func_args, dict) else None
                if not isinstance(epistemic_context, dict) or not epistemic_context:
                    if trace_logger:
                        trace_logger.metadata["root_identification_capable"] = False
                        trace_logger.metadata.setdefault("root_identification_blocking_reasons", [])
                        if "MISSING_EXPLICIT_EPISTEMIC_CONTEXT" not in trace_logger.metadata["root_identification_blocking_reasons"]:
                            trace_logger.metadata["root_identification_blocking_reasons"].append("MISSING_EXPLICIT_EPISTEMIC_CONTEXT")
                        trace_logger.metadata["missing_epistemic_context_count"] = trace_logger.metadata.get("missing_epistemic_context_count", 0) + 1
                    query_text_for_context = self._query_from_tool_args(func_name, func_args)
                    epistemic_context = {
                        "explicit": False,
                        "active_subgoal_ids": [],
                        "action_role": "TEST" if func_name not in {"read_chunk", "read_chunks"} else "VERIFY",
                        "propositions": [{
                            "subject": query_text_for_context,
                            "predicate": "query_about",
                            "object": "",
                            "relation_id": None,
                            "stance": "HYPOTHESIS",
                            "supporting_evidence_ids": [],
                            "missing_constraint_ids": ["relation_specific_support"],
                            "extractor_mode": "inferred",
                        }],
                    }
                else:
                    if not self._epistemic_props(epistemic_context, "all"):
                        if trace_logger:
                            trace_logger.metadata["root_identification_capable"] = False
                            trace_logger.metadata.setdefault("root_identification_blocking_reasons", [])
                            if "EPISTEMIC_CONTEXT_WITHOUT_PROPOSITIONS" not in trace_logger.metadata["root_identification_blocking_reasons"]:
                                trace_logger.metadata["root_identification_blocking_reasons"].append("EPISTEMIC_CONTEXT_WITHOUT_PROPOSITIONS")
                    epistemic_context = {**epistemic_context, "explicit": True}
                    if trace_logger:
                        trace_logger.metadata.setdefault("root_identification_capable", True)

                if self.verbose:
                    print(f"Tool: {func_name}")
                    print(f"  Args: {func_args}")

                search_history_start = len(context.search_history)
                read_chunk_ids_start = set(context.read_chunk_ids)
                call_order = len([n for n in getattr(trace_logger, "nodes", []) if n["type"] == "plan_query"]) + 1 if trace_logger else 1
                visible_span_ids = []
                for delivery in context.context_deliveries:
                    visible_span_ids.extend(delivery.get("span_ids", []) or [])
                decision_node = self._trace_decision_record(
                    trace_logger,
                    analysis_node,
                    input_context_node,
                    func_name,
                    func_args,
                    epistemic_context,
                    loop_count,
                    call_order,
                    tc.get("id"),
                    list(dict.fromkeys(visible_span_ids)),
                )

                try:
                    tool_result, tool_log = self.tools.execute(func_name, context, **func_args)
                except Exception as e:
                    tool_result = f"Error executing tool: {str(e)}"
                    tool_log = {"retrieved_tokens": 0, "error": str(e)}

                new_evidence_nodes = self._trace_tool_evidence(
                    trace_logger,
                    analysis_node,
                    decision_node,
                    func_name,
                    func_args,
                    tool_result,
                    tool_log,
                    context,
                    search_history_start,
                    read_chunk_ids_start,
                    loop_count,
                    tc.get("id"),
                    raw_arguments,
                    arguments_parse_error,
                )
                evidence_nodes.extend(new_evidence_nodes)

                if self.verbose:
                    output_preview = tool_result[:300] + "..." if len(tool_result) > 300 else tool_result
                    print(f"  Result: {output_preview}")
                    if tool_log.get("retrieved_tokens", 0) > 0:
                        print(f"  Tokens: {tool_log['retrieved_tokens']}")
                    print()

                tool_message_index = len(messages)
                tool_result_payload = tool_log.get("tool_result") if isinstance(tool_log, dict) else None
                delivered_span_ids = []
                if isinstance(tool_result_payload, dict):
                    for result_item in tool_result_payload.get("results", []) or []:
                        if isinstance(result_item, dict):
                            delivered_span_ids.extend(result_item.get("returned_span_ids", []) or [])
                            delivered_span_ids.extend(
                                span.get("span_id") for span in result_item.get("matched_spans", []) or []
                                if isinstance(span, dict) and span.get("span_id")
                            )
                    delivered_span_ids.extend(tool_result_payload.get("diagnostics", {}).get("returned_span_ids", []) or [])
                context.record_context_delivery(
                    llm_call_id=analysis_node,
                    tool_call_id=tc["id"],
                    message_index=tool_message_index,
                    span_ids=delivered_span_ids,
                    text=tool_result,
                    metadata={"tool_name": func_name, "tool_status": tool_log.get("status")},
                )
                self._trace_evidence_assessments(
                    trace_logger,
                    decision_node,
                    epistemic_context or {},
                    list(dict.fromkeys(delivered_span_ids)),
                    tool_result_payload or {},
                )
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
