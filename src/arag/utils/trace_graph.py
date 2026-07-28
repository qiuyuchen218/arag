"""Execution provenance graph and HTML visualization for ARAG runs."""

import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from arag.utils.trace_error import normalize_error


class TraceGraph:
    """Execution-only trace graph for one QA sample."""

    trace_schema_version = "2.0"
    EVENT_TYPES = {"question", "llm_call", "plan_query", "retriever_call", "read_call", "context_snapshot", "branch_fork", "evaluation", "answer", "error"}
    ARTIFACT_TYPES = {"document", "retrieved_chunk", "evidence_span", "tool_result", "answer"}
    EPISTEMIC_TYPES = {"subgoal", "claim", "evidence_set", "hypothesis", "constraint", "query_intent", "commitment_event"}
    NODE_TYPES = EVENT_TYPES | ARTIFACT_TYPES | EPISTEMIC_TYPES
    EDGE_TYPES = {
        "next", "next_in_branch", "invokes", "executes", "returns", "retrieves", "reads",
        "contains", "consumes", "consumed_by", "generates", "delivered_in_context",
        "delivered_into", "available_in_context",
        "cites", "member_of", "supports", "contradicts", "jointly_supports", "depends_on",
        "answers_subgoal", "decomposes_to", "targets_subgoal", "proposed_for",
        "proposes", "motivates", "queries_for", "updates", "resolves_candidate_for", "influences",
        "forked_from", "inherits", "invalidates", "rejects", "supersedes",
        "failed_with", "evaluates",
    }

    def __init__(
        self,
        sample_id: str = None,
        dataset: str = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.sample_id = str(sample_id) if sample_id is not None else None
        self.dataset = dataset
        self.created_at = self._utc_now()
        self.metadata = metadata or {}
        self.reset(clear_metadata=False)

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def reset(self, clear_metadata: bool = True):
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, Any]] = []
        self._type_counters: Dict[str, int] = {}
        self._step_counter = 0
        self._artifact_counter = 0
        self._last_event_id = None
        self._chunk_index: Dict[str, str] = {}
        self._context_edge_index: Dict[tuple, Dict[str, Any]] = {}
        if clear_metadata:
            self.metadata = {}
            self.created_at = self._utc_now()

    def timestamp(self) -> str:
        return self._utc_now()

    @staticmethod
    def _clean_json(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): TraceGraph._clean_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [TraceGraph._clean_json(v) for v in value]
        if isinstance(value, tuple):
            return [TraceGraph._clean_json(v) for v in value]
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        try:
            json.dumps(value, ensure_ascii=False, allow_nan=False)
            return value
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _safe_id_part(value: Any) -> str:
        text = str(value) if value is not None else "unknown"
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)
        return safe[:90] or "unknown"

    def _new_id(self, node_type: str) -> str:
        prefixes = {
            "question": "q", "llm_call": "llm", "plan_query": "pq",
            "retriever_call": "ret", "read_call": "read", "context_snapshot": "ctx",
            "branch_fork": "fork", "evaluation": "eval", "retrieved_chunk": "chunk",
            "document": "doc", "evidence_span": "span", "tool_result": "toolres",
            "subgoal": "sg", "claim": "claim", "evidence_set": "evset",
            "hypothesis": "hyp", "constraint": "con", "query_intent": "qi",
            "commitment_event": "commit",
            "answer": "ans", "error": "err",
        }
        self._type_counters[node_type] = self._type_counters.get(node_type, 0) + 1
        return f"{prefixes.get(node_type, 'n')}_{self._type_counters[node_type]:03d}"

    def add_node(
        self,
        node_type: str,
        content: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        status: str = "success",
        node_id: str = None,
    ) -> str:
        if node_type not in self.NODE_TYPES:
            raise ValueError(f"Unsupported trace node type: {node_type}")
        metadata = self._clean_json(metadata or {})
        node_id = node_id or self._new_id(node_type)
        if any(node["id"] == node_id for node in self.nodes):
            raise ValueError(f"Duplicate trace node id: {node_id}")
        node = {
            "id": node_id,
            "type": node_type,
            "content": self._clean_json(content),
            "metadata": metadata,
            "timestamp": self.timestamp(),
            "status": status,
            "branch_id": metadata.get("branch_id", "b0"),
        }
        if node_type in self.EVENT_TYPES:
            self._step_counter += 1
            node["step_index"] = self._step_counter
        else:
            self._artifact_counter += 1
            node["created_index"] = self._artifact_counter
            node["step_index"] = None
        self.nodes.append(node)
        return node_id

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if not source or not target:
            return
        if edge_type not in self.EDGE_TYPES:
            raise ValueError(f"Unsupported trace edge type: {edge_type}")
        metadata = self._clean_json(metadata or {})
        if edge_type == "available_in_context":
            key = (source, target, edge_type)
            if key in self._context_edge_index:
                edge = self._context_edge_index[key]
                occurrence = dict(metadata)
                occurrence.pop("semantics", None)
                edge["metadata"].setdefault("occurrences", []).append(occurrence)
                edge["metadata"]["num_occurrences"] = len(edge["metadata"]["occurrences"])
                return
            metadata.setdefault("semantics", "available_to_model_not_proven_used")
            metadata.setdefault("occurrences", [dict(metadata)])
            metadata["num_occurrences"] = 1
        edge = {"source": source, "target": target, "type": edge_type, "metadata": metadata}
        self.edges.append(edge)
        if edge_type == "available_in_context":
            self._context_edge_index[(source, target, edge_type)] = edge

    def link_event_next(self, target: str, metadata: Optional[Dict[str, Any]] = None):
        target_node = self._node(target)
        if target_node["type"] not in self.EVENT_TYPES:
            return
        if self._last_event_id and self._last_event_id != target:
            source_node = self._node(self._last_event_id)
            next_metadata = {
                "from_step": source_node["step_index"],
                "to_step": target_node["step_index"],
                "timestamp": target_node["timestamp"],
                "branch_id": target_node.get("branch_id", "b0"),
                **(metadata or {}),
            }
            self.add_edge(self._last_event_id, target, "next", next_metadata)
            self.add_edge(self._last_event_id, target, "next_in_branch", next_metadata)
        self._last_event_id = target

    add_temporal_next = link_event_next

    def latest_node_id(self, node_type: str) -> Optional[str]:
        for node in reversed(self.nodes):
            if node.get("type") == node_type:
                return node.get("id")
        return None

    def _node(self, node_id: str) -> Dict[str, Any]:
        for node in self.nodes:
            if node["id"] == node_id:
                return node
        raise ValueError(f"Unknown trace node id: {node_id}")

    def add_question(self, question: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        existing = self.latest_node_id("question")
        if existing:
            return existing
        node_id = self.add_node("question", question, metadata or {"role": "user_input"})
        self.link_event_next(node_id)
        return node_id

    def add_llm_call(
        self,
        model: str = None,
        loop: int = 0,
        message: Optional[Dict[str, Any]] = None,
        response: Optional[Dict[str, Any]] = None,
        forced: bool = False,
        reason: str = None,
        status: str = "success",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        message = message or {}
        response = response or {}
        tool_calls = message.get("tool_calls", []) or []
        node_metadata = {
            "loop": loop,
            "model": model,
            "input_tokens": response.get("input_tokens", 0),
            "output_tokens": response.get("output_tokens", 0),
            "cost": response.get("cost", 0),
            "tool_calls": tool_calls,
            "tool_call_count": len(tool_calls),
            "forced": forced,
            "termination_reason": reason,
            "response_preview": self._short_text(message.get("content", ""), 1000),
        }
        node_metadata.update(metadata or {})
        node_id = self.add_node("llm_call", f"model={model}; loop={loop}", node_metadata, status=status)
        self.link_event_next(node_id)
        return node_id

    def add_plan_query(
        self,
        llm_id: str,
        query: str,
        tool_name: str,
        arguments: Dict[str, Any],
        loop: int,
        tool_call_id: str = None,
        call_order: int = 1,
        raw_arguments: str = None,
        arguments_parse_error: str = None,
    ) -> str:
        metadata = {
            "loop": loop, "source": "tool_call_argument", "tool_name": tool_name,
            "arguments": arguments, "raw_arguments": raw_arguments,
            "arguments_parse_error": arguments_parse_error,
        }
        node_id = self.add_node("plan_query", query, metadata)
        self.add_edge(llm_id, node_id, "invokes", {
            "loop": loop, "tool_call_id": tool_call_id, "tool_name": tool_name,
            "call_order": call_order, "timestamp": self.timestamp(),
        })
        self.link_event_next(node_id)
        return node_id

    def add_retriever_call(
        self,
        plan_query_id: str,
        tool_name: str,
        query: str,
        arguments: Dict[str, Any],
        loop: int,
        tool_call_id: str = None,
        call_order: int = 1,
        operation: str = "search",
        status: str = "success",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        node_metadata = {
            "loop": loop, "tool_name": tool_name, "arguments": arguments, "query": query,
            "tool_call_id": tool_call_id, "call_order": call_order, "operation": operation,
        }
        node_metadata.update(metadata or {})
        node_type = "read_call" if operation == "read" else "retriever_call"
        node_id = self.add_node(node_type, f"{tool_name}({query})", node_metadata, status=status)
        self.add_edge(plan_query_id, node_id, "executes", {
            "loop": loop, "tool_call_id": tool_call_id, "tool_name": tool_name,
            "operation": operation, "timestamp": self.timestamp(),
        })
        self.link_event_next(node_id)
        return node_id

    def _chunk_key(self, metadata: Dict[str, Any], dedupe_key: str = None) -> str:
        if dedupe_key:
            return dedupe_key
        corpus = metadata.get("corpus_id") or metadata.get("dataset") or self.dataset or "corpus"
        chunk_id = metadata.get("chunk_id")
        doc_id = metadata.get("doc_id")
        if chunk_id is not None:
            return f"{corpus}:chunk:{chunk_id}"
        if doc_id is not None:
            return f"{corpus}:doc:{doc_id}"
        return f"occurrence:{len(self._chunk_index) + 1}"

    def add_retrieval_result(
        self,
        retriever_call_id: str,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
        occurrence: Optional[Dict[str, Any]] = None,
        dedupe_key: str = None,
        edge_type: str = "retrieves",
    ) -> str:
        metadata = dict(metadata or {})
        occurrence = dict(occurrence or {})
        key = self._chunk_key(metadata, dedupe_key)
        operation = occurrence.get("operation") or metadata.get("operation")
        is_read = operation == "read" or metadata.get("source") == "read_chunk"
        if key in self._chunk_index:
            node_id = self._chunk_index[key]
            node = self._node(node_id)
            stable = node["metadata"]
            stable["retrieval_count"] = stable.get("retrieval_count", 0) + (0 if is_read else 1)
            stable["read_count"] = stable.get("read_count", 0) + (1 if is_read else 0)
            if is_read:
                node["content"] = content or node["content"]
                stable["full_content_available"] = bool(content)
        else:
            chunk_id = metadata.get("chunk_id")
            readable = f"chunk_{self._safe_id_part(chunk_id)}" if chunk_id is not None else None
            node_id = readable if readable and not any(n["id"] == readable for n in self.nodes) else None
            stable = {
                "chunk_id": metadata.get("chunk_id"), "doc_id": metadata.get("doc_id"),
                "corpus_id": metadata.get("corpus_id") or metadata.get("dataset") or self.dataset,
                "full_content_available": bool(content) if is_read else bool(metadata.get("full_content_available")),
                "first_seen_loop": metadata.get("loop") or occurrence.get("loop"),
                "retrieval_count": 0 if is_read else 1,
                "read_count": 1 if is_read else 0,
            }
            stable.update({k: v for k, v in metadata.items() if k in {"title", "source", "tokens"}})
            node_id = self.add_node("retrieved_chunk", content, stable, node_id=node_id)
            self._chunk_index[key] = node_id
        edge_metadata = {
            "loop": occurrence.get("loop") or metadata.get("loop"),
            "tool_call_id": occurrence.get("tool_call_id"),
            "tool_name": occurrence.get("tool_name") or metadata.get("tool_name"),
            "query": occurrence.get("query") or metadata.get("query"),
            "rank": occurrence.get("rank") or metadata.get("rank"),
            "raw_score": occurrence.get("raw_score", metadata.get("score")),
            "retrieved_at": self.timestamp(),
            "operation": operation or ("read" if is_read else "search"),
            "result_metadata": occurrence.get("result_metadata"),
        }
        self.add_edge(retriever_call_id, node_id, edge_type, edge_metadata)
        return node_id

    def add_evidence(self, content: Any = None, metadata: Optional[Dict[str, Any]] = None,
                     dedupe_key: str = None) -> str:
        return self.add_retrieval_result("", content, metadata, {}, dedupe_key)

    def add_context_snapshot(self, llm_id: str, context: Optional[Dict[str, Any]] = None):
        context = self._clean_json(context or {})
        node_id = self.add_node("context_snapshot", {
            "visible_chunk_ids": context.get("visible_chunk_ids", []),
            "delivered_span_ids": context.get("delivered_span_ids", []),
            "message_index": context.get("message_index"),
            "text_hash": context.get("text_hash"),
        }, context)
        if llm_id:
            self._node(llm_id)["metadata"]["input_context_snapshot_id"] = node_id
            self._node(llm_id)["metadata"]["context"] = context
            self.add_edge(node_id, llm_id, "consumed_by", {
                "semantics": "input_context_visible_to_llm_not_proven_used",
                "timestamp": self.timestamp(),
            })
            self.link_event_next(node_id)
        return node_id

    def add_evidence_span(self, span: Dict[str, Any], source_node_id: str = None) -> str:
        span = self._clean_json(span)
        node_id = span.get("span_id") or self._new_id("evidence_span")
        if any(n["id"] == node_id for n in self.nodes):
            return node_id
        identity = (
            span.get("doc_id"),
            span.get("chunk_id"),
            span.get("sentence_id"),
            span.get("start_offset"),
            span.get("end_offset"),
            span.get("content_hash") or span.get("text"),
        )
        if any(v is not None for v in identity):
            for existing in self.nodes:
                if existing.get("type") != "evidence_span":
                    continue
                md = existing.get("metadata", {})
                existing_identity = (
                    md.get("doc_id"),
                    md.get("chunk_id"),
                    md.get("sentence_id"),
                    md.get("start_offset"),
                    md.get("end_offset"),
                    md.get("content_hash") or existing.get("content"),
                )
                if existing_identity == identity:
                    if source_node_id:
                        self.add_edge(source_node_id, existing["id"], "contains", {"timestamp": self.timestamp(), "deduped_span_id": node_id})
                    return existing["id"]
        added = self.add_node("evidence_span", span.get("text", ""), span, node_id=node_id)
        if source_node_id:
            self.add_edge(source_node_id, added, "contains", {"timestamp": self.timestamp()})
        return added

    def add_tool_result_node(self, tool_call_id: str, tool_result: Dict[str, Any]) -> str:
        node_id = tool_result.get("call_id") or self._new_id("tool_result")
        node_id = f"toolres_{self._safe_id_part(node_id)}"
        if any(n["id"] == node_id for n in self.nodes):
            return node_id
        added = self.add_node("tool_result", tool_result.get("rendered_text", ""), tool_result, node_id=node_id)
        self.add_edge(tool_call_id, added, "returns", {"timestamp": self.timestamp()})
        for item in tool_result.get("results", []) or []:
            spans = item.get("matched_spans", []) or []
            if "returned_span_ids" in item and item.get("returned_text"):
                # ReadReceipt spans are also stored on read chunk metadata; keep the receipt link here.
                pass
            for span in spans:
                if span.get("sentence_id") is None and len(str(span.get("text", "") or "")) > 800:
                    continue
                sid = self.add_evidence_span(span, added)
                self.add_edge(added, sid, "contains", {"timestamp": self.timestamp()})
        return added

    def add_claim_assessment(self, assessment: Dict[str, Any], generated_by: str = None) -> str:
        claim = self._clean_json(assessment.get("claim", {}))
        claim_id = claim.get("claim_id") or self._new_id("claim")
        source = generated_by or claim.get("generated_by")
        source_step = None
        if source and any(n["id"] == source for n in self.nodes):
            source_step = self._node(source).get("step_index")
        node_id = self.add_node("claim", claim.get("content", ""), {
            **claim,
            "step_index": source_step,
            "support_vector": assessment.get("support_vector"),
            "defect_vector": assessment.get("defect_vector"),
            "raw_score": assessment.get("raw_score"),
            "calibrated_score": assessment.get("calibrated_score"),
            "verifier_mode": assessment.get("verifier_mode"),
            "authoritative": assessment.get("authoritative"),
            "verifier_is_real": assessment.get("verifier_is_real"),
            "verifier_decision_capable": assessment.get("verifier_decision_capable"),
            "verifier_calibrated": assessment.get("verifier_calibrated"),
            "verifier_authoritative_for_repair": assessment.get("verifier_authoritative_for_repair"),
            "verifier_input_span_ids": assessment.get("verifier_input_span_ids"),
            "verifier_input_doc_ids": assessment.get("verifier_input_doc_ids"),
            "verifier_input_token_count": assessment.get("verifier_input_token_count"),
            "verifier_input_hash": assessment.get("verifier_input_hash"),
            "evidence_set_span_ids": assessment.get("evidence_set_span_ids"),
            "evidence_isolation_valid": assessment.get("evidence_isolation_valid"),
            "evidence_leakage_span_ids": assessment.get("evidence_leakage_span_ids"),
        }, status=assessment.get("status", "UNCERTAIN"), node_id=claim_id)
        if source:
            self.add_edge(source, node_id, "generates", {"timestamp": self.timestamp()})
        evset = assessment.get("best_evidence_set") or {}
        evset_id = evset.get("evidence_set_id")
        if evset_id:
            ev_node = self.add_node("evidence_set", evset_id, evset, node_id=evset_id)
            self.add_edge(ev_node, node_id, "jointly_supports", {
                "status": assessment.get("status"),
                "support_vector": assessment.get("support_vector"),
                "timestamp": self.timestamp(),
            })
            for span_id in evset.get("evidence_span_ids", []) or []:
                if any(n["id"] == span_id for n in self.nodes):
                    self.add_edge(span_id, ev_node, "member_of", {"timestamp": self.timestamp()})
        for dep in claim.get("dependencies", []) or []:
            if any(n["id"] == dep for n in self.nodes):
                self.add_edge(node_id, dep, "depends_on", {"timestamp": self.timestamp()})
        return node_id

    def add_subgoal_node(self, subgoal: Dict[str, Any], question_node_id: str = None) -> str:
        subgoal = self._clean_json(subgoal)
        node_id = subgoal.get("subgoal_id") or self._new_id("subgoal")
        if any(n["id"] == node_id for n in self.nodes):
            return node_id
        added = self.add_node("subgoal", subgoal.get("content", ""), subgoal, status=subgoal.get("status", "open"), node_id=node_id)
        if question_node_id:
            self.add_edge(question_node_id, added, "decomposes_to", {"timestamp": self.timestamp()})
        for dep in subgoal.get("dependencies", []) or []:
            if any(n["id"] == dep for n in self.nodes):
                self.add_edge(added, dep, "depends_on", {"timestamp": self.timestamp()})
        return added

    def add_hypothesis_node(self, hypothesis: Dict[str, Any]) -> str:
        hypothesis = self._clean_json(hypothesis)
        node_id = hypothesis.get("hypothesis_id") or self._new_id("hypothesis")
        if any(n["id"] == node_id for n in self.nodes):
            node = self._node(node_id)
            node["metadata"].update(hypothesis)
            node["content"] = hypothesis.get("content") or hypothesis.get("canonical_entity") or node.get("content")
            node["status"] = hypothesis.get("status", node.get("status"))
            return node_id
        added = self.add_node("hypothesis", hypothesis.get("content", ""), hypothesis, status=hypothesis.get("status", "proposed"), node_id=node_id)
        target = hypothesis.get("target_subgoal_id")
        if target and any(n["id"] == target for n in self.nodes):
            self.add_edge(added, target, "proposed_for", {"timestamp": self.timestamp()})
        source = hypothesis.get("source_event_id") or hypothesis.get("generated_by")
        if source and any(n["id"] == source for n in self.nodes):
            self.add_edge(source, added, "proposes", {"timestamp": self.timestamp()})
            if self._node(source)["type"] == "plan_query" and not hypothesis.get("posthoc_summary"):
                self.add_edge(added, source, "motivates", {"timestamp": self.timestamp()})
        return added

    def add_query_intent_node(self, intent: Dict[str, Any]) -> str:
        intent = self._clean_json(intent)
        node_id = intent.get("query_intent_id") or self._new_id("query_intent")
        if any(n["id"] == node_id for n in self.nodes):
            return node_id
        added = self.add_node("query_intent", intent.get("normalized_query") or intent.get("raw_query", ""), intent, status=intent.get("query_mode", "unknown"), node_id=node_id)
        pq = intent.get("source_plan_query_id")
        if pq and any(n["id"] == pq for n in self.nodes):
            self.add_edge(pq, added, "generates", {"timestamp": self.timestamp()})
        sg = intent.get("target_subgoal_id")
        if sg and any(n["id"] == sg for n in self.nodes):
            self.add_edge(added, sg, "targets_subgoal", {"timestamp": self.timestamp()})
        return added

    def add_commitment_event_node(self, event: Dict[str, Any]) -> str:
        event = self._clean_json(event)
        node_id = event.get("commitment_event_id") or self._new_id("commitment_event")
        if any(n["id"] == node_id for n in self.nodes):
            return node_id
        added = self.add_node("commitment_event", event.get("candidate_entity", ""), event, status="premature" if event.get("is_premature") else "committed", node_id=node_id)
        hyp = event.get("hypothesis_id")
        if hyp and any(n["id"] == hyp for n in self.nodes):
            self.add_edge(hyp, added, "generates", {"timestamp": self.timestamp()})
        source = event.get("source_event_id")
        if source and any(n["id"] == source for n in self.nodes):
            self.add_edge(added, source, "motivates", {"timestamp": self.timestamp()})
        target = event.get("target_subgoal_id")
        if target and any(n["id"] == target for n in self.nodes):
            self.add_edge(added, target, "proposed_for", {"timestamp": self.timestamp()})
        return added

    def add_intermediate_claims(self, llm_id: str, message: str, loop: int) -> List[str]:
        return []

    def add_error(self, parent_id: str, raw_error: Any, stage: str, loop: int = 0,
                  termination_reason: str = "", fatal: bool = True) -> str:
        normalized = normalize_error(raw_error, termination_reason)
        error_id = self.add_node("error", str(raw_error), {
            **normalized, "raw_exception": str(raw_error), "severity": "fatal" if fatal else "warning",
            "stage": stage, "loop": loop, "termination_reason": termination_reason,
        }, status="failed" if fatal else "warning")
        self.add_edge(parent_id, error_id, "failed_with", {
            "stage": stage, "fatal": fatal, "timestamp": self.timestamp(),
        })
        self.link_event_next(error_id)
        self.metadata["normalized_termination_reason"] = normalized["normalized_termination_reason"]
        self.metadata["debug_summary"] = normalized["debug_summary"]
        return error_id

    def add_answer(self, llm_id: str, answer: str, loop: int, termination_reason: str,
                   evidence_nodes: Optional[List[str]] = None, failed: bool = False,
                   raw_error: str = "") -> str:
        usable = bool(str(answer or "").strip()) and not str(answer).lower().startswith("error:") and not failed
        answer_id = self.add_node("answer", answer if usable else "", {
            "pred_answer": answer if usable else "", "termination_reason": termination_reason,
            "is_error_answer": not usable, "failure_reason": termination_reason if not usable else None,
            "raw_error": raw_error or None, "loop": loop,
        }, status="success" if usable else "failed")
        if llm_id:
            self.add_edge(llm_id, answer_id, "generates", {
                "loop": loop, "generation_order": 1, "timestamp": self.timestamp(),
            })
        self.link_event_next(answer_id)
        self.metadata["final_answer"] = answer if usable else ""
        return answer_id

    def finalize_metadata(self):
        node_counts: Dict[str, int] = {}
        edge_counts: Dict[str, int] = {}
        for node in self.nodes:
            node_counts[node["type"]] = node_counts.get(node["type"], 0) + 1
        for edge in self.edges:
            edge_counts[edge["type"]] = edge_counts.get(edge["type"], 0) + 1
        llm_nodes = [n for n in self.nodes if n["type"] == "llm_call"]
        errors = [n for n in self.nodes if n["type"] == "error"]
        retrieval_edges = [e for e in self.edges if e["type"] in {"retrieves", "reads"}]
        subgoals = [n for n in self.nodes if n["type"] == "subgoal"]
        hypotheses = [n for n in self.nodes if n["type"] == "hypothesis"]
        query_intents = [n for n in self.nodes if n["type"] == "query_intent"]
        claims = [n for n in self.nodes if n["type"] == "claim"]
        required_subgoals = [n for n in subgoals if n["metadata"].get("required", True)]
        resolved_required = [n for n in required_subgoals if n.get("status") == "resolved"]
        tool_calls_by_name: Dict[str, int] = {}
        for node in self.nodes:
            if node["type"] in {"retriever_call", "read_call"}:
                name = node["metadata"].get("tool_name", "unknown")
                tool_calls_by_name[name] = tool_calls_by_name.get(name, 0) + 1
        self.metadata.update({
            "trace_schema_version": self.trace_schema_version,
            "sample_id": self.sample_id,
            "dataset": self.dataset,
            "num_event_nodes": sum(1 for n in self.nodes if n["type"] in self.EVENT_TYPES),
            "num_artifact_nodes": sum(1 for n in self.nodes if n["type"] in self.ARTIFACT_TYPES),
            "num_nodes_by_type": node_counts,
            "num_edges_by_type": edge_counts,
            "loops": max([n["metadata"].get("loop", 0) or 0 for n in self.nodes] or [0]),
            "tool_call_count": sum(tool_calls_by_name.values()),
            "tool_calls_by_name": tool_calls_by_name,
            "retrieval_occurrence_count": len(retrieval_edges),
            "unique_chunk_count": node_counts.get("retrieved_chunk", 0),
            "read_occurrence_count": sum(1 for e in retrieval_edges if e["metadata"].get("operation") == "read"),
            "empty_retrieval_count": sum(
                1 for n in errors
                if n["metadata"].get("stage") == "retrieval"
                and (
                    n["metadata"].get("termination_reason") == "no_retrieval"
                    or "empty retrieval" in str(n.get("content", "")).lower()
                )
            ),
            "llm_call_count": len(llm_nodes),
            "total_input_tokens": sum(n["metadata"].get("input_tokens", 0) or 0 for n in llm_nodes),
            "total_output_tokens": sum(n["metadata"].get("output_tokens", 0) or 0 for n in llm_nodes),
            "total_cost": sum(n["metadata"].get("cost", 0) or 0 for n in llm_nodes),
            "termination_reason": self.metadata.get("termination_reason"),
            "answer_generated": any(n["type"] == "answer" and n["status"] == "success" for n in self.nodes),
            "has_runtime_error": any(n["metadata"].get("severity") == "fatal" for n in errors),
            "runtime_error_types": sorted(set(n["metadata"].get("error_type", "unknown_error") for n in errors)),
            "number_of_subgoals": len(subgoals),
            "number_of_hypotheses": len(hypotheses),
            "number_of_query_intents": len(query_intents),
            "dependency_edge_count": sum(1 for e in self.edges if e["type"] == "depends_on"),
            "required_subgoal_coverage": len(resolved_required) / max(len(required_subgoals), 1),
            "authoritative_claim_count": sum(1 for n in claims if n["metadata"].get("authoritative")),
            "unassessed_claim_count": sum(1 for n in claims if n.get("status") == "UNASSESSED"),
            "semantic_validation_warnings": self.metadata.get("semantic_validation_warnings", []),
        })
        self.metadata.setdefault("normalized_termination_reason", "success" if not errors else "unknown_error")
        self.metadata.setdefault("debug_summary", "Execution completed successfully." if not errors else "Execution failed.")

    def validate(self):
        self.metadata["trace_schema_version"] = self.trace_schema_version
        json.dumps({"nodes": self.nodes, "edges": self.edges, "metadata": self.metadata},
                   ensure_ascii=False, allow_nan=False)
        node_ids = [node["id"] for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Trace validation failed: duplicate node ids")
        id_set = set(node_ids)
        questions = [n for n in self.nodes if n["type"] == "question"]
        answers = [n for n in self.nodes if n["type"] == "answer"]
        if len(questions) != 1:
            raise ValueError(f"Trace validation failed: expected exactly one question, found {len(questions)}")
        if len(answers) > 1:
            raise ValueError(f"Trace validation failed: expected at most one answer, found {len(answers)}")
        steps = [n["step_index"] for n in self.nodes if n["type"] in self.EVENT_TYPES]
        if steps != sorted(steps) or len(steps) != len(set(steps)):
            raise ValueError("Trace validation failed: event step_index values must be strictly increasing")
        for node in self.nodes:
            if node["type"] not in self.NODE_TYPES:
                raise ValueError(f"Trace validation failed: illegal node type {node['type']}")
        for edge in self.edges:
            if edge["type"] not in self.EDGE_TYPES:
                raise ValueError(f"Trace validation failed: illegal edge type {edge['type']}")
            if edge["source"] not in id_set or edge["target"] not in id_set:
                raise ValueError(f"Trace validation failed: missing edge endpoint {edge}")
            source_type = self._node(edge["source"])["type"]
            target_type = self._node(edge["target"])["type"]
            if edge["type"] in {"retrieves", "reads"} and (source_type not in {"retriever_call", "read_call"} or target_type != "retrieved_chunk"):
                raise ValueError("Trace validation failed: retrieves/reads must connect tool call to retrieved_chunk")
            if edge["type"] == "executes" and (source_type != "plan_query" or target_type not in {"retriever_call", "read_call"}):
                raise ValueError("Trace validation failed: executes must connect plan_query to tool call")
            if edge["type"] == "generates" and source_type not in {"llm_call", "answer", "plan_query", "hypothesis"}:
                raise ValueError("Trace validation failed: generates source must be llm_call, answer, plan_query, or hypothesis")
        self.validate_v2()

    def validate_v2(self):
        errors = []
        id_set = {node["id"] for node in self.nodes}
        by_id = {node["id"]: node for node in self.nodes}
        event_types = self.EVENT_TYPES
        for node in self.nodes:
            if node["type"] not in event_types and node.get("step_index") is not None:
                errors.append(f"artifact_or_epistemic_step_index:{node['id']}")
            if node["type"] == "context_snapshot":
                delivered = node.get("metadata", {}).get("delivered_span_ids", []) or []
                if len(delivered) != len(set(delivered)):
                    errors.append(f"duplicate_delivered_span_ids:{node['id']}")
                missing = [sid for sid in delivered if sid not in id_set]
                if missing:
                    errors.append(f"missing_delivered_span:{node['id']}:{missing[:3]}")
            if node["type"] == "evidence_span":
                md = node.get("metadata", {})
                text = str(node.get("content", "") or "")
                start = md.get("start_offset", 0) or 0
                end = md.get("end_offset", 0) or 0
                if end and start and int(end) < int(start):
                    errors.append(f"invalid_span_offsets:{node['id']}")
                if md.get("sentence_id") is None and len(text) > 800:
                    errors.append(f"duplicate_full_chunk_span:{node['id']}")
        identities = {}
        for node in self.nodes:
            if node["type"] != "evidence_span":
                continue
            md = node.get("metadata", {})
            identity = (
                md.get("doc_id"),
                md.get("chunk_id"),
                md.get("sentence_id"),
                md.get("start_offset"),
                md.get("end_offset"),
                md.get("content_hash") or node.get("content"),
            )
            if identity in identities:
                errors.append(f"duplicate_artifact_identity:{identities[identity]}:{node['id']}")
            identities[identity] = node["id"]
        for node in self.nodes:
            if node["type"] != "claim":
                continue
            md = node.get("metadata", {})
            evset = None
            for edge in self.edges:
                if edge["type"] == "jointly_supports" and edge["target"] == node["id"]:
                    evset = by_id.get(edge["source"])
                    break
            if evset:
                ev_ids = set(evset.get("metadata", {}).get("evidence_span_ids", []) or [])
                in_ids = set(md.get("verifier_input_span_ids", []) or [])
                if ev_ids != in_ids:
                    errors.append(f"evidence_set_verifier_leak:{node['id']}")
            if md.get("evidence_isolation_valid") is False:
                errors.append(f"evidence_isolation_invalid:{node['id']}")
            claim = md
            if claim.get("claim_type") in {"answer_claim", "factual_claim"} and not claim.get("dependencies"):
                errors.append(f"missing_expected_dependency:{node['id']}")
        for edge in self.edges:
            if edge["type"] == "next_in_branch":
                s, t = by_id[edge["source"]], by_id[edge["target"]]
                if s.get("branch_id", "b0") != t.get("branch_id", "b0"):
                    errors.append(f"next_in_branch_cross_branch:{edge['source']}->{edge['target']}")
            if edge["type"] == "consumed_by":
                s, t = by_id[edge["source"]], by_id[edge["target"]]
                if s["type"] != "context_snapshot" or t["type"] != "llm_call":
                    errors.append(f"bad_consumed_by:{edge['source']}->{edge['target']}")
                if (s.get("step_index") or 0) >= (t.get("step_index") or 0):
                    errors.append(f"context_not_before_llm:{edge['source']}->{edge['target']}")
            if edge["type"] in {"proposes", "motivates", "generates", "updates", "executes", "returns", "targets_subgoal"}:
                s, t = by_id[edge["source"]], by_id[edge["target"]]
                s_step = s.get("step_index") or s.get("metadata", {}).get("step_index") or s.get("metadata", {}).get("first_proposed_at")
                t_step = t.get("step_index") or t.get("metadata", {}).get("step_index") or t.get("metadata", {}).get("first_proposed_at")
                if s_step is not None and t_step is not None and int(s_step) > int(t_step):
                    errors.append(f"causal_edge_temporal_invalid:{edge['source']}->{edge['target']}")
                if s.get("branch_id", "b0") != t.get("branch_id", "b0"):
                    errors.append(f"causal_edge_cross_branch:{edge['source']}->{edge['target']}")
        incoming_next = {}
        outgoing_next = {}
        for edge in self.edges:
            if edge["type"] == "next_in_branch":
                incoming_next[edge["target"]] = incoming_next.get(edge["target"], 0) + 1
                outgoing_next[edge["source"]] = outgoing_next.get(edge["source"], 0) + 1
        for node_id, count in incoming_next.items():
            if count > 1:
                errors.append(f"multiple_next_predecessors:{node_id}")
        for node_id, count in outgoing_next.items():
            if count > 1:
                errors.append(f"multiple_next_successors:{node_id}")
        for node in self.nodes:
            if node["type"] == "evidence_set":
                span_ids = node.get("metadata", {}).get("evidence_span_ids", []) or []
                missing = [sid for sid in span_ids if sid not in id_set]
                if missing:
                    errors.append(f"evidence_set_missing_span:{node['id']}:{missing[:3]}")
                docs = sorted({by_id[sid].get("metadata", {}).get("doc_id") for sid in span_ids if sid in by_id})
                docs = [str(d) for d in docs if d is not None]
                expected = sorted(str(d) for d in (node.get("metadata", {}).get("unique_doc_ids", []) or []))
                if docs != expected:
                    errors.append(f"evidence_set_doc_mismatch:{node['id']}")
        repair_plan = self.metadata.get("repair_plan") or {}
        root = repair_plan.get("root_cause_node")
        rollback = repair_plan.get("rollback_checkpoint")
        if root and root in by_id:
            if by_id[root].get("metadata", {}).get("posthoc_summary"):
                errors.append(f"root_is_posthoc_summary:{root}")
            if rollback and rollback in by_id:
                r_step = by_id[root].get("step_index") or by_id[root].get("metadata", {}).get("step_index") or by_id[root].get("metadata", {}).get("first_proposed_at")
                rb_step = by_id[rollback].get("step_index") or by_id[rollback].get("metadata", {}).get("step_index")
                if r_step is not None and rb_step is not None and int(rb_step) >= int(r_step):
                    errors.append(f"rollback_not_before_root:{rollback}:{root}")
                root_llm = repair_plan.get("root_generator_llm")
                if root_llm:
                    consumed = any(e["type"] == "consumed_by" and e["source"] == rollback and e["target"] == root_llm for e in self.edges)
                    if not consumed:
                        errors.append(f"rollback_not_consumed_by_root_generator:{rollback}:{root_llm}")
        if errors:
            self.metadata["trace_valid"] = False
            self.metadata["validation_errors"] = errors
            raise ValueError("Trace v2 validation failed: " + "; ".join(errors[:5]))
        self.metadata["trace_valid"] = True
        self.metadata["validation_errors"] = []

    def to_dict(self) -> Dict[str, Any]:
        self.finalize_metadata()
        graph = {
            "trace_schema_version": self.trace_schema_version,
            "sample_id": self.sample_id,
            "dataset": self.dataset,
            "created_at": self.created_at,
            "nodes": self.nodes,
            "edges": self.edges,
            "metadata": self.metadata,
        }
        return self._clean_json(graph)

    def save_json(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        graph = self.to_dict()
        try:
            self.validate()
            graph = self.to_dict()
        except ValueError as exc:
            self.metadata["trace_valid"] = False
            self.metadata.setdefault("validation_errors", [str(exc)])
            self.metadata["trace_validation_error"] = str(exc)
            graph = self.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(graph, f, ensure_ascii=False, indent=2, allow_nan=False)

    @staticmethod
    def _short_text(value: Any, max_chars: int = 120) -> str:
        if value is None:
            return ""
        text = value if isinstance(value, str) else json.dumps(TraceGraph._clean_json(value), ensure_ascii=False)
        text = " ".join(text.split())
        return text[:max_chars] + ("..." if len(text) > max_chars else "")

    def save_html(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        graph = self.to_dict()
        nodes = graph["nodes"]
        edges = graph["edges"]
        node_by_id = {node["id"]: node for node in nodes}
        event_nodes = sorted(
            [n for n in nodes if n["type"] in self.EVENT_TYPES],
            key=lambda n: n.get("step_index") or 0,
        )
        artifact_nodes = [n for n in nodes if n["type"] in self.ARTIFACT_TYPES]
        epistemic_nodes = [n for n in nodes if n["type"] in self.EPISTEMIC_TYPES]
        positions: Dict[str, Dict[str, int]] = {}
        x_gap, y_gap = 210, 115
        for idx, node in enumerate(event_nodes):
            positions[node["id"]] = {"x": 90 + idx * x_gap, "y": 170}
        for idx, node in enumerate(artifact_nodes):
            source_edges = [e for e in edges if e["target"] == node["id"] and e["type"] in {"retrieves", "reads"}]
            source = source_edges[0]["source"] if source_edges else None
            base_x = positions.get(source, {"x": 90 + (idx % 6) * x_gap})["x"]
            positions[node["id"]] = {"x": base_x, "y": 330 + (idx % 5) * y_gap}
        for idx, node in enumerate(epistemic_nodes):
            source_edges = [e for e in edges if e["target"] == node["id"] or e["source"] == node["id"]]
            anchor = None
            if source_edges:
                edge = source_edges[0]
                anchor = edge["source"] if edge["source"] != node["id"] else edge["target"]
            base_x = positions.get(anchor, {"x": 120 + (idx % 8) * x_gap})["x"]
            positions[node["id"]] = {"x": base_x, "y": 650 + (idx % 6) * y_gap}
        width = max([p["x"] for p in positions.values()] or [900]) + 220
        height = max([p["y"] for p in positions.values()] or [600]) + 180
        colors = {
            "question": "#2563eb", "llm_call": "#7c3aed", "plan_query": "#0891b2",
            "retriever_call": "#ea580c", "read_call": "#f59e0b", "context_snapshot": "#0d9488",
            "retrieved_chunk": "#16a34a", "evidence_span": "#15803d", "claim": "#be123c",
            "answer": "#dc2626", "error": "#991b1b",
        }
        edge_lines = []
        for edge in edges:
            s, t = positions.get(edge["source"]), positions.get(edge["target"])
            if not s or not t:
                continue
            label = edge["type"]
            if edge["type"] in {"retrieves", "reads"}:
                md = edge["metadata"]
                label = f'{edge["type"]} {md.get("tool_name", "")} r{md.get("rank", "")}'
            edge_lines.append(f'''
<line x1="{s["x"]}" y1="{s["y"]}" x2="{t["x"]}" y2="{t["y"]}" stroke="#94a3b8" stroke-width="1.6" marker-end="url(#arrow)" />
<text x="{(s["x"] + t["x"]) / 2}" y="{(s["y"] + t["y"]) / 2 - 6}" class="edge-label">{html.escape(label)}</text>''')
        detail_blocks, node_blocks = [], []
        for node in nodes:
            pos = positions[node["id"]]
            color = colors.get(node["type"], "#64748b")
            label = f'{node["id"]} · {node["type"]} · {node["status"]}'
            preview = self._short_text(node.get("content"), 90)
            detail_id = f'detail-{node["id"]}'
            node_blocks.append(f'''
<g class="node" onclick="showDetail('{detail_id}')">
  <rect x="{pos["x"] - 82}" y="{pos["y"] - 34}" width="164" height="68" rx="7" fill="{color}" opacity="0.94" />
  <text x="{pos["x"]}" y="{pos["y"] - 9}" text-anchor="middle" class="node-title">{html.escape(label)}</text>
  <foreignObject x="{pos["x"] - 72}" y="{pos["y"]}" width="144" height="28"><div xmlns="http://www.w3.org/1999/xhtml" class="node-preview">{html.escape(preview)}</div></foreignObject>
</g>''')
            detail_blocks.append(f'<pre id="{detail_id}" class="detail-block">{html.escape(json.dumps(node, ensure_ascii=False, indent=2))}</pre>')
        summary = html.escape(json.dumps(graph["metadata"], ensure_ascii=False, indent=2))
        legend = " ".join(f'<span class="legend-item"><span style="background:{c}"></span>{t}</span>' for t, c in colors.items())
        html_doc = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8" /><title>ARAG Trace - {html.escape(str(self.sample_id))}</title>
<style>
body{{margin:0;font-family:Arial,Helvetica,sans-serif;color:#0f172a;background:#f8fafc}}
header{{padding:18px 24px;background:#0f172a;color:white}} header h1{{margin:0 0 8px;font-size:20px}} header p{{margin:4px 0;color:#cbd5e1;font-size:13px}}
.summary{{padding:14px 24px;background:white;border-bottom:1px solid #e2e8f0}} .summary pre{{white-space:pre-wrap;margin:8px 0 0;font-size:12px}}
.legend{{padding:10px 24px;background:#f1f5f9;border-bottom:1px solid #e2e8f0;position:sticky;top:0;z-index:5}} .legend-item{{display:inline-flex;align-items:center;margin-right:16px;font-size:13px}} .legend-item span{{width:12px;height:12px;border-radius:3px;margin-right:5px}}
.canvas-wrap{{overflow:auto;padding:20px}} svg{{background:white;border:1px solid #e2e8f0;border-radius:8px}} .node{{cursor:pointer}} .node-title{{fill:white;font-size:10px;font-weight:700}} .node-preview{{color:white;font-size:10px;line-height:1.2;text-align:center;overflow:hidden}} .edge-label{{fill:#475569;font-size:10px}}
.details{{padding:18px 24px;background:white;border-top:1px solid #e2e8f0}} .detail-block{{display:none;white-space:pre-wrap;background:#0f172a;color:#e2e8f0;padding:14px;border-radius:8px;max-height:420px;overflow:auto}}
</style></head><body>
<header><h1>ARAG Execution Trace</h1><p>sample_id={html.escape(str(self.sample_id))} · dataset={html.escape(str(self.dataset))} · schema={self.trace_schema_version}</p></header>
<section class="summary"><strong>Run Summary</strong><pre>{summary}</pre></section>
<div class="legend">{legend}</div><div class="canvas-wrap"><svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#94a3b8" /></marker></defs>
{''.join(edge_lines)}{''.join(node_blocks)}
</svg></div><section class="details"><h2>Node Metadata</h2>{''.join(detail_blocks)}</section>
<script>function showDetail(id){{document.querySelectorAll('.detail-block').forEach(e=>e.style.display='none');document.getElementById(id).style.display='block';}}</script>
</body></html>'''
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_doc)


TraceLogger = TraceGraph
