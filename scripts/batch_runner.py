#!/usr/bin/env python3
"""
Batch Runner for ARAG - Supports concurrent execution and checkpoint resume.

Usage:
    python scripts/batch_runner.py \
        --config configs/example.yaml \
        --questions data/questions.json \
        --output results/
"""

import os
import json
import argparse
import logging
import re
from pathlib import Path
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

from tqdm import tqdm
from arag import LLMClient, BaseAgent, ToolRegistry, Config
from arag.tools.keyword_search import KeywordSearchTool
from arag.tools.semantic_search import SemanticSearchTool
from arag.tools.read_chunk import ReadChunkTool
from arag.utils.trace_graph import TraceGraph
from arag.core.schemas import EvidenceSpan
from arag.core.schemas import content_hash as make_content_hash, stable_hash
from arag.cognition import (
    FakeQuestionDecomposer,
    OnlineHypothesisTracker,
    assess_answer,
    build_candidate_constraint_matrix,
    build_shadow_repair_plan,
    extract_structured_propositions,
    ground_relations,
    infer_failure_types,
    parse_query_intent,
)
from arag.verification import (
    ClaimSupportScorer,
    FakeVerificationBackend,
    LLMVerificationBackend,
    SimpleClaimExtractor,
    SupportConfig,
    VerificationResult,
    failure_frontier,
)
from arag.repair import BlameEngine, BranchManager, rejected_hypothesis

logging.basicConfig(level=logging.ERROR)


def _span_identity(span: EvidenceSpan) -> tuple:
    return (
        str(span.doc_id),
        str(span.chunk_id),
        span.sentence_id,
        int(span.start_offset or 0),
        int(span.end_offset or 0),
        span.content_hash or make_content_hash(span.text or ""),
    )


def _sentence_spans_from_read_chunk(chunk: Dict[str, Any]) -> List[EvidenceSpan]:
    text = chunk.get("content", "") or ""
    md = chunk.get("metadata", {}) or {}
    if not text.strip():
        return []
    doc_id = str(md.get("doc_id", chunk.get("chunk_id")))
    chunk_id = str(chunk.get("chunk_id"))
    artifact = md.get("artifact_id", f"artifact_{stable_hash(doc_id, chunk_id)}")
    spans: List[EvidenceSpan] = []
    for idx, match in enumerate(re.finditer(r"[^.!?。！？]+[.!?。！？]?", text)):
        sent = match.group(0).strip()
        if not sent:
            continue
        start, end = match.start(), match.end()
        spans.append(EvidenceSpan(
            span_id=f"span_{stable_hash(artifact, idx, start, end, sent)}",
            artifact_id=artifact,
            doc_id=doc_id,
            chunk_id=chunk_id,
            sentence_id=idx,
            text=sent,
            start_offset=start,
            end_offset=end,
            content_hash=make_content_hash(sent),
        ))
    return spans


def _claim_local_dependencies(claim_text: str, question_plan) -> List[str]:
    lower = (claim_text or "").lower()
    deps: List[str] = []
    for rel in question_plan.relations:
        predicate_terms = [t for t in re.split(r"[_\s]+", str(rel.get("predicate", "")).lower()) if t]
        if predicate_terms and any(t in lower for t in predicate_terms):
            sg = rel.get("subgoal_id")
            if sg:
                deps.append(sg)
    return list(dict.fromkeys(deps))


def _align_claim_to_relations(claim_text: str, question_plan) -> List[str]:
    lower = (claim_text or "").lower()
    aligned = []
    for rel in question_plan.relations:
        predicate_terms = [t for t in re.split(r"[_\s]+", str(rel.get("predicate", "")).lower()) if t and t not in {"date", "answer"}]
        relation_match = any(t in lower for t in predicate_terms)
        if not relation_match and rel.get("expected_output_type") == "temporal" and re.search(r"\b\d{3,4}\b", lower):
            relation_match = True
        if relation_match:
            aligned.append(rel.get("relation_id"))
    return list(dict.fromkeys([a for a in aligned if a]))


def _relation_dependency_subgoals(question_plan, relation_ids: List[str]) -> List[str]:
    relation_by_id = question_plan.relation_by_id()
    deps = []
    for rel_id in relation_ids:
        rel = relation_by_id.get(rel_id, {})
        for dep_rel_id in rel.get("dependencies", []) or []:
            dep_sg = relation_by_id.get(dep_rel_id, {}).get("subgoal_id")
            if dep_sg:
                deps.append(dep_sg)
    return list(dict.fromkeys(deps))


def _subgoal_supports(subgoal_assessments: List[Dict[str, Any]], cfg: SupportConfig) -> Dict[str, float]:
    supports = {}
    for subgoal in subgoal_assessments or []:
        status = str(subgoal.get("status", "")).upper()
        if status in {"RESOLVED", "SATISFIED"}:
            support = 1.0
        elif status in {"PARTIALLY_RESOLVED", "PARTIALLY_GROUNDED"}:
            support = max(0.0, min(cfg.dependency_threshold - 0.01, 0.5))
        else:
            support = 0.0
        supports[subgoal.get("subgoal_id")] = support
    return {k: v for k, v in supports.items() if k}


def _verification_result_from_assessment(assessment: Dict[str, Any]) -> VerificationResult:
    data = ((assessment.get("best_evidence_set") or {}).get("verifier_result") or {})
    fields = VerificationResult.__dataclass_fields__
    return VerificationResult(**{key: data[key] for key in fields if key in data})


def _resolve_subgoals(question_plan, assessments: List[Dict[str, Any]], answer_assessment: Dict[str, Any], relation_groundings: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    grounded_by_subgoal: Dict[str, List[Dict[str, Any]]] = {}
    for grounding in relation_groundings or []:
        if grounding.get("subgoal_id"):
            grounded_by_subgoal.setdefault(grounding["subgoal_id"], []).append(grounding)
    verified_by_resolution: Dict[str, List[str]] = {}
    for assessment in assessments:
        if assessment.get("evidence_status") != "VERIFIED":
            continue
        claim = assessment.get("claim", {})
        if not assessment.get("evidence_set_span_ids"):
            continue
        for sg_id in claim.get("resolves_subgoal_ids", []) or []:
            verified_by_resolution.setdefault(sg_id, []).append(claim.get("claim_id"))

    status_by_id: Dict[str, str] = {}
    subgoal_dicts = []
    for sg in question_plan.subgoals:
        data = sg.__dict__.copy()
        deps = data.get("dependencies") or []
        deps_resolved = all(status_by_id.get(dep) == "RESOLVED" for dep in deps)
        resolver_ids = verified_by_resolution.get(data["subgoal_id"], [])
        groundings = grounded_by_subgoal.get(data["subgoal_id"], [])
        grounded = any(
            g.get("status") == "SATISFIED"
            and g.get("supporting_evidence_ids")
            and (g.get("confidence", 0) or 0) >= 0.7
            for g in groundings
        )
        partial = any(g.get("status") in {"PARTIALLY_GROUNDED", "SATISFIED"} for g in groundings)
        if deps and not deps_resolved:
            data["status"] = "BLOCKED"
        elif grounded:
            data["status"] = "RESOLVED"
            data["resolved_by_claim_ids"] = resolver_ids
            data["satisfied_constraints"] = list(dict.fromkeys((data.get("satisfied_constraints") or []) + data.get("required_constraints", [])))
        elif partial:
            data["status"] = "PARTIALLY_RESOLVED"
        elif data.get("expected_output_type") == "temporal" and answer_assessment.get("slot_coverage", 0) > 0 and not deps:
            data["status"] = "UNCERTAIN"
        elif data.get("required"):
            data["status"] = "FAILED"
        else:
            data["status"] = "OPEN"
        status_by_id[data["subgoal_id"]] = data["status"]
        subgoal_dicts.append(data)
    return subgoal_dicts


def _blame_hypotheses_from_commitments(
    hypotheses: List[Dict[str, Any]],
    commitment_events: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_id = {h.get("hypothesis_id"): dict(h) for h in hypotheses}
    for event in commitment_events:
        hyp = by_id.get(event.get("hypothesis_id"))
        if not hyp:
            continue
        hyp["commitment_event_id"] = event.get("commitment_event_id")
        hyp["missing_constraints"] = event.get("missing_identity_constraints", hyp.get("missing_constraints", []))
        hyp["contradicted_constraints"] = event.get("contradicted_identity_constraints", [])
        hyp["commitment_state"] = "COMMITTED"
        hyp["source_event_id"] = event.get("commitment_event_id")
        hyp["step_index"] = event.get("step_index")
    return list(by_id.values())


def _relation_for_subgoal(question_plan, subgoal_id: str) -> Dict[str, Any]:
    for rel in question_plan.relations:
        data = rel if isinstance(rel, dict) else rel.__dict__
        if data.get("subgoal_id") == subgoal_id:
            return data
    return {}


def _annotate_retrieval_coverage(
    trace_logger: TraceGraph,
    question_plan,
    subgoal_assessments: List[Dict[str, Any]],
    relation_groundings: List[Dict[str, Any]],
    termination_reason: str = "",
) -> List[Dict[str, Any]]:
    if not trace_logger:
        return []
    plan_queries = [n for n in trace_logger.nodes if n.get("type") == "plan_query"]
    decisions = [n for n in trace_logger.nodes if n.get("type") == "decision_record"]
    read_nodes = [n for n in trace_logger.nodes if n.get("type") == "read_call"]
    delivered_spans = [
        n.get("id") for n in trace_logger.nodes
        if n.get("type") == "evidence_span"
    ]
    grounding_by_subgoal: Dict[str, List[Dict[str, Any]]] = {}
    for grounding in relation_groundings or []:
        grounding_by_subgoal.setdefault(grounding.get("subgoal_id"), []).append(grounding)
    assessments = []
    for sg in subgoal_assessments or []:
        if not sg.get("required") or str(sg.get("status", "")).upper() == "RESOLVED":
            continue
        sg_id = sg.get("subgoal_id")
        rel = _relation_for_subgoal(question_plan, sg_id)
        relation_id = rel.get("relation_id")
        predicate = rel.get("predicate") or sg.get("expected_output_type") or "required_relation"
        related_query_ids = []
        related_decision_ids = []
        predicate_terms = [t for t in re.split(r"[_\\s]+", str(predicate).lower()) if t and t not in {"date", "answer"}]
        for node in plan_queries:
            content = str(node.get("content", "")).lower()
            md = node.get("metadata", {}) or {}
            args = json.dumps(md.get("arguments", {}), ensure_ascii=False).lower()
            hit = any(t in content or t in args for t in predicate_terms) if predicate_terms else False
            if not hit and relation_id:
                hit = relation_id in json.dumps(md, ensure_ascii=False)
            if hit or not predicate_terms:
                related_query_ids.append(node.get("id"))
                dec = next((e.get("source") for e in trace_logger.edges if e.get("target") == node.get("id") and e.get("type") == "motivates"), None)
                if dec:
                    related_decision_ids.append(dec)
        strategies = list(dict.fromkeys(
            (n.get("metadata", {}) or {}).get("tool_name")
            for n in plan_queries
            if n.get("id") in related_query_ids and (n.get("metadata", {}) or {}).get("tool_name")
        ))
        queries = [str(n.get("content", "")).strip().lower() for n in plan_queries if n.get("id") in related_query_ids]
        query_diversity = len(set(queries)) / max(len(queries), 1)
        satisfied = any(g.get("status") == "SATISFIED" and g.get("supporting_evidence_ids") for g in grounding_by_subgoal.get(sg_id, []))
        if satisfied:
            state = "SATISFIED"
            reason = "relation_specific_evidence_grounded"
        elif not related_query_ids:
            state = "UNTESTED"
            reason = "required_relation_was_never_targeted"
        elif len(related_query_ids) >= 2 and query_diversity < 0.5:
            state = "STALLED"
            reason = "low_query_diversity_without_grounding"
        else:
            state = "EXHAUSTED" if termination_reason else "ACTIVE"
            reason = "terminated_before_required_relation_grounded" if termination_reason else "relation_not_yet_grounded"
        episode_id = trace_logger.add_retrieval_episode({
            "subgoal_id": sg_id,
            "relation_id": relation_id,
            "required_relation": predicate,
            "query_decision_ids": list(dict.fromkeys(related_decision_ids)),
            "plan_query_ids": list(dict.fromkeys(related_query_ids)),
            "strategies": strategies,
            "query_diversity": query_diversity,
            "returned_evidence_count": len(delivered_spans),
            "read_call_count": len(read_nodes),
            "coverage_state": state,
            "unresolved_reason": reason,
            "termination_reason": termination_reason,
        })
        cov_id = trace_logger.add_coverage_assessment(episode_id, {
            "subgoal_id": sg_id,
            "relation_id": relation_id,
            "required_relation": predicate,
            "coverage_state": state,
            "unresolved_reason": reason,
            "query_decision_ids": list(dict.fromkeys(related_decision_ids)),
            "plan_query_ids": list(dict.fromkeys(related_query_ids)),
            "strategies": strategies,
            "query_diversity": query_diversity,
            "relation_specific_evidence_found": satisfied,
            "termination_reason": termination_reason,
            "repair_action": "retrieve_required_relation_with_new_strategy",
        })
        assessments.append(trace_logger._node(cov_id)["metadata"])
    return assessments


def _load_api_txt(path: str = "api.txt") -> Dict[str, str]:
    """Load local API settings without exposing secrets in outputs."""
    file_path = Path(path)
    if not file_path.exists():
        return {}
    settings: Dict[str, str] = {}
    aliases = {
        "api_key": "api_key",
        "arag_api_key": "api_key",
        "openai_api_key": "api_key",
        "key": "api_key",
        "base_url": "base_url",
        "arag_base_url": "base_url",
        "openai_base_url": "base_url",
        "model": "model",
        "arag_model": "model",
    }
    fallback_token = None
    for raw in file_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            if key.startswith("export "):
                key = key[len("export "):].strip()
            normalized = aliases.get(key.lower())
            if normalized and value.strip():
                settings[normalized] = value.strip().strip('"').strip("'")
        elif fallback_token is None:
            fallback_token = line
    if fallback_token and "api_key" not in settings:
        settings["api_key"] = fallback_token
    return settings


class BatchRunner:
    """Batch runner with concurrent execution and checkpoint resume."""

    def __init__(
        self,
        config: Config,
        questions_file: str,
        output_dir: str,
        limit: int = None,
        num_workers: int = 10,
        verbose: bool = False
    ):
        self.config = config
        self.questions_file = Path(questions_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.limit = limit
        self.num_workers = num_workers
        self.verbose = verbose

        self.predictions_file = self.output_dir / "predictions.jsonl"
        self.dataset_name = self.questions_file.stem
        self.trace_dir = self.output_dir / "traces" / self.dataset_name
        self.trace_html_dir = self.output_dir / "trace_html" / self.dataset_name
        self.write_lock = Lock()

        self.questions = self._load_questions()
        self._shared_tools = self._init_shared_tools()

        prompt_file = Path(__file__).parent.parent / "src/arag/agent/prompts/default.txt"
        if prompt_file.exists():
            self._system_prompt = prompt_file.read_text()
        else:
            self._system_prompt = "You are a helpful assistant."

    def _init_shared_tools(self) -> ToolRegistry:
        """Initialize shared tools (embedding model loaded only once)."""
        data_config = self.config.get('data', {})
        chunks_file = data_config.get('chunks_file', 'data/chunks.json')
        index_dir = data_config.get('index_dir', 'data/index')

        tools = ToolRegistry()
        tools.register(KeywordSearchTool(chunks_file=chunks_file))
        tools.register(ReadChunkTool(chunks_file=chunks_file))

        index_file = Path(index_dir) / "sentence_index.pkl"
        if index_file.exists():
            embedding_config = self.config.get('embedding', {})
            print(f"Loading embedding model: {embedding_config.get('model', 'sentence-transformers/all-MiniLM-L6-v2')}")
            tools.register(SemanticSearchTool(
                chunks_file=chunks_file,
                index_dir=index_dir,
                model_name=embedding_config.get('model', 'sentence-transformers/all-MiniLM-L6-v2'),
                device=embedding_config.get('device')
            ))
            print("Embedding model loaded successfully!")
        else:
            print(f"Warning: Index not found at {index_file}, semantic search disabled")

        return tools

    def _load_questions(self) -> List[Dict[str, Any]]:
        """Load questions from file."""
        with open(self.questions_file, 'r', encoding='utf-8') as f:
            if self.questions_file.suffix == ".jsonl":
                questions = [json.loads(line) for line in f if line.strip()]
            else:
                questions = json.load(f)

        if self.limit:
            questions = questions[:self.limit]

        return questions

    def _load_completed_qids(self) -> set:
        """Load completed question IDs for checkpoint resume."""
        completed_ids = set()

        if not self.predictions_file.exists():
            return completed_ids

        try:
            with open(self.predictions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if 'question' in data and 'pred_answer' in data:
                            completed_id = data.get('qid') or data.get('id') or data.get('sample_id')
                            if completed_id is not None:
                                completed_ids.add(str(completed_id))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Warning: Error loading completed data: {e}")

        return completed_ids

    @staticmethod
    def _sample_key(item: Dict[str, Any], sample_index: int) -> str:
        explicit = item.get('qid') or item.get('id')
        return str(explicit) if explicit is not None else f"sample_{sample_index:06d}"

    def _append_prediction(self, prediction: Dict[str, Any]):
        """Append prediction to file (thread-safe)."""
        with self.write_lock:
            with open(self.predictions_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(prediction, ensure_ascii=False) + '\n')

    def _create_agent(self) -> BaseAgent:
        """Create agent instance with shared tools."""
        llm_config = self.config.get('llm', {})
        api_txt = _load_api_txt()
        api_key = llm_config.get('api_key') or os.getenv('ARAG_API_KEY')
        if not api_key:
            api_key = api_txt.get("api_key")

        client = LLMClient(
            model=llm_config.get('model') or os.getenv('ARAG_MODEL') or api_txt.get("model") or 'gpt-4o-mini',
            api_key=api_key,
            base_url=llm_config.get('base_url') or os.getenv('ARAG_BASE_URL') or api_txt.get("base_url") or 'https://api.openai.com/v1',
            reasoning_effort=llm_config.get('reasoning_effort')
        )

        agent_config = self.config.get('agent', {})

        return BaseAgent(
            llm_client=client,
            tools=self._shared_tools,
            system_prompt=self._system_prompt,
            max_loops=agent_config.get('max_loops', 10),
            max_token_budget=agent_config.get('max_token_budget', 128000),
            verbose=self.verbose
        )

    def _verification_backend(self):
        verification = self.config.get("verification", {})
        backend = verification.get("backend") or verification.get("mode", "fake")
        if backend in {"llm", "real_uncalibrated", "real_calibrated"}:
            api_txt = _load_api_txt()
            api_key = verification.get("api_key") or os.getenv("ARAG_VERIFIER_API_KEY") or os.getenv("ARAG_API_KEY")
            if not api_key:
                api_key = api_txt.get("api_key")
            if not api_key:
                raise ValueError("verification.backend=llm requires verifier api_key, ARAG_VERIFIER_API_KEY, ARAG_API_KEY, or api.txt")
            return LLMVerificationBackend(
                model=verification.get("model") or self.config.get("llm.model") or api_txt.get("model") or "gpt-4o-mini",
                api_key=api_key,
                base_url=verification.get("base_url") or self.config.get("llm.base_url") or os.getenv("ARAG_BASE_URL") or api_txt.get("base_url") or "https://api.openai.com/v1",
                timeout=verification.get("timeout", 60),
                max_retries=verification.get("max_retries", 2),
                prompt_token_budget=verification.get("prompt_token_budget", 4096),
            )
        return FakeVerificationBackend(authoritative_for_test=verification.get("authoritative_for_test", False))

    def _assess_claims(self, result: Dict[str, Any], trace_logger: TraceGraph) -> Dict[str, Any]:
        verification = self.config.get("verification", {})
        if not verification.get("enabled", False):
            return {
                "trace_schema_version": trace_logger.trace_schema_version,
                "branch_id": "b0",
                "claim_assessments": [],
                "final_claim_support": None,
                "root_bad_claims": [],
                "blame_results": [],
                "repair_history": [],
                "selected_branch": "b0",
                "total_repair_cost": 0,
                "termination_reason_v2": "verification_disabled",
            }
        weights = verification.get("weights", {})
        cfg = SupportConfig(
            beta=weights.get("beta", 1.5),
            gamma=weights.get("gamma", 1.0),
            delta=weights.get("delta", 1.0),
            rho=weights.get("rho", 0.3),
            kappa=weights.get("kappa", 0.2),
            mu=weights.get("mu", 0.5),
            verified_threshold=verification.get("verified_threshold", 0.80),
            low_support_threshold=verification.get("low_support_threshold", 0.45),
            contradiction_threshold=verification.get("contradiction_threshold", 0.70),
            uncertainty_threshold=verification.get("uncertainty_threshold", 0.65),
            relevance_threshold=verification.get("relevance_threshold", 0.50),
            verifier_prompt_token_budget=verification.get("prompt_token_budget", 4096),
        )
        question_node = trace_logger.latest_node_id("question")
        question_plan = FakeQuestionDecomposer().decompose(
            trace_logger.metadata.get("question") or result.get("question", ""),
            trace_logger.sample_id,
        )
        if not question_plan.original_question:
            # Batch results do not carry question inside `result`; recover from trace.
            questions = [n for n in trace_logger.nodes if n["type"] == "question"]
            question_plan = FakeQuestionDecomposer().decompose(
                questions[0]["content"] if questions else "",
                trace_logger.sample_id,
            )
        for subgoal in question_plan.subgoals:
            trace_logger.add_subgoal_node(subgoal.__dict__, question_node)

        spans = []
        span_by_identity = {}
        delivered_span_ids = []
        for delivery in result.get("context_deliveries", []):
            delivered_span_ids.extend(delivery.get("span_ids", []))
        for event in result.get("search_history", []):
            for item in event.get("results", []):
                for span in item.get("matched_spans", []) or []:
                    ev = EvidenceSpan(**span)
                    key = _span_identity(ev)
                    if key not in span_by_identity:
                        span_by_identity[key] = ev
                        spans.append(ev)
                        trace_logger.add_evidence_span(span)
        for chunk in result.get("read_chunks", {}).values():
            md = chunk.get("metadata", {})
            read_call_id = md.get("call_id")
            for ev in _sentence_spans_from_read_chunk(chunk):
                key = _span_identity(ev)
                if key not in span_by_identity:
                    span_by_identity[key] = ev
                    spans.append(ev)
                    trace_logger.add_evidence_span({
                        **ev.__dict__,
                        "evidence_artifact_identity": list(key),
                        "source": "read_chunk_sentence",
                        "read_call_id": read_call_id,
                    })
        existing_span_ids = {
            n.get("id") for n in trace_logger.nodes
            if n.get("type") == "evidence_span"
        }
        for sid in delivered_span_ids:
            if sid and sid not in existing_span_ids:
                trace_logger.add_evidence_span({
                    "span_id": sid,
                    "text": "",
                    "source": "context_delivery_placeholder",
                    "provenance_note": "span id was delivered in context but no sentence-level artifact was reconstructed",
                })
                existing_span_ids.add(sid)
        extractor = SimpleClaimExtractor()
        answer_node = trace_logger.latest_node_id("answer")
        claims = extractor.extract(result.get("answer", ""), generated_by=answer_node, branch_id="b0")
        relation_by_id = question_plan.relation_by_id()
        subgoal_by_relation = {rel.get("relation_id"): rel.get("subgoal_id") for rel in question_plan.relations}
        answer_subgoal_ids = [
            rel.get("subgoal_id") for rel in question_plan.relations
            if rel.get("answer_constraint") and rel.get("subgoal_id")
        ]
        identity_subgoal_ids = [
            rel.get("subgoal_id") for rel in question_plan.relations
            if rel.get("identity_constraint") and rel.get("subgoal_id")
        ]
        for claim in claims:
            aligned_relation_ids = _align_claim_to_relations(claim.content, question_plan)
            resolves_subgoal_ids = [
                subgoal_by_relation.get(rel_id)
                for rel_id in aligned_relation_ids
                if subgoal_by_relation.get(rel_id)
            ]
            claim.aligned_relation_ids = aligned_relation_ids
            claim.resolves_subgoal_ids = list(dict.fromkeys(resolves_subgoal_ids))
            if claim.claim_type == "answer_claim":
                claim.dependencies = _relation_dependency_subgoals(question_plan, aligned_relation_ids) or list(dict.fromkeys(identity_subgoal_ids))
                if not claim.dependencies and question_plan.subgoals:
                    claim.dependencies = [question_plan.subgoals[0].subgoal_id]
            elif claim.criticality > 0:
                claim.dependencies = _relation_dependency_subgoals(question_plan, aligned_relation_ids)
                if not claim.dependencies and not aligned_relation_ids:
                    claim.dependencies = [question_plan.subgoals[0].subgoal_id] if question_plan.subgoals else []
        visible_ids = list(dict.fromkeys(delivered_span_ids))
        tracker = OnlineHypothesisTracker(question_plan)
        query_intents = []
        llm_by_loop = {
            n.get("metadata", {}).get("loop"): n.get("id")
            for n in trace_logger.nodes
            if n.get("type") == "llm_call"
        }
        for node in trace_logger.nodes:
            if node.get("type") == "plan_query":
                intent = parse_query_intent(
                    question_plan,
                    str(node.get("content", "")),
                    tool_name=node.get("metadata", {}).get("tool_name"),
                    arguments=node.get("metadata", {}).get("arguments"),
                    source_plan_query_id=node.get("id"),
                    generated_by_llm_call_id=llm_by_loop.get(node.get("metadata", {}).get("loop")),
                    branch_id="b0",
                    step_index=node.get("step_index", 0),
                )
                query_intents.append(intent)
                trace_logger.add_query_intent_node(intent.__dict__)
                tracker.observe_query(intent, visible_evidence_ids=[])
        for msg in result.get("message_trace", []):
            if msg.get("role") == "assistant":
                tracker.observe_text(
                    msg.get("content", ""),
                    source_event_id=llm_by_loop.get(msg.get("loop")) or answer_node,
                    step_index=msg.get("loop", 0),
                    visible_evidence_ids=visible_ids,
                    posthoc_summary=False,
                )
        tracker.observe_text(
            result.get("answer", ""),
            source_event_id=answer_node,
            step_index=max([n.get("step_index") or 0 for n in trace_logger.nodes if n.get("type") == "answer"] or [0]),
            visible_evidence_ids=visible_ids,
            posthoc_summary=True,
        )
        hypothesis_assessments = []
        for hyp in tracker.hypothesis_dicts():
            trace_logger.add_hypothesis_node({
                **hyp,
                "content": f"The target entity may be {hyp.get('canonical_entity')}",
                "candidate_entity": hyp.get("canonical_entity"),
                "missing_constraints": [
                    cid for cid, cr in (hyp.get("constraint_results") or {}).items()
                    if cr.get("status") != "SATISFIED"
                ],
                "satisfied_constraints": [
                    cid for cid, cr in (hyp.get("constraint_results") or {}).items()
                    if cr.get("status") == "SATISFIED"
                ],
                "commitment_strength": 0.8 if hyp.get("commitment_state") == "COMMITTED" else 0.2,
            })
            hypothesis_assessments.append({
                **hyp,
                "hypothesis_id": hyp.get("hypothesis_id"),
                "candidate_entity": hyp.get("canonical_entity"),
                "missing_constraints": [
                    cid for cid, cr in (hyp.get("constraint_results") or {}).items()
                    if cr.get("status") != "SATISFIED"
                ],
                "satisfied_constraints": [
                    cid for cid, cr in (hyp.get("constraint_results") or {}).items()
                    if cr.get("status") == "SATISFIED"
                ],
            })
        commitment_events = tracker.commitment_dicts()
        for event in commitment_events:
            trace_logger.add_commitment_event_node(event)
        candidate_matrix = build_candidate_constraint_matrix(question_plan, hypothesis_assessments)

        scorer = ClaimSupportScorer(self._verification_backend(), cfg)
        preliminary_assessments = []
        for claim in claims:
            preliminary_assessments.append(scorer.score(
                claim,
                spans,
                delivered_span_ids,
                parent_supports=None,
                dependency_ids=[],
                blocked_dependency_ids=[],
                plan_semantic_valid=question_plan.semantic_valid,
            ))
        preliminary_props = extract_structured_propositions(question_plan, spans, preliminary_assessments, result.get("answer", ""))
        preliminary_groundings = ground_relations(question_plan, preliminary_props)
        preliminary_subgoals = _resolve_subgoals(question_plan, preliminary_assessments, {"slot_coverage": 0}, preliminary_groundings)
        dep_supports = _subgoal_supports(preliminary_subgoals, cfg)
        preliminary_by_claim_id = {
            a.get("claim", {}).get("claim_id"): a for a in preliminary_assessments
        }

        assessments = []
        for claim in claims:
            parent_supports = [dep_supports.get(dep, 0.0) for dep in claim.dependencies] if claim.dependencies else None
            blocked_dep_ids = [dep for dep in (claim.dependencies or []) if dep_supports.get(dep, 0.0) < cfg.dependency_threshold]
            preliminary = preliminary_by_claim_id.get(claim.claim_id, {})
            assessments.append(scorer.score(
                claim,
                spans,
                delivered_span_ids,
                parent_supports=parent_supports,
                dependency_ids=claim.dependencies,
                blocked_dependency_ids=blocked_dep_ids,
                plan_semantic_valid=question_plan.semantic_valid,
                verification_result=_verification_result_from_assessment(preliminary) if preliminary else None,
            ))
        for assessment in assessments:
            claim_node_id = trace_logger.add_claim_assessment(assessment, generated_by=answer_node)
            assessment["claim_node_id"] = claim_node_id
            assessment["evidence_set_node_id"] = assessment.get("best_evidence_set", {}).get("evidence_set_id")
        structured_propositions = extract_structured_propositions(question_plan, spans, assessments, result.get("answer", ""))
        relation_groundings = ground_relations(question_plan, structured_propositions)
        subgoal_assessments = _resolve_subgoals(question_plan, assessments, {"slot_coverage": 0}, relation_groundings)
        answer_assessment = assess_answer(question_plan, result.get("answer", ""), assessments, relation_groundings, subgoal_assessments).to_dict()
        subgoal_assessments = _resolve_subgoals(question_plan, assessments, answer_assessment, relation_groundings)
        unresolved_required = [sg["subgoal_id"] for sg in subgoal_assessments if sg.get("required") and str(sg.get("status", "")).upper() != "RESOLVED"]
        failure_types = infer_failure_types(answer_assessment, hypothesis_assessments, subgoal_assessments)
        for warning in question_plan.validation_warnings:
            if warning in {"PLAN_UNDERDECOMPOSED", "PLAN_AMBIGUOUS", "PLAN_UNMAPPED_CONSTRAINT"}:
                failure_types.append(warning)
        failure_types = list(dict.fromkeys(failure_types))
        root_bad = failure_frontier(assessments)
        consistency_errors = []
        if root_bad and not failure_types:
            consistency_errors.append("ROOT_BAD_CLAIMS_WITHOUT_FAILURE_TYPES")
        satisfied_grounding_subgoals = {
            g.get("subgoal_id") for g in relation_groundings
            if g.get("status") == "SATISFIED" and g.get("supporting_evidence_ids")
        }
        for sg in subgoal_assessments:
            if str(sg.get("status", "")).upper() == "RESOLVED" and sg.get("subgoal_id") not in satisfied_grounding_subgoals:
                consistency_errors.append(f"SUBGOAL_RESOLVED_WITHOUT_VERIFIED_GROUNDING:{sg.get('subgoal_id')}")
        unsupported_claim_ids = {
            a.get("claim", {}).get("claim_id") for a in assessments
            if a.get("evidence_status") in {"UNSUPPORTED", "CONTRADICTED", "INVALID_PROVENANCE"}
        }
        if answer_assessment.get("support_status") == "VERIFIED" and answer_assessment.get("critical_answer_claim_id") in unsupported_claim_ids:
            consistency_errors.append("ANSWER_VERIFIED_FROM_UNSUPPORTED_CLAIM")
        if consistency_errors:
            failure_types = list(dict.fromkeys(failure_types + ["TRACE_CONSISTENCY_ERROR"]))
        final_support = min((a["raw_score"] for a in assessments if a["claim"].get("criticality", 1.0) > 0), default=None)
        coverage_assessments = _annotate_retrieval_coverage(
            trace_logger,
            question_plan,
            subgoal_assessments,
            relation_groundings,
            result.get("termination_reason", ""),
        )
        trace = trace_logger.to_dict()
        blame = []
        root_bad_hypotheses = list(dict.fromkeys(
            e["hypothesis_id"] for e in commitment_events
            if e.get("is_premature") and "PREMATURE_ENTITY_COMMITMENT" in failure_types
        ))[:3]
        if failure_types:
            blame = BlameEngine().score_cognitive(
                failure_types,
                [sg for sg in subgoal_assessments if sg["subgoal_id"] in unresolved_required],
                _blame_hypotheses_from_commitments(hypothesis_assessments, commitment_events),
                trace,
            )
        elif root_bad:
            by_id = {a["claim"]["claim_id"]: a for a in assessments}
            blame = BlameEngine().score(by_id[root_bad[0]], trace)
        repair_history = []
        selected_branch = "b0"
        total_repair_cost = 0
        repair_enabled = bool(self.config.get("repair.enabled", False))
        repair_dry_run = bool(self.config.get("repair.dry_run", True))
        allow_non_authoritative = bool(self.config.get("repair.allow_non_authoritative_repair", False))
        authoritative = all(a.get("authoritative") for a in assessments) if assessments else False
        can_repair = (authoritative or allow_non_authoritative) and bool(failure_types or root_bad)
        repair_plan = None
        if repair_enabled and repair_dry_run and blame:
            repair_plan = build_shadow_repair_plan(
                failure_types,
                root_bad_hypotheses,
                unresolved_required,
                blame,
                question_plan.to_dict(),
                candidate_matrix=candidate_matrix,
                commitment_events=commitment_events,
                claim_assessments=assessments,
            )
            manager = BranchManager()
            repair_history = manager.to_dict()["branches"]
        elif repair_enabled and blame and can_repair:
            manager = BranchManager()
            best = blame[0]
            branch = manager.fork(
                "b0",
                best["node_id"],
                best["node_id"],
                "estimated_root_bad_claim_repair",
                inherited_claim_ids=[a["claim"]["claim_id"] for a in assessments if a["status"] == "VERIFIED"],
                inherited_evidence_ids=list(dict.fromkeys(delivered_span_ids)),
                constraints=[rejected_hypothesis(root_bad[0], "low_support", best["node_id"], "root bad claim under current evidence")],
                inherited_nodes=[a.get("claim_node_id") for a in assessments if a["status"] == "VERIFIED" and a.get("claim_node_id")],
                invalidated_node_ids=best.get("affected_downstream_nodes", []),
                active_unresolved_subgoals=unresolved_required,
                repair_instruction="rollback before candidate root and re-solve only affected unresolved subgoals",
                branch_budget=int(best.get("repair_cost", 1) or 1),
            )
            # This conservative controller only selects a repair branch after an
            # executor has completed it. Until then b0 remains selected.
            repair_history = manager.to_dict()["branches"]
            total_repair_cost = best["repair_cost"]
        elif root_bad:
            manager = BranchManager()
            repair_history = manager.to_dict()["branches"]
        else:
            manager = BranchManager()
            manager.select("b0")
            repair_history = manager.to_dict()["branches"]
        if not assessments or not all(a.get("authoritative") for a in assessments):
            termination_v2 = "verification_not_authoritative" if claims else "no_answer_no_claims"
        elif answer_assessment.get("completeness_status") == "INCOMPLETE":
            termination_v2 = "answer_incomplete"
        elif unresolved_required:
            termination_v2 = "unresolved_required_subgoal"
        elif not claims and not str(result.get("answer", "")).strip():
            termination_v2 = "no_answer_no_claims"
        elif not root_bad:
            termination_v2 = "all_critical_claims_verified"
        else:
            termination_v2 = "root_bad_claims_found"
        repair_eligible_final = bool(
            can_repair
            and repair_plan
            and repair_plan.get("diagnostic_status") == "REPAIRABLE_ROOT_FOUND"
            and repair_plan.get("success_criteria")
        )
        return {
            "trace_schema_version": trace_logger.trace_schema_version,
            "branch_id": selected_branch,
            "claim_assessments": assessments,
            "question_plan": question_plan.to_dict(),
            "answer_assessment": answer_assessment,
            "subgoal_assessments": subgoal_assessments,
            "hypothesis_assessments": hypothesis_assessments,
            "query_intents": [qi.__dict__ for qi in query_intents],
            "commitment_events": commitment_events,
            "candidate_constraint_matrix": candidate_matrix,
            "structured_propositions": structured_propositions,
            "relation_groundings": relation_groundings,
            "coverage_assessments": coverage_assessments,
            "failure_frontier": root_bad,
            "final_claim_support": final_support,
            "root_bad_claims": root_bad,
            "root_bad_hypotheses": root_bad_hypotheses,
            "unresolved_required_subgoals": unresolved_required,
            "failure_types": failure_types,
            "plan_validation_warnings": question_plan.validation_warnings,
            "consistency_errors": consistency_errors,
            "blame_results": blame,
            "repair_plan": repair_plan,
            "repair_history": repair_history,
            "selected_branch": selected_branch,
            "total_repair_cost": total_repair_cost,
            "verifier_mode": (assessments[0].get("verifier_mode") if assessments else verification.get("mode", "disabled")),
            "verifier_authoritative": authoritative,
            "repair_eligible": repair_eligible_final,
            "termination_reason_v2": termination_v2,
        }

    @staticmethod
    def _safe_path_part(value: Any) -> str:
        text = str(value) if value is not None else "unknown"
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)
        return safe[:120] or "unknown"

    def _process_one(
        self,
        item: Dict[str, Any],
        agent: BaseAgent,
        sample_index: int = 0,
    ) -> Dict[str, Any]:
        """Process one question."""
        qid = item.get('qid') or item.get('id')
        sample_id = self._sample_key(item, sample_index)
        safe_sample_id = self._safe_path_part(sample_id)

        question = item.get('question', '')
        gold_answer = item.get('answer', item.get('gold_answer', ''))

        trace_logger = TraceGraph(
            sample_id=str(sample_id),
            dataset=self.dataset_name,
            metadata={
                "qid": qid,
                "sample_index": sample_index,
                "questions_file": str(self.questions_file),
            },
        )

        question_node = trace_logger.add_question(question, {
            "qid": qid,
            "sample_index": sample_index,
            "dataset": self.dataset_name,
        })

        trace_path = self.trace_dir / f"{safe_sample_id}.json"
        trace_html_path = self.trace_html_dir / f"{safe_sample_id}.html"

        try:
            result = agent.run(question, trace_logger=trace_logger)
            v2_result = self._assess_claims(result, trace_logger)
            trace_logger.metadata.update({
                "final_answer": result.get("answer", ""),
                "pred_answer": result.get("answer", ""),
                "termination_reason": result.get("termination_reason", ""),
                "total_cost": result.get("total_cost", 0),
                "loops": result.get("loops", 0),
                "total_retrieved_tokens": result.get("total_retrieved_tokens", 0),
                "raw_error": result.get("raw_error"),
                **v2_result,
            })
            trace_logger.save_json(trace_path)
            trace_logger.save_html(trace_html_path)

            return {
                'qid': qid,
                'sample_id': str(sample_id),
                'question': question,
                'trajectory': result['trajectory'],
                'gold_answer': gold_answer,
                'pred_answer': result['answer'],
                'total_cost': result['total_cost'],
                'loops': result['loops'],
                'total_retrieved_tokens': result.get('total_retrieved_tokens', 0),
                'retrieval_logs': result.get('retrieval_logs', []),
                'chunks_read_count': result.get('chunks_read_count', 0),
                'chunks_read_ids': result.get('chunks_read_ids', []),
                'read_chunks': result.get('read_chunks', {}),
                'search_history': result.get('search_history', []),
                'message_trace': result.get('message_trace', []),
                'final_messages': result.get('final_messages', []),
                'termination_reason': result.get('termination_reason', ''),
                'trace_path': str(trace_path),
                'trace_html_path': str(trace_html_path),
                **v2_result,
            }
        except Exception as e:
            error_answer = f"Error: {str(e)}"
            parent = trace_logger.latest_node_id("llm_call") or question_node
            trace_logger.add_error(parent, e, "batch_runner", 0, "error")
            if trace_logger.latest_node_id("answer") is None:
                trace_logger.add_answer(None, "", 0, "error", failed=True, raw_error=str(e))
            trace_logger.metadata.update({
                "final_answer": "",
                "pred_answer": error_answer,
                "termination_reason": "error",
                "error": str(e),
            })
            trace_logger.save_json(trace_path)
            trace_logger.save_html(trace_html_path)

            return {
                'qid': qid,
                'sample_id': str(sample_id),
                'question': question,
                'trajectory': [],
                'gold_answer': gold_answer,
                'pred_answer': error_answer,
                'total_cost': 0,
                'loops': 0,
                'total_retrieved_tokens': 0,
                'retrieval_logs': [],
                'chunks_read_count': 0,
                'chunks_read_ids': [],
                'read_chunks': {},
                'search_history': [],
                'message_trace': [],
                'final_messages': [],
                'termination_reason': 'error',
                'error': str(e),
                'trace_path': str(trace_path),
                'trace_html_path': str(trace_html_path),
            }

    def run(self):
        """Run batch processing."""
        completed_ids = self._load_completed_qids()

        pending = [
            (idx, q) for idx, q in enumerate(self.questions)
            if self._sample_key(q, idx) not in completed_ids
        ]

        print(f"Total questions: {len(self.questions)}")
        print(f"Completed: {len(completed_ids)}")
        print(f"Pending: {len(pending)}")

        if not pending:
            print("All questions completed!")
            return

        print(f"Starting with {self.num_workers} workers...")

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {}

            for sample_index, item in pending:
                agent = self._create_agent()
                future = executor.submit(self._process_one, item, agent, sample_index)
                futures[future] = self._sample_key(item, sample_index)

            with tqdm(total=len(pending), desc="Processing") as pbar:
                for future in as_completed(futures):
                    qid = futures[future]
                    try:
                        result = future.result()
                        self._append_prediction(result)
                    except Exception as e:
                        print(f"Error processing {qid}: {e}")
                    pbar.update(1)

        print(f"\nResults saved to: {self.predictions_file}")
        print(f"Trace JSON saved under: {self.trace_dir}")
        print(f"Trace HTML saved under: {self.trace_html_dir}")


def main():
    parser = argparse.ArgumentParser(description="ARAG Batch Runner")
    parser.add_argument("--config", "-c", required=True, help="Config file path")
    parser.add_argument("--questions", "-q", required=True, help="Questions file path")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--limit", "-l", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--workers", "-w", type=int, default=10, help="Number of workers")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--enable-claim-verification", action="store_true", help="Enable v2 claim extraction and verification")
    parser.add_argument("--enable-repair", action="store_true", help="Enable estimated blame and append-only repair branch planning")
    parser.add_argument("--max-repair-branches", type=int, default=None, help="Maximum repair branches")
    parser.add_argument("--max-repair-cost", type=float, default=None, help="Maximum repair cost")
    parser.add_argument("--trace-schema-version", default=None, help="Trace schema version to write")

    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if args.enable_claim_verification:
        config.set("verification.enabled", True)
    if args.enable_repair:
        config.set("repair.enabled", True)
    if args.max_repair_branches is not None:
        config.set("repair.max_branches", args.max_repair_branches)
    if args.max_repair_cost is not None:
        config.set("repair.max_cost", args.max_repair_cost)
    if args.trace_schema_version:
        TraceGraph.trace_schema_version = args.trace_schema_version

    runner = BatchRunner(
        config=config,
        questions_file=args.questions,
        output_dir=args.output,
        limit=args.limit,
        num_workers=args.workers,
        verbose=args.verbose
    )

    runner.run()


if __name__ == "__main__":
    main()
