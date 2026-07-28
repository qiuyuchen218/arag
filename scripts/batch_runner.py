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
    infer_failure_types,
    parse_query_intent,
)
from arag.verification import (
    ClaimSupportScorer,
    FakeVerificationBackend,
    LLMVerificationBackend,
    SimpleClaimExtractor,
    SupportConfig,
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


def _resolve_subgoals(question_plan, assessments: List[Dict[str, Any]], answer_assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
    verified_by_dep: Dict[str, List[str]] = {}
    for assessment in assessments:
        if assessment.get("status") != "VERIFIED":
            continue
        claim = assessment.get("claim", {})
        if not assessment.get("evidence_set_span_ids"):
            continue
        for dep in claim.get("dependencies", []) or []:
            verified_by_dep.setdefault(dep, []).append(claim.get("claim_id"))

    status_by_id: Dict[str, str] = {}
    subgoal_dicts = []
    for sg in question_plan.subgoals:
        data = sg.__dict__.copy()
        deps = data.get("dependencies") or []
        deps_resolved = all(status_by_id.get(dep) == "RESOLVED" for dep in deps)
        resolver_ids = verified_by_dep.get(data["subgoal_id"], [])
        if deps and not deps_resolved:
            data["status"] = "BLOCKED"
        elif resolver_ids:
            data["status"] = "RESOLVED"
            data["resolved_by_claim_ids"] = resolver_ids
            data["satisfied_constraints"] = list(dict.fromkeys((data.get("satisfied_constraints") or []) + data.get("required_constraints", [])))
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
        completed_qids = set()

        if not self.predictions_file.exists():
            return completed_qids

        try:
            with open(self.predictions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if 'question' in data and 'pred_answer' in data:
                            qid = data.get('qid') or data.get('id')
                            if qid is not None:
                                completed_qids.add(qid)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Warning: Error loading completed data: {e}")

        return completed_qids

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
            if claim.claim_type == "answer_claim":
                claim.dependencies = list(dict.fromkeys(identity_subgoal_ids + answer_subgoal_ids))
            elif claim.criticality > 0:
                claim.dependencies = _claim_local_dependencies(claim.content, question_plan) or identity_subgoal_ids[:1]
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
        preliminary_subgoals = [sg.__dict__ for sg in question_plan.subgoals]
        dep_supports = {sg["subgoal_id"]: 0.0 for sg in preliminary_subgoals if sg.get("required")}
        assessments = []
        for claim in claims:
            parent_supports = [dep_supports.get(dep, 0.0) for dep in claim.dependencies] if claim.dependencies else None
            assessments.append(scorer.score(claim, spans, delivered_span_ids, parent_supports=parent_supports))
        for assessment in assessments:
            claim_node_id = trace_logger.add_claim_assessment(assessment, generated_by=answer_node)
            assessment["claim_node_id"] = claim_node_id
            assessment["evidence_set_node_id"] = assessment.get("best_evidence_set", {}).get("evidence_set_id")
        answer_assessment = assess_answer(question_plan, result.get("answer", ""), assessments).to_dict()
        subgoal_assessments = _resolve_subgoals(question_plan, assessments, answer_assessment)
        unresolved_required = [sg["subgoal_id"] for sg in subgoal_assessments if sg.get("required") and sg.get("status") != "resolved"]
        failure_types = infer_failure_types(answer_assessment, hypothesis_assessments, subgoal_assessments)
        root_bad = failure_frontier(assessments)
        final_support = min((a["raw_score"] for a in assessments if a["claim"].get("criticality", 1.0) > 0), default=None)
        trace = trace_logger.to_dict()
        blame = []
        root_bad_hypotheses = [
            e["hypothesis_id"] for e in commitment_events
            if e.get("is_premature") and "PREMATURE_ENTITY_COMMITMENT" in failure_types
        ][:3]
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
            "final_claim_support": final_support,
            "root_bad_claims": root_bad,
            "root_bad_hypotheses": root_bad_hypotheses,
            "unresolved_required_subgoals": unresolved_required,
            "failure_types": failure_types,
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
        sample_id = qid if qid is not None else f"sample_{sample_index:06d}"
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
        completed_qids = self._load_completed_qids()

        pending = [q for q in self.questions
                   if (q.get('qid') or q.get('id')) not in completed_qids]

        print(f"Total questions: {len(self.questions)}")
        print(f"Completed: {len(completed_qids)}")
        print(f"Pending: {len(pending)}")

        if not pending:
            print("All questions completed!")
            return

        print(f"Starting with {self.num_workers} workers...")

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {}

            for sample_index, item in enumerate(pending):
                agent = self._create_agent()
                future = executor.submit(self._process_one, item, agent, sample_index)
                futures[future] = item.get('qid') or item.get('id')

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
