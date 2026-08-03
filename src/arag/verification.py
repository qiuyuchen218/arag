"""Claim extraction and support scoring for ARAG v2."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import requests

from arag.core.schemas import EvidenceSpan, clean_json, stable_hash
from arag.cognition import classify_claim


@dataclass
class Claim:
    claim_id: str
    content: str
    generated_by: Optional[str] = None
    branch_id: str = "b0"
    step_index: int = 0
    dependencies: List[str] = field(default_factory=list)
    cited_evidence_ids: List[str] = field(default_factory=list)
    criticality: float = 1.0
    status: str = "UNCERTAIN"
    claim_type: str = "answer_claim"
    aligned_relation_ids: List[str] = field(default_factory=list)
    resolves_subgoal_ids: List[str] = field(default_factory=list)


@dataclass
class EvidenceSet:
    evidence_set_id: str
    claim_id: str
    evidence_span_ids: List[str]
    unique_doc_ids: List[str]
    joint: bool = False
    verifier_result: Dict[str, Any] = field(default_factory=dict)
    minimal_sufficient: bool = False
    branch_id: str = "b0"


class SimpleClaimExtractor:
    """Offline extractor for public final-answer text.

    It deliberately avoids hidden chain-of-thought and only splits observable
    text into sentence-level claims.
    """

    def extract(self, text: str, generated_by: str = None, branch_id: str = "b0") -> List[Claim]:
        claims = []
        normalized = re.sub(r"\n\s*(\d+[\.)]|[-*])\s*", ". ", text or "")
        normalized = re.sub(r":\s*(\d+[\.)])", ": ", normalized)
        parts = re.split(r"(?<=[.!?。！？])\s+|\n+", normalized)
        for idx, sentence in enumerate(parts):
            content = re.sub(r"^\s*(?:[-*]|\d+[\.)])\s*", "", sentence.strip())
            content = re.sub(r"\s+", " ", content).strip()
            content = re.sub(r":\.$", ":", content)
            if not content:
                continue
            claim_type, criticality = classify_claim(content)
            claims.append(Claim(
                claim_id=f"claim_{stable_hash(branch_id, generated_by, idx, content)}",
                content=content,
                generated_by=generated_by,
                branch_id=branch_id,
                step_index=idx,
                criticality=criticality,
                claim_type=claim_type,
            ))
        answer_claims = [c for c in claims if c.claim_type == "answer_claim"]
        if not answer_claims and claims:
            for claim in reversed(claims):
                if claim.criticality > 0 and claim.claim_type not in {"process_claim", "meta_claim", "incomplete_fragment", "retrieval_coverage_claim"}:
                    claim.claim_type = "answer_claim"
                    claim.criticality = max(claim.criticality, 0.8)
                    break
        return claims


@dataclass
class VerificationResult:
    p_entail: float
    p_contradict: float
    p_insufficient: float
    relevance: float = 1.0
    explanation: str = ""
    verifier_model: str = "fake"
    prompt_hash: str = ""
    raw_result: Dict[str, Any] = field(default_factory=dict)
    verifier_mode: str = "fake_test"
    authoritative: bool = False
    calibrated: bool = False
    evidence_entailment: Optional[float] = None
    world_knowledge_plausibility: Optional[float] = None
    evidence_relevance: Optional[float] = None
    evidence_contradiction: Optional[float] = None
    insufficient_evidence: Optional[float] = None
    referenced_span_ids: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return clean_json(asdict(self))


class VerificationBackend:
    def verify(self, claim: Claim, evidence: List[EvidenceSpan], dependencies: List[Claim] = None, claim_type: str = None) -> VerificationResult:
        raise NotImplementedError


class FakeVerificationBackend(VerificationBackend):
    def __init__(self, default: VerificationResult = None, overrides: Dict[str, VerificationResult] = None, authoritative_for_test: bool = False):
        self.default = default or VerificationResult(
            0.9, 0.02, 0.08,
            explanation="deterministic fake",
            verifier_mode="fake_test",
            authoritative=authoritative_for_test,
            calibrated=False,
        )
        self.overrides = overrides or {}
        self.authoritative_for_test = authoritative_for_test

    def verify(self, claim: Claim, evidence: List[EvidenceSpan], dependencies: List[Claim] = None, claim_type: str = None) -> VerificationResult:
        result = self.overrides.get(claim.claim_id) or self.overrides.get(claim.content) or self.default
        if result.verifier_mode == "fake_test" and self.authoritative_for_test:
            result.authoritative = True
        return result


class LLMVerificationBackend(VerificationBackend):
    def __init__(self, model: str, api_key: str, base_url: str = "https://api.openai.com/v1", timeout: int = 60, max_retries: int = 2, prompt_token_budget: int = 4096):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.prompt_token_budget = prompt_token_budget

    def verify(self, claim: Claim, evidence: List[EvidenceSpan], dependencies: List[Claim] = None, claim_type: str = None) -> VerificationResult:
        evidence_payload = [{"span_id": e.span_id, "doc_id": e.doc_id, "text": e.text} for e in evidence]
        while _rough_tokens(json.dumps(evidence_payload, ensure_ascii=False)) > self.prompt_token_budget and evidence_payload:
            longest = max(range(len(evidence_payload)), key=lambda i: len(evidence_payload[i].get("text", "")))
            text = evidence_payload[longest]["text"]
            evidence_payload[longest]["text"] = text[: max(256, int(len(text) * 0.75))]
        prompt = {
            "claim": claim.content,
            "claim_type": claim_type or claim.claim_type,
            "dependencies": [{"claim_id": d.claim_id, "content": d.content, "status": d.status} for d in (dependencies or [])],
            "evidence": evidence_payload,
            "instructions": (
                "Use ONLY the listed evidence spans. Do not use world knowledge as entailment. "
                "Return JSON with evidence_entailment, world_knowledge_plausibility, evidence_relevance, "
                "evidence_contradiction, insufficient_evidence, referenced_span_ids, explanation. "
                "evidence_entailment means the EvidenceSet itself entails the claim."
            ),
        }
        prompt_hash = stable_hash(prompt)
        raw, data = {}, {}
        last_error = None
        for _ in range(max(1, self.max_retries)):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": "You are a strict claim verifier. Output only JSON. Do not use hidden reasoning."},
                            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                        ],
                        "temperature": 0,
                        "response_format": {"type": "json_object"},
                    },
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                raw = resp.json()
                content = raw["choices"][0]["message"]["content"]
                data = json.loads(content)
                break
            except Exception as exc:
                last_error = str(exc)
        if not data:
            return VerificationResult(0.0, 0.0, 1.0, explanation=f"verifier_failed: {last_error}", verifier_model=self.model, prompt_hash=prompt_hash, raw_result={"error": last_error}, verifier_mode="real_uncalibrated", authoritative=False)
        return VerificationResult(
            _as_probability(data.get("evidence_entailment", data.get("p_entail", 0)), default=0.0),
            _as_probability(data.get("evidence_contradiction", data.get("p_contradict", 0)), default=0.0),
            _as_probability(data.get("insufficient_evidence", data.get("p_insufficient", 1)), default=1.0),
            _as_probability(data.get("evidence_relevance", data.get("relevance", 1)), default=1.0),
            str(data.get("explanation", "")),
            self.model,
            prompt_hash,
            raw,
            "real_uncalibrated",
            True,
            False,
            evidence_entailment=_as_probability(data.get("evidence_entailment", data.get("p_entail", 0)), default=0.0),
            world_knowledge_plausibility=_as_probability(data.get("world_knowledge_plausibility", 0), default=0.0),
            evidence_relevance=_as_probability(data.get("evidence_relevance", data.get("relevance", 1)), default=1.0),
            evidence_contradiction=_as_probability(data.get("evidence_contradiction", data.get("p_contradict", 0)), default=0.0),
            insufficient_evidence=_as_probability(data.get("insufficient_evidence", data.get("p_insufficient", 1)), default=1.0),
            referenced_span_ids=[str(x) for x in data.get("referenced_span_ids", []) if x],
        )


def _as_probability(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    text = str(value).strip().lower()
    aliases = {
        "very high": 0.95,
        "high": 0.85,
        "medium": 0.5,
        "moderate": 0.5,
        "low": 0.15,
        "very low": 0.05,
        "true": 1.0,
        "yes": 1.0,
        "false": 0.0,
        "no": 0.0,
    }
    if text in aliases:
        return aliases[text]
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match:
        return max(0.0, min(1.0, float(match.group(0))))
    return default


def entropy_u(p_entail: float, p_contradict: float, p_insufficient: float) -> float:
    vals = [max(float(v), 0.0) for v in (p_entail, p_contradict, p_insufficient)]
    total = sum(vals) or 1.0
    probs = [v / total for v in vals]
    ent = -sum(p * math.log(p, 3) for p in probs if p > 0)
    return max(0.0, min(1.0, ent))


@dataclass
class SupportConfig:
    beta: float = 1.5
    gamma: float = 1.0
    delta: float = 1.0
    rho: float = 0.3
    kappa: float = 0.2
    mu: float = 0.5
    verified_threshold: float = 0.80
    low_support_threshold: float = 0.45
    contradiction_threshold: float = 0.70
    uncertainty_threshold: float = 0.65
    entailment_threshold: float = 0.80
    provenance_threshold: float = 1.0
    dependency_threshold: float = 0.70
    relevance_threshold: float = 0.50
    verifier_prompt_token_budget: int = 4096


class ClaimSupportScorer:
    def __init__(self, backend: VerificationBackend, config: SupportConfig = None):
        self.backend = backend
        self.config = config or SupportConfig()

    def score(
        self,
        claim: Claim,
        spans: List[EvidenceSpan],
        delivered_span_ids: List[str],
        parent_supports: List[float] = None,
        dependency_ids: List[str] = None,
        blocked_dependency_ids: List[str] = None,
        plan_semantic_valid: bool = True,
        verification_result: VerificationResult = None,
    ) -> Dict[str, Any]:
        legal = [s for s in spans if s.span_id in set(delivered_span_ids or [])]
        selected_spans = _select_evidence_set(claim, legal, max_spans=3)
        g_prov = 1.0 if selected_spans else 0.0
        verification = verification_result or self.backend.verify(claim, selected_spans)
        e = max(0.0, min(1.0, verification.evidence_entailment if verification.evidence_entailment is not None else verification.p_entail))
        c = max(0.0, min(1.0, verification.evidence_contradiction if verification.evidence_contradiction is not None else verification.p_contradict))
        p = g_prov
        dependency_ids = list(dependency_ids if dependency_ids is not None else (claim.dependencies or []))
        blocked_dependency_ids = list(blocked_dependency_ids or [])
        h_available = bool(plan_semantic_valid)
        h = 0.0 if not plan_semantic_valid else (min(parent_supports) if parent_supports else 1.0)
        r = max((verification.evidence_relevance if verification.evidence_relevance is not None else verification.relevance) if selected_spans else 0.0, 0.0)
        unique_docs = {s.doc_id for s in selected_spans}
        d = min(1.0, len(unique_docs) / 1.0) if selected_spans else 0.0
        u = entropy_u(e, c, verification.insufficient_evidence if verification.insufficient_evidence is not None else verification.p_insufficient)
        cfg = self.config
        evset_span_ids = [s.span_id for s in selected_spans]
        verifier_input_span_ids = evset_span_ids[:]
        verifier_input_doc_ids = sorted({s.doc_id for s in selected_spans})
        verifier_input_payload = {
            "claim": claim.content,
            "evidence": [{"span_id": s.span_id, "doc_id": s.doc_id, "text": s.text} for s in selected_spans],
        }
        verifier_input_hash = stable_hash(verifier_input_payload)
        verifier_input_token_count = _rough_tokens(json.dumps(verifier_input_payload, ensure_ascii=False))
        referenced = set(verification.referenced_span_ids or _extract_span_refs(verification.explanation))
        leakage = sorted(referenced - set(evset_span_ids))
        evidence_isolation_valid = set(verifier_input_span_ids) == set(evset_span_ids) and not leakage
        if not evidence_isolation_valid:
            p = 0.0
            g_prov = 0.0
        raw = g_prov * e * ((1 - c) ** cfg.beta) * (p ** cfg.gamma) * (h ** cfg.delta) * (r ** cfg.rho) * (d ** cfg.kappa) * ((1 - u) ** cfg.mu)
        if not verification.authoritative:
            status = "UNASSESSED"
            diagnostic_status = "FAKE_SUPPORTED" if verification.verifier_mode in {"fake_test", "heuristic_diagnostic"} and e >= cfg.entailment_threshold else "UNASSESSED"
            evidence_status = "UNASSESSED"
            reasoning_status = "UNASSESSED"
        elif not g_prov:
            evidence_status = "INVALID_PROVENANCE"
            reasoning_status = _reasoning_status(plan_semantic_valid, h, blocked_dependency_ids, claim.aligned_relation_ids, cfg)
            status = _overall_status(evidence_status, reasoning_status)
            diagnostic_status = status
        elif c >= cfg.contradiction_threshold:
            evidence_status = "CONTRADICTED"
            reasoning_status = _reasoning_status(plan_semantic_valid, h, blocked_dependency_ids, claim.aligned_relation_ids, cfg)
            status = _overall_status(evidence_status, reasoning_status)
            diagnostic_status = status
        elif not evidence_isolation_valid:
            evidence_status = "INVALID_PROVENANCE"
            reasoning_status = _reasoning_status(plan_semantic_valid, h, blocked_dependency_ids, claim.aligned_relation_ids, cfg)
            status = _overall_status(evidence_status, reasoning_status)
            diagnostic_status = status
        elif verification.calibrated and raw >= cfg.verified_threshold and h >= cfg.dependency_threshold:
            evidence_status = "VERIFIED"
            reasoning_status = _reasoning_status(plan_semantic_valid, h, blocked_dependency_ids, claim.aligned_relation_ids, cfg)
            status = _overall_status(evidence_status, reasoning_status)
            diagnostic_status = status
        elif (
            e >= cfg.entailment_threshold
            and c <= cfg.contradiction_threshold
            and p >= cfg.provenance_threshold
            and r >= cfg.relevance_threshold
            and u <= cfg.uncertainty_threshold
        ):
            evidence_status = "VERIFIED"
            reasoning_status = _reasoning_status(plan_semantic_valid, h, blocked_dependency_ids, claim.aligned_relation_ids, cfg)
            status = _overall_status(evidence_status, reasoning_status)
            diagnostic_status = status
        elif e < cfg.entailment_threshold and c < cfg.contradiction_threshold:
            evidence_status = "UNSUPPORTED"
            reasoning_status = _reasoning_status(plan_semantic_valid, h, blocked_dependency_ids, claim.aligned_relation_ids, cfg)
            status = _overall_status(evidence_status, reasoning_status)
            diagnostic_status = status
        elif u >= cfg.uncertainty_threshold:
            evidence_status = "VERIFIER_UNCERTAIN"
            reasoning_status = _reasoning_status(plan_semantic_valid, h, blocked_dependency_ids, claim.aligned_relation_ids, cfg)
            status = _overall_status(evidence_status, reasoning_status)
            diagnostic_status = status
        else:
            evidence_status = "VERIFIER_UNCERTAIN"
            reasoning_status = _reasoning_status(plan_semantic_valid, h, blocked_dependency_ids, claim.aligned_relation_ids, cfg)
            status = _overall_status(evidence_status, reasoning_status)
            diagnostic_status = status
        claim.status = status
        selected_docs = sorted({s.doc_id for s in selected_spans})
        minimal_sufficient = (
            verification.authoritative
            and evidence_isolation_valid
            and e >= cfg.entailment_threshold
            and r >= cfg.relevance_threshold
            and p >= cfg.provenance_threshold
            and c <= cfg.contradiction_threshold
        )
        return clean_json({
            "claim": asdict(claim),
            "support_vector": {
                "E": e, "C": c, "P": p, "H": h, "R": r, "D": d, "U": u,
                "details": {
                    "H": {
                        "available": h_available,
                        "reason": "plan uncertain" if not plan_semantic_valid else ("no dependencies" if not dependency_ids else "actual dependency supports"),
                        "dependency_ids": dependency_ids,
                        "blocked_dependency_ids": blocked_dependency_ids,
                    },
                    "R": {"available": bool(selected_spans), "reason": "verifier relevance over selected EvidenceSet" if selected_spans else "no legal evidence"},
                    "D": {"available": bool(selected_spans), "reason": "unique docs from selected EvidenceSet" if selected_spans else "no legal evidence"},
                    "evidence_isolation_valid": evidence_isolation_valid,
                    "evidence_leakage_span_ids": leakage,
                },
            },
            "defect_vector": {"1-E": 1 - e, "C": c, "1-P": 1 - p, "1-H": 1 - h, "1-R": 1 - r, "1-D": 1 - d, "U": u},
            "provenance_gate": {"G_prov": g_prov, "delivered_in_context": bool(legal), "evidence_exists": bool(spans)},
            "raw_score": raw,
            "calibrated_score": raw,
            "calibrated": verification.calibrated,
            "verifier_mode": verification.verifier_mode,
            "authoritative": verification.authoritative,
            "verifier_is_real": verification.verifier_mode.startswith("real_"),
            "verifier_decision_capable": verification.authoritative,
            "verifier_calibrated": verification.calibrated,
            "verifier_authoritative_for_repair": verification.authoritative and not verification.calibrated,
            "repair_eligible": verification.authoritative and status in {"UNSUPPORTED", "CONTRADICTED", "INVALID_PROVENANCE", "DEPENDENCY_BLOCKED", "DEPENDENCY_BROKEN", "UNCERTAIN"},
            "diagnostic_status": diagnostic_status,
            "evidence_status": evidence_status,
            "reasoning_status": reasoning_status,
            "overall_status": status,
            "verifier_model": verification.verifier_model,
            "prompt_hash": verification.prompt_hash,
            "verifier_input_span_ids": verifier_input_span_ids,
            "verifier_input_doc_ids": verifier_input_doc_ids,
            "verifier_input_token_count": verifier_input_token_count,
            "verifier_input_hash": verifier_input_hash,
            "evidence_set_span_ids": evset_span_ids,
            "evidence_isolation_valid": evidence_isolation_valid,
            "evidence_leakage_span_ids": leakage,
            "world_knowledge_plausibility": verification.world_knowledge_plausibility,
            "status": status,
            "best_evidence_set": asdict(EvidenceSet(
                evidence_set_id=f"evset_{stable_hash(claim.claim_id, [s.span_id for s in selected_spans])}",
                claim_id=claim.claim_id,
                evidence_span_ids=[s.span_id for s in selected_spans],
                unique_doc_ids=selected_docs,
                joint=len(selected_spans) > 1,
                verifier_result=verification.as_dict(),
                minimal_sufficient=minimal_sufficient if verification.authoritative else None,
                branch_id=claim.branch_id,
            )),
        })


def _select_evidence_set(claim: Claim, legal: List[EvidenceSpan], max_spans: int = 3) -> List[EvidenceSpan]:
    terms = {t.lower() for t in re.findall(r"[A-Za-z0-9]{4,}", claim.content or "")}
    scored = []
    seen = set()
    for idx, span in enumerate(legal):
        identity = (span.doc_id, span.chunk_id, span.sentence_id, span.start_offset, span.end_offset, span.content_hash or span.text)
        if identity in seen:
            continue
        seen.add(identity)
        text_terms = set(re.findall(r"[A-Za-z0-9]{4,}", (span.text or "").lower()))
        overlap = len(terms & text_terms)
        date_bonus = 1 if DATE_LIKE_RE.search(span.text or "") and DATE_LIKE_RE.search(claim.content or "") else 0
        scored.append((-(overlap + date_bonus), idx, span))
    scored.sort()
    return [s for _, _, s in scored[:max_spans]]


DATE_LIKE_RE = re.compile(r"\b\d{3,4}\b")


def _extract_span_refs(text: str) -> List[str]:
    return re.findall(r"\bspan_[0-9a-fA-F]+\b", text or "")


def _reasoning_status(plan_semantic_valid: bool, h: float, blocked_dependency_ids: List[str], aligned_relation_ids: List[str], cfg: SupportConfig) -> str:
    if not plan_semantic_valid:
        return "PLAN_UNCERTAIN"
    if not aligned_relation_ids and blocked_dependency_ids:
        return "UNALIGNED_TO_PLAN"
    if blocked_dependency_ids or h < cfg.dependency_threshold:
        return "DEPENDENCY_BLOCKED"
    return "USABLE"


def _overall_status(evidence_status: str, reasoning_status: str) -> str:
    if evidence_status != "VERIFIED":
        return evidence_status
    if reasoning_status != "USABLE":
        return reasoning_status
    return "VERIFIED"


def _rough_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)


def failure_frontier(assessments: List[Dict[str, Any]]) -> List[str]:
    failed = {
        a["claim"]["claim_id"] for a in assessments
        if a.get("authoritative")
        and a.get("claim", {}).get("criticality", 1.0) > 0
        and a.get("claim", {}).get("claim_type") not in {"process_claim", "meta_claim", "incomplete_fragment", "retrieval_coverage_claim"}
        and a.get("status") in {
            "UNSUPPORTED", "CONTRADICTED", "INVALID_PROVENANCE", "DEPENDENCY_BLOCKED", "DEPENDENCY_BROKEN", "UNCERTAIN"
        }
    }
    by_id = {a["claim"]["claim_id"]: a["claim"] for a in assessments}
    resolving_claims_by_subgoal: Dict[str, List[str]] = {}
    for assessment in assessments:
        claim = assessment.get("claim", {})
        claim_id = claim.get("claim_id")
        if not claim_id:
            continue
        for subgoal_id in claim.get("resolves_subgoal_ids", []) or []:
            resolving_claims_by_subgoal.setdefault(subgoal_id, []).append(claim_id)
    roots = []
    for cid in sorted(failed):
        parents = set()
        for dep in by_id.get(cid, {}).get("dependencies", []) or []:
            if dep in by_id:
                parents.add(dep)
            parents.update(resolving_claims_by_subgoal.get(dep, []))
        parents.discard(cid)
        if not parents.intersection(failed):
            roots.append(cid)
    return roots
