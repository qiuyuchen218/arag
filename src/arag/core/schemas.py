"""Structured, serializable records used by ARAG v2 observability.

The v2 protocol keeps the old rendered tool text for LLM compatibility while
also preserving stable IDs, provenance, spans, ranks, and diagnostics for
claim-level verification and local rollback.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_json(value: Any) -> Any:
    if is_dataclass(value):
        return clean_json(asdict(value))
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean_json(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
        return value
    except (TypeError, ValueError):
        return str(value)


def stable_hash(*parts: Any, length: int = 16) -> str:
    payload = json.dumps(clean_json(parts), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def artifact_id(corpus_version: str, doc_id: Any, chunk_id: Any) -> str:
    return f"artifact_{stable_hash(corpus_version or 'corpus', doc_id, chunk_id)}"


@dataclass
class EvidenceSpan:
    span_id: str
    artifact_id: str
    doc_id: str
    chunk_id: str
    sentence_id: Optional[int]
    text: str
    start_offset: int = 0
    end_offset: int = 0
    content_hash: str = ""

    @classmethod
    def from_text(
        cls,
        text: str,
        corpus_version: str,
        doc_id: Any,
        chunk_id: Any,
        sentence_id: Optional[int] = None,
        start_offset: int = 0,
        end_offset: Optional[int] = None,
    ) -> "EvidenceSpan":
        end = len(text) if end_offset is None else end_offset
        aid = artifact_id(corpus_version, doc_id, chunk_id)
        sid = f"span_{stable_hash(aid, sentence_id, start_offset, end, text)}"
        return cls(sid, aid, str(doc_id), str(chunk_id), sentence_id, text, start_offset, end, content_hash(text))


@dataclass
class SearchHit:
    artifact_id: str
    doc_id: str
    chunk_id: str
    sentence_ids: List[int] = field(default_factory=list)
    matched_spans: List[EvidenceSpan] = field(default_factory=list)
    retrieval_channel: str = "lexical"
    raw_score: float = 0.0
    calibrated_score: Optional[float] = None
    candidate_rank: Optional[int] = None
    final_rank: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReadReceipt:
    branch_id: str
    call_id: str
    artifact_id: str
    returned_span_ids: List[str]
    returned_text: str
    content_hash: str
    truncated: bool = False
    from_cache: bool = False
    tokens: int = 0


@dataclass
class ToolCallRecord:
    call_id: str
    branch_id: str
    parent_llm_call_id: Optional[str]
    tool_name: str
    query: str
    arguments: Dict[str, Any]
    corpus_version: str = "default"
    index_version: Optional[str] = None
    timestamp: str = field(default_factory=utc_now)
    step_index: int = 0


@dataclass
class ToolResult:
    call_id: str
    tool_name: str
    status: str
    rendered_text: str
    results: List[Dict[str, Any]] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    retrieved_tokens: int = 0
    error: Optional[str] = None

    def to_log(self) -> Dict[str, Any]:
        return {
            "tool_result": self.to_dict(),
            "status": self.status,
            "retrieved_tokens": self.retrieved_tokens,
            "error": self.error,
            **self.diagnostics,
        }

    def to_legacy_tuple(self) -> tuple[str, Dict[str, Any]]:
        return self.rendered_text, self.to_log()

    def to_dict(self) -> Dict[str, Any]:
        return clean_json(asdict(self))


def result_dict(item: Any) -> Dict[str, Any]:
    return clean_json(item)
