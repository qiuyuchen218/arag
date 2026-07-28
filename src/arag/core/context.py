"""Agent execution context for ARAG."""

from typing import Any, Dict, List, Set
from dataclasses import dataclass, field

from arag.core.schemas import clean_json, content_hash, stable_hash, utc_now


@dataclass
class RetrievalLog:
    """Log entry for a retrieval operation."""
    tool_name: str
    tokens: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentContext:
    """Context manager for agent execution state."""

    def __init__(self):
        self.branch_id: str = "b0"
        # Token statistics
        self.total_retrieved_tokens: int = 0
        self.retrieval_logs: List[RetrievalLog] = []

        # State management
        self.read_chunk_ids: Set[str] = set()
        self.search_history: List[Dict[str, Any]] = []

        # Observability state exported with each prediction for debugging.
        self.read_chunks: Dict[str, Dict[str, Any]] = {}
        self.corpus_cache_read_ids: Set[str] = set()
        self.branch_read_chunk_ids: Dict[str, Set[str]] = {self.branch_id: set()}
        self.context_deliveries: List[Dict[str, Any]] = []

    def add_retrieval_log(
        self,
        tool_name: str,
        tokens: int,
        metadata: Dict[str, Any] = None
    ):
        """Add a retrieval log entry."""
        log = RetrievalLog(
            tool_name=tool_name,
            tokens=tokens,
            metadata=metadata or {}
        )
        self.retrieval_logs.append(log)
        self.total_retrieved_tokens += tokens

    def add_search_event(
        self,
        tool_name: str,
        query: Dict[str, Any],
        results: List[Dict[str, Any]],
        metadata: Dict[str, Any] = None,
    ):
        """Record structured search results for later error analysis."""
        self.search_history.append({
            "tool_name": tool_name,
            "query": query,
            "results": clean_json(results),
            "metadata": metadata or {},
        })

    def mark_chunk_as_read(
        self,
        chunk_id: str,
        content: str = None,
        tokens: int = 0,
        metadata: Dict[str, Any] = None,
    ):
        """Mark chunk as read and optionally store its full content."""
        chunk_id = str(chunk_id)
        self.read_chunk_ids.add(chunk_id)
        self.corpus_cache_read_ids.add(chunk_id)
        self.branch_read_chunk_ids.setdefault(self.branch_id, set()).add(chunk_id)
        if content is not None:
            self.read_chunks[chunk_id] = {
                "chunk_id": chunk_id,
                "content": content,
                "tokens": tokens,
                "metadata": metadata or {},
            }

    def has_branch_read_chunk(self, chunk_id: str, branch_id: str = None) -> bool:
        branch_id = branch_id or self.branch_id
        return str(chunk_id) in self.branch_read_chunk_ids.get(branch_id, set())

    def record_context_delivery(
        self,
        llm_call_id: str,
        tool_call_id: str,
        message_index: int,
        span_ids: List[str],
        text: str,
        branch_id: str = None,
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        delivery = {
            "delivery_id": f"ctxdel_{stable_hash(branch_id or self.branch_id, llm_call_id, tool_call_id, message_index, span_ids, text)}",
            "branch_id": branch_id or self.branch_id,
            "llm_call_id": llm_call_id,
            "tool_call_id": tool_call_id,
            "message_index": message_index,
            "span_ids": list(dict.fromkeys(str(s) for s in span_ids or [])),
            "text_hash": content_hash(text or ""),
            "timestamp": utc_now(),
            "semantics": "delivered_to_context_not_proven_used",
            "metadata": metadata or {},
        }
        self.context_deliveries.append(delivery)
        return delivery

    def is_chunk_read(self, chunk_id: str) -> bool:
        """Check if chunk has been read."""
        return str(chunk_id) in self.read_chunk_ids

    def add_read_chunk(self, chunk_id: str, content: str = None):
        """Alias for mark_chunk_as_read."""
        self.mark_chunk_as_read(chunk_id, content=content)

    def has_read_chunk(self, chunk_id: str) -> bool:
        """Alias for is_chunk_read."""
        return self.is_chunk_read(chunk_id)

    def get_read_chunk(self, chunk_id: str):
        """Return stored chunk content if available."""
        chunk = self.read_chunks.get(str(chunk_id))
        if chunk is None:
            return None
        return chunk.get("content", "")

    def reset(self):
        """Reset context for new query."""
        self.retrieval_logs = []
        self.read_chunk_ids = set()
        self.search_history = []
        self.read_chunks = {}
        self.corpus_cache_read_ids = set()
        self.branch_read_chunk_ids = {self.branch_id: set()}
        self.context_deliveries = []
        self.total_retrieved_tokens = 0

    def get_summary(self) -> Dict[str, Any]:
        """Get context summary."""
        return {
            "total_retrieved_tokens": self.total_retrieved_tokens,
            "retrieval_logs": [
                {
                    "tool_name": log.tool_name,
                    "tokens": log.tokens,
                    "metadata": log.metadata
                }
                for log in self.retrieval_logs
            ],
            "chunks_read_count": len(self.read_chunk_ids),
            "chunks_read_ids": list(self.read_chunk_ids),
            "read_chunks": self.read_chunks,
            "search_history": self.search_history,
            "branch_id": self.branch_id,
            "corpus_cache_read_ids": list(self.corpus_cache_read_ids),
            "branch_read_chunk_ids": {
                branch: sorted(ids) for branch, ids in self.branch_read_chunk_ids.items()
            },
            "context_deliveries": self.context_deliveries,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Export context as dictionary."""
        return self.get_summary()
