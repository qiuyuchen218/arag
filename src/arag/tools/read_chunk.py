"""Read chunk tool - retrieve full document content."""

import json
import re
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

from arag.core.schemas import EvidenceSpan, ReadReceipt, ToolResult, artifact_id, content_fingerprint, content_hash, result_dict, stable_hash
from arag.tools.base import BaseTool

if TYPE_CHECKING:
    from arag.core.context import AgentContext

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False


class ReadChunkTool(BaseTool):
    """Read full content of document chunks."""

    def __init__(self, chunks_file: str):
        self.chunks_file = chunks_file
        self.chunks = self._load_chunks()
        self.corpus_version = content_fingerprint({
            "chunks_file": chunks_file,
            "chunks": self.chunks,
        })
        self.chunks_dict = {str(c['id']): c for c in self.chunks}

        if not HAS_TIKTOKEN:
            raise ImportError("tiktoken required. Install: pip install tiktoken")
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")

    def _load_chunks(self) -> List[Dict[str, Any]]:
        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if data and isinstance(data[0], dict):
            chunks = []
            by_doc_pos = {}
            for i, item in enumerate(data):
                chunk = {
                    **item,
                    "id": str(item.get("id", item.get("chunk_id", i))),
                    "text": str(item.get("text", item.get("content", ""))),
                    "doc_id": str(item.get("doc_id") or item.get("document_id") or item.get("id", i)),
                    "position_in_doc": item.get("position_in_doc", i),
                    "metadata": item.get("metadata", {}),
                }
                chunks.append(chunk)
                by_doc_pos.setdefault(chunk["doc_id"], []).append(chunk)
            for doc_chunks in by_doc_pos.values():
                doc_chunks.sort(key=lambda c: c.get("position_in_doc", 0))
                for idx, chunk in enumerate(doc_chunks):
                    chunk["prev_chunk_id"] = doc_chunks[idx - 1]["id"] if idx > 0 else None
                    chunk["next_chunk_id"] = doc_chunks[idx + 1]["id"] if idx + 1 < len(doc_chunks) else None
            return chunks

        chunks = []
        for item in data:
            if isinstance(item, str):
                parts = item.split(':', 1)
                if len(parts) == 2:
                    chunks.append({'id': parts[0], 'text': parts[1], "doc_id": parts[0], "position_in_doc": len(chunks), "metadata": {}})
        return chunks

    def _split_sentences(self, text: str) -> List[Tuple[int, str, int, int]]:
        spans = []
        for match in re.finditer(r'[^.!?\n]+[.!?\n]*', text or ""):
            sentence = match.group(0).strip()
            if sentence:
                spans.append((len(spans), sentence, match.start(), match.end()))
        if not spans and text:
            spans.append((0, text, 0, len(text)))
        return spans

    @property
    def name(self) -> str:
        return "read_chunk"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "read_chunk",
                "description": """Read the complete content of document chunks by their IDs.

This tool returns the full text of the specified chunks, allowing you to examine the complete context and details that are not visible in search snippets.

IMPORTANT: Search results (keyword_search and semantic_search) only show abbreviated snippets marked with "..." - they are NOT sufficient for answering questions. You MUST use read_chunk to get the full content before formulating your answer.

STRATEGY:
- Always read promising chunks identified by your searches
- Make sure to read the most relevant chunks to gather complete information
- If information seems incomplete or truncated, read adjacent chunks (+/- 1)
- Reading full text is essential for accurate answers

Note: Previously read chunks will be marked as already seen to avoid redundant information.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "chunk_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of chunk IDs to retrieve (e.g., ['0', '24', '172'])"
                        },
                        "sentence_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Optional sentence IDs to return."
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["full", "span", "window"],
                            "default": "full"
                        },
                        "window_before": {"type": "integer", "default": 0},
                        "window_after": {"type": "integer", "default": 0},
                        "max_tokens": {"type": "integer", "default": 0},
                        "return_cached": {"type": "boolean", "default": True},
                        "epistemic_context": {
                            "type": "object",
                            "description": "Required public decision context. Include at least one proposition that this read is exploring, testing, verifying, committing, or using as a premise.",
                            "properties": {
                                "action_role": {
                                    "type": "string",
                                    "enum": ["EXPLORE", "TEST", "VERIFY", "DISAMBIGUATE", "USE_AS_PREMISE", "COMMIT"]
                                },
                                "active_subgoal_ids": {"type": "array", "items": {"type": "string"}},
                                "purpose": {"type": "string"},
                                "propositions": {
                                    "type": "array",
                                    "minItems": 1,
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "subject": {"type": "string"},
                                            "predicate": {"type": "string"},
                                            "object": {"type": "string"},
                                            "relation_id": {"type": "string"},
                                            "stance": {
                                                "type": "string",
                                                "enum": ["HYPOTHESIS", "PARTIALLY_SUPPORTED", "COMMITTED"]
                                            },
                                            "supporting_evidence_ids": {"type": "array", "items": {"type": "string"}},
                                            "missing_constraint_ids": {"type": "array", "items": {"type": "string"}}
                                        },
                                        "required": ["subject", "predicate", "object", "stance"]
                                    }
                                }
                            },
                            "required": ["action_role", "active_subgoal_ids", "purpose", "propositions"],
                            "additionalProperties": True
                        }
                    },
                    "required": ["chunk_ids", "epistemic_context"]
                }
            }
        }

    def execute(
        self,
        context: 'AgentContext',
        chunk_ids: List[str] = None,
        chunk_id: str = None,
        sentence_ids: List[int] = None,
        mode: str = "full",
        window_before: int = 0,
        window_after: int = 0,
        max_tokens: int = 0,
        return_cached: bool = True,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Read chunks.

        Args:
            context: Agent execution context
            chunk_ids: List of chunk IDs to read
            chunk_id: Single chunk ID (for backward compatibility)
        """
        if chunk_ids is None:
            if chunk_id is not None:
                chunk_ids = [str(chunk_id)]
            else:
                result = ToolResult("read_invalid", self.name, "invalid_arguments", "Error: No chunk IDs provided", [], {}, 0, 0, "missing_chunk_ids")
                return result.to_legacy_tuple()

        chunk_ids = [str(cid) for cid in chunk_ids]

        result_parts = []
        new_chunks_read = []
        already_read = []
        receipts = []
        returned_span_ids = []
        total_tokens = 0
        status = "success"
        call_id = f"read_{stable_hash(getattr(context, 'branch_id', 'b0'), chunk_ids, sentence_ids, mode, len(context.retrieval_logs))}"

        for cid in chunk_ids:
            from_cache = cid in context.corpus_cache_read_ids
            branch_already_read = context.has_branch_read_chunk(cid)
            if branch_already_read:
                already_read.append(cid)

            if cid in self.chunks_dict:
                chunk = self.chunks_dict[cid]
                content = chunk["text"]
                sentence_rows = self._split_sentences(content)
                selected_rows = sentence_rows
                if sentence_ids:
                    wanted = set(int(s) for s in sentence_ids)
                    selected_rows = [row for row in sentence_rows if row[0] in wanted]
                if mode == "window" and sentence_ids:
                    wanted = set()
                    for sid in sentence_ids:
                        for idx in range(max(0, sid - int(window_before or 0)), sid + int(window_after or 0) + 1):
                            wanted.add(idx)
                    selected_rows = [row for row in sentence_rows if row[0] in wanted]
                elif mode == "span" and not sentence_ids:
                    selected_rows = sentence_rows[:1]
                selected_text = content if mode == "full" and not sentence_ids else "\n".join(row[1] for row in selected_rows)
                chunk_tokens = len(self.tokenizer.encode(selected_text))
                truncated = False
                if max_tokens and chunk_tokens > max_tokens:
                    token_ids = self.tokenizer.encode(selected_text)[:max_tokens]
                    selected_text = self.tokenizer.decode(token_ids)
                    chunk_tokens = max_tokens
                    truncated = True
                    status = "success_truncated"

                spans = [
                    EvidenceSpan.from_text(row[1], self.corpus_version, chunk.get("doc_id", cid), cid, row[0], row[2], row[3])
                    for row in (selected_rows if selected_rows else [(None, selected_text, 0, len(selected_text))])
                ]
                returned_span_ids.extend(span.span_id for span in spans)
                result_parts.append(f"\n{'='*80}")
                result_parts.append(f"[Chunk {cid}]")
                if from_cache:
                    result_parts.append("(from cache; content delivered again for this context)")
                result_parts.append(f"{'-'*80}")
                result_parts.append(selected_text)
                result_parts.append(f"{'='*80}")

                total_tokens += chunk_tokens

                context.mark_chunk_as_read(
                    cid,
                    content=selected_text,
                    tokens=chunk_tokens,
                    metadata={
                        "source": "read_chunk",
                        "doc_id": chunk.get("doc_id", cid),
                        "artifact_id": artifact_id(self.corpus_version, chunk.get("doc_id", cid), cid),
                        "span_ids": [span.span_id for span in spans],
                        "from_cache": from_cache,
                        "truncated": truncated,
                    }
                )
                new_chunks_read.append(cid)
                receipts.append(ReadReceipt(
                    branch_id=getattr(context, "branch_id", "b0"),
                    call_id=call_id,
                    artifact_id=artifact_id(self.corpus_version, chunk.get("doc_id", cid), cid),
                    returned_span_ids=[span.span_id for span in spans],
                    returned_text=selected_text,
                    content_hash=content_hash(selected_text),
                    truncated=truncated,
                    from_cache=from_cache,
                    tokens=chunk_tokens,
                ))
            else:
                result_parts.append(f"\n[Chunk {cid}] - Not found")
                status = "not_found"

        tool_result = "\n".join(result_parts)

        context.add_retrieval_log(
            tool_name="read_chunk",
            tokens=total_tokens,
            metadata={
                "chunk_ids_requested": chunk_ids,
                "new_chunks_read": new_chunks_read,
                "already_read": already_read
            }
        )

        tool_log = ToolResult(
            call_id=call_id,
            tool_name=self.name,
            status=status,
            rendered_text=tool_result,
            results=[result_dict(r) for r in receipts],
            diagnostics={
                "new_chunks_count": len(new_chunks_read),
                "already_read_count": len(already_read),
                "chunk_ids": chunk_ids,
                "returned_span_ids": returned_span_ids,
                "corpus_version": self.corpus_version,
                "cache_semantics": "from_cache_means_no_disk_read_not_context_delivery",
            },
            retrieved_tokens=total_tokens,
        ).to_log()

        return tool_result, tool_log
