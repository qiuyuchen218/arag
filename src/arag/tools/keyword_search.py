"""Keyword search tool - exact text matching."""

import json
import math
import re
from collections import Counter
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

from arag.core.schemas import EvidenceSpan, SearchHit, ToolResult, artifact_id, result_dict, stable_hash
from arag.tools.base import BaseTool

if TYPE_CHECKING:
    from arag.core.context import AgentContext

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False


class KeywordSearchTool(BaseTool):
    """Keyword search using exact text matching."""

    def __init__(self, chunks_file: str):
        self.chunks_file = chunks_file
        self.chunks = self._load_chunks()
        self.corpus_version = stable_hash(chunks_file, len(self.chunks))
        self._prepare_index()

        if not HAS_TIKTOKEN:
            raise ImportError("tiktoken required. Install: pip install tiktoken")
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")

    def _load_chunks(self) -> List[Dict[str, Any]]:
        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if data and isinstance(data[0], dict):
            chunks = []
            for i, item in enumerate(data):
                chunk_id = str(item.get("id", item.get("chunk_id", i)))
                chunks.append({
                    **item,
                    "id": chunk_id,
                    "text": str(item.get("text", item.get("content", ""))),
                    "doc_id": str(item.get("doc_id") or item.get("document_id") or chunk_id),
                    "position_in_doc": item.get("position_in_doc", i),
                    "metadata": item.get("metadata", {}),
                })
            return chunks

        chunks = []
        for item in data:
            if isinstance(item, str):
                parts = item.split(':', 1)
                if len(parts) == 2:
                    chunks.append({'id': parts[0], 'text': parts[1], "doc_id": parts[0], "position_in_doc": len(chunks), "metadata": {}})
        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        spans = []
        start = 0
        for match in re.finditer(r'[^.!?\n]+[.!?\n]*', text):
            sentence = match.group(0).strip()
            if sentence:
                spans.append((len(spans), sentence, match.start(), match.end()))
            start = match.end()
        if not spans and text:
            spans.append((0, text, 0, len(text)))
        return spans

    @staticmethod
    def _tokens(text: str) -> List[str]:
        return re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)

    def _prepare_index(self):
        self._doc_freq = Counter()
        self._chunk_tokens = {}
        for chunk in self.chunks:
            toks = self._tokens(chunk["text"])
            self._chunk_tokens[chunk["id"]] = toks
            self._doc_freq.update(set(toks))
        self._avgdl = sum(len(v) for v in self._chunk_tokens.values()) / max(len(self._chunk_tokens), 1)

    def _parse_keywords(self, keywords: List[str]) -> Tuple[str, List[str]]:
        raw = " ".join(str(k).strip() for k in keywords or [] if str(k).strip())
        if not raw:
            return "OR", []
        mode = "AND" if re.search(r"\bAND\b", raw, re.I) else "OR"
        terms = [t.strip().strip('"') for t in re.split(r"\bAND\b|\bOR\b|,", raw, flags=re.I) if t.strip()]
        return mode, list(dict.fromkeys(terms))

    def _find_spans(self, chunk: Dict[str, Any], terms: List[str]) -> List[EvidenceSpan]:
        text = chunk["text"]
        spans = []
        for sentence_id, sentence, start, end in self._split_sentences(text):
            sent_hits = []
            for term in terms:
                pattern = re.escape(term.lower())
                if re.match(r"^\w+$", term, flags=re.UNICODE):
                    pattern = rf"\b{pattern}\b"
                if re.search(pattern, sentence.lower(), flags=re.UNICODE):
                    sent_hits.append(term)
            if sent_hits:
                spans.append(EvidenceSpan.from_text(
                    sentence,
                    self.corpus_version,
                    chunk.get("doc_id", chunk["id"]),
                    chunk["id"],
                    sentence_id,
                    start,
                    end,
                ))
        return spans

    def _bm25(self, query_terms: List[str], chunk_id: str) -> float:
        toks = self._chunk_tokens.get(chunk_id, [])
        tf = Counter(toks)
        n_docs = max(len(self.chunks), 1)
        score = 0.0
        k1, b = 1.5, 0.75
        for term in query_terms:
            for tok in self._tokens(term):
                df = self._doc_freq.get(tok, 0)
                if df == 0:
                    continue
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                freq = tf.get(tok, 0)
                denom = freq + k1 * (1 - b + b * len(toks) / max(self._avgdl, 1e-9))
                score += idf * (freq * (k1 + 1)) / denom if denom else 0.0
        return score

    @property
    def name(self) -> str:
        return "keyword_search"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "keyword_search",
                "description": """Search for document chunks using keyword-based exact text matching (case-insensitive). Returns chunk IDs and abbreviated sentence snippets where the keywords appear.

IMPORTANT: This tool matches keywords literally in the text. Use SHORT, SPECIFIC terms (1-3 words maximum). Each keyword is matched independently.

RETURNS: Abbreviated snippets marked with "..." showing where keywords appear. These snippets help you identify relevant chunks, but you MUST use read_chunk to get the full text for answering questions.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of keywords to search. Each keyword should be 1-3 words maximum."
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of top-ranked chunks to return (default: 5, max: 20)",
                            "default": 5
                        }
                    },
                    "required": ["keywords"]
                }
            }
        }

    def execute(self, context: 'AgentContext', keywords: List[str] = None, query: str = None, top_k: int = 5) -> Tuple[str, Dict[str, Any]]:
        top_k = max(0, min(int(top_k or 5), 20))
        if keywords is None and query is not None:
            keywords = [query]
        mode, terms = self._parse_keywords(keywords or [])
        call_id = f"kw_{stable_hash(getattr(context, 'branch_id', 'b0'), terms, top_k, len(context.search_history))}"
        if not terms or top_k <= 0:
            result = ToolResult(call_id, self.name, "empty", "No keyword query provided.", [], {"query_mode": mode}, 0, 0)
            context.add_search_event(self.name, {"keywords": keywords or [], "top_k": top_k}, [], result.diagnostics)
            return result.to_legacy_tuple()

        scored_chunks = []
        for chunk in self.chunks:
            spans = self._find_spans(chunk, terms)
            found_terms = []
            lowered = chunk["text"].lower()
            for term in terms:
                pattern = re.escape(term.lower())
                if re.match(r"^\w+$", term, flags=re.UNICODE):
                    pattern = rf"\b{pattern}\b"
                if re.search(pattern, lowered, flags=re.UNICODE):
                    found_terms.append(term)
            if (mode == "AND" and len(found_terms) != len(terms)) or not found_terms:
                continue
            raw_score = self._bm25(found_terms, chunk["id"]) + 0.05 * sum(len(s.text) for s in spans)
            scored_chunks.append((chunk, raw_score, spans, found_terms))

        scored_chunks.sort(key=lambda x: (-x[1], str(x[0]["doc_id"]), str(x[0]["id"])))
        candidates = []
        for rank, (chunk, score, spans, found_terms) in enumerate(scored_chunks, start=1):
            candidates.append((rank, chunk, score, spans, found_terms))

        collapsed = []
        seen_docs = set()
        for item in candidates:
            _, chunk, _, _, _ = item
            doc_id = chunk.get("doc_id", chunk["id"])
            if doc_id in seen_docs and len(collapsed) >= max(1, top_k // 2):
                continue
            seen_docs.add(doc_id)
            collapsed.append(item)
            if len(collapsed) >= top_k:
                break
        top_chunks = collapsed

        if not top_chunks:
            tool_result = f"No results found for keywords: {keywords}"
            context.add_search_event(
                tool_name="keyword_search",
                query={"keywords": keywords, "top_k": top_k},
                results=[],
                metadata={"chunks_found": 0, "candidate_count": 0, "query_mode": mode}
            )
            tool_log = ToolResult(call_id, self.name, "empty", tool_result, [], {"chunks_found": 0, "candidate_count": 0, "query_mode": mode}, 0, 0).to_log()
            return tool_result, tool_log

        result_parts = []
        search_results = []

        for final_rank, (candidate_rank, chunk, score, spans, found_terms) in enumerate(top_chunks, start=1):
            matched_sentences = [s.text for s in spans[:5]]
            if matched_sentences:
                matched_text = "... " + " ... ".join(matched_sentences) + " ..."
            else:
                matched_text = "(no exact sentence match)"
            hit = SearchHit(
                artifact_id=artifact_id(self.corpus_version, chunk.get("doc_id", chunk["id"]), chunk["id"]),
                doc_id=str(chunk.get("doc_id", chunk["id"])),
                chunk_id=str(chunk["id"]),
                sentence_ids=[s.sentence_id for s in spans if s.sentence_id is not None],
                matched_spans=spans[:5],
                retrieval_channel="lexical",
                raw_score=float(score),
                candidate_rank=candidate_rank,
                final_rank=final_rank,
                metadata={"keywords_found": found_terms, "title": chunk.get("title"), "candidate_state": "top_k"},
            )
            item_dict = result_dict(hit)
            item_dict["score"] = score
            item_dict["matched_sentences"] = matched_sentences
            search_results.append(item_dict)
            result_parts.append(f"Chunk ID: {chunk['id']}, Matched keywords in chunk: {matched_text}")

        tool_result = "\n\n".join(result_parts)

        all_matched_sentences = []
        for _, _, _, spans, _ in top_chunks:
            all_matched_sentences.extend([s.text for s in spans])

        if all_matched_sentences:
            sentences_text = "\n".join(all_matched_sentences)
            retrieved_tokens = len(self.tokenizer.encode(sentences_text))
        else:
            retrieved_tokens = 0

        context.add_retrieval_log(
            tool_name="keyword_search",
            tokens=retrieved_tokens,
            metadata={
                "keywords": keywords,
                "chunks_found": len(top_chunks),
                "candidate_count": len(candidates),
                "chunk_ids": [c[1]['id'] for c in top_chunks],
            }
        )
        context.add_search_event(
            tool_name="keyword_search",
            query={"keywords": keywords, "top_k": top_k},
            results=search_results,
            metadata={
                "chunks_found": len(top_chunks),
                "candidate_count": len(candidates),
                "retrieved_tokens": retrieved_tokens,
                "query_mode": mode,
                "internal_queries": [{"mode": mode, "terms": terms}],
            }
        )

        tool_log = ToolResult(
            call_id=call_id,
            tool_name=self.name,
            status="success",
            rendered_text=tool_result,
            results=search_results,
            diagnostics={
                "chunks_found": len(top_chunks),
                "candidate_count": len(candidates),
                "candidate_count_not_top_k": max(len(candidates) - len(top_chunks), 0),
                "chunk_ids": [c[1]["id"] for c in top_chunks],
                "query_mode": mode,
                "internal_queries": [{"mode": mode, "terms": terms}],
                "corpus_version": self.corpus_version,
            },
            retrieved_tokens=retrieved_tokens,
        ).to_log()
        return tool_result, tool_log
