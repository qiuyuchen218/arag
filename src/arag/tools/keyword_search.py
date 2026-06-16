"""Keyword search tool - exact text matching."""

import json
import re
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

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

        if not HAS_TIKTOKEN:
            raise ImportError("tiktoken required. Install: pip install tiktoken")
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")

    def _load_chunks(self) -> List[Dict[str, Any]]:
        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if data and isinstance(data[0], dict):
            return data

        chunks = []
        for item in data:
            if isinstance(item, str):
                parts = item.split(':', 1)
                if len(parts) == 2:
                    chunks.append({'id': parts[0], 'text': parts[1]})
        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'[.!?\n]+', text)
        return [s.strip() for s in sentences if s.strip()]

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

    def execute(self, context: 'AgentContext', keywords: List[str], top_k: int = 5) -> Tuple[str, Dict[str, Any]]:
        top_k = min(top_k, 20)

        scored_chunks = []
        for chunk in self.chunks:
            text = chunk['text']
            text_lower = text.lower()
            chunk_id = chunk['id']

            matches = []
            total_score = 0

            for keyword in keywords:
                keyword_lower = keyword.lower()
                count = text_lower.count(keyword_lower)
                if count > 0:
                    matches.append(keyword)
                    total_score += count * len(keyword)

            if total_score > 0:
                sentences = self._split_sentences(text)
                matched_sentences = []

                for sentence in sentences:
                    sentence_lower = sentence.lower()
                    if any(keyword.lower() in sentence_lower for keyword in matches):
                        matched_sentences.append(sentence)

                scored_chunks.append({
                    'chunk_id': chunk_id,
                    'score': total_score,
                    'matched_sentences': matched_sentences[:5],
                    'keywords_found': matches
                })

        scored_chunks.sort(key=lambda x: x['score'], reverse=True)
        top_chunks = scored_chunks[:top_k]

        if not top_chunks:
            tool_result = f"No results found for keywords: {keywords}"
            context.add_search_event(
                tool_name="keyword_search",
                query={"keywords": keywords, "top_k": top_k},
                results=[],
                metadata={"chunks_found": 0}
            )
            tool_log = {"retrieved_tokens": 0, "chunks_found": 0}
            return tool_result, tool_log

        result_parts = []
        search_results = []

        for item in top_chunks:
            if item['matched_sentences']:
                matched_text = "... " + " ... ".join(item['matched_sentences']) + " ..."
            else:
                matched_text = "(no exact sentence match)"

            search_results.append({
                "chunk_id": item["chunk_id"],
                "score": item["score"],
                "keywords_found": item["keywords_found"],
                "matched_sentences": item["matched_sentences"],
            })

            result_parts.append(f"Chunk ID: {item['chunk_id']}, Matched keywords in chunk: {matched_text}")

        tool_result = "\n\n".join(result_parts)

        all_matched_sentences = []
        for item in top_chunks:
            all_matched_sentences.extend(item['matched_sentences'])

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
                "chunk_ids": [c['chunk_id'] for c in top_chunks]
            }
        )
        context.add_search_event(
            tool_name="keyword_search",
            query={"keywords": keywords, "top_k": top_k},
            results=search_results,
            metadata={
                "chunks_found": len(top_chunks),
                "retrieved_tokens": retrieved_tokens,
            }
        )

        tool_log = {
            "retrieved_tokens": retrieved_tokens,
            "chunks_found": len(top_chunks),
            "chunk_ids": [c["chunk_id"] for c in top_chunks],
        }
        return tool_result, tool_log