"""Semantic search tool - embedding-based similarity matching."""

import os
import pickle
import threading
import hashlib
import numpy as np
from typing import Dict, List, Any, Tuple, TYPE_CHECKING

from arag.core.schemas import EvidenceSpan, SearchHit, ToolResult, artifact_id, content_fingerprint, result_dict, stable_hash
from arag.tools.base import BaseTool

if TYPE_CHECKING:
    from arag.core.context import AgentContext

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class SemanticSearchTool(BaseTool):
    """Semantic search using embedding similarity."""

    _embedding_lock = threading.Lock()

    def __init__(
        self,
        chunks_file: str,
        index_dir: str = "index",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = None
    ):
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("sentence-transformers required. Install: pip install sentence-transformers")

        if not HAS_TIKTOKEN:
            raise ImportError("tiktoken required. Install: pip install tiktoken")

        self.chunks_file = chunks_file
        self.index_dir = index_dir
        self.model_name = model_name
        self.device = self._resolve_device(device)

        self.embedding_model = SentenceTransformer(model_name, device=self.device)
        self._load_index()
        self.corpus_version = content_fingerprint({
            "chunks_file": chunks_file,
            "chunks": self.chunks,
        })
        self.index_version = content_fingerprint({
            "index_dir": index_dir,
            "model_name": model_name,
            "corpus_version": self.corpus_version,
            "sentences": self.sentences,
            "sentence_to_chunk": self.sentence_to_chunk,
            "embeddings_shape": getattr(self.embeddings, "shape", None),
            "embeddings_dtype": str(getattr(self.embeddings, "dtype", "")),
            "embeddings_hash": hashlib.sha256(getattr(self.embeddings, "tobytes", lambda: b"")()).hexdigest(),
        })
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")

    @property
    def name(self) -> str:
        return "semantic_search"

    @staticmethod
    def _resolve_device(device: str = None) -> str:
        if device and str(device).startswith("cuda"):
            try:
                import torch
                if not torch.cuda.is_available():
                    print(f"Warning: requested device {device} but CUDA is unavailable; falling back to cpu")
                    return "cpu"
            except Exception:
                print(f"Warning: requested device {device} but CUDA availability could not be checked; falling back to cpu")
                return "cpu"
        return device

    def _load_index(self):
        index_file = os.path.join(self.index_dir, "sentence_index.pkl")

        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Index not found: {index_file}")

        with open(index_file, 'rb') as f:
            index_data = pickle.load(f)

        self.sentences = index_data['sentences']
        self.embeddings = index_data['embeddings']
        self.sentence_to_chunk = index_data['sentence_to_chunk']
        self.chunks = index_data['chunks']

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "semantic_search",
                "description": """Semantic search using embedding similarity. Matches your query against sentences in each chunk via vector similarity.

WHEN TO USE:
- When keyword search fails to find relevant information
- When exact wording in documents is unknown
- For conceptual/meaning-based matching

RETURNS: Abbreviated snippets with matched sentences. Use read_chunk to get full text for answering.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query describing what information you're looking for"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of most relevant results to return (default: 5, max: 20)",
                            "default": 5
                        },
                        "epistemic_context": {
                            "type": "object",
                            "description": "Required public decision context. Include at least one proposition that this retrieval is exploring, testing, verifying, committing, or using as a premise.",
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
                    "required": ["query", "epistemic_context"]
                }
            }
        }

    def execute(self, context: 'AgentContext', query: str, top_k: int = 5, candidate_pool: int = None) -> Tuple[str, Dict[str, Any]]:
        top_k = min(top_k, 20)
        candidate_pool = max(candidate_pool or top_k * 5, top_k)
        call_id = f"sem_{stable_hash(getattr(context, 'branch_id', 'b0'), query, top_k, len(context.search_history))}"

        with self._embedding_lock:
            query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)[0]

        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:candidate_pool]

        chunk_sentences = {}
        for idx in top_indices:
            sentence = self.sentences[idx]
            chunk_id = self.sentence_to_chunk[idx]
            similarity = float(similarities[idx])

            if chunk_id not in chunk_sentences:
                chunk_sentences[chunk_id] = []
            chunk_sentences[chunk_id].append({
                'sentence': sentence,
                'similarity': similarity,
                'position': int(idx)
            })

        chunk_scores = []
        for chunk_id, sents in chunk_sentences.items():
            max_similarity = max(s['similarity'] for s in sents)
            chunk_scores.append((chunk_id, max_similarity, sents))

        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        diverse = []
        seen_docs = set()
        for candidate_rank, item in enumerate(chunk_scores, start=1):
            chunk_id = item[0]
            doc_id = str(self.chunks[chunk_id].get("doc_id", chunk_id)) if isinstance(self.chunks, dict) else str(chunk_id)
            if doc_id in seen_docs and len(diverse) >= max(1, top_k // 2):
                continue
            seen_docs.add(doc_id)
            diverse.append((candidate_rank, *item))
            if len(diverse) >= top_k:
                break
        top_chunks = diverse

        if not top_chunks:
            context.add_search_event(
                tool_name="semantic_search",
                query={"query": query, "top_k": top_k},
                results=[],
                metadata={"chunks_found": 0}
            )
            result = ToolResult(call_id, self.name, "empty", f"No results for: {query}", [], {"chunks_found": 0, "candidate_pool": candidate_pool}, 0, 0)
            return result.to_legacy_tuple()

        result_parts = []
        search_results = []

        for final_rank, (candidate_rank, chunk_id, max_sim, sents) in enumerate(top_chunks, start=1):
            chunk_text = self.chunks[chunk_id]['text']
            sents_sorted = sorted(sents, key=lambda x: chunk_text.find(x['sentence']))
            matched_text = "... " + " ... ".join([s['sentence'] for s in sents_sorted]) + " ..."
            spans = [
                EvidenceSpan.from_text(
                    s["sentence"],
                    self.corpus_version,
                    self.chunks[chunk_id].get("doc_id", chunk_id),
                    chunk_id,
                    int(s.get("position", 0)),
                    max(chunk_text.find(s["sentence"]), 0),
                    max(chunk_text.find(s["sentence"]), 0) + len(s["sentence"]),
                )
                for s in sents_sorted[:5]
            ]
            hit = SearchHit(
                artifact_id=artifact_id(self.corpus_version, self.chunks[chunk_id].get("doc_id", chunk_id), chunk_id),
                doc_id=str(self.chunks[chunk_id].get("doc_id", chunk_id)),
                chunk_id=str(chunk_id),
                sentence_ids=[int(s.get("position", 0)) for s in sents_sorted],
                matched_spans=spans,
                retrieval_channel="dense",
                raw_score=float(max_sim),
                candidate_rank=candidate_rank,
                final_rank=final_rank,
                metadata={"score_semantics": "raw_cosine_similarity_not_probability"},
            )
            item_dict = result_dict(hit)
            item_dict["score"] = max_sim
            item_dict["matched_sentences"] = sents_sorted
            search_results.append(item_dict)
            result_parts.append(f"Chunk ID: {chunk_id} (Similarity: {max_sim:.3f})\nMatched: {matched_text}")

        tool_result = "\n\n".join(result_parts)

        all_matched = []
        for _, _, _, sents in top_chunks:
            all_matched.extend([s['sentence'] for s in sents])

        retrieved_tokens = len(self.tokenizer.encode("\n".join(all_matched))) if all_matched else 0

        context.add_retrieval_log(
            tool_name="semantic_search",
            tokens=retrieved_tokens,
            metadata={
                "query": query,
                "chunks_found": len(top_chunks),
                "chunk_ids": [chunk_id for _, chunk_id, _, _ in top_chunks],
            }
        )
        context.add_search_event(
            tool_name="semantic_search",
            query={"query": query, "top_k": top_k},
            results=search_results,
            metadata={
                "chunks_found": len(top_chunks),
                "retrieved_tokens": retrieved_tokens,
                "candidate_pool": candidate_pool,
                "embedding_model": self.model_name,
                "index_version": self.index_version,
                "score_semantics": "raw_cosine_similarity_not_probability",
            }
        )

        return ToolResult(
            call_id=call_id,
            tool_name=self.name,
            status="success",
            rendered_text=tool_result,
            results=search_results,
            diagnostics={
                "retrieved_tokens": retrieved_tokens,
                "chunks_found": len(top_chunks),
                "candidate_pool": candidate_pool,
                "embedding_model": self.model_name,
                "index_version": self.index_version,
                "corpus_version": self.corpus_version,
                "score_semantics": "raw_cosine_similarity_not_probability",
                "fusion": "dense_only",
                "reranker": None,
            },
            retrieved_tokens=retrieved_tokens,
        ).to_legacy_tuple()
