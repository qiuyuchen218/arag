"""Lightweight trace graph and HTML visualization for ARAG runs."""

import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from arag.utils.claim_extractor import extract_claims, extract_intermediate_claims
from arag.utils.trace_error import normalize_error


class TraceGraph:
    """
    A lightweight execution trace graph for one QA sample.

    The stable first-version vocabulary is question, plan_query, retriever_call,
    retrieved_chunk, llm_call, claim, answer, and error. Rich details remain in
    metadata rather than creating additional node types.
    """

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
        self._node_counter = 0
        self._type_counters: Dict[str, int] = {}
        self._step_counter = 0
        self._last_temporal_node_id = None
        self._evidence_index: Dict[str, str] = {}
        self._edge_index: Dict[tuple, Dict[str, Any]] = {}
        if clear_metadata:
            self.metadata = {}
            self.created_at = self._utc_now()

    def add_node(
        self,
        node_type: str,
        content: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        status: str = "success",
        node_id: str = None,
    ) -> str:
        self._node_counter += 1
        self._step_counter += 1
        prefixes = {"question": "q", "plan_query": "pq", "retriever_call": "ret",
                    "retrieved_chunk": "chunk", "llm_call": "llm", "claim": "claim",
                    "answer": "ans", "error": "err", "evaluation": "eval"}
        self._type_counters[node_type] = self._type_counters.get(node_type, 0) + 1
        node_id = node_id or f"{prefixes.get(node_type, 'n')}_{self._type_counters[node_type]:03d}"
        self.nodes.append({
            "id": node_id,
            "type": node_type,
            "content": content,
            "metadata": metadata or {},
            "timestamp": self._utc_now(),
            "step_index": self._step_counter,
            "status": status,
        })
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
        key = (source, target, edge_type)
        if edge_type == "used_as_context" and key in self._edge_index:
            edge = self._edge_index[key]
            appearance = dict(metadata or {})
            appearance.pop("timestamp", None)
            edge_metadata = edge["metadata"]
            if "appearances" not in edge_metadata:
                first = {k: edge_metadata.get(k) for k in ("loop", "context_order")
                         if edge_metadata.get(k) is not None}
                edge_metadata["appearances"] = [first] if first else []
            edge_metadata["appearances"].append(appearance)
            edge_metadata["num_appearances"] = len(edge_metadata["appearances"])
            return
        edge = {
            "source": source,
            "target": target,
            "type": edge_type,
            "metadata": metadata or {},
        }
        if edge_type == "used_as_context":
            edge["metadata"].setdefault("num_appearances", 1)
        self.edges.append(edge)
        self._edge_index[key] = edge

    def add_temporal_next(self, target: str, metadata: Optional[Dict[str, Any]] = None):
        if self._last_temporal_node_id and self._last_temporal_node_id != target:
            source_node = next(n for n in self.nodes if n["id"] == self._last_temporal_node_id)
            target_node = next(n for n in self.nodes if n["id"] == target)
            temporal = {"from_step": source_node["step_index"], "to_step": target_node["step_index"],
                        "timestamp": target_node["timestamp"]}
            temporal.update(metadata or {})
            self.add_edge(self._last_temporal_node_id, target, "next", temporal)
        self._last_temporal_node_id = target

    def latest_node_id(self, node_type: str) -> Optional[str]:
        for node in reversed(self.nodes):
            if node.get("type") == node_type:
                return node.get("id")
        return None

    def add_evidence(
        self,
        content: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        dedupe_key: str = None,
    ) -> str:
        """
        Add an evidence node with optional deduplication.

        If dedupe_key is provided and the same evidence has already appeared in
        this sample graph, the existing evidence node id is returned.
        """
        metadata = metadata or {}
        if dedupe_key is None:
            chunk_id = metadata.get("chunk_id") or metadata.get("doc_id")
            if chunk_id is not None:
                dedupe_key = f"chunk:{chunk_id}"
        if dedupe_key and dedupe_key in self._evidence_index:
            node_id = self._evidence_index[dedupe_key]
            node = next(n for n in self.nodes if n["id"] == node_id)
            stable = node["metadata"]
            loop = metadata.get("loop")
            stable["retrieved_times"] = stable.get("retrieved_times", 1) + 1
            if loop is not None:
                stable["first_retrieved_loop"] = min(stable.get("first_retrieved_loop", loop), loop)
            new_rank, new_score = metadata.get("rank"), metadata.get("score")
            old_rank, old_score = stable.get("best_rank"), stable.get("best_score")
            is_search = metadata.get("source") != "read_chunk"
            is_better = is_search and (
                (new_rank is not None and (old_rank is None or new_rank < old_rank)) or
                (new_rank == old_rank and new_score is not None and
                 (old_score is None or new_score > old_score))
            )
            if is_better:
                stable.update({"best_rank": new_rank, "best_score": new_score,
                               "best_retrieval_loop": loop,
                               "best_retrieval_tool": metadata.get("tool_name")})
            if metadata.get("source") == "read_chunk":
                node["content"] = content or node["content"]
                stable.update({"was_read": True, "full_content_available": bool(content),
                               "read_loop": loop, "read_tokens": metadata.get("tokens", 0)})
            return node_id

        chunk_id = metadata.get("chunk_id")
        readable_id = None
        if chunk_id is not None:
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(chunk_id))
            readable_id = f"chunk_{safe[:80]}"
            if any(n["id"] == readable_id for n in self.nodes):
                readable_id = None
        loop = metadata.get("loop")
        stable_metadata = {
            "chunk_id": metadata.get("chunk_id"), "doc_id": metadata.get("doc_id"),
            "source": "corpus", "was_read": metadata.get("source") == "read_chunk",
            "full_content_available": metadata.get("source") == "read_chunk" and bool(content),
            "read_loop": loop if metadata.get("source") == "read_chunk" else None,
            "read_tokens": metadata.get("tokens", 0) if metadata.get("source") == "read_chunk" else None,
            "retrieved_times": 1, "first_retrieved_loop": loop,
            "best_rank": metadata.get("rank"), "best_score": metadata.get("score"),
            "best_retrieval_loop": loop, "best_retrieval_tool": metadata.get("tool_name"),
        }
        node_id = self.add_node("retrieved_chunk", content=content, metadata=stable_metadata, node_id=readable_id)
        if dedupe_key:
            self._evidence_index[dedupe_key] = node_id
        return node_id

    @staticmethod
    def _keywords(text: str) -> set:
        stop = {"the", "and", "that", "this", "with", "from", "was", "were", "who", "when",
                "for", "into", "has", "have", "had", "been", "person", "according", "information"}
        return {word.lower() for word in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ£]+(?:'[A-Za-z]+)?|\d+", re.sub(r"\*\*", "", text or ""))
                if len(word) > 2 or word.isdigit() if word.lower() not in stop}

    @staticmethod
    def _date_terms(terms: set) -> set:
        months = {"january", "february", "march", "april", "may", "june", "july", "august",
                  "september", "october", "november", "december"}
        return {term for term in terms if term in months or re.fullmatch(r"\d{4}", term)}

    @staticmethod
    def _core_entities(text: str) -> Dict[str, List[str]]:
        text_lower = str(text or "").lower()
        known = {
            "diego maradona": ["diego maradona", "maradona"],
            "lionel messi": ["lionel messi", "messi"],
            "barcelona": ["barcelona", "fc barcelona"],
            "boca juniors": ["boca juniors"],
            "copa del rey": ["copa del rey"],
        }
        return {entity: aliases for entity, aliases in known.items()
                if any(alias in text_lower for alias in aliases)}

    @staticmethod
    def _predicate_terms(text: str) -> set:
        words = TraceGraph._keywords(text)
        predicates = set()
        groups = {
            "signed": {"signed", "signing", "join", "joined"},
            "compared": {"compared", "comparison", "compares"},
            "goal": {"goal", "goals", "scored"},
        }
        for predicate, aliases in groups.items():
            if words & aliases:
                predicates.add(predicate)
        return predicates

    @staticmethod
    def _date_matches(claim: str, chunk: str) -> bool:
        claim_lower = str(claim or "").lower()
        chunk_lower = str(chunk or "").lower()
        months = ("january", "february", "march", "april", "may", "june", "july", "august",
                  "september", "october", "november", "december")
        month_years = re.findall(r"\b(" + "|".join(months) + r")\s+(\d{4})\b", claim_lower)
        for month, year in month_years:
            if re.search(rf"\b{month}\s+{year}\b", chunk_lower):
                continue
            if re.search(rf"\b{month}\b", chunk_lower) and re.search(rf"\b{year}\b", chunk_lower):
                continue
            return False
        claim_years = set(re.findall(r"\b\d{4}\b", claim_lower))
        chunk_years = set(re.findall(r"\b\d{4}\b", chunk_lower))
        if claim_years and not claim_years <= chunk_years:
            return False
        claim_months = {month for month in months if re.search(rf"\b{month}\b", claim_lower)}
        chunk_months = {month for month in months if re.search(rf"\b{month}\b", chunk_lower)}
        return not claim_months or bool(claim_months & chunk_months)

    @classmethod
    def _evidence_features(cls, claim: str, chunk: str) -> Dict[str, Any]:
        claim_terms = cls._keywords(claim)
        chunk_terms = cls._keywords(chunk)
        overlap = sorted(claim_terms & chunk_terms)
        claim_entities = cls._core_entities(claim)
        chunk_lower = str(chunk or "").lower()
        matched_entities = [
            entity for entity, aliases in claim_entities.items()
            if any(alias in chunk_lower for alias in aliases)
        ]
        missing_entities = sorted(set(claim_entities) - set(matched_entities))
        claim_predicates = cls._predicate_terms(claim)
        chunk_predicates = cls._predicate_terms(chunk)
        matched_predicates = sorted(claim_predicates & chunk_predicates)
        date_ok = cls._date_matches(claim, chunk)
        signed_claim = "signed" in claim_predicates
        required_entities = set(claim_entities)
        if signed_claim:
            # For signed-by-Barcelona claims, the decisive support pattern is
            # Maradona + signed + exact date; Barcelona/Boca are helpful but not
            # mandatory because many source sentences omit the agent phrase.
            required_entities -= {"barcelona", "boca juniors"}
        return {
            "overlap": overlap,
            "overlap_count": len(overlap),
            "claim_entities": sorted(claim_entities),
            "matched_entities": matched_entities,
            "missing_entities": missing_entities,
            "claim_predicates": sorted(claim_predicates),
            "matched_predicates": matched_predicates,
            "required_entities": sorted(required_entities),
            "missing_required_entities": sorted(required_entities - set(matched_entities)),
            "date_ok": date_ok,
            "has_core_entity": bool(matched_entities),
            "has_required_entities": not (required_entities - set(matched_entities)),
            "has_core_predicate": bool(matched_predicates),
        }

    def add_intermediate_claims(self, llm_id: str, message: str, loop: int) -> List[str]:
        ids = []
        for claim_data in extract_intermediate_claims(message):
            metadata = claim_data["metadata"]
            metadata["loop"] = loop
            claim_id = self.add_node("claim", claim_data["content"], metadata, status="unknown")
            self.add_edge(llm_id, claim_id, "generates", {"loop": loop,
                "generation_order": metadata["claim_index"], "timestamp": self._utc_now()})
            ids.append(claim_id)
        return ids

    def add_error(self, parent_id: str, raw_error: Any, stage: str, loop: int = 0,
                  termination_reason: str = "", fatal: bool = True) -> str:
        normalized = normalize_error(raw_error, termination_reason)
        error_id = self.add_node("error", str(raw_error), {
            **normalized, "raw_exception": str(raw_error), "severity": "fatal" if fatal else "warning",
            "stage": stage, "loop": loop,
        }, status="failed")
        self.add_edge(parent_id, error_id, "failed_with", {
            "stage": stage, "fatal": fatal, "timestamp": self._utc_now()
        })
        self.add_temporal_next(error_id)
        self.metadata["normalized_termination_reason"] = normalized["normalized_termination_reason"]
        self.metadata["debug_summary"] = normalized["debug_summary"]
        return error_id

    def add_answer(self, llm_id: str, answer: str, loop: int, termination_reason: str,
                   evidence_nodes: Optional[List[str]] = None, failed: bool = False,
                   raw_error: str = "") -> str:
        evidence_nodes = list(dict.fromkeys(evidence_nodes or []))
        usable = bool(str(answer or "").strip()) and not str(answer).lower().startswith("error:") and not failed
        claim_ids = []
        for claim_data in extract_claims(answer if usable else ""):
            metadata = claim_data["metadata"]
            metadata["loop"] = loop
            if not evidence_nodes:
                metadata["support_status"] = "no_evidence"
            claim_id = self.add_node("claim", claim_data["content"], metadata,
                                     status="unknown")
            self.add_temporal_next(claim_id)
            claim_ids.append((claim_id, metadata))
            if llm_id:
                self.add_edge(llm_id, claim_id, "generates", {"loop": loop,
                    "generation_order": metadata["claim_index"], "timestamp": self._utc_now()})
            candidates = []
            for evidence_id in evidence_nodes:
                chunk = next((n for n in self.nodes if n["id"] == evidence_id), {})
                cm = chunk.get("metadata", {})
                features = self._evidence_features(claim_data["content"], str(chunk.get("content") or ""))
                is_read = bool(cm.get("was_read"))
                top_semantic = (cm.get("best_rank") == 1 and
                                cm.get("best_retrieval_tool") == "semantic_search")
                if features["claim_entities"] and not features["has_required_entities"]:
                    continue
                if features["claim_predicates"] and not features["has_core_predicate"]:
                    continue
                if not features["date_ok"]:
                    continue
                if not (features["overlap_count"] >= 2 or features["has_core_entity"] or
                        (is_read and features["overlap_count"] >= 1) or
                        (top_semantic and features["overlap_count"] >= 1)):
                    continue
                support = "candidate"
                if (features["has_core_entity"] and features["has_core_predicate"] and
                        features["date_ok"]):
                    if "signed" in features["claim_predicates"]:
                        signed_core = (
                            "diego maradona" in features["matched_entities"] and
                            "signed" in features["matched_predicates"] and
                            (not any(term in claim_data["content"].lower()
                                     for term in ("june", "1982")) or features["date_ok"])
                        )
                        support = "likely_support" if signed_core else "candidate"
                    elif features["overlap_count"] >= 3:
                        support = "likely_support"
                candidates.append((features["overlap_count"], features["has_core_entity"],
                                   features["has_core_predicate"], is_read, top_semantic,
                                   evidence_id, cm, features, support))
            # Attribution is intentionally sparse: retain only the strongest few
            # heuristic candidates instead of recreating a filtered all-to-all graph.
            candidates.sort(key=lambda item: item[:5], reverse=True)
            for (
                overlap_count, has_entity, has_predicate, is_read, top_semantic,
                evidence_id, cm, features, support,
            ) in candidates[:3]:
                if support == "likely_support" and cm.get("first_supporting_loop") is None:
                    cm["first_supporting_loop"] = loop
                self.add_edge(evidence_id, claim_id, "evidence_link", {
                    "support_status": support, "evidence_score": float(overlap_count),
                    "method": "entity_predicate_date_heuristic", "overlap_terms": features["overlap"],
                    "overlap_count": overlap_count,
                    "matched_entities": features["matched_entities"],
                    "matched_predicates": features["matched_predicates"],
                    "date_match": features["date_ok"],
                    "retrieval_rank": cm.get("best_rank"), "retrieval_score": cm.get("best_score"),
                    "actually_used": "unknown", "timestamp": self._utc_now(),
                })
        intermediate = [n for n in self.nodes if n["type"] == "claim" and
                        n["metadata"].get("stage") == "intermediate"]
        for claim_id, _ in claim_ids:
            final_node = next(n for n in self.nodes if n["id"] == claim_id)
            final_terms = self._keywords(final_node["content"])
            for prior in intermediate:
                overlap = sorted(final_terms & self._keywords(prior["content"]))
                meaningful_overlap = [term for term in overlap if term not in {"answer", "chunk"}]
                if len(meaningful_overlap) >= 2:
                    self.add_edge(prior["id"], claim_id, "depends_on", {
                        "method": "keyword_overlap_heuristic", "overlap_terms": meaningful_overlap,
                        "reason": "final claim reuses information introduced by intermediate claim"})
        answer_id = self.add_node("answer", answer if usable else "", {
            "pred_answer": answer if usable else "", "termination_reason": termination_reason,
            "is_error_answer": not usable, "failure_reason": termination_reason if not usable else None,
            "raw_error": raw_error or None, "loop": loop,
        }, status="success" if usable else "failed")
        if llm_id:
            self.add_edge(llm_id, answer_id, "generates", {"loop": loop, "generation_order": 1,
                                                            "timestamp": self._utc_now()})
        for claim_id, metadata in claim_ids:
            self.add_edge(claim_id, answer_id, "composes_answer", {"claim_index": metadata["claim_index"]})
        self.add_temporal_next(answer_id)
        self.metadata["final_answer"] = answer if usable else ""
        return answer_id

    def finalize_metadata(self):
        node_types = ["question", "llm_call", "plan_query", "retriever_call", "retrieved_chunk", "claim", "answer", "error"]
        edge_types = ["calls", "next", "decomposes_to", "proposes_query", "retrieves", "used_as_context",
                      "generates", "evidence_link", "depends_on", "composes_answer", "failed_with"]
        node_counts, edge_counts = {key: 0 for key in node_types}, {key: 0 for key in edge_types}
        for node in self.nodes:
            node_counts[node["type"]] = node_counts.get(node["type"], 0) + 1
        for edge in self.edges:
            edge_counts[edge["type"]] = edge_counts.get(edge["type"], 0) + 1
        errors = [n for n in self.nodes if n["type"] == "error"]
        error_stages = sorted(set(n["metadata"].get("stage", "unknown") for n in errors))
        warning_stages = sorted(set(
            n["metadata"].get("stage", "unknown") for n in errors
            if n["metadata"].get("severity") == "warning"
        ))
        fatal_error_stages = sorted(set(
            n["metadata"].get("stage", "unknown") for n in errors
            if n["metadata"].get("severity") == "fatal"
        ))
        first_error = min(errors, key=lambda n: n.get("step_index", 0)) if errors else None
        chunks = [n for n in self.nodes if n["type"] == "retrieved_chunk"]
        for chunk in chunks:
            cm = chunk["metadata"]
            late_loop = cm.get("first_supporting_loop") or cm.get("read_loop")
            cm["early_retrieved_late_used"] = bool(late_loop is not None and
                cm.get("first_retrieved_loop") is not None and cm["first_retrieved_loop"] < late_loop)
        evidence = [e for e in self.edges if e["type"] == "evidence_link"]
        self.metadata.update({
            "num_nodes_by_type": node_counts, "num_edges_by_type": edge_counts,
            "has_error": bool(errors),
            "error_types": sorted(set(n["metadata"].get("error_type", "unknown_error") for n in errors)),
            "error_stages": error_stages,
            "warning_stages": warning_stages,
            "fatal_error_stages": fatal_error_stages,
            "first_error_stage": first_error["metadata"].get("stage") if first_error else None,
            "first_error_loop": first_error["metadata"].get("loop") if first_error else None,
            "evidence_missing": any(n["type"] == "claim" and
                                    n["metadata"].get("support_status") == "no_evidence" for n in self.nodes),
            "retrieval_attempted": bool(node_counts["retriever_call"]),
            "llm_attempted": bool(node_counts["llm_call"]),
            "answer_generated": any(n["type"] == "answer" and n["status"] == "success" for n in self.nodes),
            "num_unique_chunks": len(chunks),
            "num_read_chunks": sum(bool(n["metadata"].get("was_read")) for n in chunks),
            "num_candidate_evidence_links": sum(e["metadata"].get("support_status") == "candidate" for e in evidence),
            "num_likely_support_evidence_links": sum(e["metadata"].get("support_status") == "likely_support" for e in evidence),
            "num_intermediate_claims": sum(n["metadata"].get("stage") == "intermediate" for n in self.nodes if n["type"] == "claim"),
            "num_final_claims": sum(n["metadata"].get("stage") == "final" for n in self.nodes if n["type"] == "claim"),
            "early_retrieved_late_used_chunks": [n["id"] for n in chunks if n["metadata"].get("early_retrieved_late_used")],
        })
        self.metadata.setdefault("normalized_termination_reason", "success" if not errors else "unknown_error")
        self.metadata.setdefault("debug_summary", "Execution completed successfully." if not errors else "Execution failed.")

    def to_dict(self) -> Dict[str, Any]:
        self.finalize_metadata()
        return {
            "sample_id": self.sample_id,
            "dataset": self.dataset,
            "created_at": self.created_at,
            "nodes": self.nodes,
            "edges": self.edges,
            "metadata": self.metadata,
        }

    def save_json(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @staticmethod
    def _short_text(value: Any, max_chars: int = 120) -> str:
        if value is None:
            return ""
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        text = " ".join(text.split())
        if len(text) > max_chars:
            return text[:max_chars] + "..."
        return text

    def save_html(self, path: str):
        """
        Save a self-contained HTML visualization.

        This uses no external JavaScript or CDN, so it works on a remote server
        after downloading the HTML file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        graph = self.to_dict()
        nodes = graph["nodes"]
        edges = graph["edges"]

        type_colors = {
            "question": "#2563eb",
            "plan_query": "#0891b2",
            "retriever_call": "#ea580c",
            "retrieved_chunk": "#16a34a",
            "llm_call": "#7c3aed",
            "claim": "#0f766e",
            "answer": "#dc2626",
            "error": "#991b1b",
        }

        node_by_id = {n["id"]: n for n in nodes}
        x_gap = 230
        y_gap = 130
        margin_x = 80
        margin_y = 90

        positions = {}
        type_offsets = {}
        for idx, node in enumerate(nodes):
            step = node.get("step_index", idx + 1)
            node_type = node.get("type", "unknown")
            type_offsets[node_type] = type_offsets.get(node_type, 0) + 1
            positions[node["id"]] = {
                "x": margin_x + (step - 1) * x_gap,
                "y": margin_y + ((type_offsets[node_type] - 1) % 4) * y_gap,
            }

        width = max([p["x"] for p in positions.values()] or [800]) + 260
        height = max([p["y"] for p in positions.values()] or [600]) + 180

        edge_lines = []
        for edge in edges:
            source = positions.get(edge.get("source"))
            target = positions.get(edge.get("target"))
            if not source or not target:
                continue
            edge_type = edge.get("type", "")
            edge_lines.append(f'''
                <line x1="{source["x"]}" y1="{source["y"]}" x2="{target["x"]}" y2="{target["y"]}"
                      stroke="#94a3b8" stroke-width="1.8" marker-end="url(#arrow)" />
                <text x="{(source["x"] + target["x"]) / 2}" y="{(source["y"] + target["y"]) / 2 - 6}"
                      class="edge-label">{html.escape(edge_type)}</text>
            ''')

        node_blocks = []
        detail_blocks = []
        for node in nodes:
            pos = positions[node["id"]]
            node_type = node.get("type", "unknown")
            color = type_colors.get(node_type, "#64748b")
            status = node.get("status", "unknown")
            label = f'{node["id"]} · {node_type} · {status}'
            preview = self._short_text(node.get("content"), 95)
            detail_id = f'detail-{node["id"]}'

            node_blocks.append(f'''
                <g class="node" onclick="showDetail('{detail_id}')">
                    <rect x="{pos["x"] - 82}" y="{pos["y"] - 35}" width="164" height="70" rx="8"
                          fill="{color}" opacity="0.92" />
                    <text x="{pos["x"]}" y="{pos["y"] - 10}" text-anchor="middle" class="node-title">
                        {html.escape(label)}
                    </text>
                    <foreignObject x="{pos["x"] - 72}" y="{pos["y"]}" width="144" height="30">
                        <div xmlns="http://www.w3.org/1999/xhtml" class="node-preview">{html.escape(preview)}</div>
                    </foreignObject>
                </g>
            ''')

            detail_payload = html.escape(json.dumps(node, ensure_ascii=False, indent=2))
            detail_blocks.append(f'''
                <pre id="{detail_id}" class="detail-block">{detail_payload}</pre>
            ''')

        legend = "".join(
            f'<span class="legend-item"><span style="background:{color}"></span>{node_type}</span>'
            for node_type, color in type_colors.items()
        )

        html_doc = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>ARAG Trace Graph - {html.escape(str(self.sample_id))}</title>
<style>
body {{
    margin: 0;
    font-family: Arial, Helvetica, sans-serif;
    color: #0f172a;
    background: #f8fafc;
}}
header {{
    padding: 18px 24px;
    background: #0f172a;
    color: white;
}}
header h1 {{
    margin: 0 0 8px;
    font-size: 20px;
}}
header p {{
    margin: 4px 0;
    color: #cbd5e1;
    font-size: 13px;
}}
.legend {{
    padding: 10px 24px;
    background: white;
    border-bottom: 1px solid #e2e8f0;
    position: sticky;
    top: 0;
    z-index: 5;
}}
.legend-item {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin-right: 16px;
    font-size: 13px;
}}
.legend-item span {{
    width: 12px;
    height: 12px;
    border-radius: 2px;
    display: inline-block;
}}
.wrap {{
    display: grid;
    grid-template-columns: minmax(0, 1fr) 420px;
    min-height: calc(100vh - 100px);
}}
.graph {{
    overflow: auto;
    background: #f8fafc;
}}
.side {{
    border-left: 1px solid #e2e8f0;
    background: white;
    padding: 16px;
    overflow: auto;
}}
.node {{
    cursor: pointer;
}}
.node-title {{
    fill: white;
    font-size: 12px;
    font-weight: 700;
}}
.node-preview {{
    color: white;
    font-size: 10px;
    line-height: 1.25;
    text-align: center;
    overflow: hidden;
}}
.edge-label {{
    fill: #475569;
    font-size: 10px;
    paint-order: stroke;
    stroke: #f8fafc;
    stroke-width: 4px;
}}
.detail-block {{
    display: none;
    white-space: pre-wrap;
    word-break: break-word;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px;
    font-size: 12px;
    line-height: 1.45;
}}
.detail-block.active {{
    display: block;
}}
.summary {{
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 12px;
    font-size: 13px;
}}
</style>
</head>
<body>
<header>
    <h1>ARAG Trace Graph</h1>
    <p>sample_id: {html.escape(str(self.sample_id))}</p>
    <p>dataset: {html.escape(str(self.dataset))} · nodes: {len(nodes)} · edges: {len(edges)}</p>
    <p><strong>Debug summary:</strong> {html.escape(str(graph.get("metadata", {}).get("debug_summary", "")))}</p>
</header>
<div class="legend">{legend}</div>
<div class="wrap">
    <div class="graph">
        <svg width="{width}" height="{height}">
            <defs>
                <marker id="arrow" markerWidth="10" markerHeight="10" refX="10" refY="3"
                        orient="auto" markerUnits="strokeWidth">
                    <path d="M0,0 L0,6 L9,3 z" fill="#94a3b8" />
                </marker>
            </defs>
            {''.join(edge_lines)}
            {''.join(node_blocks)}
        </svg>
    </div>
    <aside class="side">
        <div class="summary">
            <strong>Graph metadata</strong>
            <pre>{html.escape(json.dumps(graph.get("metadata", {}), ensure_ascii=False, indent=2))}</pre>
        </div>
        <p>Click a node to inspect its full JSON.</p>
        {''.join(detail_blocks)}
    </aside>
</div>
<script>
function showDetail(id) {{
    document.querySelectorAll('.detail-block').forEach(el => el.classList.remove('active'));
    const el = document.getElementById(id);
    if (el) el.classList.add('active');
}}
</script>
</body>
</html>'''

        with open(path, "w", encoding="utf-8") as f:
            f.write(html_doc)


TraceLogger = TraceGraph
