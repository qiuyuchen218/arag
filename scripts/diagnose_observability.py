#!/usr/bin/env python3
import argparse
import json
import re
import string
from pathlib import Path


def normalize_answer(s):
    if s is None:
        return ""
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def flatten_search_text(pred):
    parts = []
    for event in pred.get("search_history", []):
        for result in event.get("results", []):
            parts.append(str(result.get("chunk_id", "")))
            for sent in result.get("matched_sentences", []):
                if isinstance(sent, dict):
                    parts.append(sent.get("sentence", ""))
                else:
                    parts.append(str(sent))
    return "\n".join(parts)


def flatten_read_text(pred):
    parts = []
    for chunk in pred.get("read_chunks", {}).values():
        parts.append(chunk.get("content", ""))
    return "\n".join(parts)


def flatten_assistant_trace(pred):
    parts = []
    for event in pred.get("message_trace", []):
        if event.get("role") == "assistant":
            parts.append(event.get("content", ""))
    return "\n".join(parts)


def classify(pred):
    gold = pred.get("gold_answer") or pred.get("answer") or ""
    pred_answer = pred.get("pred_answer", "")

    gold_norm = normalize_answer(gold)
    pred_norm = normalize_answer(pred_answer)

    search_text_norm = normalize_answer(flatten_search_text(pred))
    read_text_norm = normalize_answer(flatten_read_text(pred))
    trace_text_norm = normalize_answer(flatten_assistant_trace(pred))

    searched_gold = bool(gold_norm and gold_norm in search_text_norm)
    read_gold = bool(gold_norm and gold_norm in read_text_norm)
    trace_mentions_gold = bool(gold_norm and gold_norm in trace_text_norm)
    answer_contains_gold = bool(gold_norm and gold_norm in pred_norm)

    chunks_read_count = pred.get("chunks_read_count", 0)
    searched_chunks = []
    for event in pred.get("search_history", []):
        for result in event.get("results", []):
            cid = result.get("chunk_id")
            if cid is not None:
                searched_chunks.append(str(cid))

    if answer_contains_gold:
        label = "correct_exact"
    elif read_gold:
        label = "read_gold_but_answer_wrong"
    elif searched_gold and chunks_read_count == 0:
        label = "searched_gold_but_did_not_read"
    elif searched_gold:
        label = "searched_gold_but_read_wrong_chunk_or_reasoned_wrong"
    elif chunks_read_count == 0 and pred_answer:
        label = "no_chunk_read_but_answered"
    elif chunks_read_count == 0:
        label = "no_chunk_read_and_failed"
    else:
        label = "retrieval_or_evidence_missing"

    return {
        "qid": pred.get("qid"),
        "gold_answer": gold,
        "pred_answer": pred_answer,
        "label": label,
        "searched_gold_string": searched_gold,
        "read_gold_string": read_gold,
        "trace_mentions_gold": trace_mentions_gold,
        "answer_contains_gold": answer_contains_gold,
        "searched_chunk_ids": sorted(set(searched_chunks), key=searched_chunks.index),
        "chunks_read_ids": pred.get("chunks_read_ids", []),
        "termination_reason": pred.get("termination_reason", ""),
        "loops": pred.get("loops", 0),
        "llm_accuracy": pred.get("llm_accuracy"),
        "contain_accuracy": pred.get("contain_accuracy"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rows = [classify(pred) for pred in load_jsonl(args.predictions)]

    counts = {}
    for row in rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1

    print("Diagnosis summary")
    print("=" * 60)
    for label, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{label}: {count}")

    if args.output:
        out = Path(args.output)
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nSaved diagnosis to: {out}")


if __name__ == "__main__":
    main()