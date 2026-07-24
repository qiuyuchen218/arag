"""Generic optional claim splitting helpers.

The execution trace graph does not call this module. It remains as a small,
dataset-agnostic compatibility helper for older imports and offline experiments.
"""

import re
from typing import Dict, List


_LEADING_PREAMBLES = re.compile(
    r"^(?:based on the information retrieved|according to the document|therefore|answer)\s*:?,?\s*",
    re.IGNORECASE,
)
_META_TALK = re.compile(
    r"\b(?:let me|I need to|I will search|I'll search|tool call|search results?|chunk \d+)\b",
    re.IGNORECASE,
)


def normalize_claim(content: str) -> str:
    """Remove common answer wrappers without adding domain-specific meaning."""
    content = re.sub(r"\*\*", "", str(content or "")).strip()
    content = _LEADING_PREAMBLES.sub("", content).strip()
    content = re.sub(r"^(?:it states|this indicates|this suggests)\s+(?:that\s+)?", "", content,
                     flags=re.IGNORECASE).strip()
    content = re.sub(r'^[\"“](.+?)[\"”]\.?$', r"\1", content).strip()
    return content


def _sentences(text: str) -> List[str]:
    text = re.sub(r"\*\*", "", str(text or "")).strip()
    if not text:
        return []
    parts = re.split(r'([.!?]["”]?)\s+(?=[A-Z0-9])', text)
    sentences: List[str] = []
    for index in range(0, len(parts), 2):
        sentence = parts[index]
        if index + 1 < len(parts):
            sentence += parts[index + 1]
        for subpart in re.split(r"\s*;\s*|[\r\n]+", sentence):
            cleaned = normalize_claim(subpart)
            if cleaned:
                sentences.append(cleaned)
    return sentences


def extract_claims(answer: str) -> List[Dict[str, object]]:
    text = str(answer or "").strip()
    if not text or text.lower().startswith("error:"):
        return []
    claims = []
    for part in _sentences(text):
        if _META_TALK.search(part):
            continue
        claims.append({
            "content": part,
            "metadata": {
                "claim_index": len(claims) + 1,
                "source": "final_answer_sentence",
                "stage": "final",
            },
        })
    return claims


def extract_intermediate_claims(message: str) -> List[Dict[str, object]]:
    claims = []
    seen = set()
    for part in _sentences(message):
        key = part.lower()
        if key in seen or _META_TALK.search(part):
            continue
        seen.add(key)
        claims.append({
            "content": part,
            "metadata": {
                "claim_index": len(claims) + 1,
                "source": "assistant_message_sentence",
                "stage": "intermediate",
            },
        })
    return claims
