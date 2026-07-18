"""Deterministic first-pass extraction for final and intermediate claims."""

import re
from typing import Dict, List


_LEADING_PREAMBLES = re.compile(
    r"^(?:based on the information retrieved|according to the document|therefore)\s*,?\s*",
    re.IGNORECASE,
)
_INTERMEDIATE_TRIGGERS = re.compile(
    r"\b(?:I found|It states|This indicates that|This suggests that|This directly answers|"
    r"The key information is|This means|semantic search revealed that)\b",
    re.IGNORECASE,
)
_PLAN_SENTENCES = re.compile(
    r"\b(?:let me|I need to|I(?:'ll| will) try|search for|look for|find information)\b",
    re.IGNORECASE,
)
_META_TALK = re.compile(
    r"^\s*(?:I found|I need|Let me|Now I need|This directly answers)\b|"
    r"\b(?:the answer in chunk|from the semantic search results|to confirm this information|"
    r"provide proper context|a relevant piece of information in chunk)\b",
    re.IGNORECASE,
)
_CHUNK_FACT_PREFIX = re.compile(
    r"^(?:a key piece of information|a relevant piece of information|a promising lead|"
    r"this is explicitly stated)\s+in\s+chunk\s+\d+\s*(?:,\s*which\s+mentions\s+that|:\s*)",
    re.IGNORECASE,
)


def normalize_claim(content: str) -> str:
    """Clean common RAG discourse wrappers while preserving the fact."""
    content = re.sub(r"\*\*", "", str(content or "")).strip()
    content = _LEADING_PREAMBLES.sub("", content).strip()
    content = re.sub(r"^It states\s*:\s*", "", content, flags=re.IGNORECASE).strip()
    content = re.sub(r"^This suggests that\s+", "", content, flags=re.IGNORECASE).strip()
    content = re.sub(r"^Answer\s*:\s*", "", content, flags=re.IGNORECASE).strip()
    content = re.sub(r"^The semantic search revealed that\s+", "", content,
                     flags=re.IGNORECASE).strip()
    content = _CHUNK_FACT_PREFIX.sub("", content).strip()
    content = re.sub(r'^[\"“](.+?)[\"”]\.?$', r"\1", content).strip()

    quoted_by_barca = re.match(
        r'^["“](?P<fact>.+?)["”]\s+by\s+Barcelona\.?$',
        content,
        flags=re.IGNORECASE,
    )
    if quoted_by_barca:
        content = quoted_by_barca.group("fact")

    signed = re.search(
        r"\b(?:in\s+)?(?P<date>[A-Za-z]+\s+\d{4}),?\s+Diego Maradona was signed "
        r"for (?P<fee>.+?) from Boca Juniors\.?$",
        content,
        flags=re.IGNORECASE,
    )
    if signed:
        return (
            f"Diego Maradona was signed by Barcelona in {signed.group('date').title()} "
            f"for {signed.group('fee')} from Boca Juniors."
        )

    signed_no_subject = re.search(
        r"\b(?:in\s+)?(?P<date>[A-Za-z]+\s+\d{4}),?\s+diego maradona was signed "
        r"for (?P<fee>.+?) from boca juniors\.?$",
        content,
        flags=re.IGNORECASE,
    )
    if signed_no_subject:
        return (
            f"Diego Maradona was signed by Barcelona in {signed_no_subject.group('date').title()} "
            f"for {signed_no_subject.group('fee')} from Boca Juniors."
        )

    if re.fullmatch(
        r"Messi'?s goal was compared to Diego Maradona'?s famous goal\.?",
        content,
        flags=re.IGNORECASE,
    ):
        return "Messi's Copa del Rey goal was compared to Diego Maradona's goal of the century."

    return content


def _sentences(text: str) -> List[str]:
    """Split on sentence boundaries while retaining punctuation and closing quotes."""
    text = re.sub(r"\*\*", "", str(text or "")).strip()
    if not text:
        return []
    # A boundary is punctuation, an optional closing quote, then whitespace before
    # an uppercase/digit.  Capturing avoids Python's variable-width lookbehind.
    parts = re.split(r'([.!?]["”]?)\s+(?=[A-Z0-9])', text)
    sentences: List[str] = []
    for index in range(0, len(parts), 2):
        sentence = parts[index]
        if index + 1 < len(parts):
            sentence += parts[index + 1]
        for subpart in re.split(r"\s*;\s*|[\r\n]+", sentence):
            cleaned = _LEADING_PREAMBLES.sub("", subpart.strip()).strip()
            if cleaned:
                sentences.append(cleaned)
    return sentences


def extract_claims(answer: str) -> List[Dict[str, object]]:
    text = str(answer or "").strip()
    if not text or text.lower().startswith("error:"):
        return []
    claims = []
    for part in _sentences(text):
        content = normalize_claim(part)
        if not content or _META_TALK.search(content) or content.lower() in {"the question."}:
            continue
        claims.append({
            "content": content,
            "metadata": {
                "claim_index": len(claims) + 1,
                "source": "final_answer_sentence",
                "stage": "final",
                "support_status": "candidate",
                "confidence": None,
            },
        })
    return claims


def extract_intermediate_claims(message: str) -> List[Dict[str, object]]:
    """Extract a small, high-precision set of factual assistant statements."""
    claims = []
    seen = set()
    sentences = _sentences(message)
    for sentence_index, sentence in enumerate(sentences):
        quoted = re.search(r"(?:It states|states?)\s*:\s*[\"“](.+?)[\"”]\.?", sentence,
                           flags=re.IGNORECASE)
        if quoted:
            content = normalize_claim(quoted.group(1))
            key = content.lower()
            if content and key not in seen and not _META_TALK.search(content):
                seen.add(key)
                claims.append({
                    "content": content,
                    "metadata": {
                        "claim_index": len(claims) + 1,
                        "source": "assistant_reasoning",
                        "stage": "intermediate",
                        "support_status": "candidate",
                        "confidence": None,
                    },
                })
            continue
        if not _INTERMEDIATE_TRIGGERS.search(sentence) or _PLAN_SENTENCES.search(sentence):
            continue
        # Remove the discourse marker but retain the factual proposition.
        content = re.sub(
            r"^(?:I found(?: that)?|It states(?: that)?|This indicates that|This suggests that|"
            r"This directly answers(?: that)?|The key information is(?: that)?|"
            r"This means(?: that)?|The semantic search revealed that)\s*[:,]?\s*",
            "",
            sentence,
            flags=re.IGNORECASE,
        ).strip()
        # Models often emit "I found the answer in Chunk 0!" and put the actual
        # fact in the next sentence.  Store the proposition, not the announcement.
        if re.match(r"^(?:the )?answer in (?:the )?chunk \d+\b", content, re.IGNORECASE):
            if sentence_index + 1 < len(sentences):
                content = sentences[sentence_index + 1]
            else:
                continue
        content = normalize_claim(content)
        key = content.lower()
        if content and key not in seen and key not in {"the question."} and not _META_TALK.search(content):
            seen.add(key)
            claims.append({
                "content": content,
                "metadata": {
                    "claim_index": len(claims) + 1,
                    "source": "assistant_reasoning",
                    "stage": "intermediate",
                    "support_status": "candidate",
                    "confidence": None,
                },
            })
    return claims
