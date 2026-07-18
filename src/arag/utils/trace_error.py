"""Error classification for execution trace graphs."""

import re
from typing import Any, Dict


def normalize_error(raw_error: Any = "", termination_reason: str = "") -> Dict[str, str]:
    raw = str(raw_error or "")
    reason = str(termination_reason or "")
    text = f"{raw} {reason}".lower()

    error_type, subtype, normalized = "unknown_error", "unknown", "unknown_error"
    if ("proxyerror" in text or "proxy" in text and "connection refused" in text
            or "127.0.0.1" in text and "connection refused" in text):
        error_type, subtype, normalized = (
            "infrastructure_error", "proxy_connection_refused", "infrastructure_error"
        )
    elif "timeout" in text or "timed out" in text:
        error_type, subtype, normalized = "llm_api_error", "timeout", "infrastructure_error"
    elif "rate limit" in text or "ratelimit" in text or "429" in text:
        error_type, subtype, normalized = "llm_api_error", "rate_limit", "infrastructure_error"
    elif "unauthorized" in text or "authentication" in text or "401" in text:
        error_type, subtype, normalized = "llm_api_error", "authentication_error", "infrastructure_error"
    elif "max retries exceeded" in text or "connection refused" in text:
        error_type, subtype, normalized = "llm_api_error", "unknown", "infrastructure_error"
    elif "empty retrieval" in text or "empty_tool_result" in text or "no retrieval" in text:
        error_type, subtype, normalized = "no_retrieval", "empty_tool_result", "retrieval_error"
    elif "empty answer" in text:
        error_type, subtype, normalized = "generation_error", "empty_answer", "generation_error"
    elif "max_loops_exceeded" in text:
        error_type, subtype, normalized = "max_loops_exceeded", "unknown", "max_loops_exceeded"

    if subtype == "proxy_connection_refused":
        summary = "LLM call failed before retrieval because the local proxy connection was refused."
        proxy = re.search(r"127\.0\.0\.1(?::\d+)?", raw)
        if proxy:
            summary = f"LLM call failed before retrieval because the local proxy {proxy.group(0)} was refused."
    elif error_type == "no_retrieval":
        summary = "The retriever call completed but returned no evidence."
    elif subtype == "empty_answer":
        summary = "The LLM call completed without a usable answer."
    else:
        summary = f"Execution failed during {error_type.replace('_', ' ')}: {raw or reason}".strip()

    return {
        "error_type": error_type,
        "error_subtype": subtype,
        "normalized_termination_reason": normalized,
        "debug_summary": summary,
    }
