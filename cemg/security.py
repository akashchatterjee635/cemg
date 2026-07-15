"""
cemg/security.py
-----------------
Content sanitisation for anything written to CEMG that originated from
an external tool result (a webpage, an email, a document, an API response).

Why this exists: unlike a normal prompt injection which only affects the
CURRENT turn, a poisoned CEMG entry is stored and re-injected into every
FUTURE session that retrieves it as trusted system-prompt context, until
it decays out. This is an active attack class against agent memory
systems -- the mitigation here is a defense-in-depth measure, not a
complete solution. A real production system should also structurally
separate "data" the agent reads from "instructions" the agent follows,
ideally at the model/tooling level -- this module only reduces the
blast radius of what gets permanently written to the graph.
"""

from __future__ import annotations

import re

MAX_STORED_LENGTH = 300

# Deliberately simple, auditable pattern list -- not a claim of
# completeness. Any text matching these gets redacted before storage.
_INJECTION_MARKERS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"ignore (all|the|any) (previous|prior|above) instructions?",
        r"disregard (all|the|any) (previous|prior|above)",
        r"system\s*:\s*you (are|must|should)",
        r"new instructions?\s*:",
        r"\byou are now\b",
        r"\bact as\b.{0,40}\bunrestricted\b",
        r"reveal (your|the) (system prompt|instructions)",
    ]
]


def sanitize_external_content(text: str, max_len: int = MAX_STORED_LENGTH) -> str:
    """
    Truncate and redact potential injection payloads from text that came
    from an external tool result, before it is written to CEMG.

    Call this on tool OUTPUT text before it flows into `reasoning` or
    `action` fields. Never call it on text the agent itself generated
    (its own thoughts) -- only on text the agent read from the outside
    world, since that's the untrusted channel.
    """
    if not text:
        return text

    truncated = text[:max_len]
    for pattern in _INJECTION_MARKERS:
        truncated = pattern.sub("[REDACTED-POTENTIAL-INJECTION]", truncated)

    return truncated


def is_external_source(context_hint: str) -> bool:
    """
    Heuristic: which tool types read untrusted external content and
    therefore need their results sanitised before storage.

    Extend this list as you add real tools -- anything that fetches
    content the agent didn't author itself (web pages, emails, files
    from outside your own trusted pipeline, third-party API responses)
    belongs here.
    """
    external_tool_hints = {"web_search", "read_url", "read_email", "fetch_api", "read_file"}
    return context_hint in external_tool_hints
