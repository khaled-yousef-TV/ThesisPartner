"""DeepSeek API service — thesis analysis, chat, suggestions.

Uses the OpenAI-compatible chat completions endpoint.
"""

from __future__ import annotations

from typing import Any
import json as _json
import re as _re

import httpx

DEEPSEEK_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_CHAT_URL = f"{DEEPSEEK_BASE}/chat/completions"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _build_request(
    system: str,
    user: str,
    model: str,
    max_tokens: int = 8192,
    temperature: float = 0.7,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }


async def _chat(
    api_key: str,
    payload: dict[str, Any],
    timeout: float = 300.0,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(DEEPSEEK_CHAT_URL, headers=headers, json=payload)
        if r.status_code != 200:
            return {"ok": False, "error": f"DeepSeek HTTP {r.status_code}: {r.text[:500]}"}
        data = r.json()
        choices = data.get("choices", [])
        if not choices:
            return {"ok": False, "error": "DeepSeek returned empty choices"}
        content = choices[0].get("message", {}).get("content", "")
        return {"ok": True, "text": content}


def _parse_json_response(text: str) -> dict[str, Any]:
    """Extract JSON from a model response, handling code fences."""
    s = text.strip()
    if s.startswith("```"):
        s = _re.sub(r"^```[a-zA-Z0-9]*\s*\n", "", s)
        s = _re.sub(r"\n```\s*$", "", s).strip()
    try:
        return _json.loads(s)
    except _json.JSONDecodeError:
        m = _re.search(r"\{[\s\S]*\}", s)
        if m:
            return _json.loads(m.group(0))
        raise


# ---------------------------------------------------------------------------
# APA Review
# ---------------------------------------------------------------------------

async def analyze_draft(
    *,
    api_key: str,
    model: str,
    section_path: str,
    note: str | None,
    draft: str,
) -> dict[str, Any]:
    """APA 7th edition review of a thesis draft."""
    if not api_key:
        return {"ok": False, "error": "DEEPSEEK_API_KEY is not set"}

    system = (
        "You are an APA 7th edition specialist. Review the thesis draft for APA citation issues.\n"
        "Scope: in-text citations and reference list entries ONLY. Do not report grammar or spelling.\n"
        "Return a JSON object with key 'apa7' containing an array of issues. "
        "Each issue: {\"issue\": str, \"suggestion\": str, \"excerpt\": str}.\n"
        "The excerpt must be an exact verbatim substring from the draft, or empty string.\n"
        "Prefer omitting a row over guessing. Return ONLY valid JSON, no commentary."
    )

    user = (
        f"section_path: {section_path}\n"
        f"author_note: {note or ''}\n\n"
        f"<<<DRAFT>>>\n{draft}\n<<<END_DRAFT>>>\n"
    )

    payload = _build_request(system, user, model, max_tokens=4096, temperature=0.2)
    result = await _chat(api_key, payload, timeout=120.0)
    if not result.get("ok"):
        return {"ok": False, "error": result.get("error", "APA review failed"), "data": None}

    try:
        data = _parse_json_response(result["text"])
        items = data.get("apa7", []) if isinstance(data, dict) else []
        sanitized = []
        for it in items:
            if not isinstance(it, dict):
                continue
            ex = it.get("excerpt", "")
            issue = (it.get("issue") or "").lower()
            sug = (it.get("suggestion") or "").lower()
            if "&" in str(ex) and ("ampersand" in issue or "ampersand" in sug):
                continue
            sanitized.append(it)
        return {"ok": True, "data": {"apa7": sanitized}, "raw_text": result["text"]}
    except (_json.JSONDecodeError, KeyError) as exc:
        return {"ok": False, "error": f"APA review parse failed: {exc}", "data": None, "raw": result["text"][:2000]}


# ---------------------------------------------------------------------------
# Theme Fit
# ---------------------------------------------------------------------------

async def theme_fit_manuscript(
    *,
    api_key: str,
    model: str,
    sections_snapshot: str,
    thesis_memory: str,
) -> dict[str, Any]:
    """Thematic fit across all section drafts."""
    if not api_key:
        return {"ok": False, "error": "DEEPSEEK_API_KEY is not set"}

    system = (
        "You are a thesis argument coach. Review the manuscript snapshot for thematic coherence.\n"
        "Return a JSON object: {\"theme\": {\"summary\": str, \"strengths\": [str], "
        "\"gaps\": [str], \"suggestions\": [str]}, \"sectionNotes\": str}.\n"
        "Ground all observations in visible text. If content is sparse, say so. "
        "Return ONLY valid JSON, no commentary."
    )

    user = (
        f"<<<SECTIONS_SNAPSHOT>>>\n{sections_snapshot}\n<<<END_SECTIONS_SNAPSHOT>>>\n\n"
        f"<<<THESIS_MEMORY>>>\n{thesis_memory or '(none)'}\n<<<END_THESIS_MEMORY>>>\n"
    )

    payload = _build_request(system, user, model, max_tokens=4096, temperature=0.3)
    result = await _chat(api_key, payload, timeout=120.0)
    if not result.get("ok"):
        return {"ok": False, "error": result.get("error", "Theme fit failed"), "data": None}

    try:
        data = _parse_json_response(result["text"])
        theme = data.get("theme") if isinstance(data, dict) else {}
        if not isinstance(theme, dict):
            theme = {}
        return {
            "ok": True,
            "data": {
                "theme": {
                    "summary": theme.get("summary", ""),
                    "strengths": theme.get("strengths", []),
                    "gaps": theme.get("gaps", []),
                    "suggestions": theme.get("suggestions", []),
                },
                "sectionNotes": str(data.get("sectionNotes", "") if isinstance(data, dict) else ""),
            },
        }
    except (_json.JSONDecodeError, KeyError) as exc:
        return {"ok": False, "error": f"Theme fit parse failed: {exc}", "data": None}


# ---------------------------------------------------------------------------
# Section Suggestion
# ---------------------------------------------------------------------------

async def suggest_section(
    *,
    api_key: str,
    model: str,
    section_path: str,
    section_label: str,
    target_draft: str,
    sections_snapshot: str,
    thesis_memory: str,
) -> dict[str, Any]:
    """Writing guidance for one thesis section."""
    if not api_key:
        return {"ok": False, "error": "DEEPSEEK_API_KEY is not set"}

    system = (
        "You are a thesis writing coach. Provide structured guidance for one section.\n"
        "Return a JSON object: {\"summary\": str, \"outline\": [{\"heading\": str, \"points\": [str]}], "
        "\"gapsToAddress\": [str], \"connections\": [str], \"nextSteps\": [str]}.\n"
        "Ground advice in visible drafts. If content is sparse, give a sensible starter outline. "
        "Return ONLY valid JSON, no commentary."
    )

    user = (
        f"target_section: {section_path} ({section_label})\n\n"
        f"<<<TARGET_DRAFT>>>\n{target_draft or '(No draft yet)'}\n<<<END_TARGET_DRAFT>>>\n\n"
        f"<<<SECTIONS_SNAPSHOT>>>\n{sections_snapshot}\n<<<END_SECTIONS_SNAPSHOT>>>\n\n"
        f"<<<THESIS_MEMORY>>>\n{thesis_memory or '(none)'}\n<<<END_THESIS_MEMORY>>>\n"
    )

    payload = _build_request(system, user, model, max_tokens=4096, temperature=0.4)
    result = await _chat(api_key, payload, timeout=120.0)
    if not result.get("ok"):
        return {"ok": False, "error": result.get("error", "Section suggestion failed"), "data": None}

    try:
        data = _parse_json_response(result["text"])
        return {"ok": True, "data": data if isinstance(data, dict) else {}}
    except (_json.JSONDecodeError, KeyError) as exc:
        return {"ok": False, "error": f"Section suggestion parse failed: {exc}", "data": None}


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

async def chat_about_suggestion(
    *,
    api_key: str,
    model: str,
    section_path: str,
    section_label: str,
    suggestion: dict[str, Any],
    user_message: str,
    history: list[dict[str, str]],
    thesis_context: str,
) -> dict[str, Any]:
    """Chat about a section suggestion."""
    if not api_key:
        return {"ok": False, "error": "DEEPSEEK_API_KEY is not set"}

    suggestion_text = _json.dumps(suggestion, ensure_ascii=False, indent=2)
    system = (
        "You help refine a Masters thesis section. Be concise and practical.\n"
        f"Section: {section_path} ({section_label})\n"
        f"Current suggestion:\n{suggestion_text}\n"
        f"Thesis context:\n{thesis_context or '(none yet)'}\n"
    )

    messages: list[dict[str, str]] = []
    for turn in history:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role in ("user", "assistant") and content.strip():
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + messages,
        "max_tokens": 4096,
        "temperature": 0.7,
    }
    result = await _chat(api_key, payload, timeout=120.0)
    if not result.get("ok"):
        return {"ok": False, "error": result.get("error", "Chat failed")}
    return {"ok": True, "text": result.get("text", "")}


async def chat_turn(
    *,
    api_key: str,
    model: str,
    user_message: str,
    thesis_context: str,
) -> dict[str, Any]:
    """General thesis chat."""
    if not api_key:
        return {"ok": False, "error": "DEEPSEEK_API_KEY is not set"}

    system = (
        "You help refine a Masters thesis. Be concise and practical.\n"
        f"Thesis context so far:\n{thesis_context or '(none yet)'}\n"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": 4096,
        "temperature": 0.7,
    }
    result = await _chat(api_key, payload, timeout=120.0)
    if not result.get("ok"):
        return {"ok": False, "error": result.get("error", "Chat failed")}
    return {"ok": True, "text": result.get("text", "")}


async def refresh_brief(
    *,
    api_key: str,
    model: str,
    memory_block: str,
) -> dict[str, Any]:
    """Summarize thesis memory into a research brief."""
    if not api_key:
        return {"ok": False, "error": "DEEPSEEK_API_KEY is not set"}
    if not memory_block.strip():
        return {"ok": True, "brief": ""}

    prompt = (
        "Summarize the following thesis memory into a short research brief: "
        "core claims, methods, key terms, open questions. Max ~400 words. Plain text.\n\n"
        f"---\n{memory_block}\n---\n"
    )

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,
        "temperature": 0.3,
    }
    result = await _chat(api_key, payload, timeout=120.0)
    if not result.get("ok"):
        return {"ok": False, "error": result.get("error", "Brief refresh failed")}
    return {"ok": True, "brief": result.get("text", "")}
