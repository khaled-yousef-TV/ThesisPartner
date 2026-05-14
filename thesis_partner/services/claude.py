from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from anthropic import AsyncAnthropic

GRAMMAR_REVIEW_SYSTEM = """You are a copy editor for academic English. You MUST call the tool `return_grammar_review` exactly once.

This request contains **only** the thesis draft inside <<<DRAFT>>> … <<<END_DRAFT>>>. No thesis memory is provided.

Scope — **language mechanics only**:
- Grammar, spelling, punctuation, typos, agreement, tense, articles, and sentence clarity where something is **objectively wrong or risky** in context.
- **Do not** report APA, citation style, parenthetical vs narrative citations, reference list formatting, or DOI/URL style — another pass handles citations.
- **Do not** suggest stylistic rewrites, "sounds better", optional synonyms, or subjective tone when mechanics are already acceptable.

Precision (reduce false positives):
- **Prefer recall = low, precision = high**: if you are not sure the draft is wrong, **omit** the row. Empty `grammar` array is acceptable.
- For every row with a non-empty `excerpt`, the `message` must be **directly verifiable** from that exact substring. If the substring already satisfies the rule you would cite (e.g. agreement is correct, punctuation is standard), **do not** output that row.
- Use a **non-empty** `excerpt` whenever you report a concrete issue — copy the **shortest span** (phrase to sentence) that contains the error. Avoid empty excerpt except for rare document-level notes you are certain about.
- **Do not** duplicate or near-duplicate the same issue for overlapping spans; keep one clearest row.
- severity: error | warning | suggestion — use **suggestion** only for clear optional improvements, not nitpicks."""


GRAMMAR_REVIEW_TOOL: dict[str, Any] = {
    "name": "return_grammar_review",
    "description": "Grammar and language mechanics only (not citations or APA).",
    "input_schema": {
        "type": "object",
        "properties": {
            "grammar": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "severity": {
                            "type": "string",
                            "description": "One of: error, warning, suggestion",
                        },
                        "message": {
                            "type": "string",
                            "description": "Issue visible in excerpt; omit vague advice. Language only, not citations.",
                        },
                        "excerpt": {
                            "type": "string",
                            "description": "Exact verbatim substring from DRAFT showing the problem; prefer non-empty for each substantive issue.",
                        },
                    },
                    "required": ["severity", "message", "excerpt"],
                },
            },
        },
        "required": ["grammar"],
    },
}

THEME_REVIEW_SYSTEM = """You are a thesis argument coach. You MUST call the tool `return_theme_review` exactly once.

The draft is inside <<<DRAFT>>> … <<<END_DRAFT>>>. <<<THESIS_CONTEXT>>> holds stored memory for alignment — use it **only** for `theme` and `sectionNotes`, not as text to quote as if in the draft.

Rules:
- theme: compare the draft to thesis_context; if thesis_context is empty, say theme checks are limited and avoid speculative gaps.
- **Ground** strengths, gaps, and suggestions in the draft (and context where used); do not invent claims the draft does not support.
- sectionNotes: how the draft fits section_path and cross-chapter cautions. Do not fabricate quotes from memory."""

THEME_REVIEW_TOOL: dict[str, Any] = {
    "name": "return_theme_review",
    "description": "Thematic fit and section notes using draft plus stored thesis context.",
    "input_schema": {
        "type": "object",
        "properties": {
            "theme": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "strengths": {"type": "array", "items": {"type": "string"}},
                    "gaps": {"type": "array", "items": {"type": "string"}},
                    "suggestions": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["summary", "strengths", "gaps", "suggestions"],
            },
            "sectionNotes": {"type": "string"},
        },
        "required": ["theme", "sectionNotes"],
    },
}

APA_REVIEW_SYSTEM = """You are an APA 7th edition specialist. You MUST call the tool `return_apa_review` exactly once.

This request contains **only** the thesis draft inside <<<DRAFT>>> … <<<END_DRAFT>>>. There is **no** thesis memory or outside context — judge APA using that draft text alone.

Scope — **citations and references only**:
- In-text citations, parenthetical vs narrative form, ampersand, commas, years, et al., and reference list entries visible in the draft.
- **Do not** report general spelling, grammar, or sentence mechanics unless they directly break APA citation punctuation (another pass handles language).

Precision (reduce false positives):
- **Prefer omitting a row** over guessing. If the citation span is ambiguous or already conforms, do not report it.
- Prefer a **non-empty** `excerpt` that is the **shortest** substring proving the issue; the `issue` text must match what appears in that excerpt.

Rules:
- Every `excerpt` must be an **exact verbatim substring** of the draft between those markers, or empty.
- Distinguish **narrative** citations (Author, year) from **parenthetical** ((Author, year)).
- Do not report "missing ampersand" if `&` already appears between author surnames in the excerpt.
- Before each row: if the issue claims X is missing, X must not appear in the excerpt.
- If a citation excerpt already contains **et al.**, do not flag "lacks et al." / "missing et al." / multiple-author et al. rules for that same span.
- Omit overlapping duplicate rows for the same span; keep the clearest single issue.
- Use `section_path` only as a hint (e.g. References vs Discussion), not as extra text to cite."""

APA_REVIEW_TOOL: dict[str, Any] = {
    "name": "return_apa_review",
    "description": "APA 7th in-text and reference-list issues visible in the DRAFT block only.",
    "input_schema": {
        "type": "object",
        "properties": {
            "apa7": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "issue": {
                            "type": "string",
                            "description": "Short APA issue; must not claim a symbol is missing if it appears in excerpt. Do not claim missing et al. if excerpt already contains et al.",
                        },
                        "suggestion": {"type": "string"},
                        "excerpt": {
                            "type": "string",
                            "description": "Exact verbatim substring from DRAFT proving the APA issue; prefer non-empty shortest span.",
                        },
                    },
                    "required": ["issue", "suggestion", "excerpt"],
                },
            },
        },
        "required": ["apa7"],
    },
}



def _strip_code_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\s*\n", "", s)
        s = re.sub(r"\n```\s*$", "", s).strip()
    return s


def _parse_json_object(text: str) -> dict[str, Any]:
    return json.loads(_strip_code_fence(text))


def _text_blocks(message: Any) -> str:
    parts: list[str] = []
    for block in getattr(message, "content", []) or []:
        btype = getattr(block, "type", None)
        if btype == "text":
            parts.append(getattr(block, "text", ""))
    return "".join(parts).strip()


def _tool_input_by_name(message: Any, tool_name: str) -> dict[str, Any] | None:
    for block in getattr(message, "content", []) or []:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == tool_name:
            raw_input = getattr(block, "input", None)
            if isinstance(raw_input, dict):
                return raw_input
    return None


def _excerpt_in_draft(excerpt: str, draft: str) -> bool:
    ex = (excerpt or "").strip()
    if not ex:
        return True
    return ex in draft


def _should_drop_spurious_apa7(it: dict[str, Any]) -> bool:
    issue = (it.get("issue") or "").lower()
    suggestion = (it.get("suggestion") or "").lower()
    excerpt = it.get("excerpt") or ""
    blob = issue + " " + suggestion

    if "&" in excerpt:
        if "ampersand" in issue or "ampersand" in suggestion:
            return True
        if "use '&'" in blob or "use &" in blob or "instead of 'and'" in blob:
            return True
        if "missing" in issue and ("&" in issue or "ampersand" in issue):
            return True

    # Citation already uses et al. but model still says it is missing (common false positive).
    if re.search(r"\bet\s+al\.?\b", excerpt, re.IGNORECASE):
        if re.search(r"\b(lacks?\s+et|missing\s+et|without\s+et|lack\s+of\s+et)\b", blob):
            return True
        if "multiple authors" in blob and re.search(
            r"\b(lacks?|missing|without|needs?\s+et|add\s+et)\b", blob
        ):
            return True
    return False


def _sanitize_review_against_draft(data: dict[str, Any], draft: str) -> dict[str, Any]:
    for key in ("grammar", "apa7"):
        items = data.get(key)
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict):
                continue
            ex = it.get("excerpt", "")
            if isinstance(ex, str) and ex.strip() and not _excerpt_in_draft(ex, draft):
                it["excerpt"] = ""

    apa = data.get("apa7")
    if isinstance(apa, list):
        data["apa7"] = [it for it in apa if isinstance(it, dict) and not _should_drop_spurious_apa7(it)]
    return data


def _empty_theme() -> dict[str, Any]:
    return {"summary": "", "strengths": [], "gaps": [], "suggestions": []}


async def _call_grammar_review(
    *,
    client: AsyncAnthropic,
    model: str,
    section_path: str,
    note: str | None,
    draft: str,
) -> dict[str, Any]:
    user = (
        "Grammar review: use ONLY the draft between markers. No citation-style feedback.\n\n"
        + f"section_path: {section_path}\n"
        + f"optional_note_from_author: {note or ''}\n\n"
        + "<<<DRAFT>>>\n"
        + f"{draft}\n"
        + "<<<END_DRAFT>>>\n"
    )
    message = await client.messages.create(
        model=model,
        max_tokens=8192,
        system=GRAMMAR_REVIEW_SYSTEM,
        tools=[GRAMMAR_REVIEW_TOOL],
        tool_choice={"type": "tool", "name": "return_grammar_review"},
        messages=[{"role": "user", "content": user}],
    )
    data = _tool_input_by_name(message, "return_grammar_review")
    if data is None:
        raw = _text_blocks(message)
        if raw:
            try:
                return {"ok": True, "data": _parse_json_object(raw)}
            except json.JSONDecodeError as exc:
                return {"ok": False, "error": f"Grammar review JSON fallback failed: {exc}", "raw": raw[:4000]}
        return {"ok": False, "error": "Grammar review: empty or missing tool"}
    return {"ok": True, "data": data}


async def _call_theme_review(
    *,
    client: AsyncAnthropic,
    model: str,
    section_path: str,
    note: str | None,
    draft: str,
    thesis_context: str,
) -> dict[str, Any]:
    user = (
        "Read <<<DRAFT>>> first, then thesis context for alignment only.\n\n"
        + "<<<DRAFT>>>\n"
        + f"{draft}\n"
        + "<<<END_DRAFT>>>\n\n"
        + f"section_path: {section_path}\n"
        + f"optional_note_from_author: {note or ''}\n\n"
        + "<<<THESIS_CONTEXT>>>\n"
        + f"{thesis_context or '(none)'}\n"
        + "<<<END_THESIS_CONTEXT>>>\n"
    )
    message = await client.messages.create(
        model=model,
        max_tokens=8192,
        system=THEME_REVIEW_SYSTEM,
        tools=[THEME_REVIEW_TOOL],
        tool_choice={"type": "tool", "name": "return_theme_review"},
        messages=[{"role": "user", "content": user}],
    )
    data = _tool_input_by_name(message, "return_theme_review")
    if data is None:
        raw = _text_blocks(message)
        if raw:
            try:
                return {"ok": True, "data": _parse_json_object(raw)}
            except json.JSONDecodeError as exc:
                return {"ok": False, "error": f"Theme review JSON fallback failed: {exc}", "raw": raw[:4000]}
        return {"ok": False, "error": "Theme review: empty or missing tool"}
    return {"ok": True, "data": data}


async def _call_apa_review(
    *,
    client: AsyncAnthropic,
    model: str,
    section_path: str,
    note: str | None,
    draft: str,
) -> dict[str, Any]:
    user = (
        "APA review: use ONLY the draft between markers. No other source text exists in this request.\n\n"
        + f"section_path: {section_path}\n"
        + f"optional_note_from_author: {note or ''}\n\n"
        + "<<<DRAFT>>>\n"
        + f"{draft}\n"
        + "<<<END_DRAFT>>>\n"
    )
    message = await client.messages.create(
        model=model,
        max_tokens=8192,
        system=APA_REVIEW_SYSTEM,
        tools=[APA_REVIEW_TOOL],
        tool_choice={"type": "tool", "name": "return_apa_review"},
        messages=[{"role": "user", "content": user}],
    )
    data = _tool_input_by_name(message, "return_apa_review")
    if data is None:
        raw = _text_blocks(message)
        if raw:
            try:
                return {"ok": True, "data": _parse_json_object(raw)}
            except json.JSONDecodeError as exc:
                return {"ok": False, "error": f"APA review JSON fallback failed: {exc}", "raw": raw[:4000]}
        return {"ok": False, "error": "APA review: empty or missing tool"}
    return {"ok": True, "data": data}


async def analyze_draft(
    *,
    api_key: str,
    model: str,
    section_path: str,
    note: str | None,
    draft: str,
    thesis_context: str,
) -> dict[str, Any]:
    if not api_key:
        return {"ok": False, "error": "ANTHROPIC_API_KEY is not set"}

    client = AsyncAnthropic(api_key=api_key)
    try:
        grammar_res, apa_res, theme_res = await asyncio.gather(
            _call_grammar_review(
                client=client,
                model=model,
                section_path=section_path,
                note=note,
                draft=draft,
            ),
            _call_apa_review(
                client=client,
                model=model,
                section_path=section_path,
                note=note,
                draft=draft,
            ),
            _call_theme_review(
                client=client,
                model=model,
                section_path=section_path,
                note=note,
                draft=draft,
                thesis_context=thesis_context,
            ),
        )

        merged: dict[str, Any] = {
            "grammar": [],
            "apa7": [],
            "theme": _empty_theme(),
            "sectionNotes": "",
        }

        g_ok = grammar_res.get("ok") and isinstance(grammar_res.get("data"), dict)
        a_ok = apa_res.get("ok") and isinstance(apa_res.get("data"), dict)
        t_ok = theme_res.get("ok") and isinstance(theme_res.get("data"), dict)

        if g_ok:
            merged["grammar"] = grammar_res["data"].get("grammar") or []
        if a_ok:
            merged["apa7"] = apa_res["data"].get("apa7") or []
        if t_ok:
            merged["theme"] = theme_res["data"].get("theme") or _empty_theme()
            merged["sectionNotes"] = theme_res["data"].get("sectionNotes") or ""

        _sanitize_review_against_draft(merged, draft)

        out: dict[str, Any] = {"ok": g_ok and t_ok, "data": merged, "raw_text": None}
        if not g_ok:
            out["grammar_warning"] = grammar_res.get("error", "Grammar review failed")
        if not a_ok:
            out["apa_warning"] = apa_res.get("error", "APA review failed")
        if not t_ok:
            out["theme_warning"] = theme_res.get("error", "Theme review failed")
        if out["ok"]:
            out.pop("grammar_warning", None)
            out.pop("theme_warning", None)
        return out
    except Exception as exc:
        return {"ok": False, "error": str(exc), "data": None}


GRAMMAR_FIX_SYSTEM = """You are a copy editor for academic English.
Return ONLY the corrected document text. Rules:
- Fix grammar, spelling, punctuation, and clear typographical errors only.
- Preserve meaning, argument, tone, paragraph breaks, headings, citations, references, numbers, and quoted material; do not rewrite for style.
- Do not add labels, markdown fences, or any preamble or postscript — only the corrected text."""


async def quick_grammar_fix(*, api_key: str, model: str, text: str) -> dict[str, Any]:
    if not api_key:
        return {"ok": False, "error": "ANTHROPIC_API_KEY is not set"}
    client = AsyncAnthropic(api_key=api_key)
    user = (
        "Correct the following text per your instructions. "
        "Output must be the full corrected text only.\n\n---\n"
        + text
        + "\n---\n"
    )
    try:
        message = await client.messages.create(
            model=model,
            max_tokens=16_384,
            system=GRAMMAR_FIX_SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        out = _text_blocks(message)
        if not out:
            return {"ok": False, "error": "Empty response from Claude"}
        out = _strip_code_fence(out)
        return {"ok": True, "text": out}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def chat_turn(
    *,
    api_key: str,
    model: str,
    user_message: str,
    thesis_context: str,
) -> dict[str, Any]:
    if not api_key:
        return {"ok": False, "error": "ANTHROPIC_API_KEY is not set"}

    system = (
        "You help refine a Masters thesis. Be concise and practical. "
        "The user stores this chat as thesis memory for later consistency checks.\n\n"
        f"Thesis context so far:\n{thesis_context or '(none yet)'}\n"
    )
    client = AsyncAnthropic(api_key=api_key)
    try:
        message = await client.messages.create(
            model=model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )
        text = _text_blocks(message)
        return {"ok": True, "text": text}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def refresh_brief(
    *,
    api_key: str,
    model: str,
    memory_block: str,
) -> dict[str, Any]:
    if not api_key:
        return {"ok": False, "error": "ANTHROPIC_API_KEY is not set"}
    if not memory_block.strip():
        return {"ok": True, "brief": ""}

    prompt = (
        "Summarize the following thesis memory into a short research brief: "
        "core claims, methods, key terms, open questions. Max ~400 words. Plain text.\n\n"
        f"---\n{memory_block}\n---\n"
    )
    client = AsyncAnthropic(api_key=api_key)
    try:
        message = await client.messages.create(
            model=model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return {"ok": True, "brief": _text_blocks(message)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
