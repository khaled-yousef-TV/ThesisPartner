from __future__ import annotations

import json
import re
from typing import Any

from anthropic import AsyncAnthropic

APA_REVIEW_SYSTEM = """You are an APA 7th edition specialist. You MUST call the tool `return_apa_review` exactly once.

This request contains **only** the thesis draft inside <<<DRAFT>>> … <<<END_DRAFT>>>. There is **no** thesis memory or outside context — judge APA using that draft text alone.

Scope — **citations and references only**:
- In-text citations, parenthetical vs narrative form, ampersand, commas, years, et al., and reference list entries visible in the draft.
- **Do not** report general spelling, grammar, or sentence mechanics unless they directly break APA citation punctuation.

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


THEME_MANUSCRIPT_SYSTEM = """You are a thesis argument coach reviewing the whole manuscript snapshot the user assembled. You MUST call the tool `return_theme_review` exactly once.

You receive two blocks:

- <<<SECTIONS_SNAPSHOT>>> — latest draft pasted per binder section. Each section is labeled by its binder path. If no draft was saved yet, the body is exactly: (No draft submitted yet.) Some long sections may end with a truncation notice; treat truncation as imperfect visibility, not absence of content.
- <<<THESIS_MEMORY>>> — brief, pasted memory, chat excerpts. Use memory for alignment with the author's stated goals and terminology; **do not** quote memory strings as if they appeared in <<<SECTIONS_SNAPSHOT>>> unless the same wording clearly appears there.

Produce:
- **theme**: thematic coherence across sections, strengths, gaps vs memory and snapshot, actionable suggestions grounded in visible text.
- **sectionNotes**: cross-cutting observations (methods vs discussion, unanswered RQs, structure, appendix vs main text). Do not fabricate quotations.

If <<<SECTIONS_SNAPSHOT>>> has little real content overall, say so clearly and avoid inventing substantive gaps."""

THEME_REVIEW_TOOL: dict[str, Any] = {
    "name": "return_theme_review",
    "description": "Thematic fit across all section drafts plus thesis memory context.",
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


def _empty_theme() -> dict[str, Any]:
    return {"summary": "", "strengths": [], "gaps": [], "suggestions": []}
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

    if re.search(r"\bet\s+al\.?\b", excerpt, re.IGNORECASE):
        if re.search(r"\b(lacks?\s+et|missing\s+et|without\s+et|lack\s+of\s+et)\b", blob):
            return True
        if "multiple authors" in blob and re.search(
            r"\b(lacks?|missing|without|needs?\s+et|add\s+et)\b", blob
        ):
            return True
    return False


def _sanitize_review_against_draft(data: dict[str, Any], draft: str) -> dict[str, Any]:
    items = data.get("apa7")
    if isinstance(items, list):
        for it in items:
            if not isinstance(it, dict):
                continue
            ex = it.get("excerpt", "")
            if isinstance(ex, str) and ex.strip() and not _excerpt_in_draft(ex, draft):
                it["excerpt"] = ""

        data["apa7"] = [
            it
            for it in items
            if isinstance(it, dict) and not _should_drop_spurious_apa7(it)
        ]
    return data


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
) -> dict[str, Any]:
    if not api_key:
        return {"ok": False, "error": "ANTHROPIC_API_KEY is not set"}

    client = AsyncAnthropic(api_key=api_key)
    try:
        apa_res = await _call_apa_review(
            client=client,
            model=model,
            section_path=section_path,
            note=note,
            draft=draft,
        )

        merged: dict[str, Any] = {"apa7": []}

        a_ok = apa_res.get("ok") and isinstance(apa_res.get("data"), dict)

        if a_ok:
            merged["apa7"] = apa_res["data"].get("apa7") or []

        _sanitize_review_against_draft(merged, draft)

        out: dict[str, Any] = {"ok": a_ok, "data": merged, "raw_text": None}
        if not a_ok:
            out["apa_warning"] = apa_res.get("error", "APA review failed")
        return out
    except Exception as exc:
        return {"ok": False, "error": str(exc), "data": None}


async def theme_fit_manuscript(
    *,
    api_key: str,
    model: str,
    sections_snapshot: str,
    thesis_memory: str,
) -> dict[str, Any]:
    if not api_key:
        return {"ok": False, "error": "ANTHROPIC_API_KEY is not set"}

    user = (
        "Assess thematic fit using only the labeled blocks.\n\n"
        + "<<<SECTIONS_SNAPSHOT>>>\n"
        + f"{sections_snapshot}\n"
        + "<<<END_SECTIONS_SNAPSHOT>>>\n\n"
        + "<<<THESIS_MEMORY>>>\n"
        + f"{thesis_memory or '(none)'}\n"
        + "<<<END_THESIS_MEMORY>>>\n"
    )
    client = AsyncAnthropic(api_key=api_key)
    try:
        message = await client.messages.create(
            model=model,
            max_tokens=8192,
            system=THEME_MANUSCRIPT_SYSTEM,
            tools=[THEME_REVIEW_TOOL],
            tool_choice={"type": "tool", "name": "return_theme_review"},
            messages=[{"role": "user", "content": user}],
        )
        data = _tool_input_by_name(message, "return_theme_review")
        if data is None:
            raw = _text_blocks(message)
            if raw:
                try:
                    obj = _parse_json_object(raw)
                    return {
                        "ok": True,
                        "data": {
                            "theme": obj.get("theme") or _empty_theme(),
                            "sectionNotes": str(obj.get("sectionNotes") or ""),
                        },
                    }
                except json.JSONDecodeError as exc:
                    return {"ok": False, "error": f"Theme fit JSON fallback failed: {exc}", "raw": raw[:4000]}
            return {"ok": False, "error": "Theme fit: empty or missing tool"}
        theme = data.get("theme") if isinstance(data.get("theme"), dict) else None
        if theme is None:
            theme = _empty_theme()
        return {
            "ok": True,
            "data": {
                "theme": theme,
                "sectionNotes": str(data.get("sectionNotes") or ""),
            },
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "data": None}


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
