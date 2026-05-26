from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from thesis_partner import db as dbmod
from thesis_partner.binder import (
    SECTION_GROUPS,
    SIDEBAR_TREE,
    VALID_SECTION_PATHS,
    all_section_paths_ordered,
    section_href,
    section_label as binder_section_label,
)
from thesis_partner.config import Settings, get_settings
from thesis_partner.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ChatRequest,
    MemoryRequest,
    SuggestSectionChatRequest,
    SuggestSectionRequest,
)
from thesis_partner.services import deepseek, gptzero

ROOT = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(ROOT / "templates"))
templates.env.globals["section_href"] = section_href


@asynccontextmanager
async def lifespan(app: FastAPI):
    conn = dbmod.connect()
    dbmod.init_db(conn)
    conn.close()
    yield


app = FastAPI(title="Thesis Partner", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")


def get_db() -> sqlite3.Connection:
    conn = dbmod.connect()
    try:
        yield conn
    finally:
        conn.close()


def load_thesis_context(conn: sqlite3.Connection, memory_limit: int = 48) -> str:
    brief_row = conn.execute("SELECT brief FROM context_brief WHERE id = 1").fetchone()
    brief = brief_row["brief"] if brief_row else ""
    rows = conn.execute(
        """
        SELECT source, section_path, role, content
        FROM memory_entries
        ORDER BY id DESC
        LIMIT ?
        """,
        (memory_limit,),
    ).fetchall()
    chunks: list[str] = []
    if brief and brief.strip():
        chunks.append("BRIEF:\n" + brief.strip())
    for r in reversed(rows):
        head = f"{r['source']}"
        if r["section_path"]:
            head += f" | {r['section_path']}"
        if r["role"]:
            head += f" | {r['role']}"
        body = (r["content"] or "")[:6000]
        chunks.append(f"[{head}]\n{body}")
    return "\n\n---\n\n".join(chunks)


def build_sections_snapshot(
    conn: sqlite3.Connection,
    *,
    paths: list[str],
    max_chars_per_section: int,
) -> str:
    rows = {
        r["section_path"]: r
        for r in conn.execute(
            "SELECT section_path, text_content, updated_at FROM section_drafts"
        ).fetchall()
    }
    parts: list[str] = []
    for path in paths:
        label = binder_section_label(path)
        row = rows.get(path)
        if not row or not (row["text_content"] or "").strip():
            parts.append(f"## {path} ({label})\n\n(No draft submitted yet.)")
            continue
        text = str(row["text_content"] or "")
        truncated = False
        if len(text) > max_chars_per_section:
            text = text[:max_chars_per_section]
            truncated = True
        block = f"## {path} ({label})\n\n{text}"
        if truncated:
            block += f"\n\n[… truncated to {max_chars_per_section:,} characters …]"
        parts.append(block)
    return "\n\n---\n\n".join(parts)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "section_groups": SECTION_GROUPS,
            "sidebar_tree": SIDEBAR_TREE,
        },
    )


@app.get("/section/{section_path:path}", response_class=HTMLResponse)
def section_draft_view(
    request: Request,
    section_path: str,
    conn: sqlite3.Connection = Depends(get_db),
) -> HTMLResponse:
    normalized = section_path.strip()
    if normalized not in VALID_SECTION_PATHS:
        raise HTTPException(status_code=404, detail="Unknown binder section.")
    row = conn.execute(
        "SELECT text_content, note, updated_at FROM section_drafts WHERE section_path = ?",
        (normalized,),
    ).fetchone()
    draft_empty = row is None or not (row["text_content"] or "").strip()
    return templates.TemplateResponse(
        request,
        "section.html",
        {
            "sidebar_tree": SIDEBAR_TREE,
            "section_path": normalized,
            "section_label_title": binder_section_label(normalized),
            "draft_empty": draft_empty,
            "draft_text": "" if draft_empty else str(row["text_content"]),
            "draft_note": None if draft_empty else (row["note"] or None),
            "draft_updated": None if draft_empty else str(row["updated_at"] or ""),
        },
    )


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(
    body: AnalyzeRequest,
    conn: sqlite3.Connection = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> AnalyzeResponse:
    if len(body.text) > settings.max_analyze_chars:
        raise HTTPException(status_code=400, detail="Text exceeds configured limit")

    gz_task = gptzero.scan_text(settings.gptzero_api_key, body.text)
    cl_task = deepseek.analyze_draft(
        api_key=settings.deepseek_api_key,
        model=settings.deepseek_model,
        section_path=body.section_path,
        note=body.note,
        draft=body.text,
    )
    gz_result, cl_result = await asyncio.gather(gz_task, cl_task)

    claude_payload: dict = cl_result if isinstance(cl_result, dict) else {"ok": False}
    gptzero_payload: dict = gz_result if isinstance(gz_result, dict) else {"ok": False}

    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO analysis_runs (section_path, note, text_content, claude_json, gptzero_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            body.section_path,
            body.note,
            body.text,
            json.dumps(claude_payload, ensure_ascii=False),
            json.dumps(gptzero_payload, ensure_ascii=False),
        ),
    )
    cur.execute(
        """
        INSERT INTO section_drafts (section_path, text_content, note, updated_at)
        VALUES (?, ?, ?, datetime('now'))
        ON CONFLICT(section_path) DO UPDATE SET
          text_content = excluded.text_content,
          note = excluded.note,
          updated_at = datetime('now')
        """,
        (body.section_path, body.text, body.note),
    )
    conn.commit()
    analysis_id = int(cur.lastrowid)

    return AnalyzeResponse(claude=claude_payload, gptzero=gptzero_payload, analysis_id=analysis_id)


@app.post("/api/theme-fit")
async def theme_fit(
    conn: sqlite3.Connection = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> JSONResponse:
    paths = all_section_paths_ordered()
    snapshot = build_sections_snapshot(
        conn,
        paths=paths,
        max_chars_per_section=settings.max_theme_fit_section_chars,
    )
    thesis_memory = load_thesis_context(conn)
    result = await deepseek.theme_fit_manuscript(
        api_key=settings.deepseek_api_key,
        model=settings.deepseek_model,
        sections_snapshot=snapshot,
        thesis_memory=thesis_memory,
    )
    if not result.get("ok"):
        raise HTTPException(status_code=502, detail=result.get("error", "Claude error"))
    return JSONResponse(result)


@app.post("/api/suggest-section")
async def suggest_section(
    body: SuggestSectionRequest,
    conn: sqlite3.Connection = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> JSONResponse:
    section_path = body.section_path.strip()
    if section_path not in VALID_SECTION_PATHS:
        raise HTTPException(status_code=400, detail="Unknown binder section.")

    row = conn.execute(
        "SELECT text_content FROM section_drafts WHERE section_path = ?",
        (section_path,),
    ).fetchone()
    target_draft = ""
    if row and (row["text_content"] or "").strip():
        target_draft = str(row["text_content"])

    other_paths = [p for p in all_section_paths_ordered() if p != section_path]
    sections_snapshot = build_sections_snapshot(
        conn,
        paths=other_paths,
        max_chars_per_section=settings.max_theme_fit_section_chars,
    )
    thesis_memory = load_thesis_context(conn)
    result = await deepseek.suggest_section(
        api_key=settings.deepseek_api_key,
        model=settings.deepseek_model,
        section_path=section_path,
        section_label=binder_section_label(section_path),
        target_draft=target_draft,
        sections_snapshot=sections_snapshot,
        thesis_memory=thesis_memory,
    )
    if not result.get("ok"):
        raise HTTPException(status_code=502, detail=result.get("error", "Claude error"))
    return JSONResponse(result)


@app.post("/api/suggest-section/chat")
async def suggest_section_chat(
    body: SuggestSectionChatRequest,
    conn: sqlite3.Connection = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> JSONResponse:
    section_path = body.section_path.strip()
    if section_path not in VALID_SECTION_PATHS:
        raise HTTPException(status_code=400, detail="Unknown binder section.")
    if len(body.message) > settings.max_chat_chars:
        raise HTTPException(status_code=400, detail="Message exceeds configured limit")

    thesis_context = load_thesis_context(conn)
    history = [{"role": t.role, "content": t.content} for t in body.history]
    result = await deepseek.chat_about_suggestion(
        api_key=settings.deepseek_api_key,
        model=settings.deepseek_model,
        section_path=section_path,
        section_label=binder_section_label(section_path),
        suggestion=body.suggestion,
        user_message=body.message,
        history=history,
        thesis_context=thesis_context,
    )
    if not result.get("ok"):
        raise HTTPException(status_code=502, detail=result.get("error", "Claude error"))
    return JSONResponse({"ok": True, "reply": str(result.get("text", ""))})


@app.post("/api/chat")
async def chat(
    body: ChatRequest,
    conn: sqlite3.Connection = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> JSONResponse:
    if len(body.message) > settings.max_chat_chars:
        raise HTTPException(status_code=400, detail="Message exceeds configured limit")

    conn.execute(
        "INSERT INTO memory_entries (source, section_path, role, content) VALUES ('chat', NULL, 'user', ?)",
        (body.message,),
    )
    conn.commit()

    thesis_context = load_thesis_context(conn)
    result = await deepseek.chat_turn(
        api_key=settings.deepseek_api_key,
        model=settings.deepseek_model,
        user_message=body.message,
        thesis_context=thesis_context,
    )
    if not result.get("ok"):
        raise HTTPException(status_code=502, detail=result.get("error", "Claude error"))

    reply = str(result.get("text", ""))
    conn.execute(
        "INSERT INTO memory_entries (source, section_path, role, content) VALUES ('chat', NULL, 'assistant', ?)",
        (reply,),
    )
    conn.commit()
    return JSONResponse({"ok": True, "reply": reply})


@app.post("/api/memory")
def add_memory(
    body: MemoryRequest,
    conn: sqlite3.Connection = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> JSONResponse:
    if len(body.content) > settings.max_memory_chars:
        raise HTTPException(status_code=400, detail="Content exceeds configured limit")
    conn.execute(
        "INSERT INTO memory_entries (source, section_path, role, content) VALUES ('paste', ?, NULL, ?)",
        (body.section_path, body.content),
    )
    conn.commit()
    return JSONResponse({"ok": True})


@app.post("/api/brief/refresh")
async def refresh_brief(
    conn: sqlite3.Connection = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> JSONResponse:
    rows = conn.execute(
        "SELECT content FROM memory_entries ORDER BY id DESC LIMIT 80"
    ).fetchall()
    block = "\n\n".join(r["content"] for r in reversed(rows))
    result = await deepseek.refresh_brief(
        api_key=settings.deepseek_api_key,
        model=settings.deepseek_model,
        memory_block=block,
    )
    if not result.get("ok"):
        raise HTTPException(status_code=502, detail=result.get("error", "Claude error"))
    brief = str(result.get("brief", ""))
    conn.execute(
        "UPDATE context_brief SET brief = ?, updated_at = datetime('now') WHERE id = 1",
        (brief,),
    )
    conn.commit()
    return JSONResponse({"ok": True, "brief": brief})
