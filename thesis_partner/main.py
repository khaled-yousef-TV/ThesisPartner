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
from thesis_partner.binder import SECTION_GROUPS, SIDEBAR_TREE
from thesis_partner.config import Settings, get_settings
from thesis_partner.schemas import AnalyzeRequest, AnalyzeResponse, ChatRequest, GrammarFixRequest, MemoryRequest
from thesis_partner.services import claude, gptzero

ROOT = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(ROOT / "templates"))


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


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(
    body: AnalyzeRequest,
    conn: sqlite3.Connection = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> AnalyzeResponse:
    if len(body.text) > settings.max_analyze_chars:
        raise HTTPException(status_code=400, detail="Text exceeds configured limit")

    thesis_context = load_thesis_context(conn)
    gz_task = gptzero.scan_text(settings.gptzero_api_key, body.text)
    cl_task = claude.analyze_draft(
        api_key=settings.anthropic_api_key,
        model=settings.claude_model,
        section_path=body.section_path,
        note=body.note,
        draft=body.text,
        thesis_context=thesis_context,
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
    conn.commit()
    analysis_id = int(cur.lastrowid)

    return AnalyzeResponse(claude=claude_payload, gptzero=gptzero_payload, analysis_id=analysis_id)

@app.post("/api/grammar-fix")
async def grammar_fix(
    body: GrammarFixRequest,
    settings: Settings = Depends(get_settings),
) -> JSONResponse:
    limit = settings.grammar_fix_max_chars
    if len(body.text) > limit:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Quick grammar fix is limited to {limit:,} characters at once "
                "(output length). Shorten the selection or run it on one section at a time."
            ),
        )
    result = await claude.quick_grammar_fix(
        api_key=settings.anthropic_api_key,
        model=settings.claude_model,
        text=body.text,
    )
    if not result.get("ok"):
        raise HTTPException(status_code=502, detail=result.get("error", "Claude error"))
    return JSONResponse({"ok": True, "text": str(result.get("text", ""))})


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
    result = await claude.chat_turn(
        api_key=settings.anthropic_api_key,
        model=settings.claude_model,
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
    result = await claude.refresh_brief(
        api_key=settings.anthropic_api_key,
        model=settings.claude_model,
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
