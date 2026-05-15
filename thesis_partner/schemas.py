from typing import Any

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    text: str = Field(..., max_length=80_000)
    section_path: str = Field(..., max_length=500)
    note: str | None = Field(None, max_length=500)


class ChatRequest(BaseModel):
    message: str = Field(..., max_length=16_000)


class MemoryRequest(BaseModel):
    content: str = Field(..., max_length=32_000)
    section_path: str | None = Field(None, max_length=500)


class AnalyzeResponse(BaseModel):
    claude: dict[str, Any]
    gptzero: dict[str, Any]
    analysis_id: int | None = None
