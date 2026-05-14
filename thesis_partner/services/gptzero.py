from __future__ import annotations

from typing import Any

import httpx


GPTZERO_URL = "https://api.gptzero.me/v2/predict/text"


async def scan_text(api_key: str, text: str) -> dict[str, Any]:
    if not api_key:
        return {"ok": False, "error": "GPTZERO_API_KEY is not set"}

    headers = {"x-api-key": api_key, "Content-Type": "application/json"}
    # API accepts several shapes; try common JSON body.
    bodies = [
        {"document": text},
        {"input_text": text},
        {"text": text},
    ]
    last_err: str | None = None
    async with httpx.AsyncClient(timeout=120.0) as client:
        for body in bodies:
            try:
                r = await client.post(GPTZERO_URL, headers=headers, json=body)
                if r.status_code == 200:
                    data = r.json()
                    return {"ok": True, "raw": data, "summary": _summarize(data)}
                last_err = f"HTTP {r.status_code}: {r.text[:500]}"
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
    return {"ok": False, "error": last_err or "Unknown GPTZero error"}


def _summarize(data: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not isinstance(data, dict):
        return out
    doc = data.get("documents")
    if isinstance(doc, list) and doc:
        first = doc[0] if isinstance(doc[0], dict) else {}
        for key in (
            "completely_generated_prob",
            "average_generated_prob",
            "predicted_class",
            "confidence_score",
            "document_classification",
        ):
            if key in first:
                out[key] = first[key]
    for key in ("completely_generated_prob", "average_generated_prob", "predicted_class"):
        if key in data:
            out[key] = data[key]
    return out
