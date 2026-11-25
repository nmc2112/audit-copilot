"""FastAPI service: upload a FEC file and ask a question."""

from __future__ import annotations

import io
import logging
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .agent import AgentResult, AuditAgent
from .config import ROOT, load_settings
from .fec import FECFormatError, load_fec, parse_fec
from .llm import LLMClient, LLMError, build_client
from .ratelimit import RateLimitMiddleware, rate_limit_from_env

logging.basicConfig(level=logging.INFO, format="%(message)s")

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
SAMPLE_FEC = ROOT / "data" / "sample_fec.txt"


def create_app(llm: LLMClient | None = None) -> FastAPI:
    settings = load_settings()
    client = llm or build_client(settings.llm)
    app = FastAPI(title="Audit Copilot", version="0.1.0",
                  description="Ask a question about a FEC ledger. Leave the file empty to use the bundled "
                              "synthetic sample (data/sample_fec.txt).")
    if (limit := rate_limit_from_env()) > 0:
        app.add_middleware(RateLimitMiddleware, limit_per_hour=limit)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "prompt_version": settings.prompt_version,
                "tools": settings.tools.enabled}

    @app.post("/analyze", response_model=AgentResult)
    async def analyze(question: Annotated[str, Form(min_length=3, max_length=1000)],
                      fec: Annotated[UploadFile | None, File()] = None) -> AgentResult:
        if fec is None or not fec.filename:
            if not SAMPLE_FEC.exists():
                raise HTTPException(422, "No FEC file uploaded and no sample available")
            ledger = load_fec(SAMPLE_FEC)
        else:
            raw_bytes = await fec.read()
            if len(raw_bytes) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "FEC file too large")
            text = raw_bytes.decode("utf-8")
            sep = "\t" if "\t" in text.splitlines()[0] else "|"
            try:
                ledger = parse_fec(pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False))
            except FECFormatError as exc:
                raise HTTPException(422, str(exc)) from exc
        try:
            return AuditAgent(ledger, settings, client).run(question)
        except LLMError as exc:  # provider overloaded or down: clear 503 instead of a bare 500
            raise HTTPException(503, "LLM temporarily unavailable, retry in a minute.") from exc

    return app
