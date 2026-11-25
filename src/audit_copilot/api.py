"""FastAPI service: upload a FEC file and ask a question."""

from __future__ import annotations

import io
import logging
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .agent import AgentResult, AuditAgent
from .config import load_settings
from .fec import FECFormatError, parse_fec
from .llm import LLMClient, build_client

logging.basicConfig(level=logging.INFO, format="%(message)s")

MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def create_app(llm: LLMClient | None = None) -> FastAPI:
    settings = load_settings()
    client = llm or build_client(settings.llm)
    app = FastAPI(title="Audit Copilot", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "prompt_version": settings.prompt_version,
                "tools": settings.tools.enabled}

    @app.post("/analyze", response_model=AgentResult)
    async def analyze(question: Annotated[str, Form(min_length=3, max_length=1000)],
                      fec: Annotated[UploadFile, File()]) -> AgentResult:
        raw_bytes = await fec.read()
        if len(raw_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "FEC file too large")
        text = raw_bytes.decode("utf-8")
        sep = "\t" if "\t" in text.splitlines()[0] else "|"
        try:
            ledger = parse_fec(pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False))
        except FECFormatError as exc:
            raise HTTPException(422, str(exc)) from exc
        return AuditAgent(ledger, settings, client).run(question)

    return app
