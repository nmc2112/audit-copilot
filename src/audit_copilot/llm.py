"""Provider-agnostic LLM clients with retries, backoff and fallback.

Any server exposing the OpenAI-compatible /chat/completions API works:
Ollama (local), Mistral, OpenAI, vLLM... A scripted FakeLLM is provided so the
agent can be tested without network access or model costs.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Protocol

import httpx
from pydantic import BaseModel, Field

from .config import LLMConfig, ModelConfig

log = logging.getLogger("audit_copilot.llm")

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict = Field(default_factory=dict)
    raw_arguments: str = ""
    parse_error: str | None = None


class LLMResponse(BaseModel):
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             json_mode: bool = False) -> LLMResponse: ...


def _parse_tool_calls(message: dict) -> list[ToolCall]:
    calls = []
    for i, tc in enumerate(message.get("tool_calls") or []):
        fn = tc.get("function", {})
        raw = fn.get("arguments") or "{}"
        if isinstance(raw, dict):  # some servers return a dict instead of a JSON string
            calls.append(ToolCall(id=tc.get("id") or f"call_{i}", name=fn.get("name", ""),
                                  arguments=raw, raw_arguments=json.dumps(raw)))
            continue
        try:
            args, error = json.loads(raw), None
        except json.JSONDecodeError as exc:
            args, error = {}, f"Invalid JSON arguments: {exc}"
        calls.append(ToolCall(id=tc.get("id") or f"call_{i}", name=fn.get("name", ""),
                              arguments=args if isinstance(args, dict) else {},
                              raw_arguments=raw, parse_error=error))
    return calls


class OpenAICompatibleClient:
    """Calls one model, retrying transient failures with exponential backoff."""

    def __init__(self, model_cfg: ModelConfig, llm_cfg: LLMConfig,
                 transport: httpx.BaseTransport | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.cfg, self.llm_cfg, self._sleep = model_cfg, llm_cfg, sleep
        headers = {"Authorization": f"Bearer {model_cfg.api_key}"} if model_cfg.api_key else {}
        self._http = httpx.Client(base_url=model_cfg.base_url, timeout=llm_cfg.timeout_s,
                                  headers=headers, transport=transport)

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             json_mode: bool = False) -> LLMResponse:
        payload: dict = {"model": self.cfg.model, "messages": messages,
                         "temperature": self.llm_cfg.temperature}
        if tools:
            payload["tools"] = tools
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        last_error: Exception | None = None
        for attempt in range(1, self.llm_cfg.max_retries + 1):
            start = time.perf_counter()
            try:
                resp = self._http.post("/chat/completions", json=payload)
                if resp.status_code in RETRYABLE_STATUS:
                    raise LLMError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()
                data = resp.json()
                message = data["choices"][0]["message"]
                usage = data.get("usage") or {}
                return LLMResponse(
                    content=message.get("content") or "",
                    tool_calls=_parse_tool_calls(message),
                    model=data.get("model", self.cfg.model),
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    latency_s=round(time.perf_counter() - start, 3),
                )
            except (httpx.TransportError, LLMError) as exc:
                last_error = exc
                wait = self.llm_cfg.backoff_s * 2 ** (attempt - 1)
                log.warning("model=%s attempt=%d failed: %s (retry in %.1fs)",
                            self.cfg.model, attempt, exc, wait)
                if attempt < self.llm_cfg.max_retries:
                    self._sleep(wait)
            except httpx.HTTPStatusError as exc:  # 4xx other than retryable: do not retry
                raise LLMError(f"Non-retryable error from {self.cfg.model}: {exc}") from exc
        raise LLMError(f"{self.cfg.model} failed after {self.llm_cfg.max_retries} attempts: {last_error}")


class FallbackClient:
    """Tries clients in order; moves to the next one when a client gives up."""

    def __init__(self, clients: list[LLMClient]):
        if not clients:
            raise ValueError("At least one LLM client is required")
        self.clients = clients

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             json_mode: bool = False) -> LLMResponse:
        errors = []
        for client in self.clients:
            try:
                return client.chat(messages, tools=tools, json_mode=json_mode)
            except LLMError as exc:
                errors.append(str(exc))
                log.warning("falling back after error: %s", exc)
        raise LLMError("All LLM clients failed: " + " | ".join(errors))


def build_client(llm_cfg: LLMConfig) -> LLMClient:
    clients: list[LLMClient] = [OpenAICompatibleClient(llm_cfg.primary, llm_cfg)]
    if llm_cfg.fallback and llm_cfg.fallback.available:
        clients.append(OpenAICompatibleClient(llm_cfg.fallback, llm_cfg))
    return FallbackClient(clients)


class FakeLLM:
    """Deterministic stand-in for tests: replays a list of scripted responses."""

    def __init__(self, script: list[LLMResponse | Callable[[list[dict]], LLMResponse]]):
        self.script = list(script)
        self.calls: list[dict] = []

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             json_mode: bool = False) -> LLMResponse:
        self.calls.append({"messages": list(messages), "tools": tools, "json_mode": json_mode})
        if not self.script:
            raise LLMError("FakeLLM script exhausted")
        step = self.script.pop(0)
        return step(messages) if callable(step) else step
