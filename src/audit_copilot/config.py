"""Typed loading of the versioned YAML configuration and prompts."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel


def _find_root() -> Path:
    """Project root holding config/ and prompts/ (works installed or from source)."""
    if os.environ.get("AUDIT_COPILOT_HOME"):
        return Path(os.environ["AUDIT_COPILOT_HOME"])
    if (Path.cwd() / "config" / "settings.yaml").exists():
        return Path.cwd()
    return Path(__file__).resolve().parents[2]


ROOT = _find_root()


class ModelConfig(BaseModel):
    base_url: str
    model: str
    api_key_env: str | None = None

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None

    @property
    def available(self) -> bool:
        return self.api_key_env is None or bool(self.api_key)


class LLMConfig(BaseModel):
    primary: ModelConfig
    fallback: ModelConfig | None = None
    timeout_s: float = 120
    max_retries: int = 3
    backoff_s: float = 1.0
    temperature: float = 0.0


class AgentConfig(BaseModel):
    max_iterations: int = 6
    max_items_per_tool: int = 20
    report_max_attempts: int = 3


class ToolsConfig(BaseModel):
    enabled: list[str] = []


class Settings(BaseModel):
    prompt_version: str
    llm: LLMConfig
    agent: AgentConfig = AgentConfig()
    tools: ToolsConfig = ToolsConfig()

    def prompt(self, name: str) -> str:
        return (ROOT / "prompts" / self.prompt_version / f"{name}.md").read_text(encoding="utf-8")


def load_settings(path: str | Path | None = None) -> Settings:
    if path is None:
        path = os.environ.get("AUDIT_COPILOT_CONFIG") or ROOT / "config" / "settings.yaml"
    path = Path(path)
    return Settings.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
