"""config.yaml loading and validation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

PROJECT_DIR = Path(__file__).resolve().parent.parent


class LLMCfg(BaseModel):
    api: str = "ollama"
    base_url: str = "http://localhost:11434"
    api_key: str = "ollama"
    model: str = "qwen3:14b"
    fallback_model: str | None = "qwen3:8b"
    temperature: float = 0
    num_ctx: int = 8192
    reduced_num_ctx: int = 4096
    max_input_tokens: int = 6000
    max_output_tokens: int = 1500
    think: bool | str | None = False
    vram_gb: float = 11
    fallback_vram_gb: float = 7
    timeout_s: float = 600
    attachment_summary_min_chars: int = 3000

    @field_validator("api")
    @classmethod
    def _api(cls, v: str) -> str:
        if v not in ("ollama", "openai"):
            raise ValueError("llm.api must be 'ollama' or 'openai'")
        return v


class EmbeddingCfg(BaseModel):
    model: str = "bge-m3"
    batch_size: int = 32
    vram_gb: float = 2
    cpu_fallback: bool = True


class GPUCfg(BaseModel):
    safety_margin_gb: float = 1.5
    monitor_interval_s: float = 1
    unload_wait_s: float = 30
    ollama_env: dict[str, str] = Field(default_factory=dict)

    @field_validator("ollama_env", mode="before")
    @classmethod
    def _env(cls, v):
        return {str(k): str(val) for k, val in (v or {}).items()}


class LanguageCfg(BaseModel):
    output: str = "zh-TW"
    english_only_threads: str = "en"
    opencc_profile: str = "s2twp"


class GLiNERCfg(BaseModel):
    model: str = "urchade/gliner_multi-v2.1"
    labels: list[str] = Field(default_factory=lambda: ["company", "person", "product model", "deadline", "project"])
    label_aliases: dict[str, str] = Field(default_factory=dict)
    use_label_aliases: bool = False
    threshold: float = 0.5
    batch_size: int = 8
    fp16: bool = True
    vram_gb: float = 2
    window_words: int = 300
    window_overlap: int = 30
    max_chars_per_attachment: int = 20000


class CleanCfg(BaseModel):
    disclaimers: list[str] = Field(default_factory=list)
    boilerplate_min_emails: int = 3
    boilerplate_min_same_sender: int = 2


class Category(BaseModel):
    name: str
    description: str


class Thresholds(BaseModel):
    review_confidence: float = 0.6
    needs_reply_days: int = 3
    deadline_days: int = 7


class ChunkCfg(BaseModel):
    target_tokens: int = 500
    sheet_rows_per_chunk: int = 30


class DiscoveryCfg(BaseModel):
    min_emails: int = 10
    samples_per_cluster: int = 5


class Config(BaseModel):
    input_dir: Path | None = None
    output_dir: Path = Path("./out")
    llm: LLMCfg = Field(default_factory=LLMCfg)
    embedding: EmbeddingCfg = Field(default_factory=EmbeddingCfg)
    gpu: GPUCfg = Field(default_factory=GPUCfg)
    language: LanguageCfg = Field(default_factory=LanguageCfg)
    gliner: GLiNERCfg = Field(default_factory=GLiNERCfg)
    clean: CleanCfg = Field(default_factory=CleanCfg)
    taxonomy: list[Category] = Field(default_factory=lambda: [Category(name="other", description="Anything else")])
    regex: dict[str, str] = Field(default_factory=dict)
    thresholds: Thresholds = Field(default_factory=Thresholds)
    chunk: ChunkCfg = Field(default_factory=ChunkCfg)
    discovery: DiscoveryCfg = Field(default_factory=DiscoveryCfg)
    internal_domains: list[str] = Field(default_factory=list)

    @field_validator("taxonomy")
    @classmethod
    def _taxonomy(cls, v: list[Category]) -> list[Category]:
        names = [c.name for c in v]
        if len(set(names)) != len(names):
            raise ValueError("taxonomy names must be unique")
        if "other" not in names:
            v.append(Category(name="other", description="Anything that does not fit above"))
        return v

    @property
    def category_names(self) -> list[str]:
        return [c.name for c in self.taxonomy]

    def hash(self) -> str:
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def resolve_config_path(path: Path | None) -> Path:
    if path is not None:
        return path
    local = Path("config.yaml")
    return local if local.exists() else PROJECT_DIR / "config.yaml"


def load_config(path: Path | None = None, **overrides) -> Config:
    p = resolve_config_path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}
    for k, v in overrides.items():
        if v is not None:
            data[k] = v
    return Config.model_validate(data or {})
