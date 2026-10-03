"""LLM client with JSON-schema output and the OOM fallback ladder.

Two transports, picked by `llm.api`:
- ollama: native /api/chat. Needed for per-request num_ctx, keep_alive, and think, which
  Ollama's OpenAI-compatible /v1 endpoint ignores.
- openai: any OpenAI-compatible server (llama-server, Azure OpenAI) via the openai client.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .gpu import OllamaAdmin, OOMError, Phase, is_oom

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
Builder = Callable[[int], list[dict]]  # max_input_tokens -> messages


class LLMError(RuntimeError):
    pass


@dataclass
class CallMeta:
    model: str
    method: str                  # primary | reload_retry | reduced_ctx | fallback
    fallback_used: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class LLMStats:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    oom_events: int = 0
    fallbacks: int = 0
    failures: int = 0
    seconds: float = 0.0
    truncation_warnings: int = 0
    by_method: dict[str, int] = field(default_factory=dict)


class LLM:
    STICKY_FALLBACK_AFTER = 3

    def __init__(self, cfg, phase: Phase | None = None, stats: LLMStats | None = None):
        self.cfg = cfg
        self.lc = cfg.llm
        self.phase = phase
        self.stats = stats or LLMStats()
        self.admin = OllamaAdmin(self.lc.base_url) if self.lc.api == "ollama" else None
        self.http = httpx.Client(timeout=self.lc.timeout_s)
        self._openai = None
        self._consecutive_fallbacks = 0
        self.sticky_fallback = bool(phase and phase.fallback and self.lc.fallback_model)
        self.loaded: set[str] = set()
        if phase is not None:
            phase.on_exit(self.unload)

    # -- transport -----------------------------------------------------------
    def _ollama_chat(self, model: str, messages: list[dict], schema: dict, num_ctx: int) -> tuple[str, int, int]:
        body = {
            "model": model, "messages": messages, "format": schema, "stream": False, "keep_alive": "15m",
            "options": {"temperature": self.lc.temperature, "num_ctx": num_ctx,
                        "num_predict": self.lc.max_output_tokens},
        }
        if self.lc.think is not None:
            body["think"] = self.lc.think
        if self.phase is not None and self.phase.device == "cpu":
            body["options"]["num_gpu"] = 0
        base = self.lc.base_url.rstrip("/").removesuffix("/v1")
        r = self.http.post(f"{base}/api/chat", json=body)
        if r.status_code >= 400:
            try:
                err = r.json().get("error", r.text)
            except ValueError:
                err = r.text
            if is_oom(RuntimeError(err)):
                raise OOMError(err)
            raise LLMError(f"ollama {r.status_code}: {err}")
        data = r.json()
        self.loaded.add(model)
        return data["message"]["content"], data.get("prompt_eval_count", 0), data.get("eval_count", 0)

    def _openai_chat(self, model: str, messages: list[dict], schema: dict, num_ctx: int) -> tuple[str, int, int]:
        if self._openai is None:
            from openai import OpenAI

            self._openai = OpenAI(base_url=self.lc.base_url, api_key=self.lc.api_key, timeout=self.lc.timeout_s)
        try:
            resp = self._openai.chat.completions.create(
                model=model, messages=messages, temperature=self.lc.temperature,
                max_tokens=self.lc.max_output_tokens,
                response_format={"type": "json_schema", "json_schema": {"name": "output", "schema": schema}},
            )
        except Exception as e:
            if is_oom(e):
                raise OOMError(str(e)) from e
            raise LLMError(str(e)) from e
        u = resp.usage
        return resp.choices[0].message.content or "", getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0)

    def _once(self, model: str, messages: list[dict], out: type[T], num_ctx: int, tries: int = 3) -> tuple[T, int, int]:
        schema = out.model_json_schema()
        last: Exception | None = None
        for attempt in range(tries):
            t0 = time.monotonic()
            try:
                send = self._ollama_chat if self.lc.api == "ollama" else self._openai_chat
                text, pt, ct = send(model, messages, schema, num_ctx)
            except httpx.TimeoutException as e:
                last = LLMError(f"timeout: {e}")
                continue
            except httpx.HTTPError as e:
                last = LLMError(f"http: {e}")
                time.sleep(2 * (attempt + 1))
                continue
            finally:
                self.stats.seconds += time.monotonic() - t0
            self.stats.calls += 1
            self.stats.prompt_tokens += pt
            self.stats.completion_tokens += ct
            if pt and pt >= num_ctx - self.lc.max_output_tokens:
                self.stats.truncation_warnings += 1
                log.warning("prompt used %s tokens of num_ctx %s; input may have been truncated", pt, num_ctx)
            try:
                return out.model_validate_json(_strip_fences(text)), pt, ct
            except ValidationError as e:
                last = LLMError(f"invalid output: {e.errors()[:2]}")
                log.info("retrying after invalid JSON from %s", model)
        raise last or LLMError("no response")

    # -- ladder ----------------------------------------------------------------
    def call(self, build: Builder, out: type[T]) -> tuple[T, CallMeta]:
        lc = self.lc
        primary = lc.fallback_model if self.sticky_fallback else lc.model
        reduced_in = max(512, lc.reduced_num_ctx - lc.max_output_tokens - 512)
        ladder: list[tuple[str, str, int, int]] = [
            ("primary", primary, lc.num_ctx, lc.max_input_tokens),
            ("reload_retry", primary, lc.num_ctx, lc.max_input_tokens),
            ("reduced_ctx", primary, lc.reduced_num_ctx, reduced_in),
        ]
        if lc.fallback_model and primary != lc.fallback_model:
            ladder.append(("fallback", lc.fallback_model, lc.reduced_num_ctx, reduced_in))
        oom_seen: Exception | None = None
        for step, (method, model, num_ctx, max_in) in enumerate(ladder):
            if step == 1:
                self.unload()      # free memory and reload the model
            if method == "fallback":
                self.unload()
            try:
                result, pt, ct = self._once(model, build(max_in), out, num_ctx)
            except OOMError as e:
                self.stats.oom_events += 1
                oom_seen = e
                log.warning("OOM on %s (%s): %s", model, method, str(e)[:200])
                continue
            fallback_used = model != lc.model
            if method == "fallback":
                self.stats.fallbacks += 1
                self._consecutive_fallbacks += 1
                if self._consecutive_fallbacks >= self.STICKY_FALLBACK_AFTER and not self.sticky_fallback:
                    self.sticky_fallback = True
                    if self.phase:
                        self.phase.note(f"{self._consecutive_fallbacks} OOMs in a row: staying on {lc.fallback_model}")
            elif step == 0:
                self._consecutive_fallbacks = 0
            self.stats.by_method[method] = self.stats.by_method.get(method, 0) + 1
            return result, CallMeta(model, method, fallback_used, pt, ct)
        self.stats.failures += 1
        raise OOMError(f"failed_oom after fallback ladder: {oom_seen}")

    def unload(self) -> None:
        if self.admin:
            for m in {self.lc.model, self.lc.fallback_model} & (self.loaded | {self.lc.model}):
                if m:
                    self.admin.unload(m)
            self.loaded.clear()


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    # some models wrap JSON in prose; take the outermost object
    if not t.startswith("{"):
        a, b = t.find("{"), t.rfind("}")
        if a >= 0 and b > a:
            t = t[a:b + 1]
    return t


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)
