"""A tiny stand-in for the Ollama HTTP API, for end-to-end tests without a GPU.

/api/chat answers with JSON generated from the request's `format` schema. Strings are written
in Simplified Chinese on purpose, so the OpenCC safety net is exercised.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

KEYWORDS = [("purchase_order", ["PO-", "訂單"]), ("rfq_quotation", ["詢價", "報價"]),
            ("issue_complaint", ["品質", "異常", "RMA"]), ("shipping_logistics", ["Shipment", "海關", "customs"]),
            ("schedule_meeting", ["meet", "sync", "會議"]), ("approval_request", ["核准", "approve"])]


class State:
    def __init__(self, models: list[str]):
        self.models = models
        self.loaded: set[str] = set()
        self.oom_models: set[str] = set()
        self.calls: list[dict] = []


def _gen(schema: dict, defs: dict, text: str):
    if "$ref" in schema:
        return _gen(defs[schema["$ref"].split("/")[-1]], defs, text)
    if "anyOf" in schema:
        options = [s for s in schema["anyOf"] if s.get("type") != "null"]
        return _gen(options[0], defs, text) if options else None
    if "enum" in schema:
        values = schema["enum"]
        for cat, words in KEYWORDS:
            if cat in values and any(w in text for w in words):
                return cat
        return "med" if "med" in values else ("zh" if "zh" in values else values[0])
    if "const" in schema:
        return schema["const"]
    t = schema.get("type")
    if t == "object":
        return {k: _gen(v, defs, text) for k, v in schema.get("properties", {}).items()}
    if t == "array":
        n = max(1, schema.get("minItems", 1))
        return [_gen(schema.get("items", {"type": "string"}), defs, text) for _ in range(n)]
    if t == "number":
        return 0.82
    if t == "integer":
        return 1
    if t == "boolean":
        return True
    return "请确认软件信息并回复"  # Simplified on purpose


def embed_text(text: str, dim: int = 32) -> list[float]:
    v = [0.0] * dim
    for i in range(len(text) - 2):
        h = int(hashlib.md5(text[i:i + 3].encode()).hexdigest()[:8], 16)
        v[h % dim] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def make_handler(state: State):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code: int, obj) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/tags":
                self._send(200, {"models": [{"name": m} for m in state.models]})
            elif self.path == "/api/ps":
                self._send(200, {"models": [{"name": m, "model": m} for m in sorted(state.loaded)]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            model = data.get("model")
            state.calls.append({"path": self.path, **{k: v for k, v in data.items() if k != "messages"}})
            if data.get("keep_alive") == 0:
                state.loaded.discard(model)
                if self.path == "/api/generate":
                    return self._send(200, {"done": True})
            if model in state.oom_models:
                return self._send(500, {"error": "llama runner process has terminated: CUDA error: out of memory"})
            if self.path == "/api/chat":
                state.loaded.add(model)
                schema = data.get("format") or {"type": "object"}
                text = "\n".join(m["content"] for m in data.get("messages", []))
                out = _gen(schema, schema.get("$defs", {}), text)
                return self._send(200, {"message": {"role": "assistant", "content": json.dumps(out, ensure_ascii=False)},
                                        "prompt_eval_count": len(text) // 2, "eval_count": 50})
            if self.path == "/api/embed":
                state.loaded.add(model)
                inputs = data["input"] if isinstance(data["input"], list) else [data["input"]]
                return self._send(200, {"embeddings": [embed_text(t) for t in inputs]})
            self._send(404, {"error": "not found"})

    return H


class FakeOllama:
    def __init__(self, models: list[str]):
        self.state = State(models)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.state))
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "FakeOllama":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
