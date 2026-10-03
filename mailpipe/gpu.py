"""GPU phase manager: VRAM monitor (NVML), budget checks, model unloading, OOM detection."""
from __future__ import annotations

import gc
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator

import httpx

log = logging.getLogger(__name__)

GB = 1024 ** 3


# --------------------------------------------------------------------------- NVML

class VRAM:
    """Thin NVML wrapper. All readings in GB; returns None when NVML is unavailable."""

    def __init__(self, index: int = 0):
        self.ok = False
        self.name = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nv = pynvml
            self._h = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = pynvml.nvmlDeviceGetName(self._h)
            self.name = name.decode() if isinstance(name, bytes) else name
            self.ok = True
        except Exception as e:
            log.debug("NVML unavailable: %s", e)

    def _mem(self):
        return self._nv.nvmlDeviceGetMemoryInfo(self._h) if self.ok else None

    def total_gb(self) -> float | None:
        m = self._mem()
        return m.total / GB if m else None

    def used_gb(self) -> float | None:
        m = self._mem()
        return m.used / GB if m else None

    def free_gb(self) -> float | None:
        m = self._mem()
        return m.free / GB if m else None

    def processes(self) -> list[dict]:
        if not self.ok:
            return []
        out = []
        try:
            for p in self._nv.nvmlDeviceGetComputeRunningProcesses(self._h):
                out.append({"pid": p.pid, "used_gb": (p.usedGpuMemory or 0) / GB})
        except Exception:
            pass
        return out


class Monitor:
    """Samples VRAM in a background thread and records the peak per phase."""

    def __init__(self, vram: VRAM, interval_s: float = 1.0):
        self.vram = vram
        self.interval = interval_s
        self.current: str | None = None
        self.peaks: dict[str, float] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if not self.vram.ok or self._thread:
            return
        self._thread = threading.Thread(target=self._run, name="vram-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def set_phase(self, name: str | None) -> None:
        with self._lock:
            self.current = name
        self.sample()

    def sample(self) -> None:
        used = self.vram.used_gb()
        if used is None:
            return
        with self._lock:
            if self.current:
                self.peaks[self.current] = max(self.peaks.get(self.current, 0.0), used)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.sample()


# --------------------------------------------------------------------------- Ollama admin

class OllamaAdmin:
    def __init__(self, base_url: str, timeout: float = 30):
        self.base = base_url.rstrip("/").removesuffix("/v1")
        self.http = httpx.Client(timeout=timeout)

    def is_up(self) -> bool:
        try:
            return self.http.get(f"{self.base}/api/tags").status_code == 200
        except httpx.HTTPError:
            return False

    def tags(self) -> list[str]:
        r = self.http.get(f"{self.base}/api/tags")
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    def loaded(self) -> list[dict]:
        try:
            r = self.http.get(f"{self.base}/api/ps")
            r.raise_for_status()
            return r.json().get("models", [])
        except httpx.HTTPError:
            return []

    def unload(self, model: str) -> None:
        try:
            # an embedding model rejects /api/generate; keep_alive=0 on /api/embed unloads it
            r = self.http.post(f"{self.base}/api/generate", json={"model": model, "keep_alive": 0})
            if r.status_code >= 400:
                self.http.post(f"{self.base}/api/embed", json={"model": model, "input": "", "keep_alive": 0})
        except httpx.HTTPError as e:
            log.warning("could not unload %s: %s", model, e)

    def unload_all(self) -> list[str]:
        names = [m.get("name") or m.get("model") for m in self.loaded()]
        for n in names:
            self.unload(n)
        return names


def model_name(name: str) -> str:
    return name if ":" in name else f"{name}:latest"


# --------------------------------------------------------------------------- OOM

class OOMError(RuntimeError):
    pass


def is_oom(exc: BaseException) -> bool:
    if isinstance(exc, OOMError):
        return True
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda" in msg or "cudamalloc" in msg or "oom" in msg.split()


def free_torch_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# --------------------------------------------------------------------------- phases

@dataclass
class Phase:
    name: str
    need_gb: float
    fallback: bool = False          # use the smaller / fallback model
    device: str = "cuda"            # cuda | cpu (for torch models and Ollama num_gpu=0)
    notes: list[str] = field(default_factory=list)
    baseline_gb: float | None = None
    peak_gb: float | None = None
    _exit: list[Callable[[], None]] = field(default_factory=list)

    def on_exit(self, fn: Callable[[], None]) -> None:
        self._exit.append(fn)

    def note(self, msg: str) -> None:
        log.warning("[%s] %s", self.name, msg)
        self.notes.append(msg)


class GPUManager:
    def __init__(self, cfg, *, cpu_only: bool = False):
        self.cfg = cfg
        self.cpu_only = cpu_only
        self.vram = VRAM()
        self.monitor = Monitor(self.vram, cfg.gpu.monitor_interval_s)
        self.ollama = OllamaAdmin(cfg.llm.base_url) if cfg.llm.api == "ollama" else None
        self.phases: dict[str, Phase] = {}
        self.monitor.start()

    def close(self) -> None:
        self.monitor.stop()

    def report(self) -> dict:
        return {
            name: {"peak_gb": round(p.peak_gb, 2) if p.peak_gb is not None else None,
                   "baseline_gb": round(p.baseline_gb, 2) if p.baseline_gb is not None else None,
                   "need_gb": p.need_gb, "fallback": p.fallback, "device": p.device, "notes": p.notes}
            for name, p in self.phases.items()
        }

    def _enough(self, need: float) -> bool | None:
        free = self.vram.free_gb()
        if free is None:
            return None
        return free >= need + self.cfg.gpu.safety_margin_gb

    def _free_idle_ollama(self, phase: Phase) -> None:
        if not self.ollama:
            return
        names = self.ollama.unload_all()
        if names:
            phase.note(f"unloaded idle Ollama models: {', '.join(names)}")
        self.wait_for_drop(self.cfg.gpu.unload_wait_s)

    def wait_for_drop(self, timeout: float, target_used_gb: float | None = None) -> None:
        if not self.vram.ok:
            time.sleep(min(timeout, 1))
            return
        deadline = time.monotonic() + timeout
        prev = self.vram.used_gb()
        while time.monotonic() < deadline:
            time.sleep(1)
            used = self.vram.used_gb()
            if target_used_gb is not None and used <= target_used_gb:
                return
            if target_used_gb is None and abs(used - prev) < 0.05:
                return
            prev = used

    @contextmanager
    def phase(self, name: str, need_gb: float, *, fallback_need_gb: float | None = None,
              cpu_ok: bool = False) -> Iterator[Phase]:
        """Run one GPU phase: budget check, then always unload on exit and verify VRAM dropped."""
        ph = Phase(name=name, need_gb=need_gb)
        self.phases[name] = ph
        if self.cpu_only:
            ph.device = "cpu"
            ph.note("--cpu-only: running on CPU")
        else:
            ok = self._enough(need_gb)
            if ok is False:
                ph.note(f"free VRAM {self.vram.free_gb():.1f}GB < need {need_gb}GB + margin")
                self._free_idle_ollama(ph)
                ok = self._enough(need_gb)
            if ok is False:
                if fallback_need_gb is not None and self._enough(fallback_need_gb):
                    ph.fallback = True
                    ph.note("switching to fallback model for this phase")
                elif cpu_ok:
                    ph.device = "cpu"
                    ph.note("not enough VRAM: running on CPU")
                elif fallback_need_gb is not None:
                    ph.fallback = True
                    ph.note("not enough VRAM even for the fallback model; trying it anyway")
                else:
                    ph.note("not enough VRAM; trying anyway, the OOM ladder will handle failures")
        ph.baseline_gb = self.vram.used_gb()
        self.monitor.set_phase(name)
        try:
            yield ph
        finally:
            for fn in reversed(ph._exit):
                try:
                    fn()
                except Exception as e:
                    log.warning("[%s] unload step failed: %s", name, e)
            free_torch_memory()
            self.monitor.sample()
            ph.peak_gb = self.monitor.peaks.get(name)
            self.monitor.set_phase(None)
            if self.vram.ok and ph.baseline_gb is not None:
                self.wait_for_drop(self.cfg.gpu.unload_wait_s, target_used_gb=ph.baseline_gb + 0.5)
                after = self.vram.used_gb()
                if after > ph.baseline_gb + 0.5:
                    ph.note(f"VRAM did not drop after phase ({after:.1f}GB used vs {ph.baseline_gb:.1f}GB before)")
