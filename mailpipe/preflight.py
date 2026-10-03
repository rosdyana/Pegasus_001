"""Preflight: dependency checks with auto-install. Every check reports OK, FIXED, WARN, SKIP, or FAIL."""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

from .config import PROJECT_DIR
from .gpu import OllamaAdmin, model_name

console = Console()
TORCH_INDEX = os.environ.get("MAILPIPE_TORCH_INDEX", "https://download.pytorch.org/whl/cu128")
TORCH_SMOKE = (
    "import json, torch\n"
    "ok = torch.cuda.is_available()\n"
    "r = {'version': torch.__version__, 'cuda': torch.version.cuda, 'available': ok}\n"
    "if ok:\n"
    "    x = torch.randn(256, 256, device='cuda'); y = (x @ x).sum().item()\n"
    "    r['device'] = torch.cuda.get_device_name(0); r['capability'] = list(torch.cuda.get_device_capability(0))\n"
    "print(json.dumps(r))\n"
)


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""
    fix: str = ""


class Preflight:
    def __init__(self, cfg, *, yes: bool = False, cpu_only: bool = False, check_input: bool = True,
                 log_dir: Path | None = None):
        self.cfg = cfg
        self.yes = yes
        self.cpu_only = cpu_only
        self.check_input = check_input
        self.log_dir = log_dir
        self.results: list[Check] = []
        self.started_ollama = False

    # -- helpers ---------------------------------------------------------------
    def add(self, name: str, status: str, detail: str = "", fix: str = "") -> Check:
        c = Check(name, status, detail, fix)
        self.results.append(c)
        color = {"OK": "green", "FIXED": "cyan", "WARN": "yellow", "SKIP": "dim", "FAIL": "red"}[status]
        console.print(f"[{color}]{status:5}[/] {name}" + (f"  [dim]{detail}[/]" if detail else ""))
        if fix and status in ("FAIL", "WARN"):
            console.print(f"      [dim]fix:[/] {fix}")
        return c

    def gpu_fail(self) -> str:
        return "WARN" if self.cpu_only else "FAIL"

    def confirm(self, question: str) -> bool:
        if self.yes:
            return True
        if not sys.stdin.isatty():
            console.print(f"[yellow]{question} -> no (non-interactive; pass --yes to approve)[/]")
            return False
        return Confirm.ask(question, default=True)

    def pip(self, *args: str) -> bool:
        cmd = [sys.executable, "-m", "pip", "install", "-q", *args]
        console.print(f"      [dim]$ {' '.join(cmd)}[/]")
        return subprocess.run(cmd).returncode == 0

    @property
    def ok(self) -> bool:
        return not any(c.status == "FAIL" for c in self.results)

    # -- checks ------------------------------------------------------------------
    def run(self) -> bool:
        console.rule("preflight")
        for fn in (self.python_version, self.virtualenv, self.nvidia, self.torch_cuda, self.packages,
                   self.ollama_binary, self.ollama_server, self.ollama_env, self.gpu_processes, self.models,
                   self.gliner_model, self.input_folder, self.output_folder, self.disk, self.free_vram):
            try:
                fn()
            except Exception as e:  # a check crashing is itself a failure, not a crash of the tool
                self.add(fn.__name__, "FAIL", f"{type(e).__name__}: {e}")
        self.write()
        return self.ok

    def python_version(self):
        v = sys.version_info
        if v >= (3, 10):
            self.add("Python >= 3.10", "OK", platform.python_version())
        else:
            self.add("Python >= 3.10", "FAIL", platform.python_version(), "install python3.12 and re-run ./run.sh")

    def virtualenv(self):
        if sys.prefix != sys.base_prefix:
            self.add("virtualenv active", "OK", sys.prefix)
        else:
            self.add("virtualenv active", "FAIL", "running on system Python", "start through ./run.sh")

    def nvidia(self):
        if not shutil.which("nvidia-smi"):
            self.add("NVIDIA driver + GPU", self.gpu_fail(), "nvidia-smi not found",
                     "install NVIDIA driver 570+: sudo ubuntu-drivers install")
            return
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True)
        if out.returncode != 0:
            self.add("NVIDIA driver + GPU", self.gpu_fail(), out.stderr.strip()[:200], "check the driver install")
            return
        line = out.stdout.strip().splitlines()[0]
        driver = line.split(",")[-1].strip()
        major = int(driver.split(".")[0]) if driver.split(".")[0].isdigit() else 0
        if major and major < 570:
            self.add("NVIDIA driver + GPU", self.gpu_fail(), line, "Blackwell needs driver 570+")
        else:
            self.add("NVIDIA driver + GPU", "OK", line)

    def _torch_probe(self) -> dict | None:
        r = subprocess.run([sys.executable, "-c", TORCH_SMOKE], capture_output=True, text=True)
        if r.returncode != 0:
            return {"error": (r.stderr.strip().splitlines() or ["import failed"])[-1]}
        return json.loads(r.stdout.strip().splitlines()[-1])

    def torch_cuda(self):
        probe = self._torch_probe()
        if probe.get("available"):
            self.add("PyTorch CUDA on GPU", "OK",
                     f"torch {probe['version']} cuda {probe['cuda']} on {probe.get('device')} sm{probe.get('capability')}")
            return
        if self.cpu_only or not shutil.which("nvidia-smi"):
            status = "OK" if "version" in probe else "FAIL"
            self.add("PyTorch (CPU)", status, probe.get("version") or probe.get("error", ""),
                     "" if status == "OK" else f"{sys.executable} -m pip install torch")
            return
        console.print(f"      torch not usable on the GPU ({probe.get('error') or probe.get('version')}); "
                      f"installing the cu128 build")
        if self.pip("--force-reinstall", "torch", "--index-url", TORCH_INDEX):
            probe = self._torch_probe()
            if probe.get("available"):
                self.add("PyTorch CUDA on GPU", "FIXED", f"torch {probe['version']} on {probe.get('device')}")
                return
        self.add("PyTorch CUDA on GPU", "FAIL", probe.get("error") or "cuda not available",
                 f"{sys.executable} -m pip install --force-reinstall torch --index-url {TORCH_INDEX}")

    def _missing_packages(self) -> list[str]:
        from importlib.metadata import PackageNotFoundError, version

        from packaging.requirements import Requirement

        missing = []
        for line in (PROJECT_DIR / "requirements.txt").read_text().splitlines():
            line = line.split("#")[0].strip()
            if not line:
                continue
            req = Requirement(line)
            try:
                v = version(req.name)
                if req.specifier and not req.specifier.contains(v, prereleases=True):
                    missing.append(f"{req.name} ({v} does not satisfy {req.specifier})")
            except PackageNotFoundError:
                missing.append(req.name)
        return missing

    def packages(self):
        missing = self._missing_packages()
        if not missing:
            self.add("Python packages", "OK", "requirements.txt satisfied")
            return
        console.print(f"      missing: {', '.join(missing)}")
        if self.pip("-r", str(PROJECT_DIR / "requirements.txt")) and not self._missing_packages():
            self.add("Python packages", "FIXED", f"installed {len(missing)}")
        else:
            self.add("Python packages", "FAIL", ", ".join(self._missing_packages()),
                     f"{sys.executable} -m pip install -r requirements.txt")

    def ollama_binary(self):
        if self.cfg.llm.api != "ollama":
            self.add("Ollama binary", "SKIP", f"llm.api = {self.cfg.llm.api}")
            return
        path = shutil.which("ollama")
        if path:
            self.add("Ollama binary", "OK", path)
        else:
            self.add("Ollama binary", "FAIL", "not on PATH", "curl -fsSL https://ollama.com/install.sh | sh")

    def ollama_server(self):
        admin = OllamaAdmin(self.cfg.llm.base_url)
        if admin.is_up():
            self.add("Ollama server", "OK", admin.base)
            return
        if self.cfg.llm.api != "ollama":
            self.add("LLM server", "FAIL", f"{self.cfg.llm.base_url} not reachable", "start the server")
            return
        if not shutil.which("ollama"):
            self.add("Ollama server", "FAIL", "not running and no binary to start it")
            return
        env = {**os.environ, **self.cfg.gpu.ollama_env}
        log_dir = self.log_dir or Path(".")
        log_dir.mkdir(parents=True, exist_ok=True)
        logf = open(log_dir / "ollama_serve.log", "ab")
        subprocess.Popen(["ollama", "serve"], env=env, stdout=logf, stderr=logf, start_new_session=True)
        for _ in range(30):
            time.sleep(1)
            if admin.is_up():
                self.started_ollama = True
                self.add("Ollama server", "FIXED", f"started `ollama serve` with VRAM settings (log: {logf.name})")
                return
        self.add("Ollama server", "FAIL", "started `ollama serve` but it did not answer in 30s",
                 f"see {logf.name}")

    def _ollama_process_env(self) -> tuple[dict[str, str] | None, str]:
        pids = subprocess.run(["pgrep", "-x", "ollama"], capture_output=True, text=True).stdout.split()
        for pid in pids:
            p = Path(f"/proc/{pid}/environ")
            try:
                raw = p.read_bytes()
            except OSError:
                continue
            env = dict(kv.split("=", 1) for kv in raw.decode(errors="replace").split("\0") if "=" in kv)
            if "OLLAMA_HOST" in env or any(k.startswith("OLLAMA_") for k in env) or "PATH" in env:
                return env, f"/proc/{pid}/environ"
        if shutil.which("systemctl"):
            out = subprocess.run(["systemctl", "show", "ollama", "--property=Environment"],
                                 capture_output=True, text=True).stdout.strip()
            if out.startswith("Environment="):
                env = dict(kv.split("=", 1) for kv in out[len("Environment="):].split() if "=" in kv)
                return env, "systemd unit"
        return None, ""

    def ollama_env(self):
        want = self.cfg.gpu.ollama_env
        if self.cfg.llm.api != "ollama" or not want:
            self.add("Ollama VRAM settings", "SKIP")
            return
        if self.started_ollama:
            self.add("Ollama VRAM settings", "OK", "set when starting ollama serve")
            return
        if not OllamaAdmin(self.cfg.llm.base_url).is_up():
            self.add("Ollama VRAM settings", "SKIP", "server not running")
            return
        env, source = self._ollama_process_env()
        lines = "\n".join(f'      Environment="{k}={v}"' for k, v in want.items())
        fix = ("sudo systemctl edit ollama   # add under [Service]:\n" + lines +
               "\n      then: sudo systemctl restart ollama")
        if env is None:
            self.add("Ollama VRAM settings", "WARN", "cannot read the Ollama server environment", fix)
            return
        wrong = {k: env.get(k) for k, v in want.items() if str(env.get(k, "")).lower() != str(v).lower()}
        if wrong:
            self.add("Ollama VRAM settings", "WARN",
                     f"from {source}: " + ", ".join(f"{k}={v or 'unset'}" for k, v in wrong.items()), fix)
        else:
            self.add("Ollama VRAM settings", "OK", f"from {source}")

    def gpu_processes(self):
        others = []
        if shutil.which("nvidia-smi"):
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 3 and "ollama" not in parts[1].lower():
                    others.append(f"{parts[1]} (pid {parts[0]}, {parts[2]} MiB)")
        admin = OllamaAdmin(self.cfg.llm.base_url)
        loaded = [m.get("name") for m in admin.loaded()] if self.cfg.llm.api == "ollama" else []
        if loaded:
            if self.confirm(f"Ollama has models loaded ({', '.join(loaded)}). Unload them?"):
                for m in loaded:
                    admin.unload(m)
                self.add("Other GPU users", "FIXED", f"unloaded {', '.join(loaded)}")
            else:
                self.add("Other GPU users", "WARN", f"Ollama models still loaded: {', '.join(loaded)}")
        if others:
            self.add("Other GPU users", "WARN", "; ".join(others), "these are left alone; free VRAM is re-checked per phase")
        elif not loaded:
            self.add("Other GPU users", "OK", "none")

    def models(self):
        if self.cfg.llm.api != "ollama":
            self.add("Ollama models pulled", "SKIP")
            return
        admin = OllamaAdmin(self.cfg.llm.base_url)
        if not admin.is_up():
            self.add("Ollama models pulled", "FAIL", "server not running")
            return
        need = [m for m in (self.cfg.llm.model, self.cfg.llm.fallback_model, self.cfg.embedding.model) if m]
        have = {model_name(t) for t in admin.tags()}
        missing = [m for m in need if model_name(m) not in have]
        if not missing:
            self.add("Ollama models pulled", "OK", ", ".join(need))
            return
        if not self.confirm(f"Pull missing Ollama models {', '.join(missing)}?"):
            self.add("Ollama models pulled", "FAIL", f"missing: {', '.join(missing)}",
                     " && ".join(f"ollama pull {m}" for m in missing))
            return
        failed = [m for m in missing if subprocess.run(["ollama", "pull", m]).returncode != 0]
        if failed:
            self.add("Ollama models pulled", "FAIL", f"pull failed: {', '.join(failed)}")
        else:
            self.add("Ollama models pulled", "FIXED", f"pulled {', '.join(missing)}")

    def gliner_model(self):
        from huggingface_hub import snapshot_download, try_to_load_from_cache

        repo = self.cfg.gliner.model
        if Path(repo).exists():
            self.add("GLiNER model cached", "OK", f"local path {repo}")
            return
        # the config alone can be cached by an interrupted download: require the weights too
        cached = isinstance(try_to_load_from_cache(repo, "gliner_config.json"), str) and any(
            isinstance(try_to_load_from_cache(repo, f), str) for f in ("pytorch_model.bin", "model.safetensors"))
        if cached:
            self.add("GLiNER model cached", "OK", repo)
            return
        console.print(f"      downloading {repo} (about 1.2 GB)")
        snapshot_download(repo)
        self.add("GLiNER model cached", "FIXED", f"downloaded {repo}")

    def input_folder(self):
        d = self.cfg.input_dir
        if not self.check_input:
            self.add("Input folder", "SKIP", "pass --input to check it")
            return
        if d is None or not Path(d).is_dir():
            self.add("Input folder", "FAIL", f"{d} is not a directory", "pass --input /path/to/mail")
            return
        from .discover import scan

        files = scan(Path(d))
        if not files:
            self.add("Input folder", "FAIL", f"no .msg or .xlsx files under {d}")
        else:
            n_msg = sum(1 for f in files if f.suffix.lower() == ".msg")
            self.add("Input folder", "OK", f"{n_msg} .msg + {len(files) - n_msg} .xlsx under {d}")

    def output_folder(self):
        d = Path(self.cfg.output_dir)
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".write_test"
            probe.write_text("ok")
            probe.unlink()
            self.add("Output folder writable", "OK", str(d.resolve()))
        except OSError as e:
            self.add("Output folder writable", "FAIL", str(e))

    def disk(self):
        d = Path(self.cfg.output_dir)
        free = shutil.disk_usage(d if d.exists() else Path(".")).free / 1024 ** 3
        self.add("Free disk space", "OK" if free >= 20 else "WARN", f"{free:.0f} GB free")

    def free_vram(self):
        from .gpu import VRAM

        v = VRAM()
        if not v.ok:
            self.add("Free VRAM", "SKIP" if self.cpu_only else "WARN", "NVML not available")
            return
        busy = [p for p in v.processes() if p["used_gb"] > 4]
        detail = f"{v.free_gb():.1f} of {v.total_gb():.1f} GB free"
        if busy:
            self.add("Free VRAM", "WARN", detail + "; another process holds > 4GB")
        else:
            self.add("Free VRAM", "OK", detail)

    # -- output ------------------------------------------------------------------
    def write(self) -> None:
        out = Path(self.cfg.output_dir)
        try:
            out.mkdir(parents=True, exist_ok=True)
            (out / "preflight.json").write_text(json.dumps({
                "ok": self.ok, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "cpu_only": self.cpu_only,
                "checks": [asdict(c) for c in self.results]}, indent=2, ensure_ascii=False))
        except OSError:
            pass
        t = Table(show_header=False, box=None)
        counts = {s: sum(1 for c in self.results if c.status == s) for s in ("OK", "FIXED", "WARN", "SKIP", "FAIL")}
        t.add_row(*(f"{k}: {v}" for k, v in counts.items()))
        console.print(t)
