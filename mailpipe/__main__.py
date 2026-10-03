"""CLI. `./run.sh [options]` runs the pipeline; subcommands handle review and search."""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

from .config import load_config

app = typer.Typer(add_completion=False, no_args_is_help=False, help="Email pipeline prototype")
console = Console()
SUBCOMMANDS = {"run", "review-export", "review-import", "search", "report", "--help", "-h"}


class Redo(str, Enum):
    clean = "clean"
    entities = "entities"
    categorize = "categorize"
    analyze = "analyze"
    embed = "embed"


def setup_logging(out_dir: Path) -> Path:
    logs = out_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / f"run_{datetime.now():%Y%m%d_%H%M%S}.log"
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    fh.setLevel(logging.INFO)
    rh = RichHandler(console=console, show_path=False, level=logging.WARNING)
    logging.basicConfig(level=logging.INFO, handlers=[fh, rh], force=True)
    for noisy in ("httpx", "httpcore", "urllib3", "jieba", "huggingface_hub", "filelock", "extract_msg",
                  "RTFDE", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return path


@app.command("run")
def run(
    input: Optional[Path] = typer.Option(None, "--input", "-i", help="Folder with .msg/.xlsx files"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output folder"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="config.yaml path"),
    check_only: bool = typer.Option(False, "--check-only", help="Run preflight (and installs) only"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Approve installs and model downloads"),
    force: bool = typer.Option(False, "--force", help="Reprocess everything, ignoring cached state"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Process at most N new files"),
    cpu_only: bool = typer.Option(False, "--cpu-only", help="No NVIDIA GPU (dev machines): GPU checks warn only"),
    skip_preflight: bool = typer.Option(False, "--skip-preflight", help="Skip dependency checks"),
    redo: List[Redo] = typer.Option([], "--redo", help="Re-run a stage for all items (repeatable)"),
):
    """Check dependencies, install what is missing, process the folder, write the report."""
    cfg = load_config(config, input_dir=input, output_dir=output)
    log_path = setup_logging(Path(cfg.output_dir))
    if not skip_preflight:
        from .preflight import Preflight

        pf = Preflight(cfg, yes=yes, cpu_only=cpu_only,
                       check_input=input is not None or not check_only, log_dir=Path(cfg.output_dir) / "logs")
        ok = pf.run()
        if not ok:
            console.print("[red]Preflight failed; fix the FAIL items above.[/] Details: "
                          f"{Path(cfg.output_dir) / 'preflight.json'}")
            raise typer.Exit(1)
    if check_only:
        console.print("[green]Preflight passed.[/]")
        return
    if cfg.input_dir is None:
        console.print("[red]No input folder: pass --input or set input_dir in config.yaml[/]")
        raise typer.Exit(2)
    from .pipeline import Pipeline

    stats = Pipeline(cfg, force=force, limit=limit, cpu_only=cpu_only, redo=[r.value for r in redo]).run()
    console.print(f"done in {stats['total_s']}s · log: {log_path}")


@app.command("review-export")
def review_export(
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """Write out/review/review_<date>.xlsx for the specialist."""
    from .review import export_review
    from .store import Store

    cfg = load_config(config, output_dir=output)
    store = Store(Path(cfg.output_dir) / "mailpipe.db")
    path = export_review(store, cfg, Path(cfg.output_dir) / "review")
    console.print(f"[green]review workbook:[/] {path}")


@app.command("review-import")
def review_import(
    file: Path = typer.Option(..., "--file", "-f", exists=True, dir_okay=False),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    reviewer: Optional[str] = typer.Option(None, "--reviewer", help="Defaults to the name on the Info sheet"),
):
    """Import the specialist's corrections, recompute metrics, refresh the report."""
    from . import report
    from .pipeline import Paths
    from .review import compute_metrics, import_review
    from .store import Store

    cfg = load_config(config, output_dir=output)
    store = Store(Path(cfg.output_dir) / "mailpipe.db")
    res = import_review(store, cfg, file, reviewer)
    console.print(f"imported from {res['reviewer']}: {res['counts']}")
    for w in res["warnings"]:
        console.print(f"[yellow]warning:[/] {w}")
    m = compute_metrics(store, cfg)
    if m.get("category_accuracy") is not None:
        console.print(f"category accuracy {m['category_accuracy']:.1%} over {m['category_n']}")
    run_id = store.scalar("SELECT MAX(run_id) FROM runs")
    if run_id:
        html, _ = report.write_reports(store, cfg, Paths(Path(cfg.output_dir)), run_id)
        console.print(f"[green]report updated:[/] {html}")


@app.command("report")
def report_cmd(
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """Regenerate report.html / report.md from the database."""
    from . import report
    from .pipeline import Paths
    from .store import Store

    cfg = load_config(config, output_dir=output)
    store = Store(Path(cfg.output_dir) / "mailpipe.db")
    run_id = store.scalar("SELECT MAX(run_id) FROM runs")
    if not run_id:
        console.print("[red]no runs yet[/]")
        raise typer.Exit(1)
    html, _ = report.write_reports(store, cfg, Paths(Path(cfg.output_dir)), run_id)
    console.print(f"[green]report:[/] {html}")


@app.command("search")
def search_cmd(
    query: str = typer.Argument(...),
    k: int = typer.Option(5, "-k"),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """Hybrid search (vectors + full text) over the chunks, to sanity-check retrieval."""
    from .embed import Embedder, search
    from .store import Store

    cfg = load_config(config, output_dir=output)
    out = Path(cfg.output_dir)
    store = Store(out / "mailpipe.db")
    for i, r in enumerate(search(store, out / "vectors", Embedder(cfg), query, k), start=1):
        console.rule(f"{i}. {r['source_type']} · {r['date'] or ''} · {r['category'] or ''} · score {r['score']}")
        console.print(r["text"][:600], markup=False)


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] not in SUBCOMMANDS:
        argv = ["run", *argv]
    app(args=argv, prog_name="run.sh")


if __name__ == "__main__":
    main()
