# mailpipe

A local prototype that turns a folder of `.msg` and `.xlsx` files into categorized, structured, LLM-ready data. It is built for one Ubuntu box with an RTX 5070 Ti (16 GB), and nothing leaves the machine.

## Usage

```bash
./run.sh --input /data/mail --output ./out            # check, install, process, report
./run.sh --check-only                                 # preflight + installs only
./run.sh --input /data/mail --output ./out --yes      # approve installs/model pulls
./run.sh --input /data/mail --output ./out --force    # reprocess everything
./run.sh --input /data/mail --output ./out --limit 50 # subset
./run.sh --input /data/mail --output ./out --redo categorize   # re-run one stage (after a taxonomy change)

./run.sh review-export --output ./out
./run.sh review-import --file ./out/review/review_2026-10-10.xlsx --output ./out
./run.sh search "PO-20251002 海關" --output ./out   # hybrid retrieval check
./run.sh report --output ./out                     # regenerate the report
```

`--redo` accepts `clean`, `entities`, `categorize`, `analyze`, `embed` and can be repeated. `--cpu-only` lets the pipeline run on a machine without an NVIDIA GPU (dev laptops). GPU checks then only warn, and GLiNER runs on CPU.

## Output

```
out/
├── mailpipe.db          SQLite: files, emails, threads, attachments, sheets, entities, labels,
│                        action_items, chunks (+ chunks_fts trigram index), reviews, progress, runs
├── vectors/             LanceDB table "chunks" (bge-m3 vectors + chunk metadata)
├── tables/              each visible sheet as Parquet
├── chunks.jsonl         portable chunk export
├── review/              review workbooks
├── topic_discovery.md   HDBSCAN clusters vs. assigned categories
├── preflight.json
├── report.html / report.md
└── logs/run_<ts>.log
```

## How it runs

1. **Pass 1 (CPU)**: discover (SHA-256), parse, clean, detect language, build threads, run regex and date extraction.
2. **P2 GLiNER**: one GPU model at a time. Each phase checks the VRAM budget, loads the model, processes the batch, unloads it, and verifies that VRAM dropped.
3. **P3 LLM**: one call per email (category, confidence, needs_reply, priority, language, action items), then thread summaries, then summaries of long attachments.
4. **Chunking (CPU)**, then **P4 embeddings**.
5. **P5 discovery**: clustering on CPU, then cluster naming with the LLM. It only runs when labels changed.
6. **Report**.

Each stage records per-item progress (`progress` table), so a re-run only does new or failed work. A re-run with nothing new makes no LLM or embedding calls. Chunks rebuild only when their source changes, and only those chunks are re-embedded.

### OOM ladder
- **LLM**:
  1. Reload the model and retry.
  2. Use `num_ctx` 4096 with trimmed input.
  3. Switch to the fallback model.
  4. Mark the item `failed_oom` and continue.
  - After 3 fallbacks in a row, the phase stays on the fallback model.
- **GLiNER**: free the cache and retry, then halve the batch, then move to CPU, then `failed_oom`.
- **Embeddings**: reload and retry, then halve the batch, then CPU (`num_gpu=0`), then `failed_oom`.

Every label and summary records `method` and `model`. The report shows VRAM peaks per phase and every fallback.

## Where this differs from the plan, and why

- **Native Ollama API by default (`llm.api: ollama`), not `/v1`.** Ollama's OpenAI-compatible endpoint ignores `num_ctx`, `keep_alive` and `think`, and the VRAM strategy depends on them. `llm.api: openai` still switches to any OpenAI-compatible server (llama-server, Azure OpenAI) with config only.
- **Date extraction.** dateparser's free-text search is noisy on mixed zh/en mail. Explicit patterns handle ISO, ROC (`115/01/15`), `10月4日`, 「下週五 / 明天」 and "next Friday". dateparser is used only on English month-name spans.
- **GLiNER word splitting.** GLiNER's default splitter treats a whole Chinese sentence as one word. A jieba-based mixed splitter replaces it, which gave tighter spans on the samples. GLiNER hits that are really PO numbers, amounts, emails or phones are dropped, because regex owns those fields.
- **Review "missing entities"** is a column on the Categories sheet (one row per email), not on the Entities sheet (one row per entity).
- **Clustering** uses scikit-learn's built-in `HDBSCAN`, so no extra package is needed.
- **Added `sentencepiece`, `protobuf`, `jieba`, `httpx` and `packaging`.** `gliner_multi-v2.1` cannot load its tokenizer without sentencepiece.

## Things to know

- **Review rows are keyed by IDs.** Re-extracting entities (`--redo entities` or `--redo clean`) gives entities new IDs, so earlier entity reviews no longer match. Category, action-item and summary reviews survive re-runs.
- **Ollama started by systemd keeps its own environment.** If preflight warns about the VRAM settings, apply the `systemctl edit ollama` lines it prints (needs sudo once).
- **Token counts are a conservative estimate**: one per Han character, about 3.5 characters per token otherwise. The report counts prompts that came close to `num_ctx`.

## Tests

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests/
python tests/make_samples.py   # regenerate the synthetic fixtures
```

- **Fixtures**: `tests/samples` holds synthetic, anonymized `.msg` files written by `tests/msgwriter.py`, a minimal CFB/MSG writer. They cover:
  - zh-TW, English and mixed text
  - quoted replies, a signature and a disclaimer
  - a thread linked through Conversation-Index and In-Reply-To
  - a nested `.msg`, plus docx, xlsx (merged and hidden sheets) and pdf attachments
  - an HTML-only body and a standalone xlsx
- **End-to-end tests** use a fake Ollama server (`tests/fake_ollama.py`). It answers in Simplified Chinese on purpose, to exercise OpenCC, and can inject OOM errors.
- **GLiNER is faked** in the tests. The real model was checked by hand on the samples.

## Open questions (from the plan)

- Does Ollama run as a systemd service on the box? If so, the VRAM env settings need a one-time `systemctl edit`. Preflight detects this and prints the lines.
- Will anything else use the GPU during runs? Preflight lists GPU processes, and each phase re-checks its VRAM budget.
- The specialist's real categories: `taxonomy` in `config.yaml` is still the placeholder.
