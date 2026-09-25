# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Ramen-themed Vision + RAG knowledge bot that runs **entirely locally and free** (no paid APIs / cloud) via Ollama. Planned features: ramen photo recognition, Japanese menu / ticket-machine (食券機) reading, ramen-knowledge Q&A with citations. QLoRA fine-tuning is an optional later phase. It is a job-hunt portfolio project, so the Before/After evidence in `docs/experiments.md` matters as much as the code.

Target hardware: RTX 3060 Ti, **8GB VRAM**. Keep models ~≤8B at 4-bit; only one 7B model fits in VRAM at a time.

## Status

Early stage — scripts only, no app, RAG pipeline, or tests yet.
- `scripts/test_vision.py`: single-image VLM smoke test; also holds shared `MODEL`, `NUM_CTX`, `load_resized()` (imported by other scripts, so run scripts from repo root as `python scripts/<name>.py`).
- `scripts/eval_baseline.py`: classifies every `samples/*.jpg` (label = filename prefix, e.g. `shio_3.jpg` → `shio`), writes `results/baseline_<variant>_<timestamp>.json`. Baseline accuracy: 17% (label-first), 4% (`--reason-first`) — details in `docs/experiments.md`.
- `scripts/eval_features.py`: asks the VLM only visible features (soup / clarity / color / noodle) as English enums and grades against `data/labels.csv` (hand-labeled; `noodle=unknown` rows are skipped). Result: soup 100%, clarity 74%, color 78%, noodle 47%. This "VLM sees, RAG judges" split is the chosen direction; next is the RAG knowledge docs.

`samples/` is gitignored: the user's photos contain GPS EXIF. Never commit originals; make EXIF-stripped, downscaled copies if images are needed in the repo.

## Commands (Windows, PowerShell)

```powershell
.\venv\Scripts\Activate.ps1                         # activate venv (Python 3.14)
pip install -r requirements.txt
python scripts/test_vision.py samples/shio_1.jpg    # VLM smoke test
python scripts/eval_baseline.py [--reason-first]    # style accuracy over all samples (~1–4 min)
python scripts/eval_features.py                     # visual-feature accuracy vs data/labels.csv (~30 s)
ollama list                                         # models: qwen2.5vl:7b (vision), bge-m3 (embeddings)
```

Environment gotchas:
- The Bash tool does not see `ollama` on PATH; use PowerShell and reload PATH first: `$env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')`.
- Set `$env:PYTHONIOENCODING='utf-8'` when running scripts from the tool shell, since output is Korean/Japanese.
- Ollama server runs at `localhost:11434` (started by the Ollama app).

## Architecture decisions

- **Images must be downscaled before sending to the VLM.** Qwen2.5-VL spends ~1 token per 28×28px patch; a 4000×3000 phone photo overflows Ollama's default 4096 context. Resize to long side 1024px (`MAX_SIDE`) and pass `options={"num_ctx": 8192}`.
- **Separate "seeing" from "judging".** Baseline showed the 7B VLM detects objects reasonably but hallucinates ramen type and even non-existent soup (see `docs/experiments.md`). Planned design: VLM extracts only visual features (soup present?, soup color, noodle thickness, toppings) → RAG over ramen-knowledge docs decides the style and cites sources.
- **Structured Output guards for the 7B VLM:** it falls into repetition loops (esp. free-form Korean), so always set `num_predict`, cap arrays with `maxItems`, and treat JSON parse failures as `invalid` rather than crashing.
- Embeddings: `bge-m3` (multilingual KO/JA/EN). Planned vector store Chroma (`chroma_db/`, gitignored), reusing patterns from the author's earlier RAG project (Chroma, FastAPI).
- Possible integration: the user's RamenLog app (`Mal-Mi-Jal/Ramen`, Spring Boot) plans receipt-OCR visit verification — a consumer for this project's vision model.
