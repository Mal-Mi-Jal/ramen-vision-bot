# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Ramen-themed Vision + RAG knowledge bot that runs **entirely locally and free** (no paid APIs / cloud) via Ollama. Planned features: ramen photo recognition, Japanese menu / ticket-machine (食券機) reading, ramen-knowledge Q&A with citations. QLoRA fine-tuning is an optional later phase. It is a job-hunt portfolio project, so the Before/After evidence in `docs/experiments.md` matters as much as the code.

Target hardware: RTX 3060 Ti, **8GB VRAM**. Keep models ~≤8B at 4-bit; only one 7B model fits in VRAM at a time.

## Status

Working RAG pipeline for photo → style over **8 styles** (shio, shoyu, miso, donkotsu, ieke, toripaitan, karai, aburasoba): on the **held-out** Wikimedia set (`samples_test/`, 25 photos) RAG = 60% vs VLM-only 28%; on the tuning set (`samples/`, 21 photos) 76% (`docs/experiments.md` 실험 6–8). No app/UI or tests yet.

Evaluation discipline: tune only against `samples/`; use `samples_test/` for final checks only (tuning on it would require a new held-out set). `samples_test/` holds CC-licensed Wikimedia Commons photos (gitignored; attribution/license per file in `data/test_sources.csv` — keep that file in sync if photos change). Known held-out weakness: cloudy-but-yellowish broth (real tori paitan, some ieke) gets `pale_gold` and the judge picks shio, ignoring `clarity`.

Scope: shoyupaitan was dropped on purpose (user: not a major style). The set of styles = the `knowledge/*.md` docs; `eval_rag.py` skips photos whose label has no doc (`knowledge.available_styles()`), so `samples/shoyupaitan_*.jpg` stay on disk but aren't graded.

Core logic lives in the `ramen_bot/` package (installed editable via `-e .` in requirements.txt, so scripts can `import ramen_bot` from anywhere):
- `vision.py`: `MODEL`, `NUM_CTX`, `load_resized()`, `extract_features()` — VLM returns only visible features (soup / clarity / color / noodle enums + a yes/no `TOPPINGS` checklist). Never ask it for the style.
- `knowledge.py`: splits `knowledge/*.md` into `## ` sections, embeds with `bge-m3` via Ollama, stores in Chroma (`chroma_db/`). Metadata per chunk: `style`, `heading`, `sources`, `soup` (from frontmatter). Chunk id = `<style>#<heading>`. "프로젝트 관찰" sections are excluded (they describe the test photos → data leakage).
- `judge.py`: features → Korean query → **hybrid retrieval** (metadata filter on `soup` + vector search over only the "사진으로 구분하는 법" sections, k=5) → candidates' criteria + "비슷한 종류와 구분" sections → LLM picks a style from the candidate enum. Sources are attached by code from the chosen style's doc, not written by the LLM. One candidate → no LLM call.

Scripts (run from repo root):
- `scripts/test_vision.py`: single-image free-form VLM smoke test.
- `scripts/eval_baseline.py`: classifies every `samples/*.jpg` (label = filename prefix, e.g. `shio_3.jpg` → `shio`), writes `results/baseline_<variant>_<timestamp>.json`. Baseline accuracy: 17% (label-first), 4% (`--reason-first`) — details in `docs/experiments.md`.
- `scripts/eval_features.py`: asks the VLM only visible features (soup / clarity / color / noodle) as English enums and grades against `data/labels.csv` (hand-labeled; `noodle=unknown` rows are skipped). Result: soup 100%, clarity 74%, color 78%, noodle 47%. This "VLM sees, RAG judges" split is the chosen direction.
- `scripts/build_index.py ["검색어"]`: rebuild Chroma from `knowledge/` (required after any doc edit); optional query prints top hits.
- `scripts/eval_rag.py [--samples DIR]`: full pipeline over `samples/` (or DIR), reports accuracy + retrieval recall (was the true style among candidates), writes `results/rag_<timestamp>.json`.
- `knowledge/*.md`: RAG source docs, one per style (filename = style label) plus `00_soup_basics.md` (清湯 vs 白湯). YAML frontmatter: `style`, `visual` (same enum vocabulary as `vision.py`), `sources`. Sections the pipeline depends on by exact heading: "사진으로 구분하는 법" (only the style's own positive cues, phrased in the same Korean words as `judge.KO` — comparisons here turned docs into retrieval "hubs") and "비슷한 종류와 구분" (comparisons). Keep sourced facts out of "프로젝트 관찰".
- Known weak spots (don't re-discover): VLM calls dark shoyu broth "cloudy"; judge picks ieke from nori alone; embeddings ignore negation ("국물 없음") — that's why soup is a metadata filter.
- `scripts/eval_baseline.py [--samples DIR]` now offers 8 styles (shoyupaitan removed), so its numbers aren't directly comparable to the 9-style 17% in 실험 4.
- `data/labels.csv` `note` column marks edge cases the user confirmed (e.g. shio_2 chicken-heavy, shio_4 clam broth looks amber).

`samples/` is gitignored: the user's photos contain GPS EXIF. Never commit originals; make EXIF-stripped, downscaled copies if images are needed in the repo.

## Commands (Windows, PowerShell)

```powershell
.\venv\Scripts\Activate.ps1                         # activate venv (Python 3.14)
pip install -r requirements.txt
python scripts/test_vision.py samples/shio_1.jpg    # VLM smoke test
python scripts/eval_baseline.py [--reason-first]    # style accuracy over all samples (~1–4 min)
python scripts/eval_features.py                     # visual-feature accuracy vs data/labels.csv (~30 s)
python scripts/build_index.py                       # rebuild Chroma after editing knowledge/
python scripts/eval_rag.py [--samples samples_test] # full RAG pipeline accuracy (~2 min); --samples = held-out folder
ollama list                                         # models: qwen2.5vl:7b (vision), bge-m3 (embeddings)
```

Environment gotchas:
- The Bash tool does not see `ollama` on PATH; use PowerShell and reload PATH first: `$env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')`.
- Set `$env:PYTHONIOENCODING='utf-8'` when running scripts from the tool shell, since output is Korean/Japanese.
- Ollama server runs at `localhost:11434` (started by the Ollama app).

## Architecture decisions

- **Images must be downscaled before sending to the VLM.** Qwen2.5-VL spends ~1 token per 28×28px patch; a 4000×3000 phone photo overflows Ollama's default 4096 context. Resize to long side 1024px (`MAX_SIDE`) and pass `options={"num_ctx": 8192}`.
- **Separate "seeing" from "judging".** Baseline showed the 7B VLM detects objects reasonably but hallucinates ramen type and even non-existent soup (see `docs/experiments.md`). Implemented in `ramen_bot/` (see Status).
- **Structured Output guards for the 7B VLM:** it falls into repetition loops (esp. free-form Korean), so always set `num_predict`, cap arrays with `maxItems`, and treat JSON parse failures as `invalid` rather than crashing.
- Embeddings: `bge-m3` (multilingual KO/JA/EN) via `ollama.embed`, passed explicitly to Chroma (Chroma's default embedding function is not used). `chroma_db/` is gitignored and rebuilt by `build_index.py`. Reuses patterns from the author's earlier RAG project (Chroma, FastAPI).
- The judge reuses `qwen2.5vl:7b` in text mode: two 7B models don't fit in 8GB VRAM together.
- Possible integration: the user's RamenLog app (`Mal-Mi-Jal/Ramen`, Spring Boot) plans receipt-OCR visit verification — a consumer for this project's vision model.
