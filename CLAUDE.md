# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Ramen-themed Vision + RAG bot that runs **entirely locally and free** (no paid APIs / cloud) via Ollama: upload a ramen photo → style + a sourced Korean explanation. Other planned features (not built): Japanese menu / ticket-machine (食券機) reading, ramen-knowledge text Q&A, optional QLoRA. It is a job-hunt portfolio project, so the evidence in `docs/experiments.md` matters as much as the code. Public repo: `Mal-Mi-Jal/ramen-vision-bot`. `README.md` is the portfolio front page; `docs/tech-notes.md` explains design rationale (incl. a Java-to-Python reading guide).

Target hardware: RTX 3060 Ti, **8GB VRAM**. Keep models ~≤8B at 4-bit; only one 7B model fits in VRAM at a time.

## Status

8 styles (shio, shoyu, miso, donkotsu, ieke, toripaitan, karai, aburasoba). Held-out Wikimedia set (`samples_test/`, 25 photos): pipeline 60% vs VLM-only 28%. Tuning set (`samples/`, 21 photos): 76%. (These are the published numbers, measured on Ollama 0.34.4 — for new work compare against "Current baseline" below.) Details in `docs/experiments.md` 실험 1–11. Non-ramen photos are rejected by a gate (실험 10). Local FastAPI demo works. Model-free logic is covered by pytest (`tests/`, ~60 tests, <1 s, no Ollama/Chroma needed). Eval results record their environment and can be compared per photo (실험 11).

Scope decisions: shoyupaitan was dropped (not a major style) — `samples/shoyupaitan_*.jpg` stay on disk but aren't graded. A tsukemen draft was started and removed (see "Adding a style").

## Architecture: "VLM sees, RAG judges"

Core logic is the `ramen_bot/` package (installed editable via `-e .` in requirements.txt). Flow: `judge.identify(path)` = `vision.is_noodle_dish` (gate) → `vision.extract_features` → `judge.judge`; `api.py` then calls `answer.explain`.

- **Gate** (`vision.is_noodle_dish`, 실험 10): a separate VLM call that picks noodle_bowl / other_food / not_food; anything but noodle_bowl → `identify` returns style `not_ramen` (no retrieval/judge), `answer.explain` returns `ok: False, kind: "not_ramen"`, and the page shows a notice instead of an error. Keep it a **separate call**: putting `is_ramen` into the feature prompt shifted existing features. It is deliberately lenient ("if unsure, noodle_bowl") — asking "are noodles visible?" rejected milky-broth bowls where noodles are submerged. Only 5 non-ramen photos were tested (also used while wording it), so a fresh negative set is still needed.

- `vision.py` — VLM (`qwen2.5vl:7b`) returns only visible features as English enums (soup / clarity / color / noodle) + a yes/no `TOPPINGS` checklist, via Ollama Structured Output. **Never ask it for the style** (VLM-only was 17%). Images are downscaled to long side 1024px (`MAX_SIDE`) with `num_ctx=8192`: Qwen2.5-VL spends ~1 token per 28×28px, and a phone photo overflows the default 4096 context.
- `knowledge.py` — splits `knowledge/*.md` into `## ` sections, embeds with `bge-m3` via `ollama.embed` (passed explicitly; Chroma's default embedder is unused), stores in Chroma (`chroma_db/`, gitignored). Chunk id = `<file stem>#<heading>`; metadata `style`, `title`, `heading`, `sources`, `soup`. "프로젝트 관찰" sections are excluded (they describe the test photos → leakage).
- `judge.py` — features → Korean query (`KO`, `TOPPING_KO`) → **hybrid retrieval**: metadata filter on `soup` + vector search over only "사진으로 구분하는 법" sections (k=5) → candidates' criteria + "비슷한 종류와 구분" → LLM (same VLM in text mode) picks from a **candidate-only enum**, short English reason first. Sources are attached by code, not the LLM. One candidate → no LLM call.
- `answer.py` — builds the Korean display payload by **assembling doc sections** (요약 / 사진으로 구분하는 법 / 유래); no LLM-written Korean (the 7B loops/hallucinates in free Korean). Hides internal `soup=…, color=…` codes.
- `api.py` + `web/index.html` — FastAPI: `GET /` page, `POST /api/identify` (multipart `file`). Sync `def` endpoint on purpose (threadpool). Upload → temp file (`delete=False` for Windows) → deleted after. Frontend HTML-escapes doc text before its mini-markdown rendering.

Structured Output guards (always): `temperature=0`, `num_predict` cap, `maxItems` on arrays, JSON parse failure → `invalid` instead of crashing.

## Knowledge docs contract (`knowledge/*.md`)

- One doc per style; **filename stem must equal the frontmatter `style`** (chunk ids use the stem, lookups use the style). Plus `00_soup_basics.md` (清湯 vs 白湯, `style: basics`).
- Frontmatter: `style`, `title`, `visual` (same enum words as `vision.py`; `visual.soup` drives the retrieval filter), `sources` (URLs).
- Headings the code depends on verbatim: "요약", "유래", "사진으로 구분하는 법" (only the style's **own positive cues**, in the same Korean words as `judge.KO` — comparison sentences here turned docs into retrieval "hubs"), "비슷한 종류와 구분" (comparisons), "프로젝트 관찰" (notes from our photos; never indexed).
- Fetch real sources and cite them; do not write ramen facts from memory.
- After any doc edit: `python scripts/build_index.py`. The set of styles is simply the docs present (`knowledge.available_styles()`).

## Adding a style

1. Write the doc per the contract above (sourced). 2. If the style needs a cue the VLM isn't asked about, add it to `vision.TOPPINGS` **and** `judge.TOPPING_KO` (both — one without the other is dead code). 3. Rebuild the index. 4. Measure on `samples/`.
- Tsukemen caveat: noodles come without soup but with a separate dipping bowl. The VLM will likely say `soup=no`, and the soup filter then leaves only aburasoba (no LLM call). Supporting it needs a new visual feature (e.g. separate dipping bowl) and a filter change, not just a doc.

## Evaluation discipline

- Tune only against `samples/`; `samples_test/` is for final checks (tuning on it would require a new held-out set). Change one thing at a time and report accuracy **and** retrieval recall.
- Labels come from filenames (`shio_3.jpg` → `shio`); `data/labels.csv` has hand-labeled visual features (+ `note` for user-confirmed edge cases). `data/test_sources.csv` lists author/license/URL for every `samples_test/` photo — keep it in sync.
- Settings already tested (실험 9, don't redo): k=3 lowers recall; 768px is ~12% faster with inconclusive accuracy (held-out 60% is at 1024); adding common toppings (soft-boiled egg) is detected well but adds query noise — only add toppings that discriminate styles.
- Known weak spots: dark shoyu broth read as "cloudy"; judge picks ieke from nori alone; cloudy-but-yellowish broth (real tori paitan, some ieke) read as `pale_gold` → shio; embeddings ignore negation ("국물 없음"), hence the soup metadata filter.

## Commands (Windows, PowerShell)

```powershell
.\venv\Scripts\Activate.ps1                         # activate venv (Python 3.14)
pip install -r requirements.txt
python scripts/build_index.py ["검색어"]            # rebuild Chroma after editing knowledge/; optional query shows top hits
uvicorn ramen_bot.api:app --port 8000               # demo at http://localhost:8000 (needs chroma_db; restart after Python edits)
python scripts/eval_rag.py [--samples samples_test] [--compare results/<baseline>.json]  # accuracy + recall (~2 min); env (Ollama version, model digests, settings, git commit) is saved in summary.env
python scripts/compare_results.py OLD.json NEW.json # per-photo diff: fixed / broken / changed, retrieval, timing; warns if env or photo set differs (no GPU)
python scripts/eval_baseline.py [--samples DIR] [--reason-first]  # VLM-only baseline (8 styles; not comparable to the 9-style 17% in 실험 4)
python scripts/eval_features.py                     # visual-feature accuracy vs data/labels.csv
python scripts/test_vision.py samples/shio_1.jpg    # free-form VLM smoke test
pip install -r requirements-dev.txt; python -m pytest  # unit tests (no Ollama/Chroma; run after every code change)
ollama list                                         # qwen2.5vl:7b (vision + judge), bge-m3 (embeddings)
```

Results are written to `results/*.json` (committed). `.claude/launch.json` has a `ramen-demo` entry for the browser preview.

Environment gotchas:
- The Bash tool does not see `ollama` on PATH; use PowerShell and reload PATH first: `$env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')`.
- Set `$env:PYTHONIOENCODING='utf-8'` when running scripts from the tool shell (Korean/Japanese output).
- Ollama server runs at `localhost:11434` (started by the Ollama app). It auto-updated 0.34.4 → 0.40.3 between 실험 9 and 10; that alone changed one held-out prediction and runs got slower (~6–7 s per feature extraction with VRAM nearly full). Compare numbers only within the same Ollama version.

## Current baseline (compare new work against this)

Measured 2026-10-20 with **Ollama 0.40.6** at commit `ed3a3b9` (실험 11; `results/rag_samples_20261020_1012.json`, `results/rag_samples_test_20261020_1017.json` — both carry `summary.env`). Run new work with `--compare <these files>`. Ollama auto-updated 0.40.3 → 0.40.6 with no code change and the held-out set went 16/26 → 18/26 (ieke_3/ieke_4 fixed, toripaitan_2 changed) — that is **not** an improvement of ours; compare only within one Ollama version. A same-condition rerun reproduced all 26 predictions exactly (temperature 0); timing varies run to run (5.7 vs 4.4 s), so don't judge speed from one run. Older numbers (README, 실험 1–9 on 0.34.4, 실험 10 on 0.40.3) are not directly comparable.

| Set | Accuracy | Retrieval recall | Time/photo |
|---|---|---|---|
| `samples/` (tuning, 21) | 16/21 = 76% | 19/21 | ~6 s |
| `samples_test/` (held-out, 26 = 실험 8's 25 + user photo `ieke_5`) | 18/26 = 69% | 23/26 | ~5 s |
| Non-ramen gate check (5 photos in a temp folder, not in the repo) | 5/5 rejected | — | +~1 s |

Tuning-set misses: donkotsu_1→ieke, karai_1→ieke, shio_4→ieke, toripaitan_1→donkotsu (a tomato read as beni shoga), toripaitan_2→shio. Held-out misses: aburasoba_1→shoyu, donkotsu_1→toripaitan, miso_1→ieke, miso_4→shoyu, shoyu_3→shio, toripaitan_1/toripaitan_2/toripaitan_3→shio (toripaitan_2 flips with each Ollama version: shio → ieke → shio). The held-out set has now been looked at several times (its failures shaped the backlog below), so **after fixing a held-out weakness, a fresh held-out set is needed** before claiming a held-out number.

## Development backlog (prioritized; pick one per session)

1. **Negative set for the gate.** Build ~20 non-ramen photos (other food incl. pasta/udon/pho, objects, scenery) in a gitignored folder with a license list like `data/test_sources.csv`; report false-accept / false-reject. The gate wording was tuned on only 5 negatives.
2. **"Nori alone → ieke"** (3 of 5 tuning misses end in ieke; 1 held-out miss, miso_1). The judge treats large nori as decisive; ieke needs spinach **and** nori. Options: a rule in the judge prompt, or doc wording in `knowledge/ieke.md` "사진으로 구분하는 법". Measure on `samples/` only.
3. **Yellowish cloudy broth → shio** (real tori paitan, some ieke; held-out toripaitan_1/3 and the toripaitan↔ieke confusions). The VLM says `pale_gold`; the judge ignores `clarity=cloudy`. Options: make shio's criteria explicitly require clear broth, or add a clarity-aware query/filter. Any doc fact (e.g. "tori paitan can look yellowish") must be sourced first.
4. ~~Automated tests (pytest)~~ — done 2026-10-14 (`tests/`). `tests/conftest.py` blocks real `ollama.chat`/`embed` and `chromadb.PersistentClient`; tests monkeypatch the names **where they are used** (e.g. `judge.is_noodle_dish`, not `vision.is_noodle_dish`, because judge.py imports it with `from ... import`). `test_knowledge.py` enforces the docs contract ("유래" is optional — `karai.md` has none). Benchmark tooling (env recording + per-photo compare) done 2026-10-20 (실험 11).
5. **Speed.** The ~2× slowdown seen on 0.40.3 is mostly gone on 0.40.6 (~5–6 s/photo, close to September); the cause (version vs VRAM pressure that day) wasn't recorded. Only worth revisiting if it regresses — measure over 2+ runs and record `ollama ps` GPU %.
6. **Tsukemen** (see "Adding a style"): needs a `dipping_bowl` visual feature and a 3-way soup filter, plus a sourced doc and photos.
7. **Ramen-knowledge text Q&A** (planned feature): text question → retrieval over all sections → short grounded answer with citations; fills the empty "After" rows for Before 1–2 in `docs/experiments.md`. Needs guards for free Korean generation (length cap, "모르면 모른다", citations by code).
8. **Japanese menu / ticket-machine (食券機) reading** (planned feature, OCR with the same VLM).

## Workflow for a change

1. Re-run the baseline if the Ollama version changed (`ollama --version`).
2. Change **one thing**; run `python -m pytest`, then `scripts/eval_rag.py --compare <baseline>` on `samples/` (+ the gate/negative check if the gate is touched) and read the per-photo diff, not just the total. Commit before measuring so `summary.env.git_dirty` is false.
3. Log it as the next 실험 N in `docs/experiments.md` — including failed attempts and reverts — and rename superseded result files (`rag_samples_v<N>a_*.json` pattern).
4. Update this file's baseline/backlog, README "한계와 다음 단계" if user-facing, then commit (noreply email). Push and Notion updates only when the user asks.

## Data and privacy

- `samples/` and `samples_test/` are gitignored: the user's photos contain GPS EXIF, and Wikimedia photos require attribution. Never commit originals; use EXIF-stripped, downscaled copies if an image must be in the repo.
- Commits use the GitHub noreply email (repo-local git config).

## Conventions

- Code comments are in Korean.
- Log each new result, reverted experiment, or notable error in `docs/experiments.md`.
- Keep diffs minimal — don't reformat whole files (an editor's format-on-save once turned a one-line change into a 300-line diff).
- Possible integration: the author's RamenLog app (Spring Boot) plans receipt-OCR visit verification — a consumer for this project's vision model.
- Personal working preferences live in `CLAUDE.local.md` (gitignored).
