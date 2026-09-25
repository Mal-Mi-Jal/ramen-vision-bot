"""시각 특징 추출 정확도 측정: VLM에게 라멘 종류는 묻지 않고, 눈에 보이는 특징만 선택형으로 묻는다.

정답: data/labels.csv (soup / clarity / color / noodle). noodle=unknown 인 사진은 면 채점에서 제외.
결과: results/features_<날짜시간>.json

사용법:
    python scripts/eval_features.py
"""
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import ollama

from test_vision import MODEL, NUM_CTX, load_resized

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT / "samples"
LABELS_CSV = ROOT / "data" / "labels.csv"
RESULTS_DIR = ROOT / "results"

FEATURES = {
    "soup": ["yes", "no"],
    "clarity": ["clear", "cloudy", "none"],
    "color": ["pale_gold", "dark_brown", "light_brown", "creamy_white", "red_orange", "none"],
    "noodle": ["thin", "medium", "thick", "unknown"],
}

# 영어로 묻는다: 7B 모델은 영어가 가장 강하고, 한국어 자유 서술은 반복 루프에 잘 빠졌다 (docs/experiments.md Before 4)
# 라멘 지식(어떤 종류가 어떤 국물인지)은 주지 않고, 각 선택지가 "어떻게 보이는지"만 정의한다
PROMPT = """Look at the ramen bowl in this photo. Answer ONLY about what is visible. Do not guess the ramen style.

- soup: "yes" if the noodles sit in liquid broth. "no" if it is a soupless mixed-noodle dish (at most a little sauce at the bottom).
- clarity: "clear" if the broth is transparent enough to see noodles through it. "cloudy" if the broth is opaque or milky. "none" if there is no soup.
- color: the broth color.
  "pale_gold" = light yellow and transparent,
  "dark_brown" = dark soy-sauce color,
  "light_brown" = beige or tan,
  "creamy_white" = milky white,
  "red_orange" = chili red or orange,
  "none" = no soup.
- noodle: noodle thickness. "thin" = about spaghetti thickness or thinner, "medium", "thick" = clearly thick and chewy, "unknown" = noodles are hidden."""

SCHEMA = {
    "type": "object",
    "properties": {name: {"type": "string", "enum": options} for name, options in FEATURES.items()},
    "required": list(FEATURES),
}


def extract(path: Path) -> dict:
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT, "images": [load_resized(path, verbose=False)]}],
        format=SCHEMA,
        options={"num_ctx": NUM_CTX, "temperature": 0, "num_predict": 128},
    )
    try:
        return json.loads(response["message"]["content"])
    except json.JSONDecodeError:
        return {name: "invalid" for name in FEATURES}


def main() -> None:
    with LABELS_CSV.open(encoding="utf-8") as f:
        labels = list(csv.DictReader(f))

    rows = []
    for i, gt in enumerate(labels, 1):
        start = time.time()
        pred = extract(SAMPLES_DIR / gt["file"])
        elapsed = time.time() - start
        marks = " ".join(
            f"{name}={pred[name]}{'' if gt[name] == 'unknown' else ('✓' if pred[name] == gt[name] else '✗(' + gt[name] + ')')}"
            for name in FEATURES
        )
        print(f"[{i}/{len(labels)}] {gt['file']:20} {marks}  ({elapsed:.1f}초)")
        rows.append({"file": gt["file"], "gt": {n: gt[n] for n in FEATURES}, "pred": pred, "seconds": round(elapsed, 1)})

    print("\n특징별 정확도 (정답이 unknown인 사진은 제외):")
    accuracy = {}
    for name in FEATURES:
        graded = [r for r in rows if r["gt"][name] != "unknown"]
        correct = sum(r["pred"][name] == r["gt"][name] for r in graded)
        accuracy[name] = correct / len(graded)
        print(f"  {name:8} {correct}/{len(graded)} = {accuracy[name]:.0%}")
    print(f"평균 소요 시간: {sum(r['seconds'] for r in rows) / len(rows):.1f}초")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"features_{datetime.now():%Y%m%d_%H%M}.json"
    out.write_text(
        json.dumps({"summary": {"model": MODEL, "accuracy": accuracy}, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n결과 저장: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
