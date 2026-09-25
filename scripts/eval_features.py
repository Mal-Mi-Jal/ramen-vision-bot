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

from ramen_bot.vision import FEATURES, MODEL, extract_features

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT / "samples"
LABELS_CSV = ROOT / "data" / "labels.csv"
RESULTS_DIR = ROOT / "results"


def main() -> None:
    with LABELS_CSV.open(encoding="utf-8") as f:
        labels = list(csv.DictReader(f))

    rows = []
    for i, gt in enumerate(labels, 1):
        start = time.time()
        pred = extract_features(SAMPLES_DIR / gt["file"])
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
