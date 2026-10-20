"""관문(gate) 평가: 라멘이 아닌 사진을 얼마나 잘 거절하고, 면 요리는 얼마나 통과시키나 (실험 12).

정답표: data/negative_labels.csv (file, category, label, description)
  category: ramen / other_noodle (통과가 정답) · broth_bowl / other_food / not_food (거절이 정답)
사진: samples_negative/ (gitignore, 사용자가 직접 찍은 사진이라 GPS 정보가 있어 올리지 않는다)
결과: results/gate_<날짜시간>.json

사용법:
    python scripts/eval_gate.py
"""
import csv
import json
import time
from datetime import datetime
from pathlib import Path

from ramen_bot.benchmark import EXPECT_PASS, collect_env, gate_passed, gate_summary, timing
from ramen_bot.judge import judge
from ramen_bot.vision import extract_features, is_noodle_dish

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "data" / "negative_labels.csv"
SAMPLES = ROOT / "samples_negative"
RESULTS_DIR = ROOT / "results"


def main() -> None:
    env = collect_env()
    print(f"측정 조건: {env}\n")
    with LABELS.open(encoding="utf-8") as f:
        labels = list(csv.DictReader(f))

    rows = []
    for i, item in enumerate(labels, 1):
        path = SAMPLES / item["file"]
        start = time.time()
        gate = is_noodle_dish(path)
        row = {"file": item["file"], "category": item["category"], "description": item["description"], "gate": gate}
        # 라멘 사진이 통과하면 종류 판별까지 해 본다 (처음 보는 사진이라 덤으로 얻는 검증)
        if item["label"] and gate_passed(gate):
            result = judge(extract_features(path))
            row.update(label=item["label"], style=result["style"], correct=result["style"] == item["label"])
        row["seconds"] = round(time.time() - start, 1)
        rows.append(row)

        expected = "통과" if item["category"] in EXPECT_PASS else "거절"
        actual = "통과" if gate_passed(gate) else "거절"
        extra = f"  종류={row['style']} ({'O' if row['correct'] else 'X'})" if "style" in row else ""
        print(f"[{i}/{len(labels)}] {'O' if expected == actual else 'X'} {item['file']} {item['category']:12} "
              f"기대={expected} 결과={actual}({gate})  {item['description']}{extra}")

    summary = gate_summary(rows)
    print("\n카테고리별 통과:")
    for category, c in summary["per_category"].items():
        print(f"  {category:12} {c['passed']}/{c['total']} 통과  (기대: {'통과' if category in EXPECT_PASS else '거절'})")
    print(f"\n잘못 통과(false accept): {len(summary['false_accept'])}장 = {summary['false_accept_rate']:.0%}  {summary['false_accept']}")
    print(f"잘못 거절(false reject): {len(summary['false_reject'])}장 = {summary['false_reject_rate']:.0%}  {summary['false_reject']}")
    t = timing(rows)
    print(f"소요 시간: 평균 {t['mean']}초, 중앙값 {t['median']}초, 최대 {t['max']}초 ({t['max_file']})")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"gate_{datetime.now():%Y%m%d_%H%M}.json"
    out.write_text(json.dumps({"summary": {**summary, "total": len(rows), "timing": t, "env": env}, "rows": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
