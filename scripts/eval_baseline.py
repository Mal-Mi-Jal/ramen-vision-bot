"""기준 정확도(baseline) 측정: samples/ 의 사진 전체를 VLM으로 분류하고 정답률을 계산한다.

정답은 파일 이름에서 읽는다: shio_1.jpg -> 정답 'shio'
결과는 results/baseline_<날짜시간>.json 으로 저장한다.

사용법:
    python scripts/eval_baseline.py                  # label을 먼저 답하게 함
    python scripts/eval_baseline.py --reason-first   # 관찰·근거를 먼저 쓰고 label은 마지막에
"""
import argparse
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import ollama

from ramen_bot.vision import MODEL, NUM_CTX, load_resized

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT / "samples"
RESULTS_DIR = ROOT / "results"

# 파일 이름 라벨 -> 모델에게 보여줄 이름. 종류 이름만 주고 특징 설명은 주지 않는다 (지식 없이 모델 실력만 측정)
LABELS = {
    "shio": "시오 (塩)",
    "shoyu": "쇼유 (醤油)",
    "miso": "미소 (味噌)",
    "donkotsu": "돈코츠 (豚骨)",
    "ieke": "이에케 (家系)",
    "toripaitan": "토리파이탄 (鶏白湯)",
    "shoyupaitan": "쇼유파이탄 (醤油白湯)",
    "karai": "카라이/매운 라멘 (辛いラーメン)",
    "aburasoba": "아부라소바/마제소바 (油そば・まぜそば)",
}

PROMPT = f"""이 라멘 사진을 보고 아래 JSON 형식으로 답해줘.
- label: 다음 중 하나만 고를 것: {", ".join(f"{k}={v}" for k, v in LABELS.items())}
- soup: 국물이 있으면 true, 국물 없는 비빔면이면 false
- soup_color: 국물 색 (예: 투명, 연한 갈색, 진한 갈색, 뽀얀 흰색, 붉은색, 없음)
- noodle: 면 굵기 (가는 면 / 중간 / 굵은 면)
- toppings: 보이는 토핑 목록
- reason: 판단 근거 한두 문장 (한국어)"""

# Structured Output: 모델 출력을 이 JSON 스키마에 강제로 맞춘다. label은 enum이라 목록 밖 답이 나올 수 없다
SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": list(LABELS)},
        "soup": {"type": "boolean"},
        "soup_color": {"type": "string"},
        "noodle": {"type": "string"},
        "toppings": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
        "reason": {"type": "string"},
    },
    "required": ["label", "soup", "soup_color", "noodle", "toppings", "reason"],
}


def true_label(path: Path) -> str:
    """shio_3.jpg -> 'shio'"""
    return re.sub(r"_\d+$", "", path.stem)


def reason_first(schema: dict) -> dict:
    """label을 맨 뒤로 보낸 스키마. LLM은 앞에서부터 차례로 생성하므로,
    관찰(국물·면·토핑)과 근거를 먼저 쓰게 한 뒤 마지막에 결론(label)을 내게 한다."""
    props = dict(schema["properties"])
    props["label"] = props.pop("label")
    return {**schema, "properties": props, "required": list(props)}


def classify(path: Path, schema: dict) -> dict:
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT, "images": [load_resized(path, verbose=False)]}],
        format=schema,
        # temperature 0: 매번 같은 답이 나오게 (재현성)
        # num_predict: 출력 토큰 상한. 작은 모델이 같은 말을 반복하는 루프에 빠져도 끝나게 한다
        options={"num_ctx": NUM_CTX, "temperature": 0, "num_predict": 512},
    )
    content = response["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # 출력이 상한에 걸려 JSON이 중간에 잘린 경우: 오답(invalid)으로 기록하고 계속 진행
        return {"label": "invalid", "raw": content[:300]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reason-first", action="store_true", help="관찰·근거를 먼저, label을 마지막에 출력")
    args = parser.parse_args()
    schema = reason_first(SCHEMA) if args.reason_first else SCHEMA
    variant = "reason_first" if args.reason_first else "label_first"

    images = sorted(SAMPLES_DIR.glob("*.jpg"))
    unknown = {true_label(p) for p in images} - set(LABELS)
    if unknown:
        raise SystemExit(f"LABELS에 없는 라벨이 있어요: {unknown}")

    rows = []
    for i, path in enumerate(images, 1):
        start = time.time()
        pred = classify(path, schema)
        elapsed = time.time() - start
        answer = true_label(path)
        ok = pred["label"] == answer
        rows.append({"file": path.name, "answer": answer, **pred, "correct": ok, "seconds": round(elapsed, 1)})
        print(f"[{i}/{len(images)}] {'O' if ok else 'X'} {path.name:22} 예측={pred['label']:12} ({elapsed:.1f}초)")

    # 종류별 정답률
    per_class = defaultdict(lambda: [0, 0])  # 라벨 -> [맞힌 수, 전체 수]
    for r in rows:
        per_class[r["answer"]][1] += 1
        per_class[r["answer"]][0] += r["correct"]

    correct = sum(r["correct"] for r in rows)
    print(f"\n전체 정답률: {correct}/{len(rows)} = {correct / len(rows):.0%}")
    print("\n종류별:")
    for label, (c, n) in sorted(per_class.items()):
        print(f"  {label:12} {c}/{n}")
    print("\n모델이 많이 고른 답:", Counter(r["label"] for r in rows).most_common())
    print(f"평균 소요 시간: {sum(r['seconds'] for r in rows) / len(rows):.1f}초")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"baseline_{variant}_{datetime.now():%Y%m%d_%H%M}.json"
    summary = {
        "model": MODEL,
        "variant": variant,
        "accuracy": correct / len(rows),
        "correct": correct,
        "total": len(rows),
    }
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
