"""RAG 적용 후(After) 정확도: 사진마다 특징 추출 -> 지식 검색 -> 종류 판단, 정답(파일 이름)과 비교한다.

먼저 python scripts/build_index.py 로 지식 DB를 만들어 두어야 한다.
결과: results/rag_<날짜시간>.json

사용법:
    python scripts/eval_rag.py
"""
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from ramen_bot.judge import identify

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT / "samples"
RESULTS_DIR = ROOT / "results"


def main() -> None:
    images = sorted(SAMPLES_DIR.glob("*.jpg"))
    rows = []
    for i, path in enumerate(images, 1):
        answer = re.sub(r"_\d+$", "", path.stem)
        start = time.time()
        result = identify(path)
        elapsed = time.time() - start
        ok = result["style"] == answer
        # 검색 재현율: 정답 종류의 문서가 검색 결과 안에 하나라도 있었나 (없으면 판단 단계가 맞힐 수 없다)
        hit = any(r.startswith(answer + "#") for r in result["retrieved"])
        rows.append({"file": path.name, "answer": answer, **result, "correct": ok, "retrieval_hit": hit,
                     "seconds": round(elapsed, 1)})
        print(f"[{i}/{len(images)}] {'O' if ok else 'X'} {path.name:20} 예측={result['style']:12} "
              f"검색{'O' if hit else 'X'}  ({elapsed:.1f}초)")
        print(f"      검색어: {result['query']}")

    correct = sum(r["correct"] for r in rows)
    hits = sum(r["retrieval_hit"] for r in rows)
    per_class = defaultdict(lambda: [0, 0])
    for r in rows:
        per_class[r["answer"]][0] += r["correct"]
        per_class[r["answer"]][1] += 1

    print(f"\n전체 정답률: {correct}/{len(rows)} = {correct / len(rows):.0%}   (RAG 전 baseline: 4/23 = 17%)")
    print(f"검색 재현율: {hits}/{len(rows)} = {hits / len(rows):.0%}  (정답 종류 문서가 검색 결과에 포함된 비율)")
    print("\n종류별:")
    for label, (c, n) in sorted(per_class.items()):
        print(f"  {label:12} {c}/{n}")
    print("\n모델이 많이 고른 답:", Counter(r["style"] for r in rows).most_common())
    print(f"평균 소요 시간: {sum(r['seconds'] for r in rows) / len(rows):.1f}초")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"rag_{datetime.now():%Y%m%d_%H%M}.json"
    summary = {"accuracy": correct / len(rows), "correct": correct, "total": len(rows),
               "retrieval_recall": hits / len(rows)}
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
