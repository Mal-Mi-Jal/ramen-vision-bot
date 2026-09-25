"""RAG 적용 후(After) 정확도: 사진마다 특징 추출 -> 지식 검색 -> 종류 판단, 정답(파일 이름)과 비교한다.

먼저 python scripts/build_index.py 로 지식 DB를 만들어 두어야 한다.
결과: results/rag_<날짜시간>.json

사용법:
    python scripts/eval_rag.py                          # samples/ 사진으로 채점
    python scripts/eval_rag.py --samples samples_test   # 다른 폴더 (튜닝에 안 쓴 새 사진)
"""
import argparse
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from ramen_bot.judge import identify
from ramen_bot.knowledge import available_styles

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


def true_label(path: Path) -> str:
    """shio_3.jpg -> 'shio'"""
    return re.sub(r"_\d+$", "", path.stem)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default="samples", help="채점할 사진 폴더 (기본: samples)")
    args = parser.parse_args()
    samples_dir = ROOT / args.samples

    styles = available_styles()
    all_images = sorted(samples_dir.glob("*.jpg"))
    # 지식 문서가 없는 종류(예: 제외한 shoyupaitan)는 맞힐 방법이 없으니 채점에서 뺀다
    images = [p for p in all_images if true_label(p) in styles]
    skipped = sorted({true_label(p) for p in all_images} - styles)
    if skipped:
        print(f"지식 문서가 없어 건너뛰는 종류: {skipped} ({len(all_images) - len(images)}장)\n")

    rows = []
    for i, path in enumerate(images, 1):
        answer = true_label(path)
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

    print(f"\n전체 정답률: {correct}/{len(rows)} = {correct / len(rows):.0%}")
    print(f"검색 재현율: {hits}/{len(rows)} = {hits / len(rows):.0%}  (정답 종류 문서가 검색 결과에 포함된 비율)")
    print("\n종류별:")
    for label, (c, n) in sorted(per_class.items()):
        print(f"  {label:12} {c}/{n}")
    print("\n모델이 많이 고른 답:", Counter(r["style"] for r in rows).most_common())
    print(f"평균 소요 시간: {sum(r['seconds'] for r in rows) / len(rows):.1f}초")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"rag_{args.samples}_{datetime.now():%Y%m%d_%H%M}.json"
    summary = {"samples": args.samples, "styles": sorted(styles), "accuracy": correct / len(rows),
               "correct": correct, "total": len(rows), "retrieval_recall": hits / len(rows)}
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
