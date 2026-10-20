"""평가 결과 두 개를 사진 단위로 비교한다 (GPU·Ollama 필요 없음, JSON만 읽는다).

사용법:
    python scripts/compare_results.py results/rag_samples_20261012_1515.json results/rag_samples_<새 결과>.json
"""
import argparse

from ramen_bot.benchmark import compare, format_report, load


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("old", help="기준선 결과 JSON")
    parser.add_argument("new", help="새 결과 JSON")
    args = parser.parse_args()
    print(format_report(compare(load(args.old), load(args.new))))


if __name__ == "__main__":
    main()
