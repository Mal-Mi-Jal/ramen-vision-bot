"""knowledge/*.md 를 임베딩해서 Chroma(chroma_db/)에 저장한다. 지식 문서를 고칠 때마다 다시 실행한다.

사용법:
    python scripts/build_index.py
    python scripts/build_index.py "시금치 김 굵은 면"   # 저장 후 검색 테스트
"""
import sys

from ramen_bot.knowledge import build_index, search

count = build_index()
print(f"청크 {count}개 저장 완료")

if len(sys.argv) > 1:
    print(f"\n검색: {sys.argv[1]}")
    for hit in search(sys.argv[1], k=5):
        print(f"  {hit['distance']:.3f}  {hit['style']:12} {hit['heading']}")
