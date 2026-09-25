"""지식(Knowledge): knowledge/*.md 를 섹션 단위로 잘라 bge-m3로 임베딩하고 Chroma에 저장·검색한다."""
import re
from pathlib import Path

import chromadb
import ollama
import yaml

ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = ROOT / "knowledge"
DB_DIR = ROOT / "chroma_db"
COLLECTION = "ramen_knowledge"
EMBED_MODEL = "bge-m3"

# "프로젝트 관찰"은 우리 테스트 사진을 보고 쓴 메모라서 검색에 넣으면 정답 누수(data leakage)가 된다
EXCLUDED_SECTIONS = {"프로젝트 관찰"}


def load_chunks() -> list[dict]:
    """각 문서를 '## 제목' 섹션 단위 청크로 자른다. 청크마다 종류(style)·섹션·출처를 메타데이터로 붙인다."""
    chunks = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        _, front, body = path.read_text(encoding="utf-8").split("---", 2)
        meta = yaml.safe_load(front)
        # 첫 섹션 앞의 '# 제목' 부분은 버리고, '## '로 시작하는 섹션만 청크로 만든다
        for section in re.split(r"^## ", body, flags=re.MULTILINE)[1:]:
            heading, _, text = section.partition("\n")
            heading = heading.split("(")[0].strip()  # "프로젝트 관찰 (우리 사진 기준)" -> "프로젝트 관찰"
            if heading in EXCLUDED_SECTIONS:
                continue
            chunks.append({
                "id": f"{path.stem}#{heading}",
                # 청크 본문 앞에 문서 제목을 붙여야 "면" 같은 짧은 섹션도 어느 라멘 얘기인지 검색에 잡힌다
                "text": f"[{meta['title']}] {heading}\n{text.strip()}",
                "style": meta["style"],
                "title": meta["title"],
                "heading": heading,
                "sources": " ".join(meta.get("sources", [])),  # Chroma 메타데이터는 리스트를 못 넣어서 문자열로
                # 문서 frontmatter의 국물 유무. 벡터 검색은 "국물 없음" 같은 부정을 잘 못 다뤄서 필터로 쓴다
                "soup": meta.get("visual", {}).get("soup", "any"),
            })
    return chunks


def embed(texts: list[str]) -> list[list[float]]:
    return ollama.embed(model=EMBED_MODEL, input=texts)["embeddings"]


def build_index() -> int:
    """지식 문서 전체를 새로 임베딩해서 Chroma에 저장한다. 문서를 고치면 다시 실행해야 한다."""
    chunks = load_chunks()
    client = chromadb.PersistentClient(path=str(DB_DIR))
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    # cosine: 벡터 방향(의미)이 얼마나 비슷한지로 거리를 잰다
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embed([c["text"] for c in chunks]),
        metadatas=[{k: c[k] for k in ("style", "title", "heading", "sources", "soup")} for c in chunks],
    )
    return len(chunks)


def get_sections(styles: list[str], heading: str) -> list[dict]:
    """검색 없이 id로 바로 꺼낸다. 예: 후보 종류들의 "비슷한 종류와 구분" 섹션"""
    collection = chromadb.PersistentClient(path=str(DB_DIR)).get_collection(COLLECTION)
    result = collection.get(ids=[f"{s}#{heading}" for s in styles])
    return [{"text": doc, **meta} for doc, meta in zip(result["documents"], result["metadatas"])]


def search(query: str, k: int = 6, where: dict | None = None) -> list[dict]:
    """질문과 의미가 가까운 청크 k개를 돌려준다. where로 메타데이터 필터를 걸 수 있다 (예: 특정 섹션만)."""
    collection = chromadb.PersistentClient(path=str(DB_DIR)).get_collection(COLLECTION)
    result = collection.query(query_embeddings=embed([query]), n_results=k, where=where)
    return [
        {"text": doc, "distance": dist, **meta}
        for doc, dist, meta in zip(result["documents"][0], result["distances"][0], result["metadatas"][0])
    ]
