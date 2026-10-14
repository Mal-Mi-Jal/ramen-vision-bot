"""knowledge.py + knowledge/*.md 문서 계약. 실제 문서를 읽지만 임베딩·Chroma는 쓰지 않는다."""
import yaml

from ramen_bot import knowledge

REQUIRED = {"요약", "사진으로 구분하는 법", "비슷한 종류와 구분"}  # "유래"는 선택 (karai 문서에는 없다)
STYLES = {"shio", "shoyu", "miso", "donkotsu", "ieke", "toripaitan", "karai", "aburasoba"}


def frontmatters():
    for path in sorted(knowledge.KNOWLEDGE_DIR.glob("*.md")):
        yield path, yaml.safe_load(path.read_text(encoding="utf-8").split("---", 2)[1])


def test_project_notes_are_never_indexed():
    # "프로젝트 관찰"은 테스트 사진을 보고 쓴 메모 -> 검색에 들어가면 정답 누수
    headings = {c["heading"] for c in knowledge.load_chunks()}
    assert "프로젝트 관찰" not in headings
    assert not any("프로젝트 관찰" in c["text"].split("\n")[0] for c in knowledge.load_chunks())


def test_heading_suffix_is_trimmed():
    # "## 프로젝트 관찰 (우리 사진 기준)", "## 소스 (국물 대신)" 처럼 괄호가 붙어도 이름만 남는다
    for c in knowledge.load_chunks():
        assert "(" not in c["heading"], c["id"]


def test_available_styles_are_the_eight_styles():
    assert knowledge.available_styles() == STYLES


def test_chunk_metadata():
    for c in knowledge.load_chunks():
        assert c["id"] == f"{c['id'].split('#')[0]}#{c['heading']}"
        assert c["text"].startswith(f"[{c['title']}] {c['heading']}\n")  # 짧은 섹션도 검색되도록 제목을 앞에 붙인다
        assert isinstance(c["sources"], str)  # Chroma 메타데이터는 리스트를 못 받는다


def test_doc_contract():
    for path, meta in frontmatters():
        if meta["style"] == "basics":
            continue
        assert path.stem == meta["style"], f"{path.name}: 파일명과 style이 다름 (청크 id가 파일명을 쓴다)"
        assert meta["visual"]["soup"] in ("yes", "no"), path.name  # 검색 필터가 쓰는 값
        assert meta.get("sources"), f"{path.name}: 출처 없음"
        headings = {c["heading"] for c in knowledge.load_chunks() if c["style"] == meta["style"]}
        assert REQUIRED <= headings, f"{path.name}: 빠진 섹션 {REQUIRED - headings}"


def test_only_aburasoba_is_soupless():
    soupless = {c["style"] for c in knowledge.load_chunks() if c["soup"] == "no"}
    assert soupless == {"aburasoba"}
