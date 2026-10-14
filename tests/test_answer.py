"""answer.py: 판단 결과 -> 화면용 설명 조립 (Chroma 대신 가짜 섹션을 쓴다)"""
import pytest

from ramen_bot import answer
from ramen_bot.knowledge import load_chunks

FAKE_DOCS = {
    ("ieke", "요약"): "요코하마의 돼지뼈 간장 라멘.",
    ("ieke", "사진으로 구분하는 법"): "- soup=yes, clarity=cloudy, color=light_brown. 탁한 국물.\n- 시금치 + 큰 김.",
    ("ieke", "유래"): "1974년 요시무라야.",
    ("shoyu", "요약"): "간장 라멘.",
}


@pytest.fixture
def fake_sections(monkeypatch):
    def get_sections(styles, heading):
        return [{"style": s, "title": f"{s} 라멘", "text": f"[{s} 라멘] {heading}\n{FAKE_DOCS[(s, heading)]}"}
                for s in styles if (s, heading) in FAKE_DOCS]
    monkeypatch.setattr(answer, "get_sections", get_sections)


def test_not_ramen(fake_sections):
    out = answer.explain({"style": "not_ramen"})
    assert out["ok"] is False
    assert out["kind"] == "not_ramen"


@pytest.mark.parametrize("style", ["invalid", None])
def test_failed_analysis(fake_sections, style):
    out = answer.explain({"style": style})
    assert out["ok"] is False
    assert "kind" not in out  # not_ramen 안내와 구분된다


def test_explain_assembles_doc_sections(fake_sections):
    result = {
        "style": "ieke",
        "features": {"soup": "yes", "clarity": "cloudy", "color": "light_brown", "noodle": "thick",
                     "toppings": {"spinach": True, "large_nori": True, "corn": False}},
        "retrieved": ["ieke#사진으로 구분하는 법 (0.412)", "shoyu#사진으로 구분하는 법 (0.455)",
                      "ieke#사진으로 구분하는 법 (0.500)"],
        "reason": "spinach and nori",
        "sources": ["https://ja.wikipedia.org/wiki/家系ラーメン"],
    }
    out = answer.explain(result)

    assert out["ok"] is True
    assert out["title"] == "ieke 라멘"
    assert out["summary"] == "요코하마의 돼지뼈 간장 라멘."
    assert out["origin"] == "1974년 요시무라야."
    assert "soup=" not in out["criteria"]  # 내부 코드는 숨긴다
    assert "탁한 국물" in out["criteria"]
    assert out["observed"] == ["탁하고 불투명한 국물 (백탕)", "연갈색·베이지색 국물", "토핑: 시금치, 큰 김"]  # 면 굵기는 안 보여줌
    assert [c["style"] for c in out["candidates"]] == ["ieke", "shoyu"]  # 중복 제거, 순서 유지
    assert out["sources"] == [{"url": "https://ja.wikipedia.org/wiki/家系ラーメン",
                               "label": "ja.wikipedia.org · 家系ラーメン"}]


def test_missing_section_is_empty_not_error(fake_sections):
    out = answer.explain({"style": "shoyu", "retrieved": []})  # 가짜 문서에 판별법·유래 없음 (실제로 karai는 유래가 없다)
    assert out["ok"] is True
    assert out["criteria"] == "" and out["origin"] == ""


def test_soupless_observed():
    assert answer._observed({"soup": "no", "color": "none"}) == ["국물 없음 (비벼 먹는 스타일)"]


# ---------- _body: 내부 코드 숨기기 ----------

@pytest.mark.parametrize("text, expected", [
    ("[t] h\nsoup=yes, clarity=cloudy, color=light_brown. 탁한 국물", "탁한 국물"),
    ("[t] h\n- soup=no. **국물 없음**", "- **국물 없음**"),
    ("[t] h\ncolor=light_brown 또는 dark_brown. 갈색", "갈색"),
    ("제목만 있고 본문 없음", ""),
])
def test_body_strips_internal_codes(text, expected):
    assert answer._body(text) == expected


def test_real_docs_have_no_codes_left_after_body():
    # 실제 지식 문서의 판별 기준 섹션에 _body를 적용하면 내부 코드가 하나도 남지 않아야 한다
    for c in load_chunks():
        if c["heading"] == "사진으로 구분하는 법":
            body = answer._body(c["text"])
            for code in ("soup=", "clarity=", "color="):
                assert code not in body, f"{c['id']}에 {code}가 화면에 노출됨"


# ---------- _source_label ----------

@pytest.mark.parametrize("url, label", [
    ("https://ja.wikipedia.org/wiki/家系ラーメン", "ja.wikipedia.org · 家系ラーメン"),
    ("https://en.wikipedia.org/wiki/Ramen/", "en.wikipedia.org · Ramen"),
    ("https://example.com/", "example.com"),
    ("not a url", "not a url"),
])
def test_source_label(url, label):
    assert answer._source_label(url) == label
