"""judge.py: 검색어 만들기, 하이브리드 검색 필터, 판단 흐름 (Ollama·Chroma는 가짜로 바꾼다)"""
import json

import pytest

from ramen_bot import judge


def chunk(style, sources="https://example.com/a"):
    """search()가 돌려주는 청크 모양의 가짜 데이터"""
    return {"style": style, "heading": judge.CRITERIA, "title": f"{style} 라멘", "text": f"{style} 판별 기준",
            "distance": 0.4, "sources": sources}


def fake_chat(content):
    """ollama.chat 대신 쓸 가짜 함수. 호출 횟수를 calls에 기록한다 (Mockito의 verify 역할)"""
    def chat(**kwargs):
        chat.calls.append(kwargs)
        return {"message": {"content": content}}
    chat.calls = []
    return chat


# ---------- features_to_query ----------

def test_query_soupless_ignores_broth_features():
    q = judge.features_to_query({"soup": "no", "clarity": "cloudy", "color": "dark_brown", "noodle": "thick"})
    assert q.startswith("국물 없음, 비벼 먹는 라멘")
    assert "탁하고" not in q and "간장색" not in q  # 국물이 없으면 국물 색·탁도는 검색어에 넣지 않는다
    assert "굵은 면" in q


def test_query_with_soup_uses_korean_words():
    q = judge.features_to_query({"soup": "yes", "clarity": "cloudy", "color": "light_brown", "noodle": "thick"})
    assert q == "탁하고 불투명한 국물 (백탕), 연갈색·베이지색 국물, 굵은 면"


def test_query_includes_only_seen_toppings():
    q = judge.features_to_query({"soup": "yes", "clarity": "clear", "color": "pale_gold", "noodle": "thin",
                                 "toppings": {"spinach": True, "large_nori": True, "corn": False}})
    assert q.endswith("토핑: 시금치, 큰 김")
    assert "옥수수" not in q


@pytest.mark.parametrize("features", [
    {},  # 특징 추출이 완전히 비었을 때
    {"soup": "invalid", "clarity": "invalid", "color": "invalid", "noodle": "invalid", "toppings": {}},  # JSON 파싱 실패
    {"soup": "yes", "toppings": {"pineapple": True}},  # 모르는 토핑
])
def test_query_does_not_crash_on_unknown_values(features):
    assert isinstance(judge.features_to_query(features), str)


# ---------- criteria_filter ----------

@pytest.mark.parametrize("soup", ["yes", "no"])
def test_criteria_filter(soup):
    conditions = judge.criteria_filter(soup)["$and"]
    assert {"heading": "사진으로 구분하는 법"} in conditions  # 판별 기준 섹션에서만 검색
    assert {"style": {"$ne": "basics"}} in conditions  # 국물 기초 문서는 후보가 아니다
    assert {"soup": soup} in conditions  # 국물 유무는 벡터가 아니라 메타데이터로 거른다


# ---------- judge() ----------

def test_single_candidate_skips_llm(monkeypatch):
    chat = fake_chat("{}")
    monkeypatch.setattr(judge, "search", lambda *a, **kw: [chunk("aburasoba", "https://x https://y")])
    monkeypatch.setattr(judge.ollama, "chat", chat)

    result = judge.judge({"soup": "no"})

    assert result["style"] == "aburasoba"
    assert result["sources"] == ["https://x", "https://y"]
    assert chat.calls == []  # 후보가 하나면 LLM을 부르지 않는다


def test_soupless_photo_filters_by_soup_no(monkeypatch):
    seen = {}
    def search(query, k, where):
        seen["where"] = where
        return [chunk("aburasoba")]
    monkeypatch.setattr(judge, "search", search)

    judge.judge({"soup": "no"})
    assert {"soup": "no"} in seen["where"]["$and"]


def test_llm_picks_among_candidates_and_code_attaches_sources(monkeypatch):
    chat = fake_chat(json.dumps({"reason": "spinach and nori", "style": "ieke"}))
    monkeypatch.setattr(judge, "search", lambda *a, **kw: [chunk("shoyu", "https://shoyu"), chunk("ieke", "https://ieke")])
    monkeypatch.setattr(judge, "get_sections", lambda styles, heading: [])
    monkeypatch.setattr(judge.ollama, "chat", chat)

    result = judge.judge({"soup": "yes", "color": "light_brown"})

    assert result["style"] == "ieke"
    assert result["sources"] == ["https://ieke"]  # LLM이 아니라 코드가 고른 종류의 문서 출처를 붙인다
    schema = chat.calls[0]["format"]
    assert schema["properties"]["style"]["enum"] == ["shoyu", "ieke"]  # 답은 검색된 후보 중에서만
    assert list(schema["properties"]) == ["reason", "style"]  # 근거를 먼저 쓰고 결론
    assert chat.calls[0]["options"]["temperature"] == 0


def test_broken_llm_json_becomes_invalid(monkeypatch):
    monkeypatch.setattr(judge, "search", lambda *a, **kw: [chunk("shoyu"), chunk("ieke")])
    monkeypatch.setattr(judge, "get_sections", lambda styles, heading: [])
    monkeypatch.setattr(judge.ollama, "chat", fake_chat('{"reason": "cut off'))  # num_predict에서 잘린 JSON

    result = judge.judge({"soup": "yes"})

    assert result["style"] == "invalid"  # 예외로 서버가 죽지 않고 invalid로 처리
    assert result["sources"] == []


# ---------- identify(): 관문 ----------

def test_gate_no_skips_feature_extraction(monkeypatch):
    def must_not_run(*a, **kw):
        raise AssertionError("관문에서 거절됐는데 다음 단계가 호출됨")
    # judge.py는 `from ramen_bot.vision import is_noodle_dish`로 가져왔으므로 judge 쪽 이름을 바꿔야 한다
    monkeypatch.setattr(judge, "is_noodle_dish", lambda path: "no")
    monkeypatch.setattr(judge, "extract_features", must_not_run)
    monkeypatch.setattr(judge, "search", must_not_run)

    result = judge.identify("photo.jpg")

    assert result["style"] == "not_ramen"
    assert result["sources"] == []


@pytest.mark.parametrize("gate", ["yes", "invalid"])
def test_gate_yes_or_invalid_continues_pipeline(monkeypatch, gate):
    # invalid(관문 응답이 깨짐)는 거절하지 않고 원래 파이프라인으로 넘긴다
    monkeypatch.setattr(judge, "is_noodle_dish", lambda path: gate)
    monkeypatch.setattr(judge, "extract_features", lambda path: {"soup": "no"})
    monkeypatch.setattr(judge, "search", lambda *a, **kw: [chunk("aburasoba")])

    result = judge.identify("photo.jpg")

    assert result["style"] == "aburasoba"
    assert result["features"] == {"soup": "no"}
