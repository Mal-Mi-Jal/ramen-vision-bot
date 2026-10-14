"""vision.py: VLM 응답 처리 (실제 VLM은 부르지 않는다)"""
import json

import pytest

from ramen_bot import judge, vision


@pytest.fixture
def fake_vlm(monkeypatch):
    """이미지 읽기와 ollama.chat을 가짜로 바꾼다. fake_vlm("응답 문자열")로 VLM 응답을 정한다"""
    monkeypatch.setattr(vision, "load_resized", lambda path, verbose=True: b"")
    def set_content(content):
        monkeypatch.setattr(vision.ollama, "chat", lambda **kw: {"message": {"content": content}})
    return set_content


@pytest.mark.parametrize("subject, expected", [
    ("noodle_bowl", "yes"),
    ("other_food", "no"),
    ("not_food", "no"),
])
def test_gate_maps_subject(fake_vlm, subject, expected):
    fake_vlm(json.dumps({"subject": subject}))
    assert vision.is_noodle_dish("photo.jpg") == expected


@pytest.mark.parametrize("content", ['{"subject": ', '{"other": "x"}'])  # 잘린 JSON, 키 없음
def test_gate_broken_response_is_invalid(fake_vlm, content):
    fake_vlm(content)
    assert vision.is_noodle_dish("photo.jpg") == "invalid"


def test_gate_schema_has_three_choices():
    assert vision.GATE_SCHEMA["properties"]["subject"]["enum"] == ["noodle_bowl", "other_food", "not_food"]


def test_extract_features_broken_json_returns_invalid(fake_vlm):
    fake_vlm('{"soup": "yes", "clar')
    features = vision.extract_features("photo.jpg")
    assert features["soup"] == "invalid"
    assert features["toppings"] == {}


def test_feature_enums_have_korean_query_words():
    # VLM이 고를 수 있는 값마다 검색어 한국어가 있어야 한다 (none/unknown은 검색어에 넣지 않는 값)
    for name in ("clarity", "color", "noodle"):
        for value in vision.FEATURES[name]:
            if value not in ("none", "unknown"):
                assert value in judge.KO, f"{name}={value}의 한국어 검색어가 judge.KO에 없음"


def test_toppings_match_korean_words():
    # VLM에 묻는 토핑과 검색어 한국어는 같이 추가해야 한다 (한쪽만 있으면 죽은 코드)
    assert set(vision.TOPPINGS) == set(judge.TOPPING_KO)
