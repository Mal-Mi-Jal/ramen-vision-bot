"""api.py: 업로드 검증·에러 처리·임시 파일 정리 (파이프라인은 가짜로 바꾼다. Spring의 MockMvc 역할)"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ramen_bot import api

client = TestClient(api.app)
NOT_RAMEN = {"features": {}, "style": "not_ramen", "reason": "", "query": "", "retrieved": [], "sources": []}


def upload(data=b"fake-jpeg", content_type="image/jpeg", name="ramen.jpg"):
    return client.post("/api/identify", files={"file": (name, data, content_type)})


@pytest.fixture
def seen_paths(monkeypatch):
    """가짜 identify: 받은 임시 파일 경로와 그 순간 파일이 있었는지를 기록한다"""
    paths = []
    def identify(path):
        paths.append((Path(path), Path(path).exists()))
        return NOT_RAMEN
    monkeypatch.setattr(api, "identify", identify)
    return paths


def test_index_page():
    assert client.get("/").status_code == 200


def test_rejects_non_image(seen_paths):
    r = upload(b"hello", content_type="text/plain", name="a.txt")
    assert r.status_code == 400
    assert seen_paths == []  # 파이프라인까지 가지 않는다


def test_rejects_too_large(seen_paths):
    r = upload(b"0" * (api.MAX_BYTES + 1))
    assert r.status_code == 413
    assert seen_paths == []


def test_success_and_temp_file_removed(seen_paths):
    r = upload(name="photo.png")
    assert r.status_code == 200
    assert r.json()["kind"] == "not_ramen"
    assert "seconds" in r.json()
    path, existed = seen_paths[0]
    assert existed and path.suffix == ".png"  # 원래 확장자를 유지한 임시 파일로 넘긴다
    assert not path.exists()  # 끝나면 지운다


def test_pipeline_error_is_500_and_temp_file_removed(monkeypatch):
    paths = []
    def broken(path):
        paths.append(Path(path))
        raise RuntimeError("Ollama 서버 꺼짐")
    monkeypatch.setattr(api, "identify", broken)

    r = upload()

    assert r.status_code == 500
    assert "Ollama 서버 꺼짐" in r.json()["detail"]
    assert not paths[0].exists()  # 에러가 나도 finally에서 지운다
