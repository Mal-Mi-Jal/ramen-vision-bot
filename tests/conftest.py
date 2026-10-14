"""모든 테스트에 자동 적용되는 설정 (pytest가 conftest.py를 자동으로 읽는다).

테스트는 Ollama·Chroma 없이 돌아야 한다. 실수로 진짜 호출이 일어나면 바로 실패시킨다.
필요한 테스트는 monkeypatch로 가짜 함수를 다시 덮어쓴다.
"""
import chromadb
import ollama
import pytest


@pytest.fixture(autouse=True)
def block_real_services(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("테스트에서 진짜 Ollama/Chroma를 호출함 -> monkeypatch로 가짜를 넣어야 한다")
    monkeypatch.setattr(ollama, "chat", blocked)
    monkeypatch.setattr(ollama, "embed", blocked)
    monkeypatch.setattr(chromadb, "PersistentClient", blocked)
