"""benchmark.py: 결과 비교 로직 (JSON만 다룬다. Ollama 필요 없음)"""
import pytest

from ramen_bot import benchmark


def row(file, answer, style, hit=True, seconds=10.0):
    return {"file": file, "answer": answer, "style": style, "correct": style == answer,
            "retrieval_hit": hit, "seconds": seconds}


def result(rows, samples="samples", env=None):
    summary = {"samples": samples}
    if env is not None:
        summary["env"] = env
    return {"summary": summary, "rows": rows}


ENV = {"ollama": "0.40.3", "models": {"qwen2.5vl:7b": "abc"}, "max_side": 1024, "num_ctx": 8192, "top_k": 5,
       "git_commit": "aaa", "git_dirty": False}


# ---------- classify (실습) ----------

@pytest.mark.parametrize("old_style, new_style, expected", [
    ("ieke", "shio", "fixed"),        # X -> O
    ("shio", "ieke", "broken"),       # O -> X
    ("ieke", "donkotsu", "changed"),  # X -> X, 틀린 답이 바뀜
    ("ieke", "ieke", None),           # X -> X, 같은 답
    ("shio", "shio", None),           # O -> O
])
def test_classify(old_style, new_style, expected):
    assert benchmark.classify(row("shio_4.jpg", "shio", old_style), row("shio_4.jpg", "shio", new_style)) == expected


# ---------- compare ----------

def test_same_total_but_different_photos():
    # 전체 정답률은 1/2로 같아도 사진별로는 하나 고쳐지고 하나 틀렸다 -> 비교 도구가 필요한 이유
    old = result([row("a_1.jpg", "shio", "shio"), row("b_1.jpg", "ieke", "shio")], env=ENV)
    new = result([row("a_1.jpg", "shio", "ieke"), row("b_1.jpg", "ieke", "ieke")], env=ENV)

    diff = benchmark.compare(old, new)

    assert diff["old"]["correct"] == diff["new"]["correct"] == 1
    assert [c["file"] for c in diff["changes"]["fixed"]] == ["b_1.jpg"]
    assert [c["file"] for c in diff["changes"]["broken"]] == ["a_1.jpg"]
    assert diff["warnings"] == []


def test_retrieval_changes():
    old = result([row("a_1.jpg", "shio", "ieke", hit=False), row("b_1.jpg", "ieke", "ieke", hit=True)], env=ENV)
    new = result([row("a_1.jpg", "shio", "ieke", hit=True), row("b_1.jpg", "ieke", "ieke", hit=False)], env=ENV)

    diff = benchmark.compare(old, new)

    assert diff["retrieval"] == {"gained": ["a_1.jpg"], "lost": ["b_1.jpg"]}


def test_warns_when_ollama_version_differs_but_not_for_code_changes():
    new_env = {**ENV, "ollama": "0.40.6", "git_commit": "bbb", "git_dirty": True}
    diff = benchmark.compare(result([row("a_1.jpg", "shio", "shio")], env=ENV),
                             result([row("a_1.jpg", "shio", "shio")], env=new_env))

    assert diff["warnings"] == ["측정 조건이 다름: ollama 0.40.3 -> 0.40.6"]  # 코드(git) 차이는 비교 대상이라 경고 안 함


def test_warns_when_env_missing_or_photos_differ():
    old = result([row("a_1.jpg", "shio", "shio"), row("old_1.jpg", "shio", "shio")])  # 예전 결과: env 없음
    new = result([row("a_1.jpg", "shio", "shio"), row("new_1.jpg", "shio", "shio")], env=ENV)

    diff = benchmark.compare(old, new)

    assert diff["old"]["total"] == 1  # 공통 사진만 채점
    assert any("env" in w for w in diff["warnings"])
    assert any("old_1.jpg" in w and "new_1.jpg" in w for w in diff["warnings"])


def test_timing():
    rows = [row("a_1.jpg", "shio", "shio", seconds=s) for s in (10.0, 12.0)] + [row("slow_1.jpg", "shio", "shio", seconds=60.0)]
    assert benchmark.timing(rows) == {"mean": 27.3, "median": 12.0, "max": 60.0, "max_file": "slow_1.jpg"}


def test_report_mentions_changes():
    old = result([row("b_1.jpg", "ieke", "shio")], env=ENV)
    new = result([row("b_1.jpg", "ieke", "ieke")], env=ENV)
    report = benchmark.format_report(benchmark.compare(old, new))
    assert "고쳐짐 X->O: 1장" in report
    assert "b_1.jpg" in report
