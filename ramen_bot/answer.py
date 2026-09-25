"""답변(Answer): 판단 결과를 사용자에게 보여줄 한국어 설명으로 만든다.

LLM에게 한국어 문장을 새로 쓰게 하지 않는다. 7B 모델은 한국어 자유 서술에서 반복·환각이 잦았다(실험 4).
대신 지식 문서의 섹션을 그대로 꺼내 조립한다 -> 화면의 모든 사실에 출처가 있다.
"""
import re

from ramen_bot.judge import KO, TOPPING_KO
from ramen_bot.knowledge import get_sections

SECTIONS = {"summary": "요약", "criteria": "사진으로 구분하는 법", "origin": "유래"}


def _body(chunk_text: str) -> str:
    """청크는 '[제목] 섹션명\\n본문' 형태로 저장돼 있다. 본문만 꺼낸다."""
    body = chunk_text.split("\n", 1)[1].strip() if "\n" in chunk_text else ""
    # "soup=yes, clarity=cloudy, color=light_brown." 같은 내부용 코드(VLM 출력과 맞추려고 문서에 넣은 것)는 사용자에게 숨긴다
    return re.sub(r"(?:(?:soup|clarity|color)=[\w*]+(?: 또는 \w+)?[,.]\s*)+", "", body)


def _observed(features: dict) -> list[str]:
    """VLM이 본 특징을 한국어 목록으로. 면 굵기는 정확도가 낮아서(47%) 보여주지 않는다."""
    if features.get("soup") == "no":
        items = ["국물 없음 (비벼 먹는 스타일)"]
    else:
        items = [KO[v] for v in (features.get("clarity"), features.get("color")) if v in KO]
    seen = [TOPPING_KO[t] for t, v in features.get("toppings", {}).items() if v and t in TOPPING_KO]
    if seen:
        items.append("토핑: " + ", ".join(seen))
    return items


def explain(result: dict) -> dict:
    """identify() 결과 -> 화면에 보여줄 설명"""
    style = result.get("style")
    if style in (None, "invalid"):
        return {"ok": False, "message": "사진을 분석하지 못했어요. 라멘 그릇이 잘 보이는 사진으로 다시 시도해 주세요."}

    doc = {}
    for key, heading in SECTIONS.items():
        found = get_sections([style], heading)
        if found:
            doc[key] = _body(found[0]["text"])
            doc["title"] = found[0]["title"]

    # 검색에 걸린 후보들 ("ieke#사진으로 구분하는 법 (0.412)" -> "ieke")
    candidate_styles = list(dict.fromkeys(r.split("#")[0] for r in result.get("retrieved", [])))
    titles = {c["style"]: c["title"] for c in get_sections(candidate_styles, SECTIONS["summary"])}

    return {
        "ok": True,
        "style": style,
        "title": doc.get("title", style),
        "summary": doc.get("summary", ""),
        "criteria": doc.get("criteria", ""),
        "origin": doc.get("origin", ""),
        "observed": _observed(result.get("features", {})),
        "candidates": [{"style": s, "title": titles.get(s, s)} for s in candidate_styles],
        "reason_en": result.get("reason", ""),  # 판단 LLM이 쓴 영어 근거 (디버그용)
        "sources": [{"url": u, "label": _source_label(u)} for u in result.get("sources", [])],
    }


def _source_label(url: str) -> str:
    """https://ja.wikipedia.org/wiki/家系ラーメン -> 'ja.wikipedia.org · 家系ラーメン'"""
    m = re.match(r"https?://([^/]+)/(?:wiki/)?(.*?)/?$", url)
    if not m:
        return url
    host, path = m.groups()
    return f"{host} · {path}" if path else host
