"""판단(Judge): VLM이 본 특징 + 검색한 지식 문서를 근거로 라멘 종류를 정한다."""
import json
from pathlib import Path

import ollama

from ramen_bot.knowledge import get_sections, search
from ramen_bot.vision import MODEL, NUM_CTX, extract_features

# 검색어는 지식 문서와 같은 한국어로 만든다 (문서 속 "사진으로 구분하는 법" 표현과 맞춤)
KO = {
    "clear": "투명한 맑은 국물 (청탕)",
    "cloudy": "탁하고 불투명한 국물 (백탕)",
    "pale_gold": "옅은 노란색·금색 국물",
    "dark_brown": "진한 갈색 간장색 국물",
    "light_brown": "연갈색·베이지색 국물",
    "creamy_white": "뽀얀 우윳빛 크림색 국물",
    "red_orange": "붉은색·주황색 매운 국물",
    "thin": "가는 면",
    "medium": "중간 굵기 면",
    "thick": "굵은 면",
}
TOPPING_KO = {
    "spinach": "시금치",
    "large_nori": "큰 김",
    "corn": "옥수수",
    "bean_sprouts": "숙주",
    "butter": "버터",
    "red_pickled_ginger": "베니쇼가(붉은 초생강)",
    "wood_ear": "목이버섯",
    "clams": "조개",
    "chili": "고춧가루·고추기름",
}

CRITERIA = "사진으로 구분하는 법"
COMPARE = "비슷한 종류와 구분"


def criteria_filter(soup: str) -> dict:
    """하이브리드 검색의 필터 부분.
    - 검색은 각 문서의 "사진으로 구분하는 법" 섹션에서만 한다. 전체 섹션에서 찾으면 모든 문서에 있는
      "국물" 섹션과 basics 문서가 상위를 차지했다 (실험 6)
    - 국물 유무(VLM 정확도 100%)는 벡터 유사도 대신 메타데이터로 거른다. 임베딩은 "국물 없음"의 '없음'을
      잘 반영하지 못해서, 국물 없는 사진에 국물 라멘 문서가 검색됐다 (실험 6)"""
    return {"$and": [{"heading": CRITERIA}, {"style": {"$ne": "basics"}}, {"soup": soup}]}


def features_to_query(f: dict) -> str:
    """특징 JSON -> 검색용 한국어 문장"""
    if f.get("soup") == "no":
        parts = ["국물 없음, 비벼 먹는 라멘"]
    else:
        parts = [KO.get(f.get("clarity"), ""), KO.get(f.get("color"), "")]
    parts.append(KO.get(f.get("noodle"), ""))
    toppings = [TOPPING_KO[t] for t, seen in f.get("toppings", {}).items() if seen and t in TOPPING_KO]
    if toppings:
        parts.append("토핑: " + ", ".join(toppings))
    return ", ".join(p for p in parts if p)


JUDGE_PROMPT = """You are a ramen expert. Decide the ramen style of a photo.
You cannot see the photo. You only get the visual features another model observed, and reference documents.

Observed features:
{features}

Reference documents (Korean):
{docs}

Each document describes how to recognize one candidate style (style=<key>).

Rules:
- Pick the candidate whose criteria best match the observed features.
- Noodle thickness from the observer is unreliable. Do not rely on it.
- First write a short reason in English (max 2 sentences) naming the style key, then choose the style."""


def judge(features: dict, k: int = 5) -> dict:
    query = features_to_query(features)
    chunks = search(query, k=k, where=criteria_filter("no" if features.get("soup") == "no" else "yes"))
    candidates = [c["style"] for c in chunks]
    result = {"query": query, "retrieved": [f"{c['style']}#{c['heading']} ({c['distance']:.3f})" for c in chunks]}

    if len(candidates) == 1:
        # 후보가 하나뿐이면 (예: 국물 없음 -> 아부라소바) LLM을 부를 필요가 없다
        verdict = {"reason": f"Only one candidate matches soup={features.get('soup')}.", "style": candidates[0]}
        return {**result, **verdict, "sources": chunks[0]["sources"].split()}

    # 후보의 판별 기준 + 후보끼리 비교하는 섹션을 함께 준다
    # 문서마다 영문 키(style=shio)를 붙여야 "시오 라멘" 문서와 답 "shio"를 헷갈리지 않는다
    compare = {c["style"]: c["text"] for c in get_sections(candidates, COMPARE)}
    docs = "\n\n".join(
        f"--- style={c['style']} | {c['title']}\n{c['text']}\n{compare.get(c['style'], '')}" for c in chunks
    )
    schema = {
        "type": "object",
        # reason을 style보다 먼저 두어 근거를 쓴 뒤에 결론을 내게 한다
        # 답은 검색된 후보 중에서만 고르게 한다 (검색이 정답을 놓치면 여기서도 못 맞힌다 -> 검색 재현율을 따로 잰다)
        "properties": {"reason": {"type": "string"}, "style": {"type": "string", "enum": candidates}},
        "required": ["reason", "style"],
    }
    response = ollama.chat(
        model=MODEL,  # 같은 qwen2.5vl을 텍스트 전용으로 쓴다. 8GB VRAM에 7B 모델 두 개는 동시에 못 올린다
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(features=json.dumps(features), docs=docs)}],
        format=schema,
        options={"num_ctx": NUM_CTX, "temperature": 0, "num_predict": 256},
    )
    try:
        verdict = json.loads(response["message"]["content"])
    except json.JSONDecodeError:
        verdict = {"reason": "", "style": "invalid"}
    # 출처는 LLM에게 쓰게 하지 않는다(지어낼 수 있음). 판단한 종류의 문서 출처를 코드가 붙인다
    sources = next((c["sources"].split() for c in chunks if c["style"] == verdict["style"]), [])
    return {**result, **verdict, "sources": sources}


def identify(path: Path) -> dict:
    """사진 한 장 -> 특징 추출 -> 문서 검색 -> 종류 판단"""
    features = extract_features(path)
    return {"features": features, **judge(features)}
