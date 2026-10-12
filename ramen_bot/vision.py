"""보기(Vision): VLM으로 사진에서 눈에 보이는 특징만 뽑는다. 라멘 종류 판단은 하지 않는다."""
import io
import json
from pathlib import Path

import ollama
from PIL import Image, ImageOps

MODEL = "qwen2.5vl:7b"
MAX_SIDE = 1024  # 이미지 긴 변 최대 픽셀. 클수록 이미지 토큰이 늘어난다 (약 28x28px당 1토큰)
NUM_CTX = 8192  # 컨텍스트 크기(토큰). Ollama 기본값 4096은 이미지+프롬프트에 빠듯하다

FEATURES = {
    "soup": ["yes", "no"],
    "clarity": ["clear", "cloudy", "none"],
    "color": ["pale_gold", "dark_brown", "light_brown", "creamy_white", "red_orange", "none"],
    "noodle": ["thin", "medium", "thick", "unknown"],
}

# 토핑 체크리스트: "보이는 토핑을 나열해"라고 하면 없는 재료를 지어냈다(파인애플, 파스타...).
# 대신 종류 구분에 단서가 되는 토핑만 골라 예/아니오로 묻는다 (knowledge/ 문서의 판별 단서에서 뽑음)
TOPPINGS = {
    "spinach": "cooked spinach or leafy greens",
    "large_nori": "large sheets of nori seaweed",
    "corn": "sweet corn kernels",
    "bean_sprouts": "bean sprouts",
    "butter": "a pat of butter",
    "red_pickled_ginger": "red pickled ginger (beni shoga)",
    "wood_ear": "black wood ear mushroom strips",
    "clams": "clams or shellfish",
    "chili": "chili flakes, chili oil, or red chili paste",
}
# 뺀 항목 (docs/experiments.md 실험 6): chopped_onion은 파를 보고도 true(17/23), raw_egg_yolk는 반숙 달걀을 보고도 true

# 영어로 묻는다: 7B 모델은 영어가 가장 강하고, 한국어 자유 서술은 반복 루프에 잘 빠졌다 (docs/experiments.md Before 4)
# 라멘 지식(어떤 종류가 어떤 국물인지)은 주지 않고, 각 선택지가 "어떻게 보이는지"만 정의한다
PROMPT = """Look at the ramen bowl in this photo. Answer ONLY about what is visible. Do not guess the ramen style.

- soup: "yes" if the noodles sit in liquid broth. "no" if it is a soupless mixed-noodle dish (at most a little sauce at the bottom).
- clarity: "clear" if the broth is transparent enough to see noodles through it. "cloudy" if the broth is opaque or milky. "none" if there is no soup.
- color: the broth color.
  "pale_gold" = light yellow and transparent,
  "dark_brown" = dark soy-sauce color,
  "light_brown" = beige or tan,
  "creamy_white" = milky white,
  "red_orange" = chili red or orange,
  "none" = no soup.
- noodle: noodle thickness. "thin" = about spaghetti thickness or thinner, "medium", "thick" = clearly thick and chewy, "unknown" = noodles are hidden.
- toppings: for each item, true only if you can clearly see it in the bowl, otherwise false.
""" + "\n".join(f"  {name}: {desc}" for name, desc in TOPPINGS.items())

SCHEMA = {
    "type": "object",
    "properties": {
        **{name: {"type": "string", "enum": options} for name, options in FEATURES.items()},
        "toppings": {
            "type": "object",
            "properties": {name: {"type": "boolean"} for name in TOPPINGS},
            "required": list(TOPPINGS),
        },
    },
    "required": [*FEATURES, "toppings"],
}


def load_resized(image_path: Path, verbose: bool = True) -> bytes:
    """이미지를 긴 변 MAX_SIDE 이하로 줄여서 JPEG 바이트로 돌려준다."""
    # exif_transpose: 폰 사진은 EXIF 회전 정보로만 세워져 있는 경우가 있어서, 실제 픽셀을 바로 세운다
    img = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    original = img.size
    img.thumbnail((MAX_SIDE, MAX_SIDE))  # 비율 유지하며 축소 (작은 이미지는 그대로)
    if verbose:
        print(f"이미지 크기: {original[0]}x{original[1]} -> {img.size[0]}x{img.size[1]}")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


# 관문(gate): 특징 추출 전에 "면 요리 사진인가"만 따로 묻는다 (실험 10).
# - 관문이 없을 때는 라멘이 아닌 사진도 "국물 없음" → 후보 1개(아부라소바) → LLM 생략으로 그대로 판별됐다
# - 특징 추출 PROMPT에 질문을 섞었더니 기존 특징(국물 유무)까지 흔들려서, 호출을 분리해 기존 동작을 그대로 둔다
# - "라멘 그릇인가"로 물으면 토핑이 수북한 마제소바를 거절해서, 넉넉하게 "면 요리인가"로 묻는다.
#   파스타 등은 통과하지만, 진짜 라멘을 거절하는 것보다 낫다
# - "면이 보이나"로 물으면 뽀얀 국물에 면이 잠긴 돈코츠·토리파이탄·미소를 거절해서,
#   "그릇 요리 / 다른 음식 / 음식 아님" 중에서 고르게 하고 애매하면 그릇 요리를 고르게 한다
GATE_PROMPT = """What does this photo mainly show? Choose one.
- "noodle_bowl": a bowl of ramen or another noodle dish. The noodles may be completely hidden under thick or milky broth \
or under toppings; it still counts. Soupless noodles and noodles with a separate dipping soup also count.
- "other_food": food or drink with no bowl of broth or noodles (for example pizza, rice, cake, coffee).
- "not_food": no food at all (for example objects, clothing, people, scenery, a screenshot).
If you are unsure, choose "noodle_bowl"."""
GATE_SCHEMA = {
    "type": "object",
    "properties": {"subject": {"type": "string", "enum": ["noodle_bowl", "other_food", "not_food"]}},
    "required": ["subject"],
}


def is_noodle_dish(path: Path) -> str:
    """사진 -> "yes" / "no" / "invalid". 판단이 깨지면(invalid) 거절하지 않고 원래 파이프라인으로 넘긴다"""
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": GATE_PROMPT, "images": [load_resized(path, verbose=False)]}],
        format=GATE_SCHEMA,
        options={"num_ctx": NUM_CTX, "temperature": 0, "num_predict": 32},
    )
    try:
        subject = json.loads(response["message"]["content"])["subject"]
    except (json.JSONDecodeError, KeyError):
        return "invalid"
    return "yes" if subject == "noodle_bowl" else "no"


def extract_features(path: Path) -> dict:
    """사진 -> {"soup": ..., "clarity": ..., "color": ..., "noodle": ..., "toppings": {...}}"""
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT, "images": [load_resized(path, verbose=False)]}],
        format=SCHEMA,
        options={"num_ctx": NUM_CTX, "temperature": 0, "num_predict": 256},
    )
    try:
        return json.loads(response["message"]["content"])
    except json.JSONDecodeError:
        return {**{name: "invalid" for name in FEATURES}, "toppings": {}}
