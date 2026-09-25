"""첫 Vision 테스트: 라멘 사진 한 장을 로컬 VLM(qwen2.5vl)에 보내고 답을 받는다.

사용법:
    python scripts/test_vision.py samples/shio_1.jpg
"""
import sys
import time
from pathlib import Path

import ollama

from ramen_bot.vision import MODEL, NUM_CTX, load_resized

PROMPT = """이 사진을 보고 한국어로 답해줘.
1. 라멘 종류 (예: 돈코츠, 쇼유, 미소, 시오, 이에케, 츠케멘 등)
2. 보이는 토핑
3. 판단 근거 (국물 색, 면 굵기 등)
라멘이 아니면 '라멘 아님'이라고만 답해."""


def main() -> None:
    if len(sys.argv) < 2:
        print("사용법: python scripts/test_vision.py <이미지 경로>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"이미지를 찾을 수 없어요: {image_path}")
        sys.exit(1)

    print(f"[{MODEL}] 분석 중... (첫 실행은 모델 로딩 때문에 느려요)")
    start = time.time()

    # images에 바이트를 넘기면 ollama 라이브러리가 base64로 인코딩해서 보내준다
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT, "images": [load_resized(image_path)]}],
        options={"num_ctx": NUM_CTX},
    )

    print(response["message"]["content"])
    print(f"\n(소요 시간: {time.time() - start:.1f}초)")


if __name__ == "__main__":
    main()
