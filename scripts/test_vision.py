"""첫 Vision 테스트: 라멘 사진 한 장을 로컬 VLM(qwen2.5vl)에 보내고 답을 받는다.

사용법:
    python scripts/test_vision.py samples/ramen.jpg
"""
import io
import sys
import time
from pathlib import Path

import ollama
from PIL import Image, ImageOps

MODEL = "qwen2.5vl:7b"
MAX_SIDE = 1024  # 이미지 긴 변 최대 픽셀. 클수록 이미지 토큰이 늘어난다 (약 28x28px당 1토큰)
NUM_CTX = 8192  # 컨텍스트 크기(토큰). Ollama 기본값 4096은 이미지+프롬프트에 빠듯하다

PROMPT = """이 사진을 보고 한국어로 답해줘.
1. 라멘 종류 (예: 돈코츠, 쇼유, 미소, 시오, 이에케, 츠케멘 등)
2. 보이는 토핑
3. 판단 근거 (국물 색, 면 굵기 등)
라멘이 아니면 '라멘 아님'이라고만 답해."""


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
