"""데모 웹 서버 (FastAPI). 사진을 올리면 라멘 종류와 한국어 설명을 돌려준다.

실행 (프로젝트 폴더에서):
    uvicorn ramen_bot.api:app --port 8000
브라우저에서 http://localhost:8000 접속
"""
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse

from ramen_bot.answer import explain
from ramen_bot.judge import identify

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
MAX_BYTES = 20 * 1024 * 1024  # 폰 원본 사진도 들어오도록 20MB까지

app = FastAPI(title="Ramen Vision Bot")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


# async def가 아니라 def로 둔다: 파이프라인(VLM 호출)이 수 초 걸리는 동기 코드라서,
# def로 두면 FastAPI가 별도 스레드에서 실행해 서버 전체가 멈추지 않는다
@app.post("/api/identify")
def identify_photo(file: UploadFile) -> dict:
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "이미지 파일만 올릴 수 있어요.")
    data = file.file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "사진이 너무 커요 (최대 20MB).")

    # 파이프라인은 파일 경로를 받으므로 임시 파일로 저장한다.
    # Windows에서는 열려 있는 임시 파일을 다른 곳에서 못 열어서, delete=False로 닫은 뒤 쓰고 직접 지운다
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "").suffix or ".jpg", delete=False) as tmp:
        tmp.write(data)
    start = time.time()
    try:
        result = identify(Path(tmp.name))
    except Exception as e:  # 깨진 이미지, Ollama 서버 꺼짐 등
        raise HTTPException(500, f"분석 중 오류가 났어요: {e}") from e
    finally:
        Path(tmp.name).unlink(missing_ok=True)

    return {**explain(result), "features": result.get("features"), "seconds": round(time.time() - start, 1)}
