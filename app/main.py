# [역할] FastAPI 애플리케이션의 진입점.
# 라우터를 등록하고 미들웨어를 설정한다.
# 실행: uvicorn app.main:app --reload
#
# 담당: 하윤

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes.chat import chat_router
from app.api.routes.metrics import metrics_router
from app.core.exceptions import LLMError

app = FastAPI(title="EcoLLM Router")


# 기반 클래스(LLMError) 하나만 등록하면 모든 하위 예외(Rate Limit, Auth 등)가 자동으로 처리된다.
# 하위 클래스를 개별 등록하면 새 예외 추가 시 이 파일도 함께 수정해야 하는 결합이 생긴다.
@app.exception_handler(LLMError)
async def llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        # "detail" 키는 FastAPI 기본 에러 형식({detail: ...})과 동일하게 맞춤.
        # 클라이언트가 422 등 다른 FastAPI 에러와 동일한 파싱 코드로 처리할 수 있다.
        content={"detail": str(exc)},
    )


# 라우터 등록 — 새 엔드포인트 추가 시 이 아래에 include_router를 추가한다.
app.include_router(chat_router)
app.include_router(metrics_router)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    # reload=True는 개발 환경 전용 — 코드 변경 시 자동 재시작
    # 프로덕션에서는 reload=False 또는 Gunicorn 사용
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
