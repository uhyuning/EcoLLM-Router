# [역할] FastAPI 애플리케이션의 진입점.
# 라우터를 등록하고 미들웨어를 설정한다.
# 실행: uvicorn app.main:app --reload
#
# 담당: 하윤

import uvicorn
from fastapi import FastAPI

app = FastAPI(title="EcoLLM Router")


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
