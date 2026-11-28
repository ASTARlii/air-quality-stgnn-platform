"""FastAPI 入口。

启动：
    .venv\\Scripts\\python.exe -m uvicorn src.main:app --reload --port 8000

接口文档：
    http://127.0.0.1:8000/docs     Swagger UI
    http://127.0.0.1:8000/openapi.json
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Air Quality STGNN Platform",
        description="基于时空图神经网络的多站点空气质量预测与污染传播可视分析平台",
        version="0.1.0",
    )
    # 允许前端跨域（前端 Vite 默认 :5173）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router, prefix="/api/v1")
    return app


app = create_app()
