from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.routes import router
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("TELECOM EGYPT ASSISTANT API")
    print("=" * 60)
    print(
        f"Model Mode: {'LOCAL (Ollama)' if settings.USE_LOCAL_MODEL else 'CLOUD (OpenAI)'}"
    )
    if settings.USE_LOCAL_MODEL:
        print(f"Ollama URL: {settings.OLLAMA_BASE_URL}")
        print(f"Model: {settings.LOCAL_MODEL_NAME}")
        print(f"Vision Model: {settings.LOCAL_VISION_MODEL_NAME}")
    print("=" * 60)

    yield

    print("\n" + "=" * 60)
    print("Shutting down Telecom Egypt Assistant API")
    print("=" * 60)

# Create FastAPI app
app = FastAPI(
    title="Telecom Egypt Intelligent Assistant API",
    description="RAG-powered chatbot with multi-lingual support and vision capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )