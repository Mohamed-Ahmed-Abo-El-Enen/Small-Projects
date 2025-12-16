from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import time
import logging
from app.model_loader import ModelLoader
from app.model import *
from utils import read_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_loader = None

config = read_json("../configuration.json")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    global model_loader
    try:
        model_loader = ModelLoader(model_name=config["MODEL_NAME"])
        logger.info("Model loaded successfully on startup")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    yield

    logger.info("Shutting down...")


app = FastAPI(
    title="LLM Evaluation API",
    description="API for LLM text generation and MCQ evaluation",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "LLM Evaluation API",
        "endpoints": ["/generate", "/evaluate_mcq", "/health"],
        "model": config["MODEL_NAME"]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model_loader is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "status": "healthy",
        "model": model_loader.model_name,
        "device": model_loader.device
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text completion"""
    if model_loader is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        start_time = time.time()

        generated_text = model_loader.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature
        )

        generation_time = time.time() - start_time

        return GenerateResponse(
            generated_text=generated_text,
            prompt=request.prompt,
            generation_time=generation_time
        )

    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate_mcq", response_model=MCQResponse)
async def evaluate_mcq(request: MCQRequest):
    """Evaluate multiple choice question"""
    if model_loader is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        start_time = time.time()

        result = model_loader.evaluate_mcq(
            question=request.question,
            choices=request.choices,
            max_new_tokens=config["MAX_NEW_TOKENS"],
            temperature=config["TEMPERATURE"],
        )

        response_time = time.time() - start_time

        is_correct = None
        if request.correct_answer:
            is_correct = result["answer"] == request.correct_answer.upper()

        return MCQResponse(
            question=request.question,
            model_answer=result["answer"],
            full_response=result["full_response"],
            correct_answer=request.correct_answer,
            is_correct=is_correct,
            response_time=response_time
        )

    except Exception as e:
        logger.error(f"MCQ evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", log_level="info", host=config["HOST"], port=config["PORT"])