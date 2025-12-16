import shutil
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
import uvicorn
from src.config import config
from src.pdf_processing import PDFProcessor
from src.vector_store import VectorStoreService
from src.pydantic_models import QuestionRequest, QuestionResponse, IngestResponse, SystemSummary
from src.workflow import QuestionGenerationWorkflow

workflow = None
pdf_processor = PDFProcessor(config.CHUNK_SIZE, config.CHUNK_OVERLAP, config.TOP_N_TABLE_CONTENT)
vector_service = VectorStoreService(config.VECTOR_DB_PATH, config.EMBEDDING_MODEL)
system_stats = {
    "documents_processed": 0,
    "questions_generated": 0,
    "total_quality_score": 0.0,
    "api_calls": 0,
    "start_time": datetime.now()
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global workflow
    workflow = QuestionGenerationWorkflow(vector_service)
    print("API started successfully!")
    yield
    print("API shutting down...")

app = FastAPI(
    title="RAG Question Generation API",
    description="Multi-agent system for generating MCQ questions from PDFs",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)):
    """Ingest a PDF file into the vector database"""
    try:
        file_path = config.UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        documents = pdf_processor.load_pdf(str(file_path))
        toc = pdf_processor.extract_table_of_contents(documents)
        doc_id = vector_service.create_vectorstore(documents)

        system_stats["documents_processed"] += len(documents)

        return IngestResponse(
            status="success",
            document_id=doc_id,
            table_of_contents=toc,
            total_chunks=len(documents)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/questions", response_model=QuestionResponse)
async def generate_questions(request: QuestionRequest):
    """Generate MCQ questions based on a query"""
    try:
        if not workflow:
            raise HTTPException(status_code=500, detail="Workflow not initialized")

        response = workflow.run(
            query=request.query,
            num_questions=request.num_questions,
            difficulty=request.difficulty
        )

        system_stats["questions_generated"] += response.total_generated
        system_stats["total_quality_score"] += (
            response.average_quality_score * response.total_generated
        )
        system_stats["api_calls"] += 1

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/summary")
async def summary():
    """Get system summary and statistics"""
    avg_quality = (
        system_stats["total_quality_score"] / system_stats["questions_generated"]
        if system_stats["questions_generated"] > 0 else 0.0
    )
    uptime = (datetime.now() - system_stats["start_time"]).total_seconds()

    return SystemSummary(
        total_documents_processed=system_stats["documents_processed"],
        vector_store="FAISS",
        embedding_model=config.EMBEDDING_MODEL,
        llm_model=config.MODEL_NAME,
        total_questions_generated=system_stats["questions_generated"],
        average_quality_score=round(avg_quality, 2),
        total_api_calls=system_stats["api_calls"],
        uptime_seconds=round(uptime, 2)
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=True)