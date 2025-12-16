from typing import List, Dict
from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    query: str = Field(..., description="Topic or concept for questions")
    num_questions: int = Field(default=5, ge=1, le=10)
    difficulty: str = Field(default="medium", pattern="^(easy|medium|hard)$")


class MCQQuestion(BaseModel):
    id: int
    question: str
    options: Dict[str, str]
    correct_answer: str
    explanation: str
    difficulty: str
    topic: str
    evaluation_score: float = 0.0


class QuestionResponse(BaseModel):
    questions: List[MCQQuestion]
    total_generated: int
    average_quality_score: float


class IngestResponse(BaseModel):
    status: str
    document_id: str
    table_of_contents: List[str]
    total_chunks: int


class SystemSummary(BaseModel):
    total_documents_processed: int
    vector_store: str
    embedding_model: str
    llm_model: str
    total_questions_generated: int
    average_quality_score: float
    total_api_calls: int
    uptime_seconds: float
