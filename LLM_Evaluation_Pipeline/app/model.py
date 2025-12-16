from pydantic import BaseModel
from typing import Dict, Optional


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 0.7


class MCQRequest(BaseModel):
    question: str
    choices: Dict[str, str]  
    correct_answer: Optional[str] = None


class GenerateResponse(BaseModel):
    generated_text: str
    prompt: str
    generation_time: float


class MCQResponse(BaseModel):
    question: str
    model_answer: str
    full_response: str
    correct_answer: Optional[str] = None
    is_correct: Optional[bool] = None
    response_time: float