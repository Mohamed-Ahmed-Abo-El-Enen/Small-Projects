from typing import List, Dict, Any, TypedDict
from src.pydantic_models import MCQQuestion


class AgentState(TypedDict):
    query: str
    num_questions: int
    difficulty: str
    retrieved_context: List[str]
    generated_questions: List[Dict[str, Any]]
    evaluated_questions: List[MCQQuestion]
    error: str
