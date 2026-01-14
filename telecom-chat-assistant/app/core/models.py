from typing import List, Optional, Dict
from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = "default"
    user_id: Optional[str] = "default"


class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    language: str
    sources: List[Dict]
    had_image: bool = False


class ConversationCreate(BaseModel):
    user_id: Optional[str] = "default"


class ConversationResponse(BaseModel):
    conversation_id: str
    created_at: str
    last_updated: str


class InitializeRequest(BaseModel):
    max_pages: Optional[int] = 50


class SystemStatus(BaseModel):
    is_initialized: bool
    model_mode: str
    model_name: str
    vision_model_name: str


class MessageHistory(BaseModel):
    role: str
    content: str
    timestamp: str
    language: Optional[str] = None
    sources: Optional[List[Dict]] = []


class ConversationHistory(BaseModel):
    conversation_id: str
    messages: List[MessageHistory]