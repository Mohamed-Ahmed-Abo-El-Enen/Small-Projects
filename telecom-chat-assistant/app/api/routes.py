import os
import shutil
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.core.config import settings
from app.core.models import (
    ChatRequest, ChatResponse, ConversationCreate, ConversationResponse,
    InitializeRequest, SystemStatus
)
from app.services.assistant import TelecomEgyptAssistant


router = APIRouter()

assistant_instance = None


def get_assistant():
    """Get or create assistant instance"""
    global assistant_instance
    if assistant_instance is None:
        raise HTTPException(
            status_code=503, 
            detail="Assistant not initialized. Please initialize first."
        )
    return assistant_instance


@router.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Telecom Egypt Intelligent Assistant API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "is_initialized": assistant_instance is not None
    }


@router.get("/status", response_model=SystemStatus)
async def get_status():
    """Get system status"""
    return SystemStatus(
        is_initialized=assistant_instance is not None,
        model_mode="local" if settings.USE_LOCAL_MODEL else "cloud",
        model_name=settings.LOCAL_MODEL_NAME if settings.USE_LOCAL_MODEL else settings.LLM_MODEL,
        vision_model_name=settings.LOCAL_VISION_MODEL_NAME if settings.USE_LOCAL_MODEL else settings.VISION_MODEL
    )


@router.post("/initialize")
async def initialize_system(request: InitializeRequest):
    """Initialize the assistant by scraping website"""
    global assistant_instance

    try:
        assistant_instance = TelecomEgyptAssistant()

        if not assistant_instance.is_initialized:
            assistant_instance.initialize_from_web(max_pages=request.max_pages)
        else:
            print("Assistant already initialized, skipping...")

        return {
            "status": "success",
            "message": f"System initialized with {request.max_pages} pages",
            "is_initialized": True,
            "is_singleton": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")


@router.post("/load-index")
async def load_existing_index():
    """Load existing FAISS index"""
    global assistant_instance

    try:
        assistant_instance = TelecomEgyptAssistant()

        if not assistant_instance.is_initialized:
            assistant_instance.load_existing_index()
        else:
            print("Assistant already initialized, skipping...")

        return {
            "status": "success",
            "message": "Index loaded successfully",
            "is_initialized": True,
            "is_singleton": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load index: {str(e)}")


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the assistant (text only)"""
    assistant = get_assistant()

    try:
        result = assistant.get_detailed_response(
            query=request.query,
            conversation_id=request.conversation_id
        )

        return ChatResponse(
            answer=result['answer'],
            conversation_id=result['conversation_id'],
            language=result['language'],
            sources=result.get('sources', []),
            had_image=result.get('had_image', False)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@router.post("/chat-with-image", response_model=ChatResponse)
async def chat_with_image(
    query: str = Form(...),
    conversation_id: str = Form("default"),
    image: Optional[UploadFile] = File(None)
):
    """Chat with the assistant (text + optional image)"""
    assistant = get_assistant()

    image_path = None

    try:
        if image:
            image_path = f"./temp_api_{image.filename}"
            with open(image_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)

        result = assistant.get_detailed_response(
            query=query,
            conversation_id=conversation_id,
            image_path=image_path
        )

        if image_path and os.path.exists(image_path):
            os.remove(image_path)

        return ChatResponse(
            answer=result['answer'],
            conversation_id=result['conversation_id'],
            language=result['language'],
            sources=result.get('sources', []),
            had_image=result.get('had_image', False)
        )
    except Exception as e:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(request: ConversationCreate):
    """Create a new conversation"""
    assistant = get_assistant()

    try:
        conv_id = assistant.create_new_conversation(user_id=request.user_id)

        conversations = assistant.get_all_conversations(user_id=request.user_id)
        conv = next((c for c in conversations if c['conversation_id'] == conv_id), None)

        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found after creation")

        return ConversationResponse(
            conversation_id=conv['conversation_id'],
            created_at=conv['created_at'],
            last_updated=conv['last_updated']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create conversation: {str(e)}")


@router.get("/conversations")
async def get_conversations(user_id: Optional[str] = None):
    """Get all conversations for a user"""
    assistant = get_assistant()

    try:
        conversations = assistant.get_all_conversations(user_id=user_id)
        return {"conversations": conversations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get conversations: {str(e)}")


@router.get("/conversations/{conversation_id}/history")
async def get_conversation_history(conversation_id: str):
    """Get conversation history"""
    assistant = get_assistant()

    try:
        history = assistant.get_conversation_history(conversation_id)
        return {"conversation_id": conversation_id, "messages": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation"""
    assistant = get_assistant()

    try:
        assistant.delete_conversation(conversation_id)
        return {"status": "success", "message": f"Conversation {conversation_id} deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete conversation: {str(e)}")


@router.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    """Upload and add a document to the knowledge base"""
    assistant = get_assistant()

    ext = Path(file.filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type {ext} not allowed. Allowed types: {settings.ALLOWED_EXTENSIONS}"
        )

    temp_path = f"./temp_upload_{file.filename}"

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        assistant.add_document(temp_path)

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return {
            "status": "success",
            "message": f"Document {file.filename} added to knowledge base",
            "filename": file.filename
        }
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Failed to add document: {str(e)}")


@router.post("/reset")
async def reset_system():
    """Reset all singleton instances"""
    global assistant_instance

    try:
        TelecomEgyptAssistant.reset_instance()
        assistant_instance = None

        return {
            "status": "success",
            "message": "All singleton instances have been reset"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset: {str(e)}")