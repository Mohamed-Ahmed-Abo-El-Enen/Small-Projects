"""
Telecom Egypt Intelligent Assistant
Main application package
"""
__version__ = "1.0.0"


from app.core.config import settings
from app.core.singleton import SingletonMeta
from app.core.logger import get_logger

__all__ = ['settings', 'SingletonMeta', 'get_logger']


from app.services.assistant import TelecomEgyptAssistant
from app.services.scraper import WebScraper
from app.services.document_processor import DocumentProcessor
from app.services.image_processor import ImageProcessor
from app.services.vector_store import VectorStoreManager
from app.services.chat_history import ChatHistoryManager
from app.services.rag_workflow import RAGWorkflowGraph
from app.services.rag_pipeline import RAGPipeline

__all__ = [
    'TelecomEgyptAssistant',
    'WebScraper',
    'DocumentProcessor',
    'ImageProcessor',
    'VectorStoreManager',
    'ChatHistoryManager',
    'RAGWorkflowGraph',
    'RAGPipeline'
]

from app.api.routes import router

__all__ = ['router']