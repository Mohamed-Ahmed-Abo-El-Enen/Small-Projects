import time
from typing import List, Dict

from app.core.config import settings
from app.core.singleton import SingletonMeta
from app.core.logger import get_logger
from app.services.scraper import WebScraper
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStoreManager
from app.services.chat_history import ChatHistoryManager
from app.services.rag_pipeline import RAGPipeline

logger = get_logger(__name__)


class TelecomEgyptAssistant(metaclass=SingletonMeta):
    """Main chatbot system with history management (Singleton)"""
    def __init__(self):
        if hasattr(self, '_initialized'):
            logger.warning("Using existing TelecomEgyptAssistant instance (Singleton)")
            return

        logger.info("Initializing TelecomEgyptAssistant (Singleton)...")

        self.vector_store_manager = VectorStoreManager()
        self.chat_history_manager = ChatHistoryManager()
        self.rag_pipeline = None
        self.is_initialized = False

        self._initialized = True
        logger.info("TelecomEgyptAssistant initialized successfully")

    def initialize_from_web(self, max_pages: int = None):
        """Initialize system by scraping website"""
        max_pages = max_pages or settings.MAX_PAGES

        logger.info("=" * 60)
        logger.info("INITIALIZING TELECOM EGYPT ASSISTANT")
        logger.info("=" * 60)

        start_time = time.time()

        try:
            logger.info(f"[1/3] Scraping website: {settings.BASE_URL} (max {max_pages} pages)")
            scraper = WebScraper(settings.BASE_URL, max_pages)
            raw_docs = scraper.crawl()
            logger.info(f"Scraped {len(raw_docs)} documents")

            logger.info("[2/3] Preparing documents and creating embeddings...")
            documents = self.vector_store_manager.prepare_documents(raw_docs)
            logger.info(f"Prepared {len(documents)} document chunks")

            logger.info("[3/3] Building vector store...")
            self.vector_store_manager.create_vector_store(documents, persist=True)
            logger.info("Vector store created successfully")

            self.rag_pipeline = RAGPipeline()
            self.is_initialized = True

            elapsed = time.time() - start_time
            logger.info("=" * 60)
            logger.info(f"SYSTEM READY! (Initialization took {elapsed:.2f}s)")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}", exc_info=True)
            raise

    def load_existing_index(self, path: str = None):
        """Load pre-built vector store"""
        logger.info("Loading existing vector store...")

        try:
            path = path or str(settings.FAISS_INDEX_PATH)
            self.vector_store_manager.load_vector_store(path)
            self.rag_pipeline = RAGPipeline()
            self.is_initialized = True
            logger.info("System ready! Vector store loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load vector store: {str(e)}", exc_info=True)
            raise

    @classmethod
    def reset_instance(cls):
        """Reset singleton instance"""
        logger.info("Resetting singleton instances...")
        SingletonMeta.clear_instance(cls)
        SingletonMeta.clear_instance(VectorStoreManager)
        SingletonMeta.clear_instance(ChatHistoryManager)
        SingletonMeta.clear_instance(RAGPipeline)
        logger.info("All singleton instances cleared")

    def add_document(self, file_path: str):
        """Add a new document to knowledge base"""
        if not self.is_initialized:
            logger.error("Attempted to add document before system initialization")
            raise ValueError("System not initialized")

        logger.info(f"Processing document: {file_path}")

        try:
            doc_processor = DocumentProcessor()
            raw_doc = doc_processor.process_document(file_path)

            if raw_doc is None:
                logger.warning(f"Failed to process document: {file_path}")
                return

            documents = self.vector_store_manager.prepare_documents([raw_doc])
            self.vector_store_manager.add_documents(documents)

            logger.info(f"Document added successfully: {file_path}")
        except Exception as e:
            logger.error(f"Error adding document {file_path}: {str(e)}", exc_info=True)
            raise

    def chat(self, query: str, conversation_id: str = "default",
             image_path: str = None) -> str:
        """Chat with the assistant"""
        if not self.is_initialized:
            logger.error("Attempted to chat before system initialization")
            return "System not initialized. Please initialize first."

        logger.info(f"Chat request | conv_id={conversation_id} | query={query[:50]}...")

        try:
            start_time = time.time()
            response = self.rag_pipeline.chat(query, conversation_id, image_path)
            elapsed = time.time() - start_time

            logger.info(f"Chat completed in {elapsed:.2f}s | conv_id={conversation_id}")
            return response
        except Exception as e:
            logger.error(f"Chat error | conv_id={conversation_id}: {str(e)}", exc_info=True)
            raise

    def get_detailed_response(self, query: str, conversation_id: str = "default",
                              image_path: str = None) -> Dict:
        """Get detailed response with metadata"""
        if not self.is_initialized:
            logger.error("Attempted to get response before system initialization")
            raise ValueError("System not initialized")

        logger.debug(f"Detailed response request | conv_id={conversation_id}")

        try:
            start_time = time.time()
            result = self.rag_pipeline.generate_answer(query, conversation_id, image_path)
            elapsed = time.time() - start_time

            logger.info(
                f"Response generated | conv_id={conversation_id} | "
                f"language={result['language']} | sources={len(result.get('sources', []))} | "
                f"duration={elapsed:.2f}s"
            )
            return result
        except Exception as e:
            logger.error(
                f"Response generation error | conv_id={conversation_id}: {str(e)}",
                exc_info=True
            )
            raise

    def create_new_conversation(self, user_id: str = "default") -> str:
        """Create a new conversation"""
        logger.info(f"Creating new conversation for user: {user_id}")

        try:
            conv_id = self.chat_history_manager.create_conversation(user_id)
            logger.info(f"Conversation created | conv_id={conv_id} | user_id={user_id}")
            return conv_id
        except Exception as e:
            logger.error(f"Failed to create conversation: {str(e)}", exc_info=True)
            raise

    def get_conversation_history(self, conversation_id: str) -> List[Dict]:
        """Get conversation history"""
        logger.debug(f"Retrieving conversation history | conv_id={conversation_id}")

        try:
            history = self.rag_pipeline.get_conversation_history(conversation_id)
            logger.debug(f"Retrieved {len(history)} messages | conv_id={conversation_id}")
            return history
        except Exception as e:
            logger.error(
                f"Failed to retrieve history | conv_id={conversation_id}: {str(e)}",
                exc_info=True
            )
            raise

    def get_all_conversations(self, user_id: str = None) -> List[Dict]:
        """Get all conversations"""
        logger.debug(f"Retrieving all conversations | user_id={user_id}")

        try:
            conversations = self.chat_history_manager.get_all_conversations(user_id)
            logger.debug(f"Retrieved {len(conversations)} conversations")
            return conversations
        except Exception as e:
            logger.error(f"Failed to retrieve conversations: {str(e)}", exc_info=True)
            raise

    def delete_conversation(self, conversation_id: str):
        """Delete a conversation"""
        logger.info(f"Deleting conversation | conv_id={conversation_id}")

        try:
            self.chat_history_manager.delete_conversation(conversation_id)
            logger.info(f"Conversation deleted | conv_id={conversation_id}")
        except Exception as e:
            logger.error(
                f"Failed to delete conversation | conv_id={conversation_id}: {str(e)}",
                exc_info=True
            )
            raise

    def visualize_workflow(self):
        """Visualize the LangGraph workflow"""
        if not self.is_initialized:
            logger.error("Attempted to visualize workflow before system initialization")
            raise ValueError("System not initialized")

        logger.info("Visualizing workflow graph...")
        self.rag_pipeline.visualize()