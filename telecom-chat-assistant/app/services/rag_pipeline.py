from typing import Dict, List
from IPython.display import Image, display

from app.services.vector_store import VectorStoreManager
from app.services.chat_history import ChatHistoryManager
from app.services.rag_workflow import RAGWorkflowGraph


class RAGPipeline:
    """RAG-based question answering system (Singleton)"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.vector_store_manager = VectorStoreManager()
        self.chat_history_manager = ChatHistoryManager()
        self.workflow_graph = RAGWorkflowGraph()

        self._initialized = True
        print("RAGPipeline initialized (Singleton)")

    def generate_answer(self, query: str, conversation_id: str = "default",
                       image_path: str = None) -> Dict:
        """Generate answer using LangGraph workflow"""
        return self.workflow_graph.process_query(query, conversation_id, image_path)

    def chat(self, query: str, conversation_id: str = "default",
             image_path: str = None) -> str:
        """Simple chat interface"""
        result = self.generate_answer(query, conversation_id, image_path)
        return result['answer']

    def get_conversation_history(self, conversation_id: str) -> List[Dict]:
        """Get conversation history"""
        return self.chat_history_manager.get_conversation_history(conversation_id)

    def visualize(self):
        """Visualize the workflow"""
        try:
            display(Image(self.workflow_graph.app.get_graph().draw_mermaid_png()))
        except Exception as e:
            print(f"Could not visualize graph: {str(e)}")
            print("Install graphviz for visualization")