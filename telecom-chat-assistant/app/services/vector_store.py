import threading
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document as LangChainDocument

from app.core.config import settings


class VectorStoreManager:
    """Manage vector store for semantic search (Singleton)"""

    _instance = None
    _initialized = False
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        print(f"Initializing embeddings: {settings.EMBEDDING_MODEL}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': str(settings.DEVICE)},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self._initialized = True
        print("VectorStoreManager initialized (Singleton)")

    def prepare_documents(self, raw_docs: List[Dict]) -> List[LangChainDocument]:
        """Convert raw documents to LangChain format and split"""
        documents = []

        for doc in raw_docs:
            if not doc.get('content'):
                continue

            metadata = {
                'source': doc['url'],
                'title': doc['title'],
                'doc_type': doc.get('source', 'website')
            }

            langchain_doc = LangChainDocument(
                page_content=doc['content'],
                metadata=metadata
            )

            documents.append(langchain_doc)

        split_docs = self.text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(split_docs)} chunks")

        return split_docs

    def create_vector_store(self, documents: List[LangChainDocument],
                           persist: bool = True) -> FAISS:
        """Create FAISS vector store from documents (thread-safe)"""
        with self._lock:
            print("Creating vector store...")

            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )

            if persist:
                self.save_vector_store()

            print("Vector store created successfully!")
            return self.vector_store

    def save_vector_store(self, path: str = None):
        """Save vector store to disk"""
        if self.vector_store:
            save_path = path or str(settings.FAISS_INDEX_PATH)
            self.vector_store.save_local(save_path)
            print(f"Vector store saved to {save_path}")

    def load_vector_store(self, path: str = None) -> FAISS:
        """Load vector store from disk"""
        try:
            load_path = path or str(settings.FAISS_INDEX_PATH)
            self.vector_store = FAISS.load_local(
                load_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"Vector store loaded from {load_path}")
            return self.vector_store
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            return None

    def add_documents(self, documents: List[LangChainDocument]):
        """Add new documents to existing vector store (thread-safe)"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")

        with self._lock:
            self.vector_store.add_documents(documents)
            print(f"Added {len(documents)} documents to vector store")
            self.save_vector_store()

    def search(self, query: str, k: int = None) -> List[Dict]:
        """Search for relevant documents"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")

        k = k or settings.TOP_K_RETRIEVAL
        results = self.vector_store.similarity_search_with_score(query, k=k)

        return [
            {
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': float(score)
            }
            for doc, score in results
        ]