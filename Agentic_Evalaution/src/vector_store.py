import os
from typing import List
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document


class VectorStoreService:
    def __init__(self, persist_directory: str, embedding_model: str):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        self.vectorstore = None
        self.index_path = os.path.join(persist_directory, "faiss_index.pkl")

    def create_vectorstore(self, documents: List[Document]) -> str:
        """Create FAISS vector store from documents"""
        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        self.vectorstore.save_local(self.persist_directory)

        return f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def load_vectorstore(self):
        """Load existing FAISS vector store"""
        if not self.vectorstore:
            try:
                self.vectorstore = FAISS.load_local(
                    self.persist_directory,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Warning: Could not load existing index: {e}")
                self.vectorstore = None

    def retrieve_context(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve relevant documents using FAISS similarity search"""
        if not self.vectorstore:
            self.load_vectorstore()

        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Please ingest documents first.")

        return self.vectorstore.similarity_search(query, k=k)

