import re
from typing import List
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document


class PDFProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, top_n_table_content: int = 20):
        self.__top_n_table_content = top_n_table_content
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_pdf(self, file_path: str) -> List[Document]:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)

        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_id": i,
                "source": file_path,
                "processed_at": datetime.now().isoformat()
            })

        return chunks

    def extract_table_of_contents(self, documents: List[Document]) -> List[str]:
        toc = []
        for doc in documents:
            text = doc.page_content
            matches = re.findall(r'^\d+\.\s+([A-Z][^\n]+)', text, re.MULTILINE)
            toc.extend(matches)

        seen = set()
        toc_unique = []
        for item in toc:
            if item not in seen:
                seen.add(item)
                toc_unique.append(item)

        return toc_unique[:self.__top_n_table_content]
