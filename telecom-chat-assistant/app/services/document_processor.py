import base64
from pathlib import Path
from typing import Optional, Dict
from bs4 import BeautifulSoup
from PIL import Image
import pytesseract
from docx import Document
from langchain_community.document_loaders import PyPDFLoader


class DocumentProcessor:
    """Process different document formats"""
    @staticmethod
    def process_pdf(file_path: str) -> str:
        """Extract text from PDF"""
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            return "\n\n".join([page.page_content for page in pages])
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            return ""

    @staticmethod
    def process_docx(file_path: str) -> str:
        """Extract text from DOCX"""
        try:
            doc = Document(file_path)
            return "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        except Exception as e:
            print(f"Error processing DOCX: {str(e)}")
            return ""

    @staticmethod
    def process_txt(file_path: str) -> str:
        """Extract text from TXT"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error processing TXT: {str(e)}")
            return ""

    @staticmethod
    def process_html(file_path: str) -> str:
        """Extract text from HTML"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                return soup.get_text(separator='\n', strip=True)
        except Exception as e:
            print(f"Error processing HTML: {str(e)}")
            return ""

    @staticmethod
    def encode_image_to_base64(image_path: str) -> str:
        """Encode image to base64 string"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image: {str(e)}")
            return ""

    @staticmethod
    def process_image_with_ocr(file_path: str) -> str:
        """Extract text from image using OCR (fallback method)"""
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image, lang='ara+eng')
            return text
        except Exception as e:
            print(f"Error processing image with OCR: {str(e)}")
            return ""

    @classmethod
    def process_document(cls, file_path: str) -> Optional[Dict]:
        """Process document based on extension"""
        path = Path(file_path)
        ext = path.suffix.lower()

        processors = {
            '.pdf': cls.process_pdf,
            '.docx': cls.process_docx,
            '.txt': cls.process_txt,
            '.html': cls.process_html,
        }

        if ext in ['.png', '.jpg', '.jpeg', '.webp', '.gif']:
            return {
                'url': file_path,
                'title': path.name,
                'content': None,
                'source': 'uploaded_image',
                'image_path': file_path,
                'is_image': True
            }

        if ext not in processors:
            print(f"Unsupported file type: {ext}")
            return None

        content = processors[ext](file_path)

        if content:
            return {
                'url': file_path,
                'title': path.name,
                'content': content,
                'source': 'uploaded_document',
                'is_image': False
            }

        return None