import os
from pathlib import Path
import dotenv

project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
dotenv_path = os.path.join(project_dir, '.env')
dotenv.load_dotenv(dotenv_path)


class Config:
    """Application configuration"""
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./data/uploads"))
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./faiss_index")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    MAX_QUESTIONS = int(os.getenv("MAX_QUESTIONS", 10))
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
    TOP_N_TABLE_CONTENT = int(os.getenv("TOP_N_TABLE_CONTENT", 20))
    SIMILARITY_SEARCH_K = int(os.getenv("SIMILARITY_SEARCH_K", 5))

    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))

    def __init__(self):
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        Path(self.VECTOR_DB_PATH).mkdir(parents=True, exist_ok=True)


config = Config()
os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
