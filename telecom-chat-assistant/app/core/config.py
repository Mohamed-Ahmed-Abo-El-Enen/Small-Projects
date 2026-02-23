import os
from typing import List
from pathlib import Path
import torch
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    OPENAI_API_KEY: str = "XXX-XXX-XXX"

    BASE_URL: str = "https://te.eg"
    MAX_PAGES: int = 30

    USE_LOCAL_MODEL: bool = True # True # For Windows os Make it False or use public ollama link from grok
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LOCAL_MODEL_NAME: str = "qwen3:latest"
    LOCAL_VISION_MODEL_NAME: str = "minicpm-v:latest"

    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    LLM_MODEL: str = "gpt-4"
    VISION_MODEL: str = "gpt-4o"

    MAX_IMAGE_SIZE_MB: int = 5
    IMAGE_RESIZE_MAX_DIMENSION: int = 336
    USE_OCR_FALLBACK: bool = True

    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RETRIEVAL: int = 5
    MAX_HISTORY_MESSAGES: int = 10

    BASE_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    FAISS_INDEX_PATH: Path = DATA_DIR / "faiss_index"
    CHAT_HISTORY_DB: Path = DATA_DIR / "chat_history.db"
    TMP_DATA_DIR: Path =  BASE_DIR / "tmp_data"

    ALLOWED_EXTENSIONS: List[str] = ['.pdf', '.docx', '.txt', '.html', '.png', '.jpg', '.jpeg']
    MAX_FILE_SIZE_MB: int = 10

    LANGUAGES: List[str] = ["ar", "en"]

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 1

    STREAMLIT_PORT: int = 8501

    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "logs"
    LOG_JSON: bool = False
    LOG_TO_FILE: bool = True
    LOG_TO_CONSOLE: bool = True

    @property
    def DEVICE(self) -> torch.device:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class Config:
        env_file = ".env"
        case_sensitive = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.FAISS_INDEX_PATH.mkdir(parents=True, exist_ok=True)

        if not self.USE_LOCAL_MODEL:
            os.environ["OPENAI_API_KEY"] = self.OPENAI_API_KEY

        print(f"Model Mode: {'LOCAL (Ollama)' if self.USE_LOCAL_MODEL else 'CLOUD (OpenAI)'}")
        if self.USE_LOCAL_MODEL:
            print(f"Ollama URL: {self.OLLAMA_BASE_URL}")
            print(f"Model: {self.LOCAL_MODEL_NAME}")


settings = Settings()