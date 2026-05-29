import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# ── Ollama ─────────────────────────────────────────────────────────────────────
OLLAMA_BASE   =  os.getenv("OLLAMA_PUBLIC_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_TEXT_MODEL", "gemma3")

VISION_MODEL  = os.getenv("OLLAMA_VISION_MODEL", "qwen3-vl:latest")
VISION_MODELS = {"gemma3", "qwen3-vl"}


TOOL_CAPABLE_MODELS = {
    "llama3.1", "llama3.2", "llama3.3",
    "qwen2.5", "qwen3"
}


def is_tool_capable(model_name: str) -> bool:
    """Does the model support native tool_calls in Ollama."""
    if not model_name:
        return False
    lname = model_name.lower()

    if "-vl" in lname or "-vision" in lname:
        return False
    bare = lname.split(":", 1)[0]
    return any(bare.startswith(prefix) for prefix in TOOL_CAPABLE_MODELS)

# ── Paths ──────────────────────────────────────────────────────────────────────
SRC_DIR          = Path(__file__).parent
PROJECT_ROOT     = SRC_DIR.parent
DATA_DIR         = Path(os.getenv("LOCAL_MCP_DATA_DIR", PROJECT_ROOT / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH          = DATA_DIR / "sessions.db"
REGISTRY_PATH    = DATA_DIR / "skills_registry.json"
UPLOADS_DIR      = DATA_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

# ── Web search ─────────────────────────────────────────────────────────────────
SEARCH_MAX_RESULTS  = 5
FETCH_MAX_CHARS     = int(os.getenv("FETCH_MAX_CHARS", "50000"))
RESEARCH_MAX_RESULTS = 3

# ── PDF / Image ────────────────────────────────────────────────────────────────
PDF_MAX_CHARS   = int(os.getenv("PDF_MAX_CHARS", "100000"))
IMAGE_MAX_SIZE  = 1024
IMAGE_QUALITY   = 85

# ── Summarize / chunking (map-reduce for long inputs) ─────────────────────────
SUMMARIZE_SINGLE_CALL_THRESHOLD = int(os.getenv("SUMMARIZE_SINGLE_CALL_THRESHOLD", "6000"))
SUMMARIZE_CHUNK_SIZE            = int(os.getenv("SUMMARIZE_CHUNK_SIZE", "4000"))
SUMMARIZE_CHUNK_OVERLAP         = int(os.getenv("SUMMARIZE_CHUNK_OVERLAP", "200"))

# ── Embeddings / FAISS ─────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
FAISS_DIR       = DATA_DIR / "faiss_index"
FAISS_DIR.mkdir(exist_ok=True)
CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 100
RETRIEVAL_K     = 4

# ── LangSmith tracing ──────────────────────────────────────────────────────────
if os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY"):
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGCHAIN_PROJECT", "LOCAL_MCP"))

# ── MCP ────────────────────────────────────────────────────────────────────────
MCP_SERVER_NAME    = "ollama-mcp"
MCP_PROTOCOL_VERSION = "2024-11-05"

# ── Streamlit ──────────────────────────────────────────────────────────────────
APP_TITLE   = "Ollama MCP Assistant"
APP_LAYOUT  = "wide"
