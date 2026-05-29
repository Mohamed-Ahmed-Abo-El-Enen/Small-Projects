from pathlib import Path
from typing import Optional
import fitz
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    FAISS_DIR,
    OLLAMA_BASE,
    RETRIEVAL_K,
)
from src.scrapper import scrape_url

from src.logger import get_logger, log_call

log = get_logger(__name__)

_INIT_SENTINEL = "__faiss_init__"
_INDEX_FILE = FAISS_DIR / "index.faiss"

_embeddings = None
_store: Optional[FAISS] = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE)
    return _embeddings


def _load_or_create_store() -> FAISS:
    """Load the FAISS index from disk or bootstrap an empty one."""
    embeddings = _get_embeddings()
    if _INDEX_FILE.exists():
        return FAISS.load_local(
            str(FAISS_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    store = FAISS.from_texts(
        [_INIT_SENTINEL],
        embeddings,
        metadatas=[{"source": _INIT_SENTINEL}],
    )
    store.save_local(str(FAISS_DIR))
    return store


def _get_store() -> FAISS:
    global _store
    if _store is None:
        _store = _load_or_create_store()
    return _store


def _save_store():
    if _store is not None:
        _store.save_local(str(FAISS_DIR))


@log_call
def index_pdf(path: str, source_label: Optional[str] = None) -> dict:
    """Chunk a PDF's text, embed each chunk, and add to the FAISS store."""
    doc = fitz.open(path)
    pages_text = [p.get_text() for p in doc]
    doc.close()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    label = source_label or Path(path).name
    docs: list[Document] = []
    for page_num, text in enumerate(pages_text, start=1):
        for chunk_idx, chunk in enumerate(splitter.split_text(text)):
            docs.append(Document(
                page_content=chunk,
                metadata={"source": label, "page": page_num, "chunk": chunk_idx},
            ))

    if not docs:
        log.warning("index_pdf: no text extracted from %s", label)
        return {"indexed_chunks": 0, "source": label, "note": "no text extracted"}

    store = _get_store()
    store.add_documents(docs)
    _save_store()
    log.info("index_pdf: %d chunks indexed from %s", len(docs), label)

    return {"indexed_chunks": len(docs), "source": label}


@log_call
def index_text(text: str, source_label: str) -> dict:
    """Chunk arbitrary text and add to the FAISS store."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(text)
    if not chunks:
        return {"indexed_chunks": 0, "source": source_label}

    docs = [
        Document(page_content=c, metadata={"source": source_label, "chunk": i})
        for i, c in enumerate(chunks)
    ]

    store = _get_store()
    store.add_documents(docs)
    _save_store()

    return {"indexed_chunks": len(chunks), "source": source_label}


@log_call
def index_url(url: str) -> dict:
    """Scrape a URL and add its paragraph-chunked text to the FAISS store."""
    data = scrape_url(url)

    if data.get("error"):
        log.warning("index_url: scrape error for %s — %s", url, data["error"])
        return {"indexed_chunks": 0, "url": url, "error": data["error"]}

    text = data.get("text", "")
    if not text.strip():
        log.warning("index_url: no text extracted from %s", url)
        return {"indexed_chunks": 0, "url": url, "note": "no text extracted"}

    title = data.get("title") or url
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(text)
    if not chunks:
        return {"indexed_chunks": 0, "url": url, "title": title}

    docs = [
        Document(
            page_content=c,
            metadata={"source": title, "url": url, "chunk": i},
        )
        for i, c in enumerate(chunks)
    ]

    store = _get_store()
    store.add_documents(docs)
    _save_store()
    log.info("index_url: %d chunks indexed from %s", len(chunks), url)

    return {"indexed_chunks": len(chunks), "url": url, "title": title}


@log_call
def retrieve(query: str, k: int = RETRIEVAL_K) -> list[dict]:
    """Similarity search — returns top-k chunks with metadata."""
    
    store = _get_store()
    results = store.similarity_search(query, k=k + 1)
    out = [
        {
            "content": r.page_content,
            "source": r.metadata.get("source"),
            "url":    r.metadata.get("url"),
            "page":   r.metadata.get("page"),
            "chunk":  r.metadata.get("chunk"),
        }
        for r in results
        if r.metadata.get("source") != _INIT_SENTINEL
    ][:k]
    log.debug("retrieve: %d hits for query=%r", len(out), query[:60])
    return out


def clear_index() -> dict:
    """Delete the on-disk FAISS index and reset the in-memory handle."""
    global _store
    _store = None
    removed = 0
    for f in FAISS_DIR.glob("*"):
        f.unlink()
        removed += 1
    return {"removed_files": removed}


def index_stats() -> dict:
    """Return basic stats about the current index."""
    try:
        store = _get_store()
        total = store.index.ntotal
        return {"total_vectors": total, "dir": str(FAISS_DIR)}
    except Exception as e:
        return {"error": str(e), "dir": str(FAISS_DIR)}
