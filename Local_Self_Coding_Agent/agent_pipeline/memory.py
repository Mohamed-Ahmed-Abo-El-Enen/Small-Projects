from __future__ import annotations

import hashlib
from pathlib import Path

from . import config

_RAG_DIR = "rag_store"
_HASH_FILE = "source.hash"
_DIM_FILE = "embed_dim"


def _configure_settings() -> None:
    """Bind LlamaIndex to the local Ollama embed + reasoning models."""
    from llama_index.core import Settings
    from llama_index.embeddings.ollama import OllamaEmbedding
    from llama_index.llms.ollama import Ollama

    Settings.embed_model = OllamaEmbedding(
        model_name=config.EMBEDDING_MODEL,
        base_url=config.OLLAMA_HOST,
    )
    Settings.llm = Ollama(
        model=config.REASONING_MODEL,
        base_url=config.OLLAMA_HOST,
        request_timeout=config.LLM_TIMEOUT,
    )


def _embedding_dim() -> int:
    """Probe the configured embedding model for its vector dimensionality."""
    from llama_index.core import Settings
    return len(Settings.embed_model.get_text_embedding("dim-probe"))


def _gather_documents(project_dir: Path, workspace_dir: Path):
    """Collect PROJECT_DOC.md and per-iteration reports as LlamaIndex Documents."""
    from llama_index.core import Document

    docs: list = []
    final_doc = project_dir / "PROJECT_DOC.md"
    if final_doc.exists():
        docs.append(Document(
            text=final_doc.read_text(encoding="utf-8"),
            metadata={"source": "PROJECT_DOC.md"},
        ))
    iter_dir = workspace_dir / "iterations"
    if iter_dir.exists():
        for r in sorted(iter_dir.glob("iter_*/report.md")):
            docs.append(Document(
                text=r.read_text(encoding="utf-8"),
                metadata={"source": r.parent.name},
            ))
    return docs


def _source_hash(docs) -> str:
    """Hash the concatenated source texts to detect changes."""
    h = hashlib.sha256()
    for d in docs:
        h.update(d.metadata.get("source", "").encode("utf-8"))
        h.update(b"\x00")
        h.update(d.text.encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()


def _persist_dir(workspace_dir: Path) -> Path:
    """Return the on-disk RAG storage path for a workspace."""
    return workspace_dir / _RAG_DIR


def _new_faiss_store(dim: int):
    """Build a fresh FAISS L2 index wrapped in a LlamaIndex vector store."""
    import faiss
    from llama_index.vector_stores.faiss import FaissVectorStore
    return FaissVectorStore(faiss_index=faiss.IndexFlatL2(dim))


def _load_or_build_index(docs, workspace_dir: Path):
    """Reuse a persisted FAISS index when source + dim match; rebuild otherwise."""
    from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
    from llama_index.vector_stores.faiss import FaissVectorStore

    persist_dir = _persist_dir(workspace_dir)
    hash_file = persist_dir / _HASH_FILE
    dim_file = persist_dir / _DIM_FILE
    new_hash = _source_hash(docs)
    cur_dim = _embedding_dim()

    if persist_dir.exists() and hash_file.exists() and dim_file.exists():
        try:
            cached_hash = hash_file.read_text(encoding="utf-8").strip()
            cached_dim = int(dim_file.read_text(encoding="utf-8").strip())
            if cached_hash == new_hash and cached_dim == cur_dim:
                vs = FaissVectorStore.from_persist_dir(str(persist_dir))
                storage = StorageContext.from_defaults(
                    vector_store=vs, persist_dir=str(persist_dir)
                )
                return load_index_from_storage(storage), "loaded from disk"
        except Exception:
            pass

    vs = _new_faiss_store(cur_dim)
    storage = StorageContext.from_defaults(vector_store=vs)
    index = VectorStoreIndex.from_documents(docs, storage_context=storage)
    persist_dir.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(persist_dir))
    try:
        hash_file.write_text(new_hash, encoding="utf-8")
        dim_file.write_text(str(cur_dim), encoding="utf-8")
    except OSError:
        pass
    return index, "rebuilt and persisted"


def query_history(project_dir: str | Path, workspace_dir: str | Path,
                  query: str, k: int = 4) -> str:
    """Return relevant snippets from prior iteration docs for a query."""
    project_dir = Path(project_dir)
    workspace_dir = Path(workspace_dir)
    try:
        docs = _gather_documents(project_dir, workspace_dir)
    except Exception:
        return ""
    if not docs:
        return ""

    try:
        _configure_settings()
        index, _ = _load_or_build_index(docs, workspace_dir)
        retriever = index.as_retriever(similarity_top_k=k)
        nodes = retriever.retrieve(query)
        return "\n\n".join(
            f"[{n.metadata.get('source', '?')}]\n{n.get_content()[:1500]}"
            for n in nodes
        )
    except Exception:
        return "\n\n".join(
            f"[{d.metadata.get('source', '?')}]\n{d.text[:1500]}"
            for d in docs[:2]
        )
