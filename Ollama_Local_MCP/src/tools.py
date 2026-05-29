import base64
import io
import time
from pathlib import Path
from typing import Any, Optional, TypedDict

import fitz
import httpx
import pytesseract
from bs4 import BeautifulSoup
from ollama import Client as _OllamaClient
from PIL import Image
from ddgs import DDGS

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama

from src.registry import get_skill, list_all

from src.config import (
    DEFAULT_MODEL,
    FETCH_MAX_CHARS,
    IMAGE_MAX_SIZE,
    IMAGE_QUALITY,
    OLLAMA_BASE,
    PDF_MAX_CHARS,
    RESEARCH_MAX_RESULTS,
    SEARCH_MAX_RESULTS,
    SUMMARIZE_CHUNK_OVERLAP,
    SUMMARIZE_CHUNK_SIZE,
    SUMMARIZE_SINGLE_CALL_THRESHOLD,
    UPLOADS_DIR,
    VISION_MODEL,
)
from src.logger import get_logger, log_call
from src.vector import clear_index, index_pdf, index_stats, index_text, index_url, retrieve
from src.scrapper import scrape_url
from src.pipeline import run_pipeline as _run
from src.pipeline import list_pipeline as _list


_REEXPORTS = (clear_index, index_pdf, index_stats, index_text, index_url, retrieve, scrape_url)

log = get_logger(__name__)


# ── Standard envelope every skill / agent returns ────────────────────────────

class SkillResult(TypedDict, total=False):
    ok: bool
    skill: str
    output: str
    metadata: dict
    data: dict
    error: Optional[str]


def envelope(
    skill: str,
    output: str = "",
    *,
    data: Optional[dict] = None,
    metadata: Optional[dict] = None,
    error: Optional[str] = None,
) -> SkillResult:
    return {
        "ok": error is None,
        "skill": skill,
        "output": output if error is None else "",
        "metadata": metadata or {},
        "data": data or {},
        "error": error,
    }


def extract_output(result: Any) -> str:
    """Pull `.output` from an envelope, or return the value unchanged."""
    if isinstance(result, dict) and "output" in result:
        return result.get("output", "") or ""
    return str(result) if result is not None else ""


# ── LangChain ChatOllama wrapper ─────────────────────────────────────────────
def _ollama_chat(
    prompt: str,
    system: str = "",
    model: str = DEFAULT_MODEL,
    history: list[dict] | None = None,
) -> str:
    messages = []
    if system:
        messages.append(SystemMessage(content=system))

    if history:
        for m in history:
            role, content = m.get("role"), m.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
            elif role == "system":
                messages.append(SystemMessage(content=content))

    messages.append(HumanMessage(content=prompt))

    llm = ChatOllama(model=model, base_url=OLLAMA_BASE)
    response = llm.invoke(messages)
    return response.content


def _encode_image(path: str) -> str:
    """Load an image, resize if oversized, return base64 JPEG string."""
    img = Image.open(path).convert("RGB")
    w, h = img.size

    if max(w, h) > IMAGE_MAX_SIZE:
        scale = IMAGE_MAX_SIZE / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=IMAGE_QUALITY)
    return base64.b64encode(buf.getvalue()).decode()


# ── Core tools ───────────────────────────────────────────────────────────────
@log_call
def chat(
    prompt: str,
    system: str = "",
    model: str = DEFAULT_MODEL,
    history: list[dict] | None = None,
) -> str:
    """Send a chat message to a local Ollama model (via ChatOllama)."""
    return _ollama_chat(prompt=prompt, system=system, model=model, history=history)


@log_call
def generate(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
) -> str:
    """Raw text completion (no chat format)."""
    response = httpx.post(
        f"{OLLAMA_BASE}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "options": {"temperature": temperature, "max_tokens": 2048},
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"]


@log_call
def list_models() -> list[str]:
    """List all locally available Ollama models."""
    response = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=10)
    response.raise_for_status()
    return [model["name"] for model in response.json().get("models", [])]


# ── Web search tools ─────────────────────────────────────────────────────────
@log_call
def web_search(query: str, max_results: int = SEARCH_MAX_RESULTS) -> list[dict]:
    """Search the web via DuckDuckGo. Returns title, url, snippet."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))

    return [
        {"title": r["title"], "url": r["href"], "snippet": r["body"]}
        for r in results
    ]


@log_call
def fetch_page(url: str, max_chars: int = FETCH_MAX_CHARS) -> str:
    """Fetch a URL and return clean plain text (HTML stripped)."""

    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36"}

    try:
        resp = httpx.get(url, headers=headers, follow_redirects=True, timeout=15)
        soup = BeautifulSoup(resp.text, "lxml")

        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = " ".join(soup.get_text(separator=" ").split())
        return text[:max_chars]
    except Exception as exc:
        return f"Error fetching {url}: {exc}"


@log_call
def research(
    query: str,
    model: str = DEFAULT_MODEL,
    max_results: int = RESEARCH_MAX_RESULTS,
) -> str:
    """Search the web then synthesize an answer with citations via Ollama."""
    results = web_search(query, max_results=max_results)

    if not results:
        return "No search results found."

    combined = f"Search query: {query}\n\n"
    for i, r in enumerate(results, 1):
        combined += f"[{i}] {r['title']}\nURL: {r['url']}\n{r['snippet']}\n\n"

    answer = _ollama_chat(
        f"Answer: '{query}'\n\n{combined}",
        system="You are a research assistant. Synthesize results into a clear "
               "factual answer. Cite sources by [number].",
        model=model,
    )

    sources = "\n".join(f"[{i+1}] {r['url']}" for i, r in enumerate(results))
    return f"{answer}\n\nSources:\n{sources}"


# ── Pdf tools (with ocr: tesseract-first, vision-model fallback) ─────────────
def _page_nums(pages: str, total: int) -> list[int]:
    if pages == "all":
        return list(range(total))
    return [int(p.strip()) - 1 for p in pages.split(",") if p.strip().isdigit()]


def _tesseract_ocr(image_path: str) -> str:
    """Run Tesseract on an image path. Returns empty string on failure."""
    try:
        img = Image.open(image_path)
        return pytesseract.image_to_string(img).strip()
    except Exception:
        return ""


def _extract_pdf_images(path: str, pages: str = "all") -> list[dict]:
    """Save every image embedded in the selected PDF pages."""
    doc = fitz.open(path)
    stem = Path(path).stem
    page_nums = _page_nums(pages, doc.page_count)

    extracted = []
    for pnum in page_nums:
        if not (0 <= pnum < doc.page_count):
            continue
        page = doc[pnum]
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            data = doc.extract_image(xref)
            ext = data.get("ext", "png")
            out_path = UPLOADS_DIR / f"{stem}_p{pnum+1}_img{img_idx}.{ext}"
            out_path.write_bytes(data["image"])
            extracted.append({
                "page":  pnum + 1,
                "index": img_idx,
                "path":  str(out_path),
            })

    doc.close()
    return extracted


@log_call
def pdf_ocr(
    path: str,
    pages: str = "all",
    vision_model: str = VISION_MODEL,
) -> list[dict]:
    """Extract every image from the PDF and OCR it."""
    extracted = _extract_pdf_images(path, pages=pages)
    results = []

    for info in extracted:
        img_path = info["path"]
        text = _tesseract_ocr(img_path)
        backend = "tesseract"

        if not text:
            try:
                text = ocr_image(img_path, model=vision_model)
                backend = "vision"
            except Exception as exc:
                text = f"(OCR failed: {exc})"
                backend = "error"

        results.append({
            "page":        info["page"],
            "image_index": info["index"],
            "path":        img_path,
            "text":        text,
            "backend":     backend,
        })

    return results


@log_call
def read_pdf(
    path: str,
    pages: str = "all",
    max_chars: int = PDF_MAX_CHARS,
    include_images: bool = False,
    vision_model: str = VISION_MODEL,
) -> str:
    """Extract text from a local PDF."""
    doc = fitz.open(path)
    total = doc.page_count
    page_nums = _page_nums(pages, total)

    chunks = [
        f"--- Page {n + 1} ---\n{doc[n].get_text()}"
        for n in page_nums
        if 0 <= n < total
    ]
    doc.close()

    full = "\n".join(chunks)

    if include_images:
        ocr = pdf_ocr(path, pages=pages, vision_model=vision_model)
        if ocr:
            full += "\n\n--- IMAGES (OCR) ---\n"
            for r in ocr:
                full += (
                    f"\n[page {r['page']}, image {r['image_index']} "
                    f"via {r['backend']}]\n{r['text']}\n"
                )

    truncated = len(full) > max_chars
    return full[:max_chars] + ("\n\n[Truncated]" if truncated else "")


@log_call
def pdf_qa(
    path: str,
    question: str,
    model: str = DEFAULT_MODEL,
    include_images: bool = False,
) -> str:
    """Read a PDF and answer a question about its content. """
    text = read_pdf(path, pages="all", max_chars=8000, include_images=include_images)
    return _ollama_chat(
        f"Based on this document, answer: {question}\n\nDOCUMENT:\n{text}",
        system="You are a document analyst. Answer accurately using only the document content.",
        model=model,
    )


@log_call
def pdf_summarize(path: str, model: str = DEFAULT_MODEL, include_images: bool = False) -> str:
    """Read a PDF and summarize it. """
    text = read_pdf(path, pages="all", include_images=include_images)
    return extract_output(summarize(text, style="bullets", model=model))


# ── Image tools ──────────────────────────────────────────────────────────────
_ollama_client = _OllamaClient(host=OLLAMA_BASE)


def _vision_call(path: str, prompt: str, model: str = VISION_MODEL) -> str:
    """Shared vision call — encodes image and sends to Ollama at OLLAMA_BASE."""
    b64 = _encode_image(path)
    resp = _ollama_client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt, "images": [b64]}],
    )
    return resp["message"]["content"]


@log_call
def describe_image(
    path: str,
    prompt: str = "Describe this image in detail.",
    model: str = VISION_MODEL,
) -> str:
    """Describe an image using a vision-capable model."""
    return _vision_call(path, prompt, model)


@log_call
def image_qa(path: str, question: str, model: str = VISION_MODEL) -> str:
    """Answer a specific question about an image (uses VISION_MODEL)."""
    return _vision_call(path, question, model)


@log_call
def ocr_image(path: str, model: str = VISION_MODEL) -> str:
    """Extract all visible text from an image via the vision model."""
    return _vision_call(
        path,
        "Extract ALL text visible in this image. Return only the extracted text, nothing else.",
        model,
    )


# ── Skill tools ──────────────────────────────────────────────────────────────
def _chunk_text(
    text: str,
    chunk_size: int = SUMMARIZE_CHUNK_SIZE,
    overlap: int = SUMMARIZE_CHUNK_OVERLAP,
) -> list[str]:
    """Split long text into overlapping chunks with a recursive character splitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    return splitter.split_text(text or "")


_STYLE_MAP = {
    "bullets":   "Return a concise bullet-point summary. No preamble.",
    "tldr":      "Return a single TL;DR sentence only.",
    "paragraph": "Return a 2-3 sentence plain prose summary.",
}


def _summarize_impl(text: str, style: str, model: str) -> str:
    """Internal recursive map-reduce."""
    system = _STYLE_MAP.get(style, _STYLE_MAP["bullets"])
    if len(text) <= SUMMARIZE_SINGLE_CALL_THRESHOLD:
        return _ollama_chat(f"Summarize this:\n\n{text}", system=system, model=model)

    chunks = _chunk_text(text)
    log.info("summarize: map-reduce %d chunks from %d chars (style=%s)", len(chunks), len(text), style)
    chunk_summaries = [
        _ollama_chat(
            f"Summarize this section (one of {len(chunks)}):\n\n{chunk}",
            system=system,
            model=model,
        )
        for chunk in chunks
    ]

    combined = "\n\n".join(
        f"[section {i + 1}/{len(chunks)}]\n{s}"
        for i, s in enumerate(chunk_summaries)
    )
    if len(combined) <= SUMMARIZE_SINGLE_CALL_THRESHOLD:
        return _ollama_chat(
            "Combine these section summaries into a single coherent summary. "
            "Do not list them as separate sections — merge into one.\n\n" + combined,
            system=system,
            model=model,
        )
    log.info("summarize: combined summaries still %d chars, recursing", len(combined))
    return _summarize_impl(combined, style=style, model=model)


def _wrap_prompt_skill(
    name: str,
    input_text: str,
    model: str,
    runner,
    extra_meta: Optional[dict] = None,
) -> SkillResult:
    """Shared envelope builder for prompt-shaped skills (summarize/translate/code_review)."""
    t0 = time.perf_counter()
    try:
        output = runner()
    except Exception as exc:
        return envelope(
            skill=name,
            error=f"{type(exc).__name__}: {exc}",
            metadata={
                "model": model,
                "input_chars": len(input_text or ""),
                "duration_s": round(time.perf_counter() - t0, 3),
                **(extra_meta or {}),
            },
        )
    return envelope(
        skill=name,
        output=output,
        metadata={
            "model": model,
            "input_chars": len(input_text or ""),
            "output_chars": len(output),
            "duration_s": round(time.perf_counter() - t0, 3),
            **(extra_meta or {}),
        },
    )


@log_call
def summarize(text: str, style: str = "bullets", model: str = DEFAULT_MODEL) -> SkillResult:
    """Summarize text via recursive map-reduce for arbitrary length."""
    text = text or ""
    return _wrap_prompt_skill(
        name="summarize",
        input_text=text,
        model=model,
        runner=lambda: _summarize_impl(text, style=style, model=model),
        extra_meta={"style": style},
    )


@log_call
def translate(text: str, target_language: str, model: str = DEFAULT_MODEL) -> SkillResult:
    """Translate text into any language."""
    return _wrap_prompt_skill(
        name="translate",
        input_text=text,
        model=model,
        runner=lambda: _ollama_chat(
            f"Translate to {target_language}. Return ONLY the translation:\n\n{text}",
            model=model,
        ),
        extra_meta={"target_language": target_language},
    )


@log_call
def code_review(code: str, language: str = "Python", model: str = DEFAULT_MODEL) -> SkillResult:
    """Review code for bugs and improvements."""
    return _wrap_prompt_skill(
        name="code_review",
        input_text=code,
        model=model,
        runner=lambda: _ollama_chat(
            f"Review this {language} code for bugs and improvements:\n\n"
            f"```{language.lower()}\n{code}\n```",
            system="You are an expert code reviewer. Be direct and specific.",
            model=model,
        ),
        extra_meta={"language": language},
    )


# ── Custom skill runner ──────────────────────────────────────────────────────
@log_call
def run_skill(
    skill_name: str,
    user_input: str,
    context: Optional[dict] = None,
) -> SkillResult:
    """Run a registered custom skill."""
    skill = get_skill(skill_name)
    if not skill:
        registered = [s["name"] for s in list_all()["skills"]]
        return envelope(
            skill=skill_name,
            error=f"Skill '{skill_name}' not found. Available: {registered or ['(none registered)']}",
            metadata={"input_chars": len(user_input or "")},
        )

    model = skill.get("model") or DEFAULT_MODEL
    prompt = skill["user_prompt_template"].replace("{input}", user_input or "")
    t0 = time.perf_counter()
    try:
        output = _ollama_chat(prompt, system=skill["system_prompt"], model=model)
    except Exception as exc:
        return envelope(
            skill=skill_name,
            error=f"{type(exc).__name__}: {exc}",
            metadata={
                "model": model,
                "input_chars": len(user_input or ""),
                "duration_s": round(time.perf_counter() - t0, 3),
            },
        )

    return envelope(
        skill=skill_name,
        output=output,
        metadata={
            "model": model,
            "input_chars": len(user_input or ""),
            "output_chars": len(output),
            "duration_s": round(time.perf_counter() - t0, 3),
            "context_keys": list((context or {}).keys()),
        },
    )


def list_custom_skills() -> dict:
    """Return all registered custom skills and agents."""
    return list_all()


def run_pipeline(user_input: str, use_retrieval: bool = True) -> dict:
    """Run the sequential LangGraph of all registered custom agents."""
    return _run(user_input, use_retrieval=use_retrieval)


def list_pipeline() -> list[dict]:
    """List registered agents in pipeline execution order."""
    return _list()


# ── Utility ──────────────────────────────────────────────────────────────────
def ollama_is_running() -> bool:
    """Quick health check — True if Ollama is reachable."""
    try:
        httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=3).raise_for_status()
        return True
    except Exception:
        return False
