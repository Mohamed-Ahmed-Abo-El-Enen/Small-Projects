from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify
from pydantic import BaseModel, Field

from .. import config, memory
from ..events import EventLogger
from ..llm import chat_json, reasoning_llm
from ..models import ProjectState, ResearchNote

SYSTEM = (
    "You are the Research Agent. You take a software idea, a target programming "
    "language, and the FULL text of relevant web pages (already fetched and "
    "cleaned to markdown — code blocks preserved). Produce a concise, factual "
    "set of research notes that the rest of the pipeline can rely on. "
    "Bias your notes toward libraries, idioms, code samples, and pitfalls "
    "SPECIFIC to the target language. Quote real code snippets from the "
    "fetched pages when they illustrate a useful pattern. Stay strictly "
    "grounded in the provided pages — do not invent sources."
)

PAGE_FETCH_TIMEOUT = 10
PAGE_MAX_CHARS = 6000
PAGE_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
SKIP_TAGS = ("script", "style", "noscript", "header", "footer", "nav",
             "aside", "form", "iframe", "svg", "button")
SKIP_SELECTORS = (
    "[class*=toc]", "[id*=toc]",
    "[class*=table-of-contents]",
    "[class*=breadcrumb]",
    "[class*=sidebar]",
    "[class*=cookie]",
    "[aria-label*=table of contents]",
    "[aria-label*=breadcrumb]",
)
MAIN_SELECTORS = (
    "main", "article",
    "[role=main]",
    "#content", "#main-content", "#body-content",
    ".content", ".main-content", ".markdown-body", ".post-content",
    ".article-body", ".answer", ".question",
)


class _Notes(BaseModel):
    notes: list[ResearchNote] = Field(default_factory=list)


def _ddg_search(query: str, k: int) -> list[dict]:
    """Run a DuckDuckGo text search, returning an empty list on any failure."""
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=k))
    except Exception:
        return []


def _compact(text: str) -> str:
    """Collapse runs of blank lines and strip trailing whitespace per line."""
    out: list[str] = []
    blank = 0
    for line in text.splitlines():
        stripped = line.rstrip()
        if stripped:
            out.append(stripped)
            blank = 0
        elif blank == 0:
            out.append("")
            blank += 1
    return "\n".join(out)


def _fetch_page(url: str) -> str:
    """Fetch a URL and return its main content as markdown with code blocks preserved."""
    if not url:
        return ""
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": PAGE_USER_AGENT,
                     "Accept": "text/html,application/xhtml+xml"},
            timeout=PAGE_FETCH_TIMEOUT,
            allow_redirects=True,
        )
    except requests.RequestException:
        return ""
    if resp.status_code != 200:
        return ""
    if not any(t in resp.headers.get("Content-Type", "").lower() for t in ("html", "xml")):
        return ""

    try:
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception:
        return ""

    for tag in soup(SKIP_TAGS):
        tag.decompose()
    for selector in SKIP_SELECTORS:
        try:
            for tag in soup.select(selector):
                tag.decompose()
        except Exception:
            pass

    main = None
    for selector in MAIN_SELECTORS:
        try:
            main = soup.select_one(selector)
        except Exception:
            main = None
        if main is not None:
            break
    if main is None:
        main = soup.body or soup
    try:
        md = markdownify(str(main), heading_style="ATX")
    except Exception:
        md = main.get_text("\n", strip=True)

    return _compact(md)[:PAGE_MAX_CHARS]


def _fetch_pages_parallel(urls: list[str], max_workers: int = 5) -> list[str]:
    """Fetch a list of URLs concurrently, preserving order."""
    if not urls:
        return []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(_fetch_page, urls))


def _build_user_prompt(idea: str, results: list[dict],
                       prior_context: str) -> str:
    """Assemble the user prompt: idea, prior context, fetched pages."""
    pieces = [
        f"# Project idea\n{idea}\n",
    ]
    if prior_context:
        pieces.append(
            "# Relevant excerpts from previous iterations (retrieved by embedding)\n"
            f"{prior_context}\n"
        )
    if results:
        pieces.append("# Fetched web pages")
        for i, r in enumerate(results, 1):
            title = r.get("title", "") or "(no title)"
            href = r.get("href") or r.get("url", "")
            content = r.get("content") or r.get("snippet") or r.get("body") or ""
            pieces.append(f"## [{i}] {title}\n{href}\n\n{content}\n")
    pieces.append(
        "Produce 3 to 8 high-signal research notes. Each note must include "
        "source URL, title, a 1-3 sentence summary, and 1-3 short snippets "
        "quoted verbatim from the fetched page (prefer code excerpts)."
    )
    return "\n".join(pieces)


def run(state: ProjectState, logger: EventLogger) -> ProjectState:
    """Search the web, fetch each result page, and produce research notes."""
    search_query = f"{state.idea} Python".strip()
    logger.log(state, "research", "searching the web for context",
               idea=state.idea[:200], query=search_query)
    raw = _ddg_search(search_query, config.RESEARCH_RESULTS)

    urls = [r.get("href") or r.get("url", "") for r in raw]
    logger.log(state, "research", f"fetching {len(urls)} pages in parallel",
               urls=[u for u in urls if u])
    contents = _fetch_pages_parallel(urls) if urls else []

    results: list[dict] = []
    fetched_chars = 0
    for r, content in zip(raw, contents):
        results.append({
            "title": r.get("title", ""),
            "href": r.get("href") or r.get("url", ""),
            "snippet": r.get("body") or r.get("snippet", ""),
            "content": content,
        })
        fetched_chars += len(content)
    logger.log(state, "research",
               f"crawled {sum(1 for c in contents if c)}/{len(contents)} pages "
               f"({fetched_chars} chars total)",
               per_page=[len(c) for c in contents])

    prior_context = ""
    if state.iteration > 1:
        prior_context = memory.query_history(
            state.project_dir, state.workspace_dir, query=state.idea, k=4
        )
        if prior_context:
            logger.log(state, "research", "retrieved prior-iteration context",
                       chars=len(prior_context))

    user = _build_user_prompt(state.idea, results, prior_context)
    parsed = chat_json(reasoning_llm(json_mode=True), SYSTEM, user, _Notes)

    state.research = parsed.notes
    logger.log(state, "research", f"collected {len(parsed.notes)} notes",
               level="result", count=len(parsed.notes))
    return state
