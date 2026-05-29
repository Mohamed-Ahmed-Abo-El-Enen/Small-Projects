import httpx
from bs4 import BeautifulSoup

from src.logger import get_logger, log_call

log = get_logger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36"
    ),
}

_DROP_TAGS = ("script", "style", "nav", "footer", "header", "aside", "form", "noscript")
_TEXT_TAGS = ("h1", "h2", "h3", "h4", "h5", "p", "li", "blockquote", "pre")


@log_call
def scrape_url(url: str, timeout: int = 20) -> dict:
    """Fetch a URL and return {url, title, text} with paragraph structure preserved."""
    try:
        resp = httpx.get(url, headers=_HEADERS, follow_redirects=True, timeout=timeout)
        resp.raise_for_status()
    except Exception as exc:
        log.warning("scrape failed: %s — %s", url, exc)
        return {"url": url, "title": "", "text": "", "error": str(exc)}

    soup = BeautifulSoup(resp.text, "lxml")

    for tag in soup(_DROP_TAGS):
        tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    title = title or url

    root = soup.find("article") or soup.find("main") or soup.body or soup

    blocks = []
    for el in root.find_all(_TEXT_TAGS):
        text = el.get_text(" ", strip=True)
        if text:
            blocks.append(text)

    if blocks:
        text = "\n\n".join(blocks)
    else:
        text = root.get_text("\n\n", strip=True)

    log.info("scraped %s — title=%r blocks=%d chars=%d", url, title[:60], len(blocks), len(text))
    return {"url": url, "title": title, "text": text}
