import requests
import time
from typing import List, Dict, Optional, Set
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

from app.core.logger import get_logger

logger = get_logger(__name__)


class WebScraper:
    """Scrape and process website content"""

    def __init__(self, base_url: str, max_pages: int = 50):
        self.base_url = base_url
        self.max_pages = max_pages
        self.visited_urls: Set[str] = set()
        self.documents: List[Dict] = []

    def is_valid_url(self, url: str) -> bool:
        """Check if URL belongs to the target domain"""
        parsed = urlparse(url)
        base_parsed = urlparse(self.base_url)
        return parsed.netloc == base_parsed.netloc

    def extract_text_from_page(self, url: str) -> Optional[Dict]:
        """Extract text content from a webpage"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

            title = soup.find('title')
            title_text = title.get_text().strip() if title else ""

            text = soup.get_text(separator='\n', strip=True)

            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = '\n'.join(lines)

            return {
                'url': url,
                'title': title_text,
                'content': text,
                'source': 'website'
            }

        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}", exc_info=True)
            return None

    def find_links(self, url: str) -> List[str]:
        """Find all links on a page"""
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')

            links = []
            for link in soup.find_all('a', href=True):
                absolute_url = urljoin(url, link['href'])
                if self.is_valid_url(absolute_url):
                    links.append(absolute_url)

            return list(set(links))

        except Exception as e:
            logger.error(f"Error finding links on {url}: {str(e)}")
            return []

    def crawl(self) -> List[Dict]:
        """Crawl website and extract content"""
        to_visit = [self.base_url]

        logger.info(f"Starting crawl of {self.base_url}...")

        while to_visit and len(self.visited_urls) < self.max_pages:
            url = to_visit.pop(0)

            if url in self.visited_urls:
                continue

            logger.debug(f"Crawling [{len(self.visited_urls) + 1}/{self.max_pages}]: {url}")

            self.visited_urls.add(url)

            doc = self.extract_text_from_page(url)
            if doc and len(doc['content']) > 100:
                self.documents.append(doc)
                logger.debug(f"Added document: {doc['title']}")

            new_links = self.find_links(url)
            for link in new_links:
                if link not in self.visited_urls and link not in to_visit:
                    to_visit.append(link)

            time.sleep(0.5)

        logger.info(f"Crawling complete! Collected {len(self.documents)} documents.")
        return self.documents