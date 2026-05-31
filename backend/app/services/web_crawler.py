import logging
import httpx
from bs4 import BeautifulSoup
import markdownify

logger = logging.getLogger(__name__)

def crawl_url_to_markdown(url: str, timeout: float = 12.0) -> str:
    """
    Crawls a given URL and converts its HTML content into clean markdown.
    Strips away non-content sections such as script, style, nav, footer, header,
    noscript, and iframe to optimize token usage and RAG semantic search quality.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    
    logger.info(f"Initiating dynamic crawl request for URL: {url}")
    try:
        # Using follow_redirects=True to handle shortened URLs or canonical redirections
        with httpx.Client(follow_redirects=True, verify=False) as client:
            response = client.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            html = response.text
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error occurred while crawling {url}: {e.response.status_code}")
        raise ValueError(f"HTTP error {e.response.status_code} from web resource.")
    except httpx.RequestError as e:
        logger.error(f"Network error occurred while crawling {url}: {e}")
        raise ValueError(f"Network connection failed: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error occurred during crawling {url}")
        raise ValueError(f"Unexpected crawler error: {e}")

    try:
        soup = BeautifulSoup(html, "html.parser")
        
        # Decompose non-content or structural layout noise tags
        noise_tags = [
            "script", "style", "nav", "footer", "header", 
            "noscript", "iframe", "aside", "svg", "form"
        ]
        for tag in soup(noise_tags):
            tag.decompose()
            
        clean_html = str(soup)
        
        # Convert clean body HTML to standard Github-style Markdown
        markdown_text = markdownify.markdownify(
            clean_html,
            heading_style="ATX",
            strip=["img"],  # Strip images as RAG text embeddings don't index images directly
            bullets="-"
        )
        
        # Strip consecutive blank lines to normalize spacing
        normalized_lines = []
        consecutive_blank = 0
        for line in markdown_text.splitlines():
            if not line.strip():
                consecutive_blank += 1
                if consecutive_blank <= 1:
                    normalized_lines.append("")
            else:
                consecutive_blank = 0
                normalized_lines.append(line.rstrip())
                
        final_markdown = "\n".join(normalized_lines).strip()
        logger.info(f"Crawl completed. Converted {len(html)} HTML chars to {len(final_markdown)} clean markdown chars.")
        return final_markdown
        
    except Exception as e:
        logger.exception(f"Failed to parse and clean HTML from URL: {url}")
        raise ValueError(f"HTML parsing and markdown conversion failed: {e}")
