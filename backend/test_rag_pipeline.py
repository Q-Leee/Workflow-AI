import sys
import os
from pathlib import Path

# Fix Windows console UTF-8 printing issues
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add backend directory to PYTHONPATH
backend_dir = Path("c:/Users/gudrb/Desktop/projects/Workflow-AI/backend").resolve()
sys.path.insert(0, str(backend_dir))

# Mock environment settings if needed
os.environ["DATABASE_URL"] = "sqlite:///test_wf.db"

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_crawler():
    print("=== Testing Web Crawler Utility ===")
    from app.services.web_crawler import crawl_url_to_markdown
    test_url = "https://raw.githubusercontent.com/run-llama/llama_index/main/README.md"
    try:
        md = crawl_url_to_markdown(test_url)
        print(f"[SUCCESS] Crawled {test_url}")
        print(f"Content snippet (first 300 chars):\n{md[:300]}\n...")
        return True
    except Exception as e:
        print(f"[FAIL] Crawler error: {e}")
        return False

def test_rag_jit():
    print("\n=== Testing JIT RAG Pipeline ===")
    from app.db.database import init_db
    init_db()  # Initialize SQLite database schema
    
    from app.services import rag
    # Using a fake user ID for testing
    user_id = "test_user_architect_123"
    question = "What is the harness project and its Docker isolation level mentioned in https://raw.githubusercontent.com/run-llama/llama_index/main/README.md?"
    
    try:
        print(f"Triggering retrieve_and_rerank with JIT Crawler for question: '{question}'")
        sources = rag.retrieve_and_rerank(
            user_id=user_id,
            question=question,
            document_id=None,
            filename=None,
            doc_type=None,
            page_min=None,
            page_max=None,
            top_k=3,
            do_rerank=False
        )
        print(f"[SUCCESS] Retrieved {len(sources)} source chunks!")
        for i, src in enumerate(sources, 1):
            print(f"[{i}] Chunk from {src.filename} (page {src.page}):")
            print(f"    Text snippet: {src.text[:150]}...")
            
        print("\n=== Testing Cache Hit (Second Query with identical URL) ===")
        # The second query should hit cache (Cache Hit log should print)
        sources_cached = rag.retrieve_and_rerank(
            user_id=user_id,
            question="Tell me more about the features in https://raw.githubusercontent.com/run-llama/llama_index/main/README.md",
            document_id=None,
            filename=None,
            doc_type=None,
            page_min=None,
            page_max=None,
            top_k=3,
            do_rerank=False
        )
        print(f"[SUCCESS] Cache retrieval fetched {len(sources_cached)} chunks!")
        return True
    except Exception as e:
        logger.exception("JIT RAG Pipeline failure")
        return False

if __name__ == "__main__":
    crawler_ok = test_crawler()
    if crawler_ok:
        test_rag_jit()
