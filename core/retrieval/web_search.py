"""
Web search module - DuckDuckGo search với error handling, rate limiting, và reranking.
"""
import time
from langchain_community.tools import DuckDuckGoSearchRun
from sentence_transformers import CrossEncoder

from config import (
    WEB_SEARCH_COOLDOWN,
    MAX_WEB_SNIPPETS,
    MIN_WEB_RESULT_LENGTH,
    RERANKER_MODEL,
)


# Initialize web search tool
_web_search_tool = None
_web_search_last_time = 0
_reranker_model = None


def get_web_search_tool():
    """Get hoặc initialize web search tool."""
    global _web_search_tool
    if _web_search_tool is None:
        _web_search_tool = DuckDuckGoSearchRun()
    return _web_search_tool


def get_reranker():
    """Get hoặc initialize reranker model."""
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder(RERANKER_MODEL)
    return _reranker_model


def web_search_with_cooldown(query: str, cooldown: int = WEB_SEARCH_COOLDOWN) -> str:
    """
    Web search với rate limiting để tránh bị block.
    
    Args:
        query: Search query
        cooldown: Cooldown seconds giữa requests
        
    Returns:
        str: Web search results (empty string nếu error)
    """
    global _web_search_last_time
    
    try:
        # Đợi nếu cần (rate limiting)
        elapsed = time.time() - _web_search_last_time
        if elapsed < cooldown:
            wait_time = cooldown - elapsed
            print(f"  ⏳ Rate limiting - đợi {wait_time:.1f}s...")
            time.sleep(wait_time)
        
        _web_search_last_time = time.time()
        
        web_search_tool = get_web_search_tool()
        result = web_search_tool.invoke(query)
        
        return result if isinstance(result, str) else str(result)
        
    except Exception as e:
        print(f"❌ Web Search lỗi: {type(e).__name__}: {str(e)[:100]}")
        return ""


def rerank_web_results(
    query: str, 
    web_results_text: str, 
    max_snippets: int = MAX_WEB_SNIPPETS
) -> str:
    """
    Extract và re-rank web snippets bằng Cross-Encoder.
    
    Args:
        query: Original search query
        web_results_text: Raw web search results
        max_snippets: Max snippets to return
        
    Returns:
        str: Reranked web snippets
    """
    if not web_results_text or len(web_results_text.strip()) == 0:
        return ""
    
    try:
        # Split snippets
        snippets = web_results_text.split('\n')
        snippets = [s.strip() for s in snippets if len(s.strip()) > 30][:15]
        
        if not snippets:
            return web_results_text  # Return original if can't parse
        
        # Re-rank bằng Cross-Encoder
        reranker = get_reranker()
        pairs = [[query, snippet] for snippet in snippets]
        scores = reranker.predict(pairs)
        
        # Get top snippets
        ranked = sorted(zip(snippets, scores), key=lambda x: x[1], reverse=True)
        top_snippets = [snippet for snippet, score in ranked[:max_snippets]]
        
        print(f"  ⭐ Web snippets re-ranked: {len(top_snippets)} snippets")
        return "\n\n".join(top_snippets)
        
    except Exception as e:
        print(f"⚠️ Reranking web results lỗi: {e}")
        return web_results_text


def get_web_search_source(query: str, use_web_fallback: bool = True) -> dict:
    """
    Execute web search với all improvements.
    
    Args:
        query: Search query
        use_web_fallback: Cho phép web search
        
    Returns:
        dict: {
            "success": bool,
            "content": str (reranked results or empty),
            "reason": str (explanation)
        }
    """
    if not use_web_fallback:
        return {
            "success": False,
            "content": "",
            "reason": "Web search disabled"
        }
    
    # 1. Web search with cooldown
    web_results = web_search_with_cooldown(query)
    
    # 2. Check if results exist
    if not web_results or len(web_results.strip()) < MIN_WEB_RESULT_LENGTH:
        return {
            "success": False,
            "content": "",
            "reason": "Web search trả về kết quả trống"
        }
    
    # 3. Re-rank web results
    reranked_web = rerank_web_results(query, web_results)
    
    if not reranked_web or len(reranked_web.strip()) < MIN_WEB_RESULT_LENGTH:
        return {
            "success": False,
            "content": "",
            "reason": "Web reranking không hiệu quả"
        }
    
    return {
        "success": True,
        "content": reranked_web,
        "reason": "Web search thành công"
    }
