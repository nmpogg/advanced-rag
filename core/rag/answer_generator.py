"""
Answer generation module - Generate final answer với fallback logic.
"""
from .llm_chains import (
    route_query,
    reflect_query,
    grade_documents,
    generate_answer,
    generate_chitchat_response,
)
from retrieval import get_web_search_source


def generate_final_answer(
    query: str,
    top_docs: list
) -> str:
    """
    Generate final answer với 4-layer fallback strategy.
    
    Args:
        query: User query
        top_docs: Top documents từ retrieval
        
    Returns:
        answer: str
    """
    if not top_docs:
        return "Không có tài liệu liên quan được tìm thấy."
    
    context = "\n\n".join(top_docs)
    final_context = context
    
    # Check nếu internal docs đã đủ
    if grade_documents(context, query):
        print("Internal docs đủ thông tin...")
        try:
            answer = generate_answer(context, query)
            return answer
        except Exception as e:
            print(f"❌ LLMs lỗi: {e}")
            return f"Lỗi: {str(e)}"
    
    # Internal docs không đủ - thử web search
    print("Internal docs không đủ, thử web search...")

    web_result = get_web_search_source(query)
    
    if web_result["success"]:
        print("Web search thành công...")
        final_context = web_result["content"]
        try:
            answer = generate_answer(final_context, query)
            return answer
        except Exception as e:
            print(f"❌ LLMs lỗi: {e}")
    else:
        print(f"Không search được do: {web_result['reason']}")
    

    answer = generate_answer(context, query)
    return answer
