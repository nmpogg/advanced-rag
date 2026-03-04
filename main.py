"""
Main RAG Pipeline - Orchestration với all modules.
"""
from config import JSON_PATH
from ingestion import load_vector_db, process_and_index_documents
from retrieval import hybrid_search_and_rerank
from generation import (
    route_query,
    reflect_query,
    generate_chitchat_response,
    generate_final_answer,
)


def main_rag_pipeline(user_query: str, vector_db) -> dict:
    """
    End-to-end RAG pipeline.
    
    Args:
        user_query: User question
        vector_db: ChromaDB vector database
        
    Returns:
        dict: {
            "answer": str,
            "source": str (internal/web_search/chitchat/error/...),
            "query": str (original),
            "refined_query": str (optimized for search),
            "doc_count": int
        }
    """
    print(f"\n{'='*60}")
    print(f"📝 CÂU HỎI: '{user_query}'")
    print(f"{'='*60}")
    
    try:
        # Router - chitchat vs RAG
        route = route_query(user_query)
        print(f"Phân loại query: {route}")
        if route == "chitchat":
            answer = generate_chitchat_response(user_query)
            return {
                "answer": answer,
                "query": user_query,
                "refined_query": None,
                "doc_count": 0
            }
        
        # Query Reflection
        refined_query = reflect_query(user_query)
        print(f"Câu hỏi tối ưu: '{refined_query}'")
        
        # Hybrid Search + Reranking
        top_docs = hybrid_search_and_rerank(refined_query, vector_db, json_path=JSON_PATH)
        
        # Generate Answer với Web Fallback
        final_answer = generate_final_answer(
            refined_query,
            top_docs
        )
        
        return {
            "answer": final_answer,
            "query": user_query,
            "refined_query": refined_query,
            "doc_count": len(top_docs)
        }
    
    except Exception as e:
        print(f"❌ PIPELINE ERROR: {type(e).__name__}: {e}")
        return {
            "answer": f"Lỗi pipeline: {str(e)}",
            "query": user_query,
            "refined_query": None,
            "doc_count": 0
        }


def run_tests():
    """Run test queries."""
    print("\n" + "="*60)
    print("🧪 RUNNING TESTS")
    print("="*60)
    
    # Load vector database
    try:
        vector_db = load_vector_db()
        print("Vector DB loaded successfully")
    except Exception as e:
        print(f"❌ Error loading vector DB: {e}")
        print("Start processing documents and creating vector DB...")
        vector_db, _ = process_and_index_documents("file.pdf")
        print("Vector DB loaded successfully")
    
    # Test cases
    test_cases = [
        {
            "name": "TEST 1: CHITCHAT",
            "query": "Model hiện tại là gì vậy?"
        },
        {
            "name": "TEST 2: RAG - INTERNAL DOCS",
            "query": "Vai trò của khuếch tán lỗi là gì?"
        },
        {
            "name": "TEST 3: WEB SEARCH",
            "query": "Giá vàng hiện tại là bao nhiêu?"
        },
        {
            "name": "TEST 4: WEB SEARCH",
            "query": "Giá Bitcoin hiện tại là bao nhiêu?"
        },
    ]
    
    for test in test_cases:
        print(test["name"])
        
        result = main_rag_pipeline(
            test["query"],
            vector_db
        )
        
        print(f"\n📝 ANSWER:")
        print(f"  {result['answer'][:200]}...")
        print()


if __name__ == "__main__":
    run_tests()






