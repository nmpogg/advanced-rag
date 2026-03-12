# File: ensemble/rrf.py
from typing import List, Dict, Any

def reciprocal_rank_fusion(
    list_of_results: List[List[Dict[str, Any]]], 
    k: int = 60, 
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Kết hợp kết quả từ nhiều retriever khác nhau bằng thuật toán RRF.
    
    Args:
        list_of_results: Danh sách chứa các list kết quả từ các retriever.
        k: Hằng số làm mượt (thường = 60 theo paper gốc).
        top_k: Số lượng kết quả trả về cuối cùng.
    """
    rrf_scores: Dict[str, float] = {}
    doc_store: Dict[str, Dict[str, Any]] = {}

    for results in list_of_results:
        # rank bắt đầu từ 1
        for rank, doc in enumerate(results, start=1):
            doc_id = doc["id"]
            
            # Nếu doc này chưa có trong bộ nhớ, thêm vào
            if doc_id not in doc_store:
                doc_store[doc_id] = {
                    "id": doc_id,
                    "content": doc["content"],
                    "metadata": doc["metadata"]
                }
                rrf_scores[doc_id] = 0.0
            
            # Cộng dồn điểm RRF cho document này
            rrf_scores[doc_id] += 1.0 / (k + rank)

    #sắp xếp các document theo điểm RRF giảm dần
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    #format
    final_results = []
    for doc_id, score in sorted_docs[:top_k]:
        final_doc = doc_store[doc_id].copy()
        final_doc["score"] = score
        final_results.append(final_doc)
        
    return final_results