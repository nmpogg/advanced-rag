# File: reranking/cross_encoder_reranker.py
import os
import sys
from typing import List, Dict, Any

from sentence_transformers import CrossEncoder

from retrieval.semantic.semantic import DenseRetriever
from retrieval.lexical.lexical import BM25Retriever
from retrieval.expansion.expansion import PRFExpansionRetriever
from ensemble.ensemble import EnsembleManager

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = None):
        """
        Khởi tạo Cross-Encoder Reranker.
        
        Args:
            model_name: Tên mô hình Cross-Encoder trên HuggingFace.
            device: Thiết bị chạy ('cuda', 'mps', 'cpu'). Nếu là None, tự động chọn thiết bị tốt nhất.
        """
        print(f"Đang khởi tạo CrossEncoderReranker với mô hình: {model_name}...")
        
        self.model = CrossEncoder(model_name, device=device)
        
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Thực hiện (rerank) danh sách các văn bản đã được truy xuất.
        
        Args:
            query: Câu hỏi gốc của người dùng.
            documents: Danh sách các chunk luật - output từ ensemble.
            top_k: Số lượng kết quả tốt nhất muốn giữ lại sau khi rerank.
        """
        if not documents:
            return []
        
        pairs = [[query, doc['content']] for doc in documents]
        
        # 2. Chạy mô hình để chấm điểm mức độ tương đồng cho từng cặp.
        # Chuyển đổi điểm số về dạng sigmoid (0 -> 1) để dễ so sánh.
        scores = self.model.predict(pairs, convert_to_tensor=True, show_progress_bar=False)
        scores = scores.cpu().numpy()
        
        reranked_docs = []
        for i, doc in enumerate(documents):
            new_doc = doc.copy()
            # Điểm của Cross-Encoder phản ánh mức độ tương đồng thực tế
            new_doc['score'] = float(scores[i]) 
            reranked_docs.append(new_doc)
            
        #sắp xếp lại toàn bộ danh sách theo điểm số mới giảm dần
        reranked_docs = sorted(reranked_docs, key=lambda x: x['score'], reverse=True)
        
        return reranked_docs[:top_k]

if __name__ == "__main__":
    #test
    dense = DenseRetriever()
    bm25 = BM25Retriever()
    prf_bm25 = PRFExpansionRetriever(base_retriever=bm25, pseudo_k=3, top_terms=5)
    
    hybrid_retriever = EnsembleManager(retrievers=[dense, prf_bm25], rrf_k=60)
    
    reranker = CrossEncoderReranker()
    
    test_query = "đi xe máy không có bằng lái phạt bao nhiêu?"
    print(f"Câu hỏi test: '{test_query}'\n")
    
    print("1. Đang chạy Dense + BM25-PRF...")
    ensemble_results = hybrid_retriever.retrieve(query=test_query, top_k=15)
    
    print("\n=== KẾT QUẢ TỪ ENSEMBLE ===")
    for i, res in enumerate(ensemble_results, 1):
        print(f"Top {i} (RRF Score: {res['score']:.4f}) | ID: {res['id']}")
        
    print("2. Đang chạy Cross-Encoder Reranker...")
    final_results = reranker.rerank(query=test_query, documents=ensemble_results, top_k=5)
    
    print("\n=== KẾT QUẢ TỪ RERANKER ===")
    for i, res in enumerate(final_results, 1):
        print(f"Top {i} (New Score: {res['score']:.4f}) | ID: {res['id']}")
        print(f"Content: {res['content'][:150]}...\n")