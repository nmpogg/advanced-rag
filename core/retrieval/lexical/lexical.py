# File: retrieval/lexical/bm25_retriever.py
import os
import pickle
import numpy as np
from pyvi import ViTokenizer
from typing import List, Dict, Any
from retrieval.base import BaseRetriever

class BM25Retriever(BaseRetriever):
    def __init__(self, index_path: str = "data/vector_store/bm25_index/bm25_model.pkl"):
        print(f"Đang khởi tạo BM25Retriever từ {index_path}...")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Không tìm thấy file index {index_path}")
            
        with open(index_path, 'rb') as f:
            index_data = pickle.load(f)
            
        self.documents = index_data["documents"]
        self.bm25_model = index_data["bm25_model"]
        print(f"Đã load BM25 thành công với {len(self.documents)} tài liệu.")

    def retrieve(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
  
        #preprocess
        query_lower = query.lower()
        tokenized_query = ViTokenizer.tokenize(query_lower).split()
        
        #get scores
        doc_scores = self.bm25_model.get_scores(tokenized_query)
        
        #top_k index có điểm cao nhất
        top_k_indices = np.argsort(doc_scores)[::-1][:top_k]
        
        #format theo base
        formatted_results = []
        for idx in top_k_indices:
            score = doc_scores[idx]
            
            if score <= 0:
                continue 
                
            doc_info = self.documents[idx]
            
            formatted_results.append({
                "id": str(doc_info["cid"]),
                "content": doc_info["content"],
                "metadata": doc_info["metadata"],
                "score": float(score)
            })
            
        return formatted_results

if __name__ == "__main__":
    #test
    retriever = BM25Retriever()
    
    test_query = "Quy định về xử phạt hành chính khi vi phạm giao thông?"
    print(f"\nQuery: '{test_query}'")
    
    results = retriever.retrieve(query=test_query, top_k=3)
    
    if not results:
        print("Không tìm thấy kết quả khớp từ khóa.")
    else:
        for i, res in enumerate(results, 1):
            print(f"\nTop {i} (BM25 Score: {res['score']:.4f})")
            print(f"ID: {res['id']}")
            print(f"Metadata: {res['metadata']}")
            print(f"Content snippet: {res['content'][:150]}...")