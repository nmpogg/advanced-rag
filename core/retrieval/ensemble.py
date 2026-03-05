from typing import List, Dict, Any

# Import class cha 
from retrieval.base import BaseRetriever
from retrieval.lexical import BM25Retriever
from retrieval.semantic import SemanticRetriever

class EnsembleRetriever(BaseRetriever):
    def __init__(self, retrievers: List[BaseRetriever], rrf_k: int = 60, pool_size: int = 15):
        """
        Khởi tạo hệ thống lai (Hybrid Search).
        
        Args:
            retrievers: Danh sách các đối tượng retriever (vd: [BM25Retriever, SemanticRetriever]).
            rrf_k: Hằng số làm mượt cho công thức RRF (mặc định 60 theo chuẩn nghiên cứu).
            pool_size: Số lượng tài liệu mỗi retriever cần lấy ra trước khi gộp. 
        """
        print(f"Đang khởi tạo Ensemble Retriever với {len(retrievers)} bộ truy xuất con...")
        self.retrievers = retrievers
        self.rrf_k = rrf_k
        self.pool_size = pool_size
        print("Khởi tạo Ensemble thành công...")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Thực hiện tìm kiếm trên tất cả các retrievers và gộp điểm bằng RRF.
        """
        rrf_scores: Dict[int, float] = {}  #tổng điểm RRF cho mỗi CID
        doc_store: Dict[int, Dict[str, Any]] = {}  #nội dung để trả về sau khi xếp hạng
        
        for retriever in self.retrievers:
            #pool_size: top k init
            results = retriever.search(query, top_k=self.pool_size)
            
            #tính RRF dựa trên rank
            for rank, res in enumerate(results, start=1):
                cid = res['cid']
                
                #nếu tài liệu lần đầu xuất hiện, lưu nội dung lại và khởi tạo điểm
                if cid not in rrf_scores:
                    rrf_scores[cid] = 0.0
                    doc_store[cid] = {
                        "text": res["text"],
                        "metadata": res["metadata"]
                    }
                
                #cộng dồn điểm RRF: 1 / (60 + thứ hạng)
                rrf_scores[cid] += 1.0 / (self.rrf_k + rank)
                
        #sắp xếp lại toàn bộ pool tài liệu theo tổng điểm RRF giảm dần
        sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        final_results = []
        for cid in sorted_cids[:top_k]:
            final_results.append({
                "cid": cid,
                "score": float(rrf_scores[cid]), # điểm lúc này là điểm RRF
                "text": doc_store[cid]["text"],
                "metadata": doc_store[cid]["metadata"]
            })
            
        return final_results

if __name__ == "__main__":

    bm25_retriever = BM25Retriever(corpus_path="evaluation/corpus.json")
    semantic_retriever = SemanticRetriever(corpus_path="evaluation/corpus.json")
    
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        pool_size=30
    )
    
    test_query = "hoạt động đường bộ bao gồm những gì?"
    print(f"\nQuery (Hybrid): '{test_query}'")
    
    #lấy ra top 3 tốt nhất
    results = hybrid_retriever.search(test_query, top_k=3)
    
    for rank, res in enumerate(results, start=1):
        print(f"\nTop {rank} - CID: {res['cid']} - RRF Score: {res['score']:.4f}")
        print(f"Metadata: {res['metadata']}")
        print(f"Trích đoạn: {res['text'][:100]}...")
            