# File: ensemble/ensemble_manager.py
from typing import List, Dict, Any

from ensemble.rrf import reciprocal_rank_fusion

from retrieval.base import BaseRetriever
from retrieval.semantic.semantic import DenseRetriever
from retrieval.lexical.lexical import BM25Retriever
from retrieval.expansion.expansion import PRFExpansionRetriever

class EnsembleManager(BaseRetriever):
    def __init__(self, retrievers: List[BaseRetriever], rrf_k: int = 60):
        """
        Khởi tạo Ensemble Manager với nhiều retriever.
        
        Args:
            retrievers: Danh sách BaseRetriever.
            rrf_k: Hằng số K cho thuật toán RRF.
        """
        self.retrievers = retrievers
        self.rrf_k = rrf_k
        print(f"Đã khởi tạo EnsembleManager với {len(self.retrievers)} retrievers.")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:

        list_of_results = []
        
        for retriever in self.retrievers:

            results = retriever.retrieve(query, top_k * 2)
            list_of_results.append(results)
        
        final_fused_results = reciprocal_rank_fusion(
            list_of_results=list_of_results, 
            k=self.rrf_k, 
            top_k=top_k
        )
        
        return final_fused_results

if __name__ == "__main__":
    #TEST
    
    dense_retriever = DenseRetriever()
    
    bm25_base = BM25Retriever()
    lexical_prf_retriever = PRFExpansionRetriever(
        base_retriever=bm25_base, 
        pseudo_k=3, top_terms=4
    )
    
    hybrid_retriever = EnsembleManager(
        retrievers=[dense_retriever, lexical_prf_retriever]
    )
    
    test_query = "đi xe máy không có bằng lái phạt bao nhiêu?"
    results = hybrid_retriever.retrieve(query=test_query, top_k=3)
    
    print("\n=== KẾT QUẢ HYBRID SEARCH (RRF) ===")
    for i, res in enumerate(results, 1):
        print(f"Top {i} (RRF Score: {res['score']:.4f}) | ID: {res['id']}")
        print(f"Content: {res['content'][:150]}...\n")