# File: retrieval/expansion/query_expansion_retriever.py
from collections import Counter
from pyvi import ViTokenizer
from typing import List, Dict, Any

from retrieval.base import BaseRetriever
from retrieval.lexical.lexical import BM25Retriever

class PRFExpansionRetriever(BaseRetriever):
    def __init__(
        self, 
        base_retriever: BaseRetriever, 
        pseudo_k: int = 3, 
        top_terms: int = 4
    ):
        """
        Khởi tạo Pseudo-Relevance Feedback (PRF) Retriever.
        
        Args:
            base_retriever: BM25Retriever instance.
            pseudo_k: Số lượng tài liệu top đầu lấy ra ở lần truy xuất 1.
            top_terms: Số lượng từ khóa quan trọng muốn thêm vào query gốc.
        """
        self.base_retriever = base_retriever
        self.pseudo_k = pseudo_k
        self.top_terms = top_terms
        
        self.stopwords = set([
            "và", "của", "các", "có", "được", "cho", "trong", "về", "là", "không", 
            "với", "những", "một", "quy_định", "khoản", "điều", "chương", "luật", 
            "này", "theo", "tại", "người", "khi", "hoặc", "từ", "để", "thì", "do",
            "khoản", "điểm", "nghị_định", "phạt", "tiền"
        ])

    def extract_important_terms(self, documents: List[Dict], original_query: str) -> str:

        #Tokenize 
        query_tokens = set(ViTokenizer.tokenize(original_query.lower()).split())
        
        all_terms = []
        for doc in documents:
            text = doc['content'].lower()
            tokens = ViTokenizer.tokenize(text).split()
            
            valid_tokens = [
                t for t in tokens 
                if len(t) > 1 and not t.isnumeric() and t not in self.stopwords and t not in query_tokens
            ]
            all_terms.extend(valid_tokens)

        term_counts = Counter(all_terms)
        
        #lấy top các từ phổ biến nhất
        most_common = term_counts.most_common(self.top_terms)
        
        # pyvi sẽ nối từ ghép bằng dấu gạch dưới (vd: vi_phạm), ta chuyển lại thành dấu cách
        expanded_terms = [term[0].replace("_", " ") for term in most_common]
        
        return " ".join(expanded_terms)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        #get list pseudo_k docs
        initial_results = self.base_retriever.retrieve(query, top_k=self.pseudo_k)
        
        if not initial_results:
            return []

        #get important term
        expanded_keywords = self.extract_important_terms(initial_results, query)
        
        expanded_query = f"{query} {expanded_keywords}"
        
        # print("\n--- [PRF Query Expansion] ---")
        # print(f"1. Query gốc       : '{query}'")
        # print(f"2. Từ khóa bóc tách: '{expanded_keywords}'")
        # print(f"3. Query mở rộng   : '{expanded_query}'")
        # print("-----------------------------\n")

        #retrieve again with expanded query
        final_results = self.base_retriever.retrieve(expanded_query, top_k=top_k)
        
        return final_results

if __name__ == "__main__":
    #test
    
    bm25_base = BM25Retriever()

    prf_retriever = PRFExpansionRetriever(
        base_retriever=bm25_base,
        pseudo_k=3,  # Lấy 3 doc đầu để phân tích từ vựng
        top_terms=4  # Rút ra 4 từ khóa quan trọng nhất
    )
    
    test_query = "đi xe máy không có bằng lái phạt bao nhiêu?"
    results = prf_retriever.retrieve(query=test_query, top_k=3)
    
    for i, res in enumerate(results, 1):
        print(f"Top {i} (Score: {res['score']:.4f}) | ID: {res['id']}")
        print(f"Content: {res['content'][:150]}...\n")