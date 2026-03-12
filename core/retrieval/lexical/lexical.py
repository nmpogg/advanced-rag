import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from pyvi import ViTokenizer

from retrieval.base import BaseRetriever

class BM25Retriever(BaseRetriever):
    def __init__(self, corpus_path: str):
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            self.corpus_data = json.load(f)
            
        self.cids = [item['cid'] for item in self.corpus_data]
        self.contents = [item['content'] for item in self.corpus_data]
        self.metadata = [item.get('metadata', {}) for item in self.corpus_data]
        
        tokenized_corpus = [self._tokenize(doc) for doc in self.contents]
        
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _tokenize(self, text: str) -> List[str]:

        tokenized_string = ViTokenizer.tokenize(text)
        return tokenized_string.lower().split()

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:

        tokenized_query = self._tokenize(query)
        
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # np.argsort trả về index từ thấp đến cao, nên cần [::-1] để đảo ngược
        top_k_indices = np.argsort(doc_scores)[::-1][:top_k]
        
        results = []
        for idx in top_k_indices:
            score = doc_scores[idx]
            if score <= 0.0:
                continue
                
            results.append({
                "cid": self.cids[idx],
                "score": float(score),
                "text": self.contents[idx],
                "metadata": self.metadata[idx]
            })
            
        return results

if __name__ == "__main__":
    retriever = BM25Retriever(corpus_path="evaluation/corpus.json")
    
    test_query = "xe chở hàng để rơi hàng hóa xuống đường có sao không?"
    print(f"\nQuery: '{test_query}'")
    
    results = retriever.search(test_query, top_k=3)
    
    
    for rank, res in enumerate(results, start=1):
        print(f"\nTop {rank} - CID: {res['cid']} - Score: {res['score']:.4f}")
        print(f"Metadata: {res['metadata']}")
        print(f"Trích đoạn: {res['text'][:150]}...")