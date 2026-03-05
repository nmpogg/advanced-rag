import json
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict, Any

from retrieval.base import BaseRetriever

class SemanticRetriever(BaseRetriever):
    def __init__(self, corpus_path: str, persist_directory: str = "./chroma_db", collection_name: str = "legal_corpus"):

        self.client = chromadb.PersistentClient(path=persist_directory)
        
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="anhtld/VN-Law-Embedding" 
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"} 
        )
        
        if self.collection.count() == 0:
            print(f"Collection '{collection_name}' trống. Đang tiến hành embedding dữ liệu...")
            self._build_index(corpus_path)
        else:
            print(f"Tải thành công Vector DB '{collection_name}' với {self.collection.count()} tài liệu.")

    def _build_index(self, corpus_path: str):

        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)
            
        cids = [str(item['cid']) for item in corpus_data] 
        contents = [item['content'] for item in corpus_data]
        metadatas = [item.get('metadata', {}) for item in corpus_data]
        
        batch_size = 200 
        for i in range(0, len(cids), batch_size):
            self.collection.add(
                ids=cids[i:i+batch_size],
                documents=contents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )
            print(f"Đã embed {min(i+batch_size, len(cids))}/{len(cids)} tài liệu.")
        print("Đã hoàn tất embedding và lưu trữ.")

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:

        chroma_results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        results = []
        
        if not chroma_results['ids'] or not chroma_results['ids'][0]:
            return results
            
        for i in range(len(chroma_results['ids'][0])):

            distance = chroma_results['distances'][0][i]
            score = 1.0 - distance 
            
            results.append({
                "cid": int(chroma_results['ids'][0][i]), 
                "score": float(score),
                "text": chroma_results['documents'][0][i],
                "metadata": chroma_results['metadatas'][0][i]
            })
            
        return results


if __name__ == "__main__":
    retriever = SemanticRetriever(corpus_path="evaluation/corpus.csv")
    
    test_query = "xe chở hàng để rơi hàng hóa xuống đường có sao không?"
    print(f"\nQuery: '{test_query}'")
    
    results = retriever.search(test_query, top_k=3)
    
    for rank, res in enumerate(results, start=1):
        print(f"\nTop {rank} - CID: {res['cid']} - Cosine Similarity: {res['score']:.4f}")
        print(f"Metadata: {res['metadata']}")
        print(f"Trích đoạn: {res['text'][:150]}...")