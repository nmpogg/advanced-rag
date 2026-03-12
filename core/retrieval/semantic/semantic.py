# File: retrieval/semantic/dense_retriever.py
import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from retrieval.base import BaseRetriever

class DenseRetriever(BaseRetriever):
    def __init__(
        self, 
        persist_directory: str = "data/vector_store/chroma_db", 
        collection_name: str = "legal_corpus",
        model_name: str = "keepitreal/vietnamese-sbert"
    ):
        """
        Khởi tạo Dense Retriever kết nối với ChromaDB.
        """
        print(f"Đang khởi tạo DenseRetriever với model: {model_name}...")
        
        self.model = SentenceTransformer(model_name)
        
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Đã kết nối ChromaDB tại '{persist_directory}', collection '{collection_name}'.")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        #encode
        query_embedding = self.model.encode(query, normalize_embeddings=True).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            # include=["documents", "metadatas", "distances"] # Mặc định trả về những trường này
        )
        
        #Format
        formatted_results = []
        
        # ChromaDB trả về list của list (vì hỗ trợ batch query), lấy phần tử [0]
        if results['ids'] and len(results['ids'][0]) > 0:
            ids = results['ids'][0]
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            
            for i in range(len(ids)):
                similarity_score = 1.0 - distances[i] 
                
                formatted_results.append({
                    "id": ids[i],
                    "content": documents[i],
                    "metadata": metadatas[i],
                    "score": similarity_score
                })
                
        return formatted_results

if __name__ == "__main__":
    # Test
    retriever = DenseRetriever()
    
    test_query = "Quy định về xử phạt hành chính khi vi phạm giao thông?"
    print(f"\nQuery: '{test_query}'")
    
    results = retriever.retrieve(query=test_query, top_k=3)
    
    if not results:
        print("Không tìm thấy kết quả. Bạn đã nhồi (ingest) dữ liệu vào ChromaDB chưa?")
    else:
        for i, res in enumerate(results, 1):
            print(f"\nTop {i} (Score: {res['score']:.4f})")
            print(f"ID: {res['id']}")
            print(f"Metadata: {res['metadata']}")
            print(f"Content snippet: {res['content'][:150]}...")