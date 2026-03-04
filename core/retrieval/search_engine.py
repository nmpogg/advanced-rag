import json
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL, TOP_K_INITIAL, TOP_K_FINAL, JSON_PATH
from ingestion import load_vector_db

reranker_model = None # for lazy loading

def get_reranker():
    global reranker_model

    if reranker_model is None:
        reranker_model = CrossEncoder(RERANKER_MODEL)
    
    return reranker_model

def semantic_search(query: str, vector_db, top_k: int) -> list:
    semantic_results = vector_db.similarity_search(query, k=top_k)
    semantic_docs = [doc.page_content for doc in semantic_results]
    print(f"Semantic search: {len(semantic_docs)} documents")

    return semantic_docs

# hàm này có thể tách ra làm nhiều kĩ thuật truy xuất, ở đây ví dụ với BM25
def keyword_search(query: str, json_path: str, top_k: int) -> list:
    
    #demo with json file
    with open(json_path, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    
    corpus = [item['content'] for item in chunks_data]
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    
    tokenized_query = query.lower().split()
    keyword_docs = bm25.get_top_n(tokenized_query, corpus, n=top_k)
    print(f"Keyword search: {len(keyword_docs)} documents")
    return keyword_docs


def tfidf(query: str, json_path: str, top_k: int) -> list:
    # Placeholder for TF-IDF search implementation
    # You can implement this using scikit-learn's TfidfVectorizer
    pass

def ensemble_results(semantic_docs: list, keyword_docs: list) -> list:

    ensembled_docs = list(set(semantic_docs + keyword_docs))
    print(f" Ensemble: {len(ensembled_docs)} documents")
    return ensembled_docs


def rerank_documents(query: str, documents: list, top_k: int) -> list:

    print("Reranking với Cross-Encoder...")
    reranker = get_reranker()
    pairs = [[query, doc] for doc in documents]
    scores = reranker.predict(pairs)
    
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    top_k_docs = [doc for doc, score in scored_docs[:top_k]]
    print(f"After reranking: {len(top_k_docs)} top documents")
    return top_k_docs


def hybrid_search_and_rerank(
    query: str, 
    vector_db, 
    json_path: str = JSON_PATH,
    top_k_initial: int = TOP_K_INITIAL,
    top_k_final: int = TOP_K_FINAL
) -> list:
    """
    Hybrid search combining semantic + keyword search with Cross-Encoder reranking.
    
    Args:
        query: Search query
        vector_db: ChromaDB vector database instance
        json_path: Path to JSON chunks file
        top_k_initial: Initial results before reranking
        top_k_final: Final top results
        
    Returns:
        list: Top K reranked documents
    """
    #semantic search
    semantic_docs = semantic_search(query, vector_db, top_k_initial)
    
    #keyword search
    keyword_docs = keyword_search(query, json_path, top_k_initial)
    
    #ensemble
    ensembled_docs = ensemble_results(semantic_docs, keyword_docs)
    
    #rerank
    top_k_docs = rerank_documents(query, ensembled_docs, top_k_final)
    
    return top_k_docs

