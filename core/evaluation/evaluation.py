import numpy as np
import pandas as pd

from retrieval.search_engine import keyword_search, semantic_search, ensemble_results, rerank_documents

def hit_k(true_cids: list, retrieved_cids: list, k: int) -> int:
    return

def precision_k(true_cids: list, retrieved_cids: list, k: int) -> float:
    hits = hit_k(true_cids, retrieved_cids, k)
    return hits / k

def recall_k(true_cids: list, retrieved_cids: list, k: int) -> float:
    hits = hit_k(true_cids, retrieved_cids, k)
    return hits / len(true_cids) if true_cids else 0.0

def f1_k(precision, recall) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * (precision) / (precision + recall)

def mrr_k(true_cids: list, retrieved_cids: list, k: int) -> float:
    for rank, cid in enumerate(retrieved_cids[:k], start=1):
        if cid in true_cids:
            return 1 / rank
    return 0.0

def map_k(true_cids: list, retrieved_cids: list, k: int) -> float:
    average_precision = 0.0
    hits = 0
    for rank, cid in enumerate(retrieved_cids[:k], start=1):
        if cid in true_cids:
            hits += 1
            average_precision += hits / rank
    return average_precision / len(true_cids) if true_cids else 0.0

def ndcg_k(true_cids: list, retrieved_cids: list, k: int) -> float:
    dcg = 0.0
    for rank, cid in enumerate(retrieved_cids[:k], start=1):
        if cid in true_cids:
            dcg += 1 / np.log2(rank + 1)
    
    ideal_dcg = sum(1 / np.log2(i + 1) for i in range(1, min(len(true_cids), k) + 1))
    
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def eval_keyword_search(true_cids: list, query: str, json_path: str, top_k: int) -> dict:
    retrieved_cids = keyword_search(query, json_path, top_k)
    
    precision = precision_k(true_cids, retrieved_cids, top_k)
    recall = recall_k(true_cids, retrieved_cids, top_k)
    f1 = f1_k(precision, recall)
    mrr = mrr_k(true_cids, retrieved_cids, top_k)
    map_score = map_k(true_cids, retrieved_cids, top_k)
    ndcg = ndcg_k(true_cids, retrieved_cids, top_k)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mrr": mrr,
        "map": map_score,
        "ndcg": ndcg
    }


if __name__ == "__main__":
    df = pd.read_excel("evaluation/data.xlsx")
    print(df.info())

