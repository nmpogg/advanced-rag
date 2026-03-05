import pandas as pd
import numpy as np
from typing import List, Dict
from tqdm import tqdm

from retrieval.base import BaseRetriever
from retrieval.lexical import BM25Retriever
from retrieval.semantic import SemanticRetriever
from retrieval.ensemble import EnsembleRetriever

def hit_k(true_cids: List[int], retrieved_cids: List[int], k: int) -> int:
    return len(set(true_cids) & set(retrieved_cids[:k]))

def hit_rate_k(true_cids: List[int], retrieved_cids: List[int], k: int) -> float:
    return 1.0 if hit_k(true_cids, retrieved_cids, k) > 0 else 0.0

def precision_k(true_cids: List[int], retrieved_cids: List[int], k: int) -> float:
    return hit_k(true_cids, retrieved_cids, k) / k

def recall_k(true_cids: List[int], retrieved_cids: List[int], k: int) -> float:
    if not true_cids: return 0.0
    return hit_k(true_cids, retrieved_cids, k) / len(true_cids)

def f1_k(true_cids: List[int], retrieved_cids: List[int], k: int) -> float:
    p = precision_k(true_cids, retrieved_cids, k)
    r = recall_k(true_cids, retrieved_cids, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)

def ap_k(true_cids: List[int], retrieved_cids: List[int], k: int) -> float:
    if not true_cids: return 0.0
    true_set = set(true_cids)
    relevant_hits = 0
    sum_precisions = 0.0
    
    for i, cid in enumerate(retrieved_cids[:k]):
        if cid in true_set:
            relevant_hits += 1
            sum_precisions += relevant_hits / (i + 1.0)
            
    return sum_precisions / min(len(true_cids), k)

def mrr_k(true_cids: List[int], retrieved_cids: List[int], k: int) -> float:
    true_set = set(true_cids)
    for rank, cid in enumerate(retrieved_cids[:k], start=1):
        if cid in true_set:
            return 1.0 / rank
    return 0.0

def ndcg_k(true_cids: List[int], retrieved_cids: List[int], k: int) -> float:
    if not true_cids: return 0.0
    true_set = set(true_cids)
    dcg = sum(1.0 / np.log2(rank + 1) for rank, cid in enumerate(retrieved_cids[:k], start=1) if cid in true_set)
    ideal_dcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(true_cids), k) + 1))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0



def evaluate_retriever(retriever: BaseRetriever, test_file_path: str, k_values: List[int] = [3, 5, 10]) -> Dict[str, Dict[str, float]]:
 
    print(f"\nĐang đọc dữ liệu test từ: {test_file_path}")
    # df = pd.read_excel(test_file_path)
    df = pd.read_csv(test_file_path)
    
    queries = df['question'].tolist()
    
    true_cids_list = []
    for val in df['cid']:
        if pd.isna(val):
            true_cids_list.append([])
        else:
            # Tách chuỗi bằng dấu phẩy và ép kiểu về int
            cids = [int(str(x).strip()) for x in str(val).split(',')]
            true_cids_list.append(cids)

    n_queries = len(queries)
    max_k = max(k_values)
    
    results = {k: {
        "hit_rate": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, 
        "map": 0.0,"mrr": 0.0, "ndcg": 0.0
    } for k in k_values}
    
    print(f"Bắt đầu đánh giá hệ thống ({n_queries} câu queries)...")
    
    for q, t_cids in tqdm(zip(queries, true_cids_list), total=n_queries, desc="Evaluating"):
        if not t_cids:
            continue
            
        retrieved_docs = retriever.search(q, top_k=max_k)
        retrieved_cids = [doc['cid'] for doc in retrieved_docs]
        
        for k in k_values:
            results[k]["hit_rate"] += hit_rate_k(t_cids, retrieved_cids, k)
            results[k]["precision"] += precision_k(t_cids, retrieved_cids, k)
            results[k]["recall"] += recall_k(t_cids, retrieved_cids, k)
            results[k]["f1"] += f1_k(t_cids, retrieved_cids, k)
            results[k]["map"] += ap_k(t_cids, retrieved_cids, k)
            results[k]["mrr"] += mrr_k(t_cids, retrieved_cids, k)
            results[k]["ndcg"] += ndcg_k(t_cids, retrieved_cids, k)
            
    for k in k_values:
        for metric in results[k]:
            results[k][metric] = results[k][metric] / n_queries
            
    return results

if __name__ == "__main__":
    print("Bắt đầu đánh giá...")
    bm25 = BM25Retriever("evaluation/corpus.json")
    semantic = SemanticRetriever("evaluation/corpus.json")
    hybrid = EnsembleRetriever([bm25, semantic], pool_size=50)
    
    models = {
        "BM25 (Lexical)": bm25,
        "Vector (Semantic)": semantic,
        "Hybrid (RRF)": hybrid
    }
    
    k_eval = [3, 5, 7]
    
    benchmark_results = {}
    for name, model in models.items():
        print(f"\n[ Đang chạy test cho: {name} ]")
        benchmark_results[name] = evaluate_retriever(model, "evaluation/ok.csv", k_values=k_eval)
        
    for name, res in benchmark_results.items():
        print(f"\n{name}:")
        for k in k_eval:
            m = res[k]

            print(f"   @Top-{k} | Hit Rate: {m['hit_rate']:.4f} | Precision: {m['precision']:.4f} | Recall: {m['recall']:.4f} | F1: {m['f1']:.4f} | MAP: {m['map']:.4f} | MRR: {m['mrr']:.4f} | NDCG: {m['ndcg']:.4f}")