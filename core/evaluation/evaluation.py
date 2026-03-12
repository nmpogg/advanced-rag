# File: evaluation/evaluate_retrieval.py
import pandas as pd
import numpy as np
import math
import time
import os
import sys
from tqdm import tqdm
from typing import List, Set, Dict

from retrieval.lexical.lexical import BM25Retriever
from retrieval.semantic.semantic import DenseRetriever
from retrieval.expansion.expansion import PRFExpansionRetriever
from ensemble.ensemble import EnsembleManager
from reranking.rerank import CrossEncoderReranker

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def calculate_metrics(retrieved_cids: List[str], ground_truth_cids: Set[str], k: int) -> Dict[str, float]:
    retrieved_k = retrieved_cids[:k]
    
    hits = len(set(retrieved_k) & ground_truth_cids)
    hit_rate = 1.0 if hits > 0 else 0.0
    
    precision = hits / k if k > 0 else 0.0
    recall = hits / len(ground_truth_cids) if len(ground_truth_cids) > 0 else 0.0
    
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    mrr = 0.0
    for rank, cid in enumerate(retrieved_k, 1):
        if cid in ground_truth_cids:
            mrr = 1.0 / rank
            break
    
    ap = 0.0
    hits_so_far = 0
    for i, cid in enumerate(retrieved_k, 1):
        if cid in ground_truth_cids:
            hits_so_far += 1
            ap += hits_so_far / i
    
    map_score = ap / min(k, len(ground_truth_cids)) if len(ground_truth_cids) > 0 else 0.0
    
    dcg = 0.0
    for i, cid in enumerate(retrieved_k, 1):
        if cid in ground_truth_cids:
            dcg += 1.0 / math.log2(i + 1)
            
    idcg = 0.0
    for i in range(1, min(k, len(ground_truth_cids)) + 1):
        idcg += 1.0 / math.log2(i + 1)
        
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    return {
        f"HitRate@{k}": hit_rate,
        f"Precision@{k}": precision,
        f"Recall@{k}": recall,
        f"F1@{k}": f1,
        f"MAP@{k}": map_score,
        f"MRR@{k}": mrr,
        f"NDCG@{k}": ndcg
    }


def evaluate_technique(name: str, retriever_func, df_val: pd.DataFrame, k: int = 5) -> dict:
    print(f"Đang đánh giá kỹ thuật: {name}")
    
    total_metrics = {
        f"HitRate@{k}": 0.0, f"Precision@{k}": 0.0, f"Recall@{k}": 0.0,
        f"F1@{k}": 0.0, f"MAP@{k}": 0.0, f"MRR@{k}": 0.0, f"NDCG@{k}": 0.0
    }
    
    start_time = time.time()
    
    for _, row in tqdm(df_val.iterrows(), total=len(df_val), desc=f"Evaluating {name}"):
        query = str(row['question'])

        ground_truths = set([cid.strip() for cid in str(row['cid']).split(',') if cid.strip()])
        

        retrieved_docs = retriever_func(query, k)
        retrieved_ids = [str(doc['id']) for doc in retrieved_docs]
        
        metrics = calculate_metrics(retrieved_ids, ground_truths, k)
        
        for key in total_metrics:
            total_metrics[key] += metrics[key]
            
    num_queries = len(df_val)
    for key in total_metrics:
        total_metrics[key] /= num_queries
        
    execution_time = time.time() - start_time
    
    result_row = {"Technique": name, "Time(s)": round(execution_time, 2)}
    result_row.update({k: round(v, 4) for k, v in total_metrics.items()})
    
    return result_row


if __name__ == "__main__":
    TOP_K = 7
    VAL_FILE = "data/benchmark/val.csv"
    OUTPUT_FILE = f"evaluation/results@{TOP_K}.csv"
    
    if not os.path.exists(VAL_FILE):
        print(f"Không tìm thấy file {VAL_FILE}")
        exit()
        
    df_val = pd.read_csv(VAL_FILE)
    print(f"Đã tải {len(df_val)} câu hỏi từ {VAL_FILE}.")

    print("\nĐang khởi tạo các mô hình truy xuất...")
    bm25 = BM25Retriever()
    dense = DenseRetriever()

    prf_bm25 = PRFExpansionRetriever(base_retriever=bm25, pseudo_k=3, top_terms=4)
    hybrid = EnsembleManager(retrievers=[dense, prf_bm25], rrf_k=60)
    reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

    results = []

    
    #BM25
    res_bm25 = evaluate_technique("BM25", lambda q, k: bm25.retrieve(q, k), df_val, TOP_K)
    results.append(res_bm25)
    
    #Semantic
    res_dense = evaluate_technique("Semantic", lambda q, k: dense.retrieve(q, k), df_val, TOP_K)
    results.append(res_dense)
    
    #BM25 + PRF Expansion
    res_prf = evaluate_technique("BM25 + Expansion", lambda q, k: prf_bm25.retrieve(q, k), df_val, TOP_K)
    results.append(res_prf)
    
    #Ensemble = Dense + BM25 PRF
    res_hybrid = evaluate_technique("Ensemble = Dense + BM25 PRF", lambda q, k: hybrid.retrieve(q, k), df_val, TOP_K)
    results.append(res_hybrid)
    
    #Reranker
    def rerank_wrapper(q, k):
        hybrid_docs = hybrid.retrieve(q, top_k=15)
        return reranker.rerank(q, documents=hybrid_docs, top_k=k)
        
    res_rerank = evaluate_technique("Reranker", rerank_wrapper, df_val, TOP_K)
    results.append(res_rerank)

    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    print("Kết quả đánh giá các kỹ thuật truy xuất:")
    print(df_results.to_markdown(index=False))
    print(f"Đã lưu file kết quả chi tiết tại: {OUTPUT_FILE}")