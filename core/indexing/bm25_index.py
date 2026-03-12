# File: indexing/build_bm25_index.py
import json
import os
import pickle
from rank_bm25 import BM25Okapi
from pyvi import ViTokenizer

def build_bm25_index(
    json_path: str = "data/processed/corpus.json",
    output_dir: str = "data/vector_store/bm25_index"
):
    """
    Đọc corpus, tách từ tiếng Việt, tạo BM25 index và lưu lại.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Không tìm thấy file {json_path}.")

    print(f"1. Đang tải dữ liệu từ {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"2. Đang tiến hành tách từ cho {len(data)} chunks...")
    # Tách từ cho toàn bộ nội dung. ViTokenizer sẽ biến "vi phạm giao thông" thành "vi_phạm giao_thông"
    tokenized_corpus = []
    for item in data:
        # Chuyển về chữ thường và tách từ
        text = str(item['content']).lower()
        tokenized_text = ViTokenizer.tokenize(text).split()
        tokenized_corpus.append(tokenized_text)

    print("3. Đang khởi tạo mô hình BM25...")
    bm25 = BM25Okapi(tokenized_corpus)

    print("4. Đang lưu Index và Document Store ra ổ cứng...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Lưu toàn bộ data (để mapping lại ID và content khi search) và model BM25
    index_data = {
        "documents": data, # Lưu lại nguyên bản gốc (cid, content, metadata)
        "bm25_model": bm25
    }
    
    output_path = os.path.join(output_dir, "bm25_model.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(index_data, f)
        
    print(f"-> Đã lưu BM25 Index thành công tại: {output_path}")

if __name__ == "__main__":
    build_bm25_index()