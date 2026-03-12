# File: indexing/build_vector_index.py
import json
import os
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np

def ingest_to_chroma(
    json_path: str = "data/processed/corpus.json",
    persist_directory: str = "data/vector_store/chroma_db",
    collection_name: str = "legal_corpus",
    model_name: str = "keepitreal/vietnamese-sbert",
    batch_size: int = 256
):
    """
    Đọc dữ liệu từ file JSON, tính toán embedding và lưu vào ChromaDB.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Không tìm thấy file {json_path}.")

    print(f"1. Đang tải dữ liệu từ {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data: List[Dict] = json.load(f)
        
    total_chunks = len(data)
    print(f"   -> Tổng số chunks cần xử lý: {total_chunks}")

    print(f"\n2. Đang khởi tạo model embedding: {model_name}...")
    model = SentenceTransformer(model_name)

    print(f"\n3. Đang kết nối ChromaDB tại '{persist_directory}'...")
    os.makedirs(persist_directory, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_directory)
    
    # Xóa collection cũ nếu tồn tại để tránh duplicate khi chạy lại script nhiều lần
    # try:
    #     client.delete_collection(name=collection_name)
    #     print(f"   -> Đã xóa collection cũ: '{collection_name}'")
    # except ValueError:
    #     pass # Collection chưa tồn tại, không sao cả

    # Tạo collection mới, set hnsw:space là cosine cho văn bản
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    print("\n4. Bắt đầu quá trình Ingestion (Chia batch để tránh tràn RAM)...")
    
    # Xử lý theo từng batch
    for i in range(0, total_chunks, batch_size):
        batch = data[i : i + batch_size]
        
        # Tách các thành phần ra thành list riêng biệt cho ChromaDB
        ids = [str(item['cid']) for item in batch] # ChromaDB yêu cầu ID dạng chuỗi (string)
        documents = [item['content'] for item in batch]
        
        # Đảm bảo metadata không chứa giá trị None hoặc list/dict phức tạp 
        # (ChromaDB chỉ nhận string, int, float, bool trong metadata)
        metadatas = []
        for item in batch:
            clean_meta = {}
            for k, v in item['metadata'].items():
                if v is not None:
                    clean_meta[k] = str(v) if not isinstance(v, (int, float, bool, str)) else v
            metadatas.append(clean_meta)

        # Encode văn bản thành vector
        # normalize_embeddings=True rất quan trọng để dùng chung với không gian 'cosine'
        print(f"   -> Đang encode batch {i//batch_size + 1} ({i} đến {min(i+batch_size, total_chunks)})...")
        embeddings = model.encode(documents, normalize_embeddings=True).tolist()

        # Thêm vào collection
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    print(f"\n5. Hoàn tất! Đã lưu {total_chunks} chunks vào ChromaDB.")
    print(f"   Bạn có thể kiểm tra thư mục: {persist_directory}")


def create_npy_from_json(
    json_path: str = "data/benchmark/corpus.json",
    npy_out_path: str = "data/benchmark/corpus_embeddings.npy",
    model_name: str = "keepitreal/vietnamese-sbert"
):
    """
    Đọc dữ liệu từ file corpus.json, mã hóa nội dung (content) thành vector
    và lưu xuống file .npy để phục vụ quá trình benchmark (evaluation).
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Không tìm thấy file {json_path}. Vui lòng kiểm tra lại.")

    print(f"1. Đang tải dữ liệu từ {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Trích xuất toàn bộ nội dung text để đưa vào mô hình
    texts = [item['content'] for item in data]
    print(f"   -> Tổng số chunks cần mã hóa: {len(texts)}")

    print(f"\n2. Đang khởi tạo model embedding: {model_name}...")
    model = SentenceTransformer(model_name)

    print("\n3. Đang mã hóa (Encoding) văn bản thành Vectors...")
    # normalize_embeddings=True rất quan trọng để tính Cosine Similarity siêu tốc
    # bằng phép nhân ma trận Dot Product trong file evaluate_retrieval.py
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    print(f"\n4. Kích thước ma trận Embeddings: {embeddings.shape}")

    # Đảm bảo thư mục lưu trữ tồn tại
    os.makedirs(os.path.dirname(npy_out_path), exist_ok=True)

    print(f"\n5. Đang lưu ma trận vào {npy_out_path}...")
    np.save(npy_out_path, embeddings)
    print("   -> Hoàn tất tạo file .npy cho Evaluation!")

if __name__ == "__main__":
    create_npy_from_json()
    # ingest_to_chroma()
