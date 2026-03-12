advanced_rag_legal/
│
├── config/
│
├── data/
│   ├── raw/
│   ├── processed/
|   |   ├── corpus.csv
|   |   └── corpus.np
│   |
|   └── benchmark/
│       ├── queries.csv
│       └── qrels.csv
│
├── indexing/
│   ├── chunking.py
│   ├── build_bm25_index.py
│   └── build_vector_index.py
│
├── retrieval/
│   ├── base_retriever.py
│   │
│   ├── lexical/
│   │   └── bm25_retriever.py
│   │
│   ├── semantic/
│   │   └── dense_retriever.py
│   │
│   ├── expansion/
│   │   └── query_expansion_retriever.py
│   │
│   └── retrieval_manager.py
│
├── ensemble/
│   ├── rrf.py
│   └── ensemble_manager.py
│
├── reranking/
│   └── cross_encoder_reranker.py
│
├── rag/
│   ├── is_chitchat.py
│   ├── query_reflection.py
│   ├── context_builder.py
│   └── rag_pipeline.py
│
├── llm/
│   ├── llm_client.py
│   └── prompts.py
│
├── memory/
│   └── chat_history.py
│
├── evaluation/
│   ├── metrics.py
│   ├── evaluate_retrieval.py
│   └── evaluate_rag.py
│
├── experiments/
│   ├── run_retrieval_experiment.py
│   └── run_rag_experiment.py
│
├── utils/
│
├── ui/
│   ├── streamlit_app.py
│   
│
└── main.py

---

## 📚 GIẢI THÍCH CHI TIẾT CẤU TRÚC UI (STREAMLIT)

### **Thư mục `ui/` - Giao diện người dùng**

#### **1. `streamlit_app.py` - Điểm vào chính của ứng dụng**
- **Chức năng**: Entry point của toàn bộ ứng dụng Streamlit
- **Các việc chính**:
  - Cấu hình trang chính (tiêu đề, icon, layout)
  - Import và kết nối các trang con từ thư mục `pages/`
  - Thiết lập sidebar với navigation menu
  - Quản lý theme (sáng/tối)
  - Khởi tạo session state toàn cục

#### **2. `config/ui_settings.py` - Cấu hình giao diện**
- **Chức năng**: Lưu trữ tất cả các cài đặt UI
- **Các việc chính**:
  - Màu sắc, font, kích thước chữ
  - Đường dẫn API backend
  - Timeout settings, page config
  - Các thông số hiển thị mặc định

---

### **Thư mục `pages/` - Các trang ứng dụng**

#### **1. `1_chat.py` - Trang chat chính**
- **Chức năng**: Giao diện chat tương tác với RAG
- **Các việc chính**:
  - Hiển thị lịch sử cuộc hội thoại
  - Input text box để người dùng nhập câu hỏi
  - Gửi câu hỏi tới pipeline RAG và nhận câu trả lời
  - Hiển thị nguồn tài liệu và độ tin cậy
  - Giữ lịch sử chat trong session state
  - Nút xóa lịch sử chat

#### **2. `2_document_management.py` - Quản lý tài liệu**
- **Chức năng**: Upload, xoá, quản lý tài liệu trong vector database
- **Các việc chính**:
  - Tạo file uploader cho PDF, TXT, DOCX
  - Xử lý tài liệu (chunking, embedding)
  - Hiển thị danh sách tài liệu đã upload
  - Thống kê: số lượng chunk, dung lượng, ngày upload
  - Nút xoá/update tài liệu
  - Hiển thị tiến độ indexing

#### **3. `3_retrieval_debug.py` - Debug retrieval**
- **Chức năng**: Tool debug để kiểm tra đầu ra từng bước retrieval
- **Các việc chính**:
  - Input test query
  - Hiển thị kết quả semantic search (vector similarity)
  - Hiển thị kết quả keyword search (BM25)
  - So sánh điểm số từ 2 phương pháp
  - Hiển thị tài liệu được rerank (từ cross-encoder)
  - Xem kết quả ensemble (kết hợp)

#### **4. `4_analytics.py` - Phân tích thống kê**
- **Chức năng**: Dashboard để theo dõi hiệu suất hệ thống
- **Các việc chính**:
  - Biểu đồ số câu hỏi được hỏi theo thời gian
  - Thống kê loại query: chitchat vs RAG
  - Tỷ lệ sử dụng nguồn: internal docs / web search
  - Thời gian phản hồi trung bình
  - Top keywords được tìm kiếm
  - Feedback từ người dùng (helpful/not helpful)

---

### **Thư mục `components/` - Các component tái sử dụng**

#### **1. `chat_interface.py` - Component chat**
- **Chức năng**: Component hiển thị giao diện chat
- **Các việc chính**:
  - Hàm hiển thị tin nhắn user (bên phải, màu xanh)
  - Hàm hiển thị tin nhắn assistant (bên trái, màu xám)
  - Hàm hiển thị loading state với spinner
  - Hàm hiển thị error messages
  - Format thời gian cho mỗi tin nhắn
  - Hỗ trợ markdown formatting trong tin nhắn

#### **2. `document_uploader.py` - Component upload**
- **Chức năng**: Component xử lý upload file
- **Các việc chính**:
  - Hiển thị file uploader widget
  - Validate format file (PDF, TXT, DOCX)
  - Validate file size
  - Hiển thị tiến độ upload
  - Xử lý lỗi upload
  - Trả về file object cho page xử lý tiếp

#### **3. `result_displayor.py` - Component hiển thị kết quả**
- **Chức năng**: Component để hiển thị kết quả tìm kiếm
- **Các việc chính**:
  - Hiển thị câu trả lời chính
  - Hiển thị danh sách tài liệu liên quan (expander)
  - Hiển thị confidence score cho mỗi tài liệu
  - Nút copy câu trả lời
  - Nút feedback (helpful/not helpful)
  - Hiển thị thời gian xử lý

#### **4. `stats_dashboard.py` - Component dashboard**
- **Chức năng**: Component hiển thị thống kê
- **Các việc chính**:
  - Các metric boxes (KPI cards)
  - Biểu đồ line/bar/pie charts
  - Hàm format số liệu cho hiển thị
  - Càng màu hoạt động (green/red) dựa trên metric

---

### **Thư mục `utils/` - Các hàm tiện ích**

#### **1. `session_manager.py` - Quản lý phiên làm việc**
- **Chức năng**: Quản lý session state của Streamlit
- **Các việc chính**:
  - Khởi tạo session variables (chat history, user settings)
  - Lưu/tải chat history từ file
  - Quản lý cache (vector DB, models)
  - Kiểm tra xung đột concurrent requests
  - Cleanup khi session kết thúc

#### **2. `formatting.py` - Hàm format dữ liệu**
- **Chức năng**: Các hàm giúp format dữ liệu trước hiển thị
- **Các việc chính**:
  - Format độ tin cậy theo %
  - Format thời gian thành readable string
  - Truncate text dài
  - Highlight keywords trong text
  - Format token count
  - Escape HTML/Markdown đặc biệt

---

### **Thư mục `assets/` - Tài nguyên tĩnh**

#### **1. `styles.css` - CSS tùy chỉnh**
- **Chức năng**: Custom CSS để styling giao diện
- **Các việc chính**:
  - Custom theme colors
  - Styling cho chat bubbles
  - Responsive design
  - Hover effects
  - Animation

#### **2. `logo.png` - Logo ứng dụng**
- **Chức năng**: Hình ảnh logo hiển thị trên sidebar/header
- **Các việc chính**:
  - Logo app tại sidebar (32x32px)
  - Favicon cho trang web

---

## 🔄 Luồng dữ liệu Streamlit UI - Core RAG

```
┌─────────────────────────────────────────────────────┐
│           Streamlit Frontend (UI Layer)              │
├──────────────┬──────────────┬──────────────┬─────────┤
│  Chat Page   │Doc Management│Debug Tools   │Analytics│
├──────────────┴──────────────┴──────────────┴─────────┤
│           Reusable Components (UI Components)        │
├─────────────────────────────────────────────────────┤
│       Session Manager + Utilities                    │
├─────────────────────────────────────────────────────┤
│  🔗 API/Function Calls to Core RAG Pipeline         │
├─────────────────────────────────────────────────────┤
│  core/main.py - main_rag_pipeline()                 │
│  core/retrieval/ - Search & Rerank                  │
│  core/generation/ - Answer Generation               │
│  core/ingestion/ - Document Loading                 │
└─────────────────────────────────────────────────────┘
```

---

## 🚀 Cách chạy ứng dụng

```bash
# Từ terminal, trong folder dự án
streamlit run ui/streamlit_app.py
```

Ứng dụng sẽ mở tại `http://localhost:8501`