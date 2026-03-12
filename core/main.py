# File: ui/streamlit_app.py
import streamlit as st
import os
import time
from dotenv import load_dotenv
import sys

# Đảm bảo Streamlit có thể tìm thấy các module trong thư mục gốc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from retrieval.semantic.semantic import DenseRetriever
from retrieval.lexical.lexical import BM25Retriever
from retrieval.expansion.expansion import PRFExpansionRetriever
from ensemble.ensemble import EnsembleManager
from reranking.rerank import CrossEncoderReranker
from llm.llm_client import LLMClient
from rag.context_builder import ContextBuilder
from rag.query_router import QueryRouter
from memory.chat_history import ChatHistory
from rag.rag_pipeline import LegalRAGPipeline

# Load biến môi trường (.env)
load_dotenv()

# ==========================================
# CẤU HÌNH TRANG WEB
# ==========================================
st.set_page_config(
    page_title="AI Tư vấn Pháp luật VN",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 1. KHỞI TẠO CACHE CHO CÁC MÔ HÌNH NẶNG (RETRIEVAL)
# ==========================================
@st.cache_resource(show_spinner="Đang tải các mô hình AI (Retrieval & Reranker) vào bộ nhớ...")
def load_heavy_models():
    """Load các mô hình tốn thời gian vào RAM 1 lần duy nhất."""
    dense = DenseRetriever()
    bm25 = BM25Retriever()
    prf_bm25 = PRFExpansionRetriever(base_retriever=bm25, pseudo_k=3, top_terms=5)
    
    # Sử dụng EnsembleManager bản mới
    hybrid = EnsembleManager(retrievers=[dense, prf_bm25], rrf_k=60)
    
    # Reranker model
    reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2") 
    return hybrid, reranker

hybrid_retriever, reranker = load_heavy_models()

# ==========================================
# 2. KHỞI TẠO SESSION STATE & PIPELINE
# ==========================================
def initialize_pipeline(model_choice):
    """Khởi tạo hoặc cập nhật lại Pipeline (Chỉ dùng Gemini)."""
    # Ép cứng provider là 'gemini'
    llm = LLMClient(provider="gemini", model_name=model_choice, temperature=0.0)
    ctx_builder = ContextBuilder(max_context_chars=4000)
    
    # Khởi tạo Router 2 trong 1
    query_router = QueryRouter(llm_client=llm)
    
    # Nếu chưa có bộ nhớ thì tạo mới, có rồi thì giữ nguyên
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ChatHistory(max_history_turns=5)
        
    pipeline = LegalRAGPipeline(
        retriever=hybrid_retriever,
        reranker=reranker,
        llm_client=llm,
        context_builder=ctx_builder,
        query_router=query_router,
        chat_history=st.session_state.chat_history,
        top_k_retrieve=30,
        top_k_rerank=5
    )
    return pipeline

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Xin chào! Tôi là AI Hỗ trợ Pháp lý. Bạn có câu hỏi nào về luật pháp Việt Nam cần giải đáp không?"}
    ]

# ==========================================
# 3. THIẾT KẾ GIAO DIỆN (UI)
# ==========================================
st.title("⚖️ Trợ Lý AI Tư Vấn Pháp Luật Việt Nam")
st.markdown("Hệ thống RAG nâng cao: **Hybrid Search** $\\rightarrow$ **Cross-Encoder** $\\rightarrow$ **Smart Query Router**")

# --- SIDEBAR: CẤU HÌNH ---
with st.sidebar:
    st.header("⚙️ Cài đặt Hệ thống")
    
    # Chỉ giữ lại tùy chọn model của Gemini
    model_name = st.selectbox("Chọn mô hình Gemini:", ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash", "gemini-1.5-pro"])
    
    # Nút áp dụng cấu hình
    if st.button("🔄 Cập nhật LLM", use_container_width=True):
        st.session_state.pipeline = initialize_pipeline(model_name)
        st.toast(f"Đã chuyển sang dùng {model_name}!", icon="✅")
        
    st.divider()
    
    # Nút xóa lịch sử
    if st.button("🗑️ Xóa Lịch Sử Trò Chuyện", type="primary", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Xin chào! Lịch sử đã được làm mới. Tôi có thể giúp gì cho bạn?"}
        ]
        if "chat_history" in st.session_state:
            st.session_state.chat_history.clear()
        st.rerun()
        
    st.divider()
    st.caption("🔍 Thông tin hệ thống:")
    st.caption("- Nhúng (Vector DB): ChromaDB")
    st.caption("- Từ vựng: BM25 + PRF Expansion")
    st.caption("- Trộn kết quả: Reciprocal Rank Fusion")
    st.caption("- Tái xếp hạng: ms-marco-MiniLM")

# Khởi tạo pipeline lần đầu tiên nếu chưa có (Mặc định dùng gemini-2.5-flash)
if "pipeline" not in st.session_state:
    st.session_state.pipeline = initialize_pipeline("gemini-2.5-flash")

# ==========================================
# 4. XỬ LÝ LUỒNG TRÒ CHUYỆN
# ==========================================
# Hiển thị lại lịch sử tin nhắn
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Nếu có thông tin bổ sung (Standalone query hoặc Trích dẫn)
        if msg.get("standalone_query") and msg["standalone_query"] != msg.get("original_query"):
            st.caption(f"💭 *AI đã hiểu lại câu hỏi: \"{msg['standalone_query']}\"*")
            
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 Xem nguồn trích dẫn pháp luật"):
                for idx, source in enumerate(msg["sources"], 1):
                    meta = source.get('metadata', {})
                    article = meta.get('article', 'Không rõ')
                    content = source.get('content', '')
                    st.markdown(f"**[{idx}] {article}**")
                    st.caption(f"{content}")

# Khung nhập liệu
if prompt := st.chat_input("Nhập câu hỏi (vd: Đi xe máy không đội mũ bảo hiểm phạt bao nhiêu?)..."):
    
    # Hiển thị tin nhắn user
    st.session_state.messages.append({"role": "user", "content": prompt, "original_query": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Hiển thị tin nhắn trợ lý
    with st.chat_message("assistant"):
        with st.spinner("Đang phân tích và tra cứu..."):
            try:
                # 🚀 GỌI LUỒNG RAG CHÍNH
                result = st.session_state.pipeline.run(query=prompt)
                
                answer = result["answer"]
                sources = result.get("sources", [])
                query_type = result.get("type", "legal")
                standalone_query = result.get("standalone_query", prompt)
                
                # In câu trả lời
                st.markdown(answer)
                
                # In thông báo "AI đã viết lại câu hỏi"
                if query_type == "legal" and standalone_query != prompt:
                    st.caption(f"💭 *AI đã hiểu lại câu hỏi: \"{standalone_query}\"*")
                
                # In nguồn trích dẫn nếu có
                if query_type == "legal" and sources:
                    with st.expander("📚 Xem nguồn trích dẫn pháp luật"):
                        for idx, source in enumerate(sources, 1):
                            meta = source.get('metadata', {})
                            article = meta.get('article', 'Không rõ')
                            content = source.get('content', '')
                            st.markdown(f"**[{idx}] {article}**")
                            st.caption(f"{content}")
                
                # In thời gian xử lý
                st.caption(f"⏱️ Tổng thời gian: {result['processing_time']:.2f}s")
                
                # Lưu vào state
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources,
                    "standalone_query": standalone_query,
                    "original_query": prompt
                })

            except Exception as e:
                error_msg = f"❌ Xin lỗi, đã xảy ra lỗi trong quá trình xử lý: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})