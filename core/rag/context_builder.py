# File: rag/context_builder.py
from typing import List, Dict, Any

class ContextBuilder:
    def __init__(self, max_context_chars: int = 4000):
        """
        Khởi tạo bộ dựng ngữ cảnh.
        
        Args:
            max_context_chars: Giới hạn độ dài ký tự tối đa của ngữ cảnh.
                               Giúp tránh việc vượt quá Context Window của LLM và tiết kiệm chi phí.
        """
        self.max_context_chars = max_context_chars

    def build_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Chuyển đổi danh sách các tài liệu (chunks) thành một chuỗi văn bản định dạng chuẩn.
        """
        if not retrieved_docs:
            return "Không tìm thấy tài liệu pháp luật nào liên quan trong cơ sở dữ liệu."
        
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(retrieved_docs, start=1):
            # Trích xuất metadata để LLM biết nó đang đọc "Điều" nào
            metadata = doc.get("metadata", {})
            article_title = metadata.get("article", f"Tài liệu {i}")
            
            # Định dạng một chunk văn bản
            chunk_text = f"[{article_title}]\nNội dung: {doc['content']}\n"
            
            # Kiểm tra độ dài để không làm tràn Prompt
            if current_length + len(chunk_text) > self.max_context_chars:
                print(f"[ContextBuilder] Ngữ cảnh đạt giới hạn ({current_length} ký tự). Đang bỏ qua các tài liệu có độ ưu tiên thấp hơn.")
                break
                
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
            
        # Nối các chunk lại với nhau bằng dải phân cách rõ ràng
        final_context = "\n---\n".join(context_parts)
        
        return final_context

if __name__ == "__main__":
    # GIẢ LẬP ĐỂ TEST
    mock_docs = [
        {
            "id": "1", 
            "content": "Phạt tiền từ 1.000.000 đồng đến 2.000.000 đồng đối với người điều khiển xe mô tô...", 
            "metadata": {"article": "Điều 5 Nghị định 100"}
        },
        {
            "id": "2", 
            "content": "Người vi phạm phải xuất trình Giấy phép lái xe khi có yêu cầu...", 
            "metadata": {"article": "Điều 82 Nghị định 100"}
        }
    ]
    
    builder = ContextBuilder()
    formatted_ctx = builder.build_context(mock_docs)
    
    print("=== NGỮ CẢNH ĐÃ ĐƯỢC FORMAT CHO LLM ===")
    print(formatted_ctx)
    print("=======================================")