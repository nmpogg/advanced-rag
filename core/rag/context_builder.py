# File: rag/context_builder.py
from typing import List, Dict, Any

class ContextBuilder:
    def __init__(self, content_length: int = 4000):
        """
        Khởi tạo bộ dựng ngữ cảnh.
        
        Args:
            content_length: Giới hạn độ dài ký tự tối đa của ngữ cảnh.
                               Giúp tránh việc vượt quá Context Window của LLM và tiết kiệm chi phí.
        """
        self.content_length = content_length

    def build_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Chuyển đổi chunks thành một chuỗi văn bản định dạng chuẩn.
        """
        if not retrieved_docs:
            return "Không tìm thấy tài liệu pháp luật nào liên quan trong cơ sở dữ liệu."
        
        context_parts = []
        current_length = 0
        
        for doc in retrieved_docs:
            metadata = doc.get("metadata", {})
            law_name = metadata.get("law_name")
            article_title = metadata.get("article")
            
            # Định dạng một chunk văn bản
            doc_type = metadata.get("type")
            if doc_type == "clause": # khoản thì thêm số khoản
                clause_num = metadata.get("clause_num", "")
                chunk_text = f"[{law_name} - {article_title} - Khoản {clause_num}]\nNội dung: {doc['content']}\n"
            else: #chỉ số điều
                chunk_text = f"[{law_name} - {article_title}]\nNội dung: {doc['content']}\n" 
            
            # Kiểm tra độ dài để không làm tràn Prompt
            if current_length + len(chunk_text) > self.content_length:
                print(f"[ContextBuilder] Ngữ cảnh đạt giới hạn ({current_length} ký tự). Chỉ lấy {len(context_parts)} document truy xuất được.")
                break
                
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
            
        # Nối các chunk lại với nhau bằng dải phân cách rõ ràng
        final_context = "\n".join(context_parts)
        
        return final_context

if __name__ == "__main__":
    # GIẢ LẬP ĐỂ TEST
    mock_docs = [
         {
            "cid": 1,
            "content": "Nghị định này quy định về: a) Xử phạt vi phạm hành chính về trật tự, an toàn giao thông trong lĩnh vực giao thông đường bộ bao gồm: hành vi vi phạm hành chính; hình thức, mức xử phạt, biện pháp khắc phục hậu quả đối với từng hành vi vi phạm hành chính; thẩm quyền lập biên bản, thẩm quyền xử phạt, mức phạt tiền cụ thể theo từng chức danh đối với hành vi vi phạm hành chính về trật tự, an toàn giao thông trong lĩnh vực giao thông đường bộ; b) Mức trừ điểm giấy phép lái xe đối với từng hành vi vi phạm hành chính; trình tự, thủ tục, thẩm quyền trừ điểm, phục hồi điểm giấy phép lái xe để quản lý việc chấp hành pháp luật về trật tự, an toàn giao thông đường bộ của người lái xe.",
            "metadata": {
            "law_name": "Nghị định 168/2024/NĐ-CP",
            "article": "Điều 1. Phạm vi điều chỉnh",
            "article_num": "1",
            "clause_num": "1",
            "type": "clause",
            "length": 679
            }
        },
        {
            "cid": 2,
            "content": "Các hành vi vi phạm hành chính trong lĩnh vực quản lý nhà nước khác liên quan đến trật tự, an toàn giao thông trong lĩnh vực giao thông đường bộ mà không quy định tại Nghị định này thì áp dụng quy định tại các Nghị định quy định về xử phạt vi phạm hành chính trong các lĩnh vực đó để xử phạt.",
            "metadata": {
            "law_name": "Nghị định 168/2024/NĐ-CP",
            "article": "Điều 1. Phạm vi điều chỉnh",
            "article_num": "1",
            "clause_num": "2",
            "type": "clause",
            "length": 292
            }
        },
        {
            "cid": 3,
            "content": "Cá nhân, tổ chức Việt Nam; cá nhân, tổ chức nước ngoài có hành vi vi phạm hành chính về trật tự, an toàn giao thông trong lĩnh vực giao thông đường bộ trên lãnh thổ nước Cộng hòa xã hội chủ nghĩa Việt Nam.",
            "metadata": {
            "law_name": "Nghị định 168/2024/NĐ-CP",
            "article": "Điều 2. Đối tượng áp dụng",
            "article_num": "2",
            "clause_num": "1",
            "type": "clause",
            "length": 205
            }
        },
        {
            "cid": 4,
            "content": "Tổ chức quy định tại khoản 1 Điều này bao gồm: a) Cơ quan nhà nước có hành vi vi phạm mà hành vi đó không thuộc nhiệm vụ quản lý nhà nước được giao; b) Đơn vị sự nghiệp công lập; c) Tổ chức chính trị - xã hội, tổ chức chính trị xã hội nghề nghiệp, tổ chức xã hội, tổ chức xã hội nghề nghiệp; d) Tổ chức kinh tế được thành lập theo quy định của Luật Doanh nghiệp gồm: doanh nghiệp tư nhân, công ty cổ phần, công ty trách nhiệm hữu hạn, công ty hợp danh và các đơn vị phụ thuộc doanh nghiệp (chi nhánh, văn phòng đại diện); đ) Tổ chức kinh tế được thành lập theo quy định của Luật Hợp tác xã gồm: tổ hợp tác, hợp tác xã, liên hiệp hợp tác xã; e) Cơ sở đào tạo lái xe, trung tâm sát hạch lái xe, cơ sở đăng kiểm xe cơ giới, xe máy chuyên dùng, cơ sở thử nghiệm, sản xuất, lắp ráp, nhập khẩu, bảo hành, bảo dưỡng xe cơ giới, xe máy chuyên dùng; g) Các tổ chức khác được thành lập theo quy định của pháp luật; h) Cơ quan, tổ chức nước ngoài được cấp có thẩm quyền của Việt Nam cho phép hoạt động trên lãnh thổ Việt Nam.",
            "metadata": {
            "law_name": "Nghị định 168/2024/NĐ-CP",
            "article": "Điều 2. Đối tượng áp dụng",
            "article_num": "2",
            "clause_num": "2",
            "type": "clause",
            "length": 1014
            }
        }
    ]
    
    builder = ContextBuilder()
    formatted_ctx = builder.build_context(mock_docs)

    print(formatted_ctx)