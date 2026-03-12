# File: rag/query_router.py
import os
import json
from typing import List, Dict, Tuple
from llm.llm_client import LLMClient
from llm.prompts import ROUTER_SYSTEM_PROMPT, build_router_user_prompt

class QueryRouter:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        print("[QueryRouter] Đã khởi tạo hệ thống định tuyến (Gộp Chitchat & Reflection).")

    def process_query(self, current_query: str, chat_history: List[Dict[str, str]]) -> Tuple[str, str]:
        """
        Gửi câu hỏi và lịch sử cho LLM để vừa phân loại, vừa xử lý.
        
        Returns:
            Tuple[str, str]: (intent, result)
            - intent: "CHITCHAT" hoặc "LEGAL"
            - result: Câu trả lời chitchat HOẶC Câu hỏi đã được viết lại (Standalone Query)
        """
        print("\n[QueryRouter] Đang phân tích ý định và ngữ cảnh...")
        
        recent_history = chat_history[-4:] if chat_history else []
        history_text = ""
        for msg in recent_history:
            role = "Người dùng" if msg["role"] == "user" else "Trợ lý"
            history_text += f"{role}: {msg['content']}\n"
            
        if not history_text:
            history_text = "Chưa có lịch sử trò chuyện."

        user_prompt = build_router_user_prompt(history_text, current_query)

        try:
            response_text = self.llm.generate(
                system_prompt=ROUTER_SYSTEM_PROMPT,
                user_prompt=user_prompt
            ).strip()
            
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
                
            # Parse JSON
            parsed_data = json.loads(response_text.strip())
            
            intent = parsed_data.get("intent", "LEGAL").upper()
            result = parsed_data.get("result", current_query)
            
            print(f"-> Intent phát hiện: {intent}")
            print(f"-> Kết quả xử lý  : {result}")
            
            return intent, result
            
        except json.JSONDecodeError as e:
            print(f"[Cảnh báo QueryRouter] Lỗi parse JSON từ LLM: {e}. Fallback về luồng LEGAL.")
            # Fallback an toàn: Coi như là câu hỏi luật và không rewrite
            return "LEGAL", current_query
            
        except Exception as e:
            print(f"[Lỗi QueryRouter] {e}. Fallback về luồng LEGAL.")
            return "LEGAL", current_query

if __name__ == "__main__":
    # TEST
    llm = LLMClient(api_key=os.environ.get("OPENAI_API_KEY"), model_name="gpt-3.5-turbo", temperature=0.0)
    router = QueryRouter(llm)
    
    mock_history = [
        {"role": "user", "content": "Đi xe máy không gương phạt bao nhiêu?"},
        {"role": "assistant", "content": "Phạt từ 100k đến 200k."}
    ]
    
    print("\n--- TEST CASE 1: LEGAL (Có nối tiếp ngữ cảnh) ---")
    intent, result = router.process_query("Vậy nếu ô tô thì sao?", mock_history)
    
    print("\n--- TEST CASE 2: CHITCHAT ---")
    intent, result = router.process_query("Cảm ơn bot nhiều nhé", mock_history)