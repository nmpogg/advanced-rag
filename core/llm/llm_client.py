# File: llm/llm_client.py
import os
from typing import Optional
from openai import OpenAI
import google.generativeai as genai

from dotenv import load_dotenv
load_dotenv()

class LLMClient:
    def __init__(
        self, 
        provider: str = "gemini", #  'openai', 'gemini', 'local'
        api_key: Optional[str] = None, 
        model_name: Optional[str] = None, 
        base_url: Optional[str] = None,
        temperature: float = 0.0
    ):
        self.provider = provider.lower()
        self.temperature = temperature
        
        print(f"[LLMClient] Đang khởi tạo kết nối với provider: {self.provider.upper()}...")

        if self.provider == "openai":
            self.model_name = model_name or "gpt-3.5-turbo"
            api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Không tìm thấy OPENAI_API_KEY.")
            self.client = OpenAI(api_key=api_key)

        elif self.provider == "gemini":
            self.model_name = model_name or "gemini-2.5-flash"
            api_key = api_key or os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Không tìm thấy GEMINI_API_KEY.")
            genai.configure(api_key=api_key)
            # Khởi tạo model Gemini
            self.client = genai.GenerativeModel(self.model_name)

        elif self.provider == "local":
            self.model_name = model_name or "llama3"
            if not base_url:
                raise ValueError("Provider 'local' yêu cầu phải truyền 'base_url' (vd: http://localhost:11434/v1).")
            self.client = OpenAI(api_key="sk-no-key-required", base_url=base_url)
            
        else:
            raise ValueError(f"Provider không hợp lệ: {provider}. Chọn 'openai', 'gemini', hoặc 'local'.")
        

    def generate(self, system_prompt: str, user_prompt: str) -> str:
 
        try:
            if self.provider in ["openai", "local"]:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=1000,
                    top_p=0.9,
                )
                return response.choices[0].message.content.strip()

            elif self.provider == "gemini":
                full_prompt = f"{system_prompt}\n\n---\n{user_prompt}"
                print(f"[LLMClient - GEMINI] Đang gửi prompt đến Gemini:\n{full_prompt}\n")
                
                response = self.client.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.temperature,
                        max_output_tokens=1000,
                        top_p=0.9,
                    )
                )
                return response.text.strip()

        except Exception as e:
            print(f"[Lỗi LLMClient - {self.provider.upper()}] Không thể tạo câu trả lời: {e}")
            return "Xin lỗi, hệ thống đang gặp sự cố khi kết nối với mô hình ngôn ngữ. Vui lòng thử lại sau."

if __name__ == "__main__":
    #test
    sys_prompt = "Bạn là trợ lý AI. Chỉ trả lời ngắn gọn trong 1 câu."
    user_question = "Thủ đô của Việt Nam là gì?"
    
    # print("\n--- TEST OPENAI ---")
    # try:
    #     llm_openai = LLMClient(provider="openai", model_name="gpt-3.5-turbo")
    #     ans = llm_openai.generate(sys_prompt, user_question)
    #     print(ans)
    # except Exception as e:
    #     print(f"Bỏ qua test OpenAI do: {e}")

    print("\n--- TEST GEMINI ---")
    try:
        llm_gemini = LLMClient(provider="gemini", model_name="gemini-2.5-flash")
        ans = llm_gemini.generate(sys_prompt, user_question)
        print(ans)
    except Exception as e:
        print(f"Bỏ qua test Gemini do: {e}")