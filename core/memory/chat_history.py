# File: memory/chat_history.py
from typing import List, Dict

class ChatHistory:
    def __init__(self, max_history_turns: int = 5):
        """
        Seassion
        
        Args:
            max_history_turns: Số lượt hỏi-đáp tối đa được lưu lại. 
                               1 turn = 1 câu hỏi của User + 1 câu trả lời của AI.
                               Ví dụ: 5 turns = 10 tin nhắn gần nhất.
        """
        self.max_history_turns = max_history_turns
        self.history: List[Dict[str, str]] = []

    def add_user_message(self, message: str):

        self.history.append({"role": "user", "content": message})
        self._prune()

    def add_assistant_message(self, message: str):

        self.history.append({"role": "assistant", "content": message})
        self._prune()

    def get_history(self) -> List[Dict[str, str]]:
        return self.history

    def _prune(self):
        """Cắt bỏ các tin nhắn cũ nếu vượt quá giới hạn (Sliding Window)"""
        max_messages = self.max_history_turns * 2
        if len(self.history) > max_messages:
            # Lấy các phần tử từ cuối mảng (mới nhất)
            self.history = self.history[-max_messages:]

    def clear(self):
        self.history = []
        print("[ChatHistory] Đã xóa toàn bộ lịch sử trò chuyện.")