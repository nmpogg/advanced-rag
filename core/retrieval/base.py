# File: retrieval/base_retriever.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Hàm trừu tượng yêu cầu mọi retriever phải trả về một list các dictionary.
        Định dạng chuẩn:
        [
            {
                "id": "cid_123",
                "content": "Nội dung điều khoản...",
                "metadata": {"article": "Điều 1", "type": "article", ...},
                "score": 0.85
            },
            ...
        ]
        """
        pass