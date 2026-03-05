from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseRetriever(ABC):
    
    @abstractmethod
    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:

        pass
    
    # for evaluaiton
    def batch_search(self, queries: List[str], top_k: int) -> List[List[Dict[str, Any]]]:

        return [self.search(q, top_k) for q in queries]