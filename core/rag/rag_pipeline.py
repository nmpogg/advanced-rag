# File: rag/rag_pipeline.py
from typing import List, Dict, Any
import time

from retrieval.base import BaseRetriever
from reranking.rerank import CrossEncoderReranker
from llm.llm_client import LLMClient
from llm.prompts import build_user_prompt, LEGAL_SYSTEM_PROMPT
from rag.context_builder import ContextBuilder

from rag.query_router import QueryRouter
from memory.chat_history import ChatHistory

class LegalRAGPipeline:
    def __init__(
        self,
        retriever: BaseRetriever,
        reranker: CrossEncoderReranker,
        llm_client: LLMClient,
        context_builder: ContextBuilder,
        query_router: QueryRouter,
        chat_history: ChatHistory,
        top_k_retrieve: int = 20,
        top_k_rerank: int = 5
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.llm_client = llm_client
        self.context_builder = context_builder
        self.query_router = query_router
        self.chat_history = chat_history
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank = top_k_rerank
        
        print("\n[RAG Pipeline] Hệ thống đã sẵn sàng...")

    def run(self, query: str) -> Dict[str, Any]:
        print(f"USER: '{query}'")
        
        total_start_time = time.time()
        
        current_history = self.chat_history.get_history()
        intent, processed_result = self.query_router.process_query(
            current_query=query, 
            chat_history=current_history
        )
        

        if intent == "CHITCHAT":
            print("\n[Chitchat] Trả về câu giao tiếp trực tiếp từ Router...")
            answer = processed_result
            
            self.chat_history.add_user_message(query)
            self.chat_history.add_assistant_message(answer)
            
            return {
                "query": query,
                "answer": answer,
                "sources": [],
                "type": "chitchat",
                "processing_time": time.time() - total_start_time
            }

        query_reflection = processed_result
        print(f"\n[Legal] Truy xuất tài liệu với Query Reflection: '{query_reflection}'")
        
        # Retrieval & Reranking
        retrieved_docs = self.retriever.retrieve(query=query_reflection, top_k=self.top_k_retrieve)
        print(f"[Legal] Danh sách tài liệu truy xuất được:")
        for i, doc in enumerate(retrieved_docs):
            print(f"  {i+1}. [{doc['metadata']['law_name']} - {doc['metadata']['article']}] (Score: {doc.get('score', 'N/A'):.4f})")
        best_docs = self.reranker.rerank(query=query_reflection, documents=retrieved_docs, top_k=self.top_k_rerank)

        print("\n[Legal] Đang gọi LLM sinh câu trả lời...")
        formatted_context = self.context_builder.build_context(best_docs)
        user_prompt = build_user_prompt(query=query_reflection, formatted_context=formatted_context)
        
        answer = self.llm_client.generate(
            system_prompt=LEGAL_SYSTEM_PROMPT,
            user_prompt=user_prompt
        )

        self.chat_history.add_user_message(query) # lưu câu gốc
        self.chat_history.add_assistant_message(answer)
        
        total_time = time.time() - total_start_time
        print(f"\n[Hoàn tất] Tổng thời gian xử lý: {total_time:.2f}s")
        
        return {
            "query": query,
            "query_reflection": query_reflection,
            "answer": answer,
            "sources": best_docs,
            "type": "legal",
            "processing_time": total_time
        }