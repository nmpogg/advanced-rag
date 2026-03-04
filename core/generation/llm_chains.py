"""
LLM Chains module - Define all prompt templates và chain definitions.
"""
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from config import LLM_MODEL, LLM_TEMPERATURE

_llm = None


def get_llm():
    """Get hoặc initialize LLM instance."""
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE
        )
    return _llm


# ROUTER CHAIN
ROUTER_PROMPT = PromptTemplate.from_template("""
Bạn là một hệ thống phân loại câu hỏi. Hãy phân loại câu hỏi của người dùng thành 2 loại:
1. "chitchat": Các câu giao tiếp thông thường, chào hỏi, hoặc không cần tra cứu tài liệu.
2. "rag": Các câu hỏi cần tra cứu kiến thức chuyên môn, thông tin cụ thể.
Chỉ trả về đúng 1 từ: "chitchat" hoặc "rag".
Câu hỏi: {query}
""")

def get_router_chain():
    """Get router chain."""
    llm = get_llm()
    router_chain = ROUTER_PROMPT | llm | StrOutputParser()
    return router_chain


def route_query(query: str) -> str:
    """
    Route query đến chitchat hoặc RAG.
    
    Args:
        query: User query
        
    Returns:
        str: "chitchat" hoặc "rag"
    """
    router_chain = get_router_chain()
    result = router_chain.invoke({"query": query})
    return result.strip().lower()


# QUERY REFLECTION CHAIN 
REFLECTION_PROMPT = PromptTemplate.from_template("""
Viết lại câu hỏi sau để tối ưu hóa cho việc tìm kiếm vector và keyword (từ khóa).
Giữ nguyên ý nghĩa nhưng làm rõ các thực thể và ngữ cảnh.
Chỉ viết lại câu hỏi, không thêm giải thích nào khác.
Câu hỏi gốc: {query}
Câu hỏi tối ưu:
""")

def get_reflection_chain():
    """Get query reflection chain."""
    llm = get_llm()
    reflection_chain = REFLECTION_PROMPT | llm | StrOutputParser()
    return reflection_chain


def reflect_query(query: str) -> str:
    """
    Optimize query để improve search results.
    
    Args:
        query: Original query
        
    Returns:
        str: Optimized query
    """
    reflection_chain = get_reflection_chain()
    refined = reflection_chain.invoke({"query": query})
    return refined.strip()


# GRADER CHAIN
GRADER_PROMPT = PromptTemplate.from_template("""
Bạn là một người chấm điểm. Hãy xem xét các tài liệu dưới đây có chứa thông tin để trả lời câu hỏi của người dùng không.
Trả lời "yes" nếu tài liệu có liên quan và đủ thông tin. Trả lời "no" nếu không.
Tài liệu: {context}
Câu hỏi: {query}
""")

def get_grader_chain():
    """Get document grader chain."""
    llm = get_llm()
    grader_chain = GRADER_PROMPT | llm | StrOutputParser()
    return grader_chain


def grade_documents(context: str, query: str) -> bool:
    """
    Grade nếu documents đủ tốt để trả lời query.
    
    Args:
        context: Document context
        query: User query
        
    Returns:
        bool: True nếu documents đủ, False nếu không
    """
    grader_chain = get_grader_chain()
    result = grader_chain.invoke({"context": context, "query": query}).strip().lower()
    return "yes" in result


# QA CHAIN
QA_PROMPT = PromptTemplate.from_template("""
Sử dụng các thông tin sau đây để trả lời câu hỏi. Nếu không biết, hãy nói là không biết, đừng bịa ra.
Thông tin: {context}
Câu hỏi: {query}
Câu trả lời:
""")

def get_qa_chain():
    """Get QA chain."""
    llm = get_llm()
    qa_chain = QA_PROMPT | llm | StrOutputParser()
    return qa_chain


def generate_answer(context: str, query: str) -> str:
    """
    Generate answer từ context.
    
    Args:
        context: Document context
        query: User query
        
    Returns:
        str: Generated answer
    """
    qa_chain = get_qa_chain()
    answer = qa_chain.invoke({"context": context, "query": query})
    return answer


# CHITCHAT CHAIN
def generate_chitchat_response(query: str) -> str:
    """
    Generate chitchat response (không cần documents).
    
    Args:
        query: User query
        
    Returns:
        str: Response text
    """
    llm = get_llm()
    response = llm.invoke(query)
    return response.content if hasattr(response, 'content') else str(response)
