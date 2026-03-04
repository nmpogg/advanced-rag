"""
Global configuration settings cho LangChain RAG system với Web Search.
"""
import os

# ========== API & Environment ==========
os.environ["GOOGLE_API_KEY"] = "your_google_api_key_here"

# ========== Model Configuration ==========
EMBEDDING_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "gemini-2.5-flash"
LLM_TEMPERATURE = 0

# ========== Vector Database Configuration ==========
CHROMA_PATH = "./chroma_db"
JSON_PATH = "./chunks_db.json"

# ========== Retrieval Configuration ==========
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_INITIAL = 10  # Initial search results before reranking
TOP_K_FINAL = 5     # Final top documents

# ========== Web Search Configuration ==========
WEB_SEARCH_COOLDOWN = 2  # seconds between web search calls
MAX_WEB_SNIPPETS = 5     # Max snippets to return from web search
MIN_WEB_RESULT_LENGTH = 20  # Minimum length for valid web result

# ========== Device Configuration ==========
DEVICE = "cpu" 
