"""
Document ingestion module - PDF loading, chunking, and indexing.
"""
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL, CHROMA_PATH, JSON_PATH, CHUNK_SIZE, CHUNK_OVERLAP, DEVICE


def get_embeddings():
    """Initialize và return HuggingFace embeddings model."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': DEVICE},
        encode_kwargs={'normalize_embeddings': True}
    )
    return embeddings


def process_and_index_documents(pdf_path: str):
    """
    Process PDF file, chunk text, và create vector database.
    
    Args:
        pdf_path: Đường dẫn tới file PDF
        
    Returns:
        tuple: (vector_db, chunks_data)
    """
    print(f"📄 Đang load PDF từ: {pdf_path}")
    
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Chunked thành {len(chunks)} chunks")
    
    # Lưu vào JSON cho Keyword Search (BM25)
    chunks_data = [
        {"id": i, "content": chunk.page_content, "metadata": chunk.metadata} 
        for i, chunk in enumerate(chunks)
    ]
    with open(JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=4)
    print(f"Lưu chunks vào JSON: {JSON_PATH}")
    
    embeddings = get_embeddings()
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=CHROMA_PATH
    )
    print(f"Đã lưu vào ChromaDB: {CHROMA_PATH}")
    
    return vector_db, chunks_data


def load_vector_db():
    """Load vector database."""
    embeddings = get_embeddings()
    vector_db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=embeddings
    )
    return vector_db
