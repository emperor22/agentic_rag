from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    BASE_URL: str = "https://openrouter.ai/api/v1"
    BASE_URL_RERANK: str = "https://openrouter.ai/api/v1/rerank"
    
    PRIMARY_MODEL: str = "x-ai/grok-4.1-fast"
    EMBEDDING_MODEL: str = "qwen/qwen3-embedding-8b"
    RERANK_MODEL: str = "cohere/rerank-4-fast"
    TEMPERATURE: float = 0.0

    RETRIEVER_K: int = 10
    RERANK_TOP_N: int = 4
    TAVILY_RESULTS: int = 3
    CHAT_HISTORY_WINDOW: int = 3
    
    MAX_RAG_RETRIES: int = 3
    MAX_WEB_RETRIES: int = 2
    HISTORY_LIMIT: int = 3
    
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    COLLECTION_NAME: str = "pdf_collection"
    
    PDF_FILENAME: str = "memory_langchain.pdf"

    class Config:
        env_file = ".env"

config = Settings()