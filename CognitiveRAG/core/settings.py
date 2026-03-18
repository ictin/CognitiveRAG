from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    provider: str = "ollama"
    ollama_base_url: str = "http://172.31.208.1:11434"
    ollama_api_path: str = "/api"
    ollama_api_key: str | None = None
    openai_api_key: str | None = None

    planner_model: str = "llama3"
    synthesis_model: str = "llama3"
    reflection_model: str = "gemma:2b"
    embedding_model: str = "nomic-embed-text"

    model_config = SettingsConfigDict(env_prefix="CRAG_LLM_", extra="ignore")


class StoreSettings(BaseSettings):
    data_dir: Path = Path("./data")
    source_documents_dir: Path = Path("./data/source_documents")
    metadata_db_path: Path = Path("./data/metadata.sqlite3")
    graph_db_path: Path = Path("./data/graph.sqlite3")
    vector_store_path: Path = Path("./data/chroma")
    episodic_db_path: Path = Path("./data/episodic.sqlite3")
    profile_db_path: Path = Path("./data/profile.sqlite3")
    task_db_path: Path = Path("./data/tasks.sqlite3")
    reasoning_db_path: Path = Path("./data/reasoning.sqlite3")
    # Explicitly configure Chroma embedding model to avoid relying on defaults
    chroma_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_backing_impl: str = "duckdb+parquet"

    model_config = SettingsConfigDict(env_prefix="CRAG_STORE_", extra="ignore")


class RetrievalSettings(BaseSettings):
    chunk_size: int = 1024
    chunk_overlap: int = 100
    bm25_top_k: int = 5
    vector_top_k: int = 5
    graph_top_k: int = 5
    episodic_top_k: int = 5
    max_context_chunks: int = 12
    web_search_enabled: bool = False

    model_config = SettingsConfigDict(env_prefix="CRAG_RETRIEVAL_", extra="ignore")


class MemorySettings(BaseSettings):
    enable_episodic_memory: bool = True
    enable_profile_memory: bool = True
    enable_task_memory: bool = True
    enable_reasoning_memory: bool = True
    max_recursion_depth: int = 3

    model_config = SettingsConfigDict(env_prefix="CRAG_MEMORY_", extra="ignore")


class AppSettings(BaseSettings):
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_prefix="CRAG_APP_", extra="ignore")


class Settings(BaseSettings):
    app: AppSettings = Field(default_factory=AppSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    store: StoreSettings = Field(default_factory=StoreSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
