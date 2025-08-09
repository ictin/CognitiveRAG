# CognitiveRAG/llm_provider.py
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_models import ChatOllama
try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    from langchain_community.embeddings import OllamaEmbeddings
from . import config

def get_llm(model_name: str):
    """
    Returns a language model instance based on the configured provider.
    """
    if config.LLM_PROVIDER == "ollama":
        return ChatOllama(model=model_name, base_url=config.OLLAMA_BASE_URL)
    elif config.LLM_PROVIDER == "openai":
        return ChatOpenAI(model=model_name, api_key=config.OPENAI_API_KEY)
    else:
        raise ValueError(f"Unsupported LLM provider: {config.LLM_PROVIDER}")

def get_embeddings():
    """
    Returns an embedding model instance based on the configured provider.
    """
    if config.LLM_PROVIDER == "ollama":
        return OllamaEmbeddings(model=config.EMBEDDING_MODEL, base_url=config.OLLAMA_BASE_URL)
    elif config.LLM_PROVIDER == "openai":
        return OpenAIEmbeddings(model=config.EMBEDDING_MODEL, api_key=config.OPENAI_API_KEY)
    else:
        raise ValueError(f"Unsupported LLM provider: {config.LLM_PROVIDER}")
