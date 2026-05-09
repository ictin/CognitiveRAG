# CognitiveRAG/llm_provider.py
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except Exception:  # pragma: no cover - handled at runtime via explicit errors
    ChatOpenAI = None
    OpenAIEmbeddings = None

try:
    from langchain_community.chat_models import ChatOllama
except Exception:  # pragma: no cover
    ChatOllama = None

try:
    from langchain_ollama import OllamaEmbeddings
except Exception:
    try:
        from langchain_community.embeddings import OllamaEmbeddings
    except Exception:  # pragma: no cover
        OllamaEmbeddings = None
from . import config

def get_llm(model_name: str):
    """
    Returns a language model instance based on the configured provider.
    This wraps the underlying model with a small sanitizer proxy so that
    provider-agnostic sanitization occurs before any invoke/chat calls.
    """
    if config.LLM_PROVIDER == "ollama":
        if ChatOllama is None:
            raise RuntimeError("ChatOllama is unavailable; install langchain-community")
        base = ChatOllama(model=model_name, base_url=config.OLLAMA_BASE_URL)
        model = base
    elif config.LLM_PROVIDER == "openai":
        if ChatOpenAI is None:
            raise RuntimeError("ChatOpenAI is unavailable; install langchain-openai")
        base = ChatOpenAI(model=model_name, api_key=config.OPENAI_API_KEY)
        model = base
    else:
        raise ValueError(f"Unsupported LLM provider: {config.LLM_PROVIDER}")

    # Lightweight proxy: sanitize inputs while preserving underlying provider contract.
    class SanitizedLLM:
        def __init__(self, underlying):
            self._underlying = underlying

        @staticmethod
        def _sanitize_prompt(prompt):
            try:
                from CognitiveRAG.llm.sanitizer import sanitize_text
                return sanitize_text(prompt)
            except Exception:
                return prompt

        def invoke(self, prompt, *args, **kwargs):
            if isinstance(prompt, str):
                prompt = self._sanitize_prompt(prompt)
            return self._underlying.invoke(prompt, *args, **kwargs)

        async def ainvoke(self, prompt, *args, **kwargs):
            if isinstance(prompt, str):
                prompt = self._sanitize_prompt(prompt)
            if hasattr(self._underlying, "ainvoke"):
                return await self._underlying.ainvoke(prompt, *args, **kwargs)
            # Compatibility fallback when async path is unavailable.
            return self.invoke(prompt, *args, **kwargs)

        # expose underlying for advanced use
        @property
        def underlying(self):
            return self._underlying

        def __getattr__(self, name):
            return getattr(self._underlying, name)

    return SanitizedLLM(model)

def get_embeddings():
    """
    Returns an embedding model instance based on the configured provider.
    """
    if config.LLM_PROVIDER == "ollama":
        if OllamaEmbeddings is None:
            raise RuntimeError("OllamaEmbeddings is unavailable")
        return OllamaEmbeddings(model=config.EMBEDDING_MODEL, base_url=config.OLLAMA_BASE_URL)
    elif config.LLM_PROVIDER == "openai":
        if OpenAIEmbeddings is None:
            raise RuntimeError("OpenAIEmbeddings is unavailable; install langchain-openai")
        return OpenAIEmbeddings(model=config.EMBEDDING_MODEL, api_key=config.OPENAI_API_KEY)
    else:
        raise ValueError(f"Unsupported LLM provider: {config.LLM_PROVIDER}")
