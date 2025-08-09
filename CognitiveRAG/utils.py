# CognitiveRAG/utils.py
import os
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any
from langchain_core.documents import Document
from nltk.tokenize import word_tokenize
import nltk
from . import config

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except (nltk.downloader.DownloadError, LookupError):
    try:
        nltk.download('punkt', quiet=True)
    except Exception:
        pass  # Fallback if NLTK download fails

# --- File I/O and Serialization ---

def save_pickle(obj: Any, path: str):
    """Saves an object to a pickle file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path: str) -> Any:
    """Loads an object from a pickle file."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

# --- Text Processing and Document Handling ---

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Splits text into chunks of a specified size with overlap."""
    # A simple splitter, can be replaced with RecursiveCharacterTextSplitter if needed
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def ingest_files_from_folder(folder_path: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    """
    Reads all supported files from a folder, chunks them, and returns
    a list of LangChain Document objects.
    """
    documents = []
    try:
        for filepath in Path(folder_path).glob("*"):
            content = ""
            if filepath.suffix.lower() in [".txt", ".md"]:
                content = filepath.read_text(encoding='utf-8', errors='ignore')
                if filepath.suffix.lower() == ".md":
                    # Simple cleanup for Markdown
                    content = "\n".join(
                        line for line in content.splitlines()
                        if not line.strip().startswith("```")
                    )
            elif filepath.suffix.lower() == ".json":
                try:
                    data = json.loads(filepath.read_text(encoding='utf-8'))
                    # Simple heuristic to find long text fields in JSON
                    def find_text(obj):
                        texts = []
                        if isinstance(obj, str) and len(obj) > 100:
                            texts.append(obj)
                        elif isinstance(obj, dict):
                            for v in obj.values():
                                texts.extend(find_text(v))
                        elif isinstance(obj, list):
                            for v in obj:
                                texts.extend(find_text(v))
                        return texts
                    content = "\n".join(find_text(data))
                except Exception:
                    continue  # Skip malformed JSON

            if content:
                chunks = chunk_text(content, chunk_size, chunk_overlap)
                for i, chunk_text in enumerate(chunks):
                    metadata = {"source": str(filepath.name), "chunk_id": i}
                    doc = Document(page_content=chunk_text, metadata=metadata)
                    documents.append(doc)
    except Exception as e:
        print(f"Warning: Error processing documents in {folder_path}: {e}")
    
    return documents

def tokenize(text: str) -> List[str]:
    """Tokenizes text for BM25."""
    try:
        return word_tokenize(text.lower())
    except Exception:
        # Fallback to simple splitting if NLTK is not available
        return text.lower().split()
