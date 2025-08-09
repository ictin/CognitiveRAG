# CognitiveRAG/main.py
import sys
import os
import uvicorn

if __name__ == "__main__":
    # Ensure the parent directory is in sys.path for package import
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.argv = [
        "uvicorn",
        "CognitiveRAG.main_server:app",
        "--host", "127.0.0.1",
        "--port", "8080",
        "--reload"
    ]
    uvicorn.main()
