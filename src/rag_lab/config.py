"""
config.py - RAG-lab 전역 설정 모듈
"""

import os
from pathlib import Path

# 기본 디렉토리 경로
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DOCS_DATA_DIR = BASE_DIR / "docs" / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

# 인덱싱 및 청킹 설정
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))
DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "3"))

# 벡터 저장소 설정
COLLECTION_NAME = "rag_lab_docs"

# 로컬 LLM (Ollama) 설정
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:2b")
