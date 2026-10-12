"""
text_splitter.py - 텍스트 청킹(Chunking) 및 문서 분할 모듈
"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_lab.config import CHUNK_SIZE, CHUNK_OVERLAP


def split_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """문서 리스트를 의미 단위가 깨지지 않도록 계층적으로 분할(청킹)합니다."""
    # 문단(\n\n) -> 줄바꿈(\n) -> 마침표/공백(. ) -> 공백( ) 순서로 우선 분할
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunked_docs = splitter.split_documents(documents)

    # 청크별 고유 메타데이터(chunk_id 등) 부여
    for idx, doc in enumerate(chunked_docs):
        doc.metadata["chunk_index"] = idx
        doc.metadata["chunk_length"] = len(doc.page_content)

    return chunked_docs
