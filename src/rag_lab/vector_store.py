"""
vector_store.py - ChromaDB 로컬 벡터 저장소 및 인덱싱/검색 관리 모듈
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any
import chromadb
from chromadb.utils import embedding_functions
from langchain_core.documents import Document

from rag_lab.config import CHROMA_DIR, COLLECTION_NAME, DEFAULT_TOP_K


class VectorStoreManager:
    """ChromaDB 로컬 인스턴스를 관리하며 인덱싱 및 유사도 검색을 수행합니다."""

    def __init__(self, persist_dir: Path = CHROMA_DIR, collection_name: str = COLLECTION_NAME):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # ChromaDB Persistent Client 초기화
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))

        # 기본 내장 임베딩 함수 (all-MiniLM-L6-v2 ONNX)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()

        # 컬렉션 로드 또는 생성 (코사인 거리 사용)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, documents: List[Document]) -> int:
        """청킹된 문서들을 벡터로 변환하여 ChromaDB에 저장합니다."""
        if not documents:
            return 0

        ids: List[str] = []
        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for idx, doc in enumerate(documents):
            source = doc.metadata.get("source", "unknown")
            chunk_idx = doc.metadata.get("chunk_index", idx)
            # 고유한 ID 부여 (예: guide.txt_chunk_0)
            doc_id = f"{source}_chunk_{chunk_idx}_{idx}"

            ids.append(doc_id)
            texts.append(doc.page_content)
            # ChromaDB 메타데이터는 int, float, str, bool만 허용
            clean_meta = {
                k: v for k, v in doc.metadata.items()
                if isinstance(v, (int, float, str, bool))
            }
            metadatas.append(clean_meta)

        # 배치 추가
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end_idx = i + batch_size
            self.collection.upsert(
                ids=ids[i:end_idx],
                documents=texts[i:end_idx],
                metadatas=metadatas[i:end_idx],
            )

        return len(documents)

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Tuple[Document, float]]:
        """질문과 가장 유사한 상위 Top-K 문서 청크를 코사인 유사도 점수와 함께 반환합니다."""
        if self.count() == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )

        matched_docs: List[Tuple[Document, float]] = []
        if not results["documents"] or not results["documents"][0]:
            return matched_docs

        docs = results["documents"][0]
        metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
        distances = results["distances"][0] if results["distances"] else [1.0] * len(docs)

        for doc_text, meta, dist in zip(docs, metadatas, distances):
            # ChromaDB 코사인 거리는 dist = 1 - cosine_similarity 이므로
            # 유사도(similarity) = 1 - dist
            similarity = max(0.0, 1.0 - dist)
            doc = Document(page_content=doc_text, metadata=meta)
            matched_docs.append((doc, similarity))

        return matched_docs

    def count(self) -> int:
        """저장된 전체 벡터 청크 수를 반환합니다."""
        return self.collection.count()

    def clear(self) -> None:
        """기존 컬렉션을 완전히 비웁니다."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
