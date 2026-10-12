"""
vector_search_demo.py
사용자 질문(Query)과 여러 텍스트 조각(Chunks) 간의 유사도를 계산하여
가장 관련 있는 문서를 찾아내는 '벡터 검색(Vector Retrieval)'의 원리 실습
"""

import sys
import numpy as np
import chromadb.utils.embedding_functions as embedding_functions

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))


def search_documents(query: str, documents: list[str], top_k: int = 3):
    embed_fn = embedding_functions.DefaultEmbeddingFunction()

    # 1. 문서 청크들과 질문을 모두 벡터로 변환
    doc_embeddings = embed_fn(documents)
    query_embedding = embed_fn([query])[0]

    query_vec = np.array(query_embedding, dtype=np.float32)

    # 2. 각 문서와 질문 사이의 코사인 유사도 계산
    scored_docs = []
    for idx, (doc, emb) in enumerate(zip(documents, doc_embeddings)):
        doc_vec = np.array(emb, dtype=np.float32)
        score = cosine_similarity(query_vec, doc_vec)
        scored_docs.append((score, idx, doc))

    # 3. 유사도 높은 순서로 정렬 (Top-K)
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    print(f"\n🔍 [검색 질문]: \"{query}\"")
    print(f"📊 [검색 결과 (Top {top_k})]:")
    for rank, (score, idx, doc) in enumerate(scored_docs[:top_k], 1):
        print(f"  {rank}위 (유사도: {score:.4f}) [문서 #{idx+1}]: {doc}")


def main():
    print("=" * 70)
    print("🔎 [RAG-lab] RAG의 첫 번째 단계: '벡터 검색(Retrieval)' 원리")
    print("=" * 70)

    # 저장된 가상의 지식 베이스(청크 문서들)
    knowledge_base = [
        "파이썬(Python)은 읽기 쉽고 간결한 문법을 가진 인기 있는 프로그래밍 언어입니다.",
        "골든 리트리버는 성격이 온순하고 사람을 잘 따르는 대형견 품종입니다.",
        "RAG(검색 증강 생성)는 LLM이 외부 지식을 검색하여 신뢰성 높은 답변을 생성하도록 돕습니다.",
        "고양이는 영역 동물로 환경 변화에 민감하며 하루 평균 12~16시간을 잡니다.",
        "벡터 데이터베이스는 텍스트 임베딩을 저장하고 고속으로 유사한 벡터를 검색합니다.",
        "아침 사과는 비타민과 유기산이 풍부하여 건강에 좋은 과일입니다."
    ]

    print(f"\n📚 현재 등록된 문서 지식 베이스 ({len(knowledge_base)}건):")
    for i, doc in enumerate(knowledge_base, 1):
        print(f"  [{i}] {doc}")

    # 테스트 1: 키워드가 정확히 일치하지 않아도 의미로 찾아내는지 검증
    # '개'나 '리트리버'라는 단어 대신 '반려견'으로 질문
    search_documents(
        query="사람을 좋아하는 순한 성격의 반려견 품종은 무엇인가요?",
        documents=knowledge_base,
        top_k=2
    )

    # 테스트 2: RAG 및 벡터 관련 질문
    search_documents(
        query="인공지능 모델이 거짓말(환각)을 하지 않게 외부 문서를 참조하는 방법",
        documents=knowledge_base,
        top_k=2
    )


if __name__ == "__main__":
    main()
