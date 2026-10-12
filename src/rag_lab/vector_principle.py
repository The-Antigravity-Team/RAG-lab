"""
vector_principle.py
텍스트 벡터화(Text Embedding)와 코사인 유사도(Cosine Similarity)의 원리를 이해하기 위한 실습 스크립트
"""

import sys
import numpy as np
import chromadb.utils.embedding_functions as embedding_functions

# Windows 콘솔 한글 및 이모지 출력 인코딩 설정
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """두 벡터 간의 코사인 유사도(Cosine Similarity)를 계산합니다.
    
    공식: cos(theta) = (A · B) / (||A|| * ||B||)
    - 1.0에 가까울수록: 의미가 매우 유사함 (같은 방향)
    - 0.0에 가까울수록: 의미적 연관이 없음 (직교)
    - -1.0에 가까울수록: 반대 의미 (반대 방향)
    """
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))


def main():
    print("=" * 70)
    print("🔬 [RAG-lab] 텍스트 벡터화(Embedding) & 유사도 계산 원리 실습")
    print("=" * 70)

    # 1. 임베딩 모델 준비 (ChromaDB 기본 내장 초경량 ONNX 모델: all-MiniLM-L6-v2)
    print("\n[Step 1] 임베딩 모델 로드 중...")
    embed_fn = embedding_functions.DefaultEmbeddingFunction()

    # 2. 테스트할 샘플 문장 목록
    sentences = [
        "강아지가 마당에서 즐겁게 뛰어놀고 있습니다.",         # A: 동물/일상
        "귀여운 개가 정원에서 신나게 달리고 있어요.",           # B: A와 단어는 다르지만 의미가 거의 같음
        "스마트폰 충전기를 콘센트에 연결하여 충전하세요.",     # C: 전자제품 (A, B와 무관)
        "대규모 언어 모델(LLM)을 이용해 문서를 검색합니다.",   # D: AI/RAG 기술
        "인공지능 딥러닝 기술로 텍스트 데이터를 분석합니다.",   # E: D와 유사한 기술 도메인
    ]

    print("\n[Step 2] 샘플 문장 목록:")
    for idx, s in enumerate(sentences):
        print(f"  [{idx + 1}] {s}")

    # 3. 텍스트 -> 벡터(숫자 배열) 변환
    print("\n[Step 3] 텍스트를 고차원 벡터로 변환(Embedding) 중...")
    embeddings = embed_fn(sentences)
    vectors = [np.array(emb, dtype=np.float32) for emb in embeddings]

    vector_dim = len(vectors[0])
    print(f" -> 각 문장이 [{vector_dim}]차원의 실수 벡터로 변환되었습니다.")
    print(f" -> [문장 1]의 벡터 앞부분 5개 값 샘플: {vectors[0][:5].round(4)}")

    # 4. 코사인 유사도 행렬 계산
    print("\n[Step 4] 문장 간 코사인 유사도(Cosine Similarity) 비교 행렬:")
    print("      " + " ".join([f"[{i+1}]   " for i in range(len(sentences))]))
    
    matrix = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        row_str = f"[{i+1}]  "
        for j in range(len(sentences)):
            sim = cosine_similarity(vectors[i], vectors[j])
            matrix[i][j] = sim
            row_str += f"{sim:6.3f} "
        print(row_str)

    # 5. 핵심 인사이트 설명
    print("\n" + "=" * 70)
    print("💡 핵심 관찰 포인트:")
    print(f"1. [동일 의미 다른 표현] [1]번과 [2]번 유사도: {matrix[0][1]:.3f} (높음)")
    print("   -> '강아지/마당/뛰어놀다'와 '개/정원/달리다'는 단어 형태는 다르지만")
    print("      벡터 공간에서 매우 가까운 위치에 배치됨을 알 수 있습니다.")
    
    print(f"2. [무관한 도메인] [1]번과 [3]번 유사도: {matrix[0][2]:.3f} (낮음)")
    print("   -> '강아지' 이야기와 '스마트폰 충전' 이야기는 의미적 거리가 멉니다.")
    
    print(f"3. [유사 도메인] [4]번과 [5]번 유사도: {matrix[3][4]:.3f} (높음)")
    print("   -> AI/LLM 관련 기술 문장들끼리 높은 유사도를 형성합니다.")
    print("=" * 70)


if __name__ == "__main__":
    main()
