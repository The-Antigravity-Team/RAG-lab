# 🧪 RAG-lab

로컬 LLM(Gemma)과 오픈소스 도구를 기반으로 한 RAG(Retrieval-Augmented Generation, 검색 증강 생성) 시스템 연구 및 실습 랩입니다.

---

## 📚 학습 문서 (Study Guides)
RAG의 각 핵심 원리는 [`docs/study/`](docs/study/) 디렉토리에 정리되어 있습니다:
- [01. RAG 개요 및 필요성](docs/study/01_rag_overview.md)
- [02. 텍스트 벡터화와 임베딩 원리](docs/study/02_text_vectorization_and_embedding.md)
- [03. 청킹(Chunking) 전략 및 문서 분할](docs/study/03_chunking_strategies.md)
- [04. 벡터 데이터베이스(ChromaDB)와 검색 원리](docs/study/04_vector_db_and_retrieval.md)
- [05. 로컬 LLM(Gemma) 연동과 RAG 증강 생성](docs/study/05_local_llm_and_gemma_rag.md)

---

## 🚀 빠른 시작 (Quick Start with `uv`)

### 1. 패키지 동기화 및 환경 준비
```bash
uv sync
```

### 2. 텍스트 벡터화 원리 실습 스크립트 실행
```bash
# 문장 벡터화 및 코사인 유사도 계산 원리 체험
uv run python src/rag_lab/vector_principle.py
```

### 3. RAG-lab CLI 도구 사용법
```bash
# 시스템 상태 확인 (ChromaDB 데이터 수, Ollama 연결 여부)
uv run rag-lab status

# docs/data 폴더 내의 모든 문서(TXT, MD, PDF) 벡터화 및 인덱싱
uv run rag-lab index

# 벡터 유사도 검색만 테스트 (Top-K Chunks)
uv run rag-lab search "Gemma 모델의 어텐션 특징"

# 엔드투엔드 RAG 질의응답 (검색 + Gemma 스트리밍 생성)
uv run rag-lab ask "RAG 시스템에서 청킹 오버랩을 두는 이유는?"

# 대화형 CLI 챗봇 실행
uv run rag-lab chat
```

---

## 🤖 로컬 Gemma 모델 구동 방법
Gemma를 로컬에서 구동하여 완전한 답변 생성을 연동하려면 [Ollama](https://ollama.com)를 설치한 후 터미널에서 다음 명령어를 실행합니다:

```bash
ollama run gemma2:2b
```
*(Ollama 서버가 켜져 있으면 `uv run rag-lab ask` 또는 `uv run rag-lab chat`에서 실시간 스트리밍으로 답변이 출력됩니다.)*
