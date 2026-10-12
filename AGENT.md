# AGENT.md - RAG-lab 프로젝트 가이드

## 1. 프로젝트 개요 (Overview)
- **프로젝트명**: `RAG-lab`
- **목적**: 로컬 LLM(Gemma)과 오픈소스 도구를 기반으로 한 RAG(Retrieval-Augmented Generation) 시스템 설계, 원리 탐구 및 점진적 구현
- **기본 철학**:
  - 원리를 먼저 명확히 이해하고(텍스트 임베딩, 벡터 유사도 등) 단계별로 확장한다.
  - 외부 의존성과 복잡도를 최소화하여 빠르게 검증(Fast Verification) 가능한 형태로 개발한다.

---

## 2. 개발 환경 및 도구 (Tech Stack & Environment)
- **패키지/환경 관리자**: `uv` (빠른 패키지 해석 및 가상환경 격리)
- **Python 버전**: `Python >= 3.12`
- **핵심 프레임워크**:
  - `LangChain` (문서 로더, 청커, 검색기, 체인 파이프라인)
  - `ChromaDB` (로컬 파일 기반 벡터 데이터베이스)
  - `Sentence-Transformers` / `Ollama Embeddings` (텍스트 임베딩 모델)
- **추론 엔진 (LLM)**:
  - 로컬 API 기반 Google `Gemma` 모델 (Ollama 연동)

---

## 3. 실행 및 개발 규칙 (Conventions & Commands)

모든 파이썬 실행 및 패키지 관리는 항상 `uv`를 사용합니다:

```bash
# 의존성 패키지 추가
uv add <package-name>

# 스크립트 실행
uv run python <script_path.py>

# 프로젝트 동기화 / 환경 생성
uv sync
```

---

## 4. 단계별 마일스톤 (Milestones)

- [x] **Milestone 0: 프로젝트 환경 셋업 (`uv`) 및 설계 가이드 구축 (`AGENT.md`)**
- [x] **Milestone 1: 텍스트 벡터화(Embedding) 원리 이해 및 코사인 유사도 검증**
  - 문장이 고차원 벡터로 변환되는 과정 시각화 (`src/rag_lab/vector_principle.py`)
  - 문장 간 코사인 유사도(Cosine Similarity) 계산 및 원리 체득 (`src/rag_lab/vector_search_demo.py`)
  - 학습 문서 작성 완료 (`docs/study/02_text_vectorization_and_embedding.md`)
- [x] **Milestone 2: 벡터 DB(ChromaDB) 연동 및 텍스트 기반 Retrieval 구현**
  - 텍스트 문서 적재 및 인덱싱 (`src/rag_lab/vector_store.py`)
  - 계층적 청킹 파이프라인 구축 (`src/rag_lab/text_splitter.py`)
  - Top-K 유사도 기반 문맥(Context) 검색 파이프라인 구축
  - 학습 문서 작성 완료 (`docs/study/03_chunking_strategies.md`, `docs/study/04_vector_db_and_retrieval.md`)
- [x] **Milestone 3: 로컬 Gemma API (Ollama) 연동 및 엔드투엔드 RAG 완성**
  - 검색된 문맥 + 질문을 Gemma 프롬프트에 결합 (`src/rag_lab/rag_chain.py`)
  - 스트리밍 응답 출력 및 오프라인 예외 처리 지원 (`src/rag_lab/llm.py`)
  - 학습 문서 작성 완료 (`docs/study/01_rag_overview.md`, `docs/study/05_local_llm_and_gemma_rag.md`)
- [x] **Milestone 4: 기술 PDF 문서 확장 (표/코드/구조화 문서 처리)**
  - PyPDF 기반 페이지/메타데이터 추출 로더 구축 (`src/rag_lab/document_loader.py`)
  - 기술 PDF 샘플 생성 및 벡터 검색 검증 완료

---

## 5. RAG-lab CLI 명령어 요약

```bash
# 1. 시스템 상태 점검 (DB 데이터 수, Ollama 연결 여부)
uv run rag-lab status

# 2. docs/data/ 폴더 내 모든 문서(TXT, MD, PDF) 벡터화 및 인덱싱
uv run rag-lab index

# 3. 질문과 유사한 문서 청크 검색 (Top-K Retrieval만 테스트)
uv run rag-lab search "질문 내용"

# 4. RAG 엔드투엔드 질의응답 (검색 + Gemma 스트리밍 생성)
uv run rag-lab ask "질문 내용"

# 5. 대화형 터미널 챗봇 실행
uv run rag-lab chat

# 6. 벡터 인덱스 초기화
uv run rag-lab reset
```
