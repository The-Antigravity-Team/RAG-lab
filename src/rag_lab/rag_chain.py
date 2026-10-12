"""
rag_chain.py - 검색 증강 생성(RAG) 파이프라인 및 스트리밍 체인 모듈
"""

import sys
from typing import Generator, List, Tuple
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rag_lab.vector_store import VectorStoreManager
from rag_lab.llm import get_llm, check_ollama_status
from rag_lab.config import OLLAMA_MODEL, DEFAULT_TOP_K

# Windows 콘솔 출력 인코딩 안전장치
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# RAG 기본 프롬프트 템플릿
RAG_SYSTEM_PROMPT = """당신은 주어진 참고 문맥만을 바탕으로 사실에 입각해 답변하는 RAG 전문 비서입니다.

[지침]:
1. 반드시 아래 [참고 문맥]에 제시된 내용만을 토대로 질문에 명확하고 친절하게 답변하세요.
2. [참고 문맥]에 질문에 대한 답이 명확히 나와 있지 않다면 추측하지 말고 "제공된 문서에서 관련 내용을 찾을 수 없습니다."라고 솔직하게 답변하세요.
3. 답변 끝에 참고한 출처 파일명(Source)을 언급해 주세요.

[참고 문맥]:
{context}

[사용자 질문]:
{question}

[답변]:"""


def format_context_docs(docs_with_scores: List[Tuple[Document, float]]) -> str:
    """검색된 문서 청크들을 출처 및 유사도 정보와 함께 프롬프트용 텍스트로 포맷팅합니다."""
    if not docs_with_scores:
        return "관련된 참고 문서를 찾지 못했습니다."

    formatted_parts = []
    for rank, (doc, score) in enumerate(docs_with_scores, start=1):
        source = doc.metadata.get("source", "알 수 없음")
        page = doc.metadata.get("page")
        page_info = f", {page}페이지" if page else ""
        part = (
            f"--- [참고자료 #{rank} | 출처: {source}{page_info} | 유사도: {score:.4f}] ---\n"
            f"{doc.page_content.strip()}"
        )
        formatted_parts.append(part)

    return "\n\n".join(formatted_parts)


class RAGPipeline:
    """검색(Retrieval)과 생성(Generation)을 결합한 통합 RAG 파이프라인"""

    def __init__(self, vector_store: VectorStoreManager = None):
        self.vector_store = vector_store or VectorStoreManager()
        self.prompt = ChatPromptTemplate.from_template(RAG_SYSTEM_PROMPT)

    def query(self, question: str, top_k: int = DEFAULT_TOP_K) -> Tuple[List[Tuple[Document, float]], Generator[str, None, None]]:
        """질문에 대해 관련 청크를 검색하고 답변 스트림을 생성합니다."""
        # 1. 문서 검색 (Retrieval)
        retrieved_docs = self.vector_store.search(question, top_k=top_k)
        context_text = format_context_docs(retrieved_docs)

        # 2. LLM 로드
        llm = get_llm()

        if llm is None:
            # Ollama 서버가 켜져 있지 않은 경우: 안내 메시지와 함께 검색된 컨텍스트와 프롬프트를 미리보기로 반환
            def fallback_stream():
                is_running, models, msg = check_ollama_status()
                yield "⚠️ [로컬 LLM (Ollama) 미실행 상태 안내]\n"
                yield f"{msg}\n"
                yield f"💡 실제 Gemma 모델을 통한 답변 생성을 원하시면 새 터미널에서 다음 명령어를 실행해 주세요:\n"
                yield f"   >>> ollama run {OLLAMA_MODEL}\n\n"
                yield "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                yield "🔎 [검색된 컨텍스트(Retrieval) 및 전송될 프롬프트 미리보기]:\n"
                yield f"{context_text}\n"
                yield "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"

            return retrieved_docs, fallback_stream()

        # 3. LLM이 준비된 경우: LangChain 체인 결합 및 스트리밍
        chain = self.prompt | llm | StrOutputParser()

        def response_stream():
            try:
                for chunk in chain.stream({"context": context_text, "question": question}):
                    yield chunk
            except Exception as e:
                yield f"\n⚠️ 답변 생성 중 오류 발생: {e}\n"

        return retrieved_docs, response_stream()
