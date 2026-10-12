"""
cli.py - RAG-lab CLI 명령줄 인터페이스
"""

import sys
import argparse
from pathlib import Path

# Windows UTF-8 출력 보장
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from rag_lab.config import DOCS_DATA_DIR, CHROMA_DIR, OLLAMA_MODEL, DEFAULT_TOP_K
from rag_lab.document_loader import load_documents_from_directory
from rag_lab.text_splitter import split_documents
from rag_lab.vector_store import VectorStoreManager
from rag_lab.llm import check_ollama_status
from rag_lab.rag_chain import RAGPipeline


def cmd_status():
    """ChromaDB 및 Ollama 로컬 LLM 연결 상태를 점검합니다."""
    print("\n" + "=" * 60)
    print("📊 [RAG-lab 시스템 상태 점검]")
    print("=" * 60)

    # 1. 벡터 스토어 상태
    v_mgr = VectorStoreManager()
    count = v_mgr.count()
    print(f"📁 저장소 경로: {CHROMA_DIR}")
    print(f"📦 인덱싱된 청크(Chunk) 수: {count}개")
    if count == 0:
        print("   ⚠️ 저장된 문서가 없습니다. 'uv run rag-lab index' 명령으로 문서를 인덱싱하세요.")

    # 2. 로컬 LLM (Ollama) 상태
    is_running, models, msg = check_ollama_status()
    print("-" * 60)
    print(f"🤖 로컬 LLM 서버: {'🟢 연결됨' if is_running else '🔴 연결 불가'}")
    print(f"   상태 메시지: {msg}")
    if is_running:
        print(f"   설치된 모델 목록: {', '.join(models) if models else '없음'}")
        if any(OLLAMA_MODEL in m for m in models):
            print(f"   ✅ 기본 지정 모델 ({OLLAMA_MODEL}) 준비 완료")
        else:
            print(f"   ⚠️ 기본 지정 모델 ({OLLAMA_MODEL})이 목록에 없습니다.")
            print(f"      'ollama run {OLLAMA_MODEL}' 명령으로 모델을 다운로드하세요.")
    else:
        print(f"   💡 모델 실행 방법: 별도 터미널에서 'ollama run {OLLAMA_MODEL}'")
    print("=" * 60 + "\n")


def cmd_index(dir_path: str = None):
    """지정된 디렉토리의 문서(TXT, MD, PDF)를 청킹 및 벡터화하여 ChromaDB에 적재합니다."""
    target_dir = Path(dir_path) if dir_path else DOCS_DATA_DIR
    print(f"\n📂 문서 인덱싱 시작: {target_dir}")

    # 1. 문서 로드
    raw_docs = load_documents_from_directory(target_dir)
    if not raw_docs:
        print("⚠️ 인덱싱할 문서(.txt, .md, .pdf)를 찾을 수 없습니다.")
        return

    print(f" -> {len(raw_docs)}개의 원본 문서(페이지)를 로드했습니다.")

    # 2. 청킹 (Chunking)
    chunks = split_documents(raw_docs)
    print(f" -> 문서를 총 {len(chunks)}개의 의미 단위 청크(Chunk)로 분할했습니다.")

    # 3. 벡터 DB 저장
    v_mgr = VectorStoreManager()
    total_added = v_mgr.add_documents(chunks)
    print(f"✅ ChromaDB 벡터 저장소에 {total_added}개 청크 임베딩 및 저장 완료!")
    print(f"   현재 전체 저장된 청크 수: {v_mgr.count()}개\n")


def cmd_search(query: str, top_k: int = DEFAULT_TOP_K):
    """순수 벡터 유사도 검색(Retrieval)만 수행하여 가장 가까운 청크를 출력합니다."""
    print(f"\n🔍 [벡터 검색] 질문: \"{query}\"")
    v_mgr = VectorStoreManager()
    if v_mgr.count() == 0:
        print("⚠️ 벡터 저장소가 비어 있습니다. 먼저 'uv run rag-lab index'를 실행하세요.")
        return

    results = v_mgr.search(query, top_k=top_k)
    print(f"📊 검색 결과 (Top {len(results)}):\n")
    for rank, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        page_str = f" (p.{page})" if page else ""
        print(f"[{rank}위] 유사도: {score:.4f} | 출처: {source}{page_str}")
        print(f"내용:\n{doc.page_content.strip()}")
        print("-" * 50)
    print()


def cmd_ask(query: str, top_k: int = DEFAULT_TOP_K):
    """검색 + Gemma 생성 파이프라인을 실행하고 결과를 실시간 스트리밍합니다."""
    print(f"\n💬 [질문]: {query}\n")
    rag = RAGPipeline()
    retrieved_docs, stream = rag.query(query, top_k=top_k)

    print("📄 [참고 문맥 출처]:")
    for rank, (doc, score) in enumerate(retrieved_docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        page_str = f", {page}페이지" if page else ""
        print(f"  - #{rank} {source}{page_str} (유사도: {score:.4f})")
    print("\n💡 [답변 생성 중...]:\n")

    for token in stream:
        sys.stdout.write(token)
        sys.stdout.flush()
    print("\n" + "=" * 60 + "\n")


def cmd_chat():
    """대화형 CLI 인터페이스를 실행합니다."""
    print("\n" + "=" * 60)
    print("🤖 [RAG-lab 대화형 질의응답 콘솔]")
    print("   종료하려면 'exit', 'quit' 또는 'q'를 입력하세요.")
    print("=" * 60)

    rag = RAGPipeline()
    while True:
        try:
            query = input("\n질문을 입력하세요 > ").strip()
            if not query:
                continue
            if query.lower() in {"exit", "quit", "q", ":q"}:
                print("👋 대화를 종료합니다.")
                break
            cmd_ask(query)
        except (KeyboardInterrupt, EOFError):
            print("\n👋 대화를 종료합니다.")
            break


def cmd_reset():
    """ChromaDB 컬렉션을 초기화합니다."""
    v_mgr = VectorStoreManager()
    v_mgr.clear()
    print("🗑️ ChromaDB 컬렉션이 초기화되었습니다.")


def main():
    parser = argparse.ArgumentParser(description="RAG-lab: 로컬 LLM & 벡터 기반 RAG 시스템")
    subparsers = parser.add_subparsers(dest="command", help="실행할 서브 명령")

    # index
    p_index = subparsers.add_parser("index", help="문서 폴더를 스캔하여 벡터 DB에 인덱싱")
    p_index.add_argument("--dir", type=str, default=None, help="문서 폴더 경로 (기본값: docs/data)")

    # search
    p_search = subparsers.add_parser("search", help="벡터 유사도 검색 (Retrieval 테스트)")
    p_search.add_argument("query", type=str, help="검색할 질문 또는 키워드")
    p_search.add_argument("-k", "--top-k", type=int, default=DEFAULT_TOP_K, help="반환할 상위 청크 수")

    # ask
    p_ask = subparsers.add_parser("ask", help="RAG 질의응답 (검색 + Gemma 생성)")
    p_ask.add_argument("query", type=str, help="질문 내용")
    p_ask.add_argument("-k", "--top-k", type=int, default=DEFAULT_TOP_K, help="반환할 상위 청크 수")

    # chat
    subparsers.add_parser("chat", help="대화형 질의응답 모드")

    # status
    subparsers.add_parser("status", help="시스템 상태(DB 및 Ollama) 점검")

    # reset
    subparsers.add_parser("reset", help="ChromaDB 인덱스 데이터 삭제")

    args = parser.parse_args()

    if args.command == "status":
        cmd_status()
    elif args.command == "index":
        cmd_index(args.dir)
    elif args.command == "search":
        cmd_search(args.query, top_k=args.top_k)
    elif args.command == "ask":
        cmd_ask(args.query, top_k=args.top_k)
    elif args.command == "chat":
        cmd_chat()
    elif args.command == "reset":
        cmd_reset()
    else:
        # 인자 없이 실행 시 도움말과 상태 출력
        parser.print_help()
        cmd_status()


if __name__ == "__main__":
    main()
