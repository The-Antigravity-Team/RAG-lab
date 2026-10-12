"""
document_loader.py - 텍스트(.txt, .md) 및 기술 PDF(.pdf) 문서 로더 모듈
"""

from pathlib import Path
from typing import List
from langchain_core.documents import Document
from pypdf import PdfReader


def load_text_file(file_path: Path) -> Document:
    """텍스트 파일(.txt, .md)을 UTF-8 또는 CP949 인코딩으로 안전하게 로드합니다."""
    encodings = ["utf-8", "utf-8-sig", "cp949"]
    content = None
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue

    if content is None:
        raise ValueError(f"파일을 읽을 수 없습니다 (인코딩 지원 실패): {file_path}")

    return Document(
        page_content=content,
        metadata={
            "source": str(file_path.name),
            "file_path": str(file_path),
            "file_type": file_path.suffix.lower().lstrip("."),
        },
    )


def load_pdf_file(file_path: Path) -> List[Document]:
    """기술 PDF 문서를 페이지별로 로드하여 메타데이터와 함께 Document 리스트로 반환합니다."""
    documents: List[Document] = []
    try:
        reader = PdfReader(str(file_path))
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": str(file_path.name),
                            "file_path": str(file_path),
                            "file_type": "pdf",
                            "page": page_num,
                            "total_pages": len(reader.pages),
                        },
                    )
                )
    except Exception as e:
        print(f"⚠️ PDF 로드 중 오류 발생 ({file_path.name}): {e}")

    return documents


def load_documents_from_directory(dir_path: Path) -> List[Document]:
    """지정된 디렉토리 내의 모든 지원 문서(.txt, .md, .pdf)를 재귀적으로 로드합니다."""
    documents: List[Document] = []
    supported_extensions = {".txt", ".md", ".pdf"}

    if not dir_path.exists():
        print(f"⚠️ 디렉토리가 존재하지 않습니다: {dir_path}")
        return documents

    for file_path in dir_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            ext = file_path.suffix.lower()
            try:
                if ext in {".txt", ".md"}:
                    doc = load_text_file(file_path)
                    documents.append(doc)
                elif ext == ".pdf":
                    pdf_docs = load_pdf_file(file_path)
                    documents.extend(pdf_docs)
            except Exception as e:
                print(f"⚠️ 문서 로드 실패 ({file_path.name}): {e}")

    return documents
