"""
llm.py - Ollama 및 로컬 Gemma 모델 연동 모듈
"""

import httpx
from typing import Tuple, List, Optional
from langchain_ollama import ChatOllama
from rag_lab.config import OLLAMA_BASE_URL, OLLAMA_MODEL


def check_ollama_status(base_url: str = OLLAMA_BASE_URL) -> Tuple[bool, List[str], str]:
    """Ollama 서버 구동 여부 및 다운로드된 모델 목록을 확인합니다.
    
    Returns:
        (is_running, model_names, status_message)
    """
    try:
        with httpx.Client(timeout=2.0) as client:
            resp = client.get(f"{base_url}/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                return True, models, "Ollama 서버가 정상 실행 중입니다."
            else:
                return False, [], f"Ollama 응답 코드 비정상: {resp.status_code}"
    except httpx.ConnectError:
        return False, [], f"Ollama 서버({base_url})에 연결할 수 없습니다. (서버 미실행)"
    except Exception as e:
        return False, [], f"Ollama 상태 확인 중 오류: {e}"


def get_llm(
    model: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    temperature: float = 0.1,
) -> Optional[ChatOllama]:
    """ChatOllama 인스턴스를 반환합니다. 서버가 꺼져 있으면 None을 반환합니다."""
    is_running, models, msg = check_ollama_status(base_url)
    if not is_running:
        return None

    return ChatOllama(
        base_url=base_url,
        model=model,
        temperature=temperature,
    )
