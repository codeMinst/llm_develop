"""
RAG 시스템의 설정 값들을 관리하는 모듈입니다.
"""
from pathlib import Path

# 경로 설정
BASE_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# 모델 설정
OLLAMA_MODEL = "llama3.1"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# 벡터 저장소 설정
VECTORSTORE_PATH = PROCESSED_DATA_DIR / "chroma_db"

# 청크 설정
CHUNK_SIZE = 800  # 한글 기준 800자
CHUNK_OVERLAP = 100

# 검색 설정
SEARCH_K = 3  # 검색할 문서 수
