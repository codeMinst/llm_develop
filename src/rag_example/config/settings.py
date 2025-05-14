"""
RAG 시스템의 설정 값들을 관리하는 모듈입니다.
"""
from pathlib import Path
import os

# 경로 설정
BASE_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# 전처리 문서 출력 디렉토리
PRE_PROC_DIR = os.path.join(DATA_DIR, 'pre_proc')

# 모델 설정
LLM_TYPE = "claude"
# API 키 가져오기
# CLAUDE_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_KEY = os.getenv("API_KEY")
OLLAMA_MODEL = "llama3.1"
CLAUDE_MODEL = "claude-3-7-sonnet-20250219"  # Claude 모델 설정

# 임베딩 모델
# EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 기존 모델
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 다국어 지원 모델
# EMBEDDING_MODEL = "xlm-r-100langs-bert-base-nli-stsb-mean-tokens"

# 벡터 저장소 설정
VECTORSTORE_PATH = PROCESSED_DATA_DIR / "chroma_db"

# 청크 설정
CHUNK_SIZE = 800  # 한글 기준 800자
CHUNK_OVERLAP = 100
CHUNK_SEPARATORS = ["--block start--", "\n\n","\n", ".", " ", ""]  # 청크 분할 시 사용할 구분자

# 검색 설정
SEARCH_K = 3  # 검색할 문서 수


