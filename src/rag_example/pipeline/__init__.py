"""
파이프라인 패키지 초기화 모듈

이 패키지는 RAG 시스템의 전체 파이프라인을 관리합니다.
파이프라인은 다음과 같은 세 가지 주요 단계로 구성됩니다:
- 수집(ingestion): 문서 로딩 및 청크 분할
- 인덱싱(indexing): 벡터 저장소 생성 및 관리
- 질의(querying): 질문 처리 및 응답 생성
"""
# 절대 경로 임포트 사용
from src.rag_example.pipeline.rag_pipeline import RAGPipeline

__all__ = [
    'RAGPipeline'
]