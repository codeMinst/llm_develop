"""
문서 수집 및 처리 모듈

이 패키지는 RAG 파이프라인의 문서 수집 및 처리 단계를 담당합니다.
주요 기능:
- 문서 파일 탐색 및 로드
- 텍스트 추출 및 정제
- 문서 청크 분할
"""
from .document_loader import DocumentLoader

__all__ = ['DocumentLoader']
