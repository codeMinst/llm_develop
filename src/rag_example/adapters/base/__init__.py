"""
문서 처리를 위한 기본 인터페이스와 타입 정의

이 패키지는 어댑터 패턴의 기본 구성 요소를 정의합니다.
주요 기능:
- 문서 어댑터 추상 클래스 정의
- 문서 기능 처리기 인터페이스 정의
- 문서 형식 타입 정의
"""
from .base import DocumentAdapter
from .feature import DocumentFeatureProcessor
from .types import DocumentType

__all__ = [
    'DocumentAdapter',
    'DocumentFeatureProcessor',
    'DocumentType',
]
