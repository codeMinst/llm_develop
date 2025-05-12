"""
문서 처리를 위한 기본 인터페이스와 타입을 정의하는 패키지입니다.
""" 
from rag_example.adapters.base.base import DocumentAdapter
from rag_example.adapters.base.feature import DocumentFeatureProcessor
from rag_example.adapters.base.types import DocumentType

__all__ = [
    'DocumentAdapter',
    'DocumentFeatureProcessor',
    'DocumentType',
]
