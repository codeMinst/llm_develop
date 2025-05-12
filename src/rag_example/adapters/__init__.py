"""
문서 처리 어댑터 패키지

이 패키지는 다양한 문서 형식을 처리하기 위한 어댑터 패턴을 구현합니다.
주요 기능:
- 문서 형식별 어댑터 제공 (PDF 등)
- 텍스트 추출 및 처리 기능
- 팩토리 패턴을 통한 어댑터 생성
"""


# 기본 인터페이스
from .base.base import DocumentAdapter

# 어댑터
from .document.pdf import PDFAdapter, TextExtractor

# 어댑터 팩토리
from .factory import get_document_adapter, DocumentAdapterError