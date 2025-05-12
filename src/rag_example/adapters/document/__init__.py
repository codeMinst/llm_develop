"""
문서 처리 어댑터 구현

이 패키지는 다양한 문서 형식을 처리하기 위한 구체적인 어댑터를 구현합니다.
주요 기능:
- PDF 문서 처리를 위한 PDFAdapter 제공
- 텍스트 추출 및 처리 기능
- 헤더/푸터 제거 및 텍스트 정제 기능
"""

from .pdf import PDFAdapter

__all__ = [
    'PDFAdapter',
]
