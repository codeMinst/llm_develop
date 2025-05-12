"""문서 처리 패키지입니다."""

# 기본 인터페이스
from rag_example.adapters.base.base import DocumentAdapter

# 어댑터
from rag_example.adapters.document.pdf import PDFAdapter, TextExtractor

# 어댑터 팩토리
from rag_example.adapters.factory import get_document_adapter, DocumentAdapterError