"""
문서 유형과 관련된 타입 정의를 포함하는 모듈입니다.
"""
from enum import Enum, auto

class DocumentType(Enum):
    """
    문서의 구조적 유형을 정의하는 열거형입니다.
    """
    STRUCTURED = auto()      # 구조화된 문서 (PDF, HTML 등)
    SEMI_STRUCTURED = auto() # 반구조화된 문서 (Markdown, CSV, JSON 등)
    UNSTRUCTURED = auto()    # 비구조화된 문서 (TXT, DOCX 등)
    
    @classmethod
    def from_extension(cls, extension: str) -> 'DocumentType':
        """
        파일 확장자를 기반으로 문서 유형을 반환합니다.
        
        Args:
            extension: 파일 확장자 (예: '.pdf', '.docx')
            
        Returns:
            문서 유형
        """
        extension = extension.lower()
        if extension.startswith('.'):
            extension = extension[1:]
            
        # 구조화된 문서
        if extension in ['pdf', 'html', 'htm', 'xml']:
            return cls.STRUCTURED
        # 반구조화된 문서
        elif extension in ['md', 'markdown', 'csv', 'json', 'yaml', 'yml']:
            return cls.SEMI_STRUCTURED
        # 비구조화된 문서
        else:
            return cls.UNSTRUCTURED
