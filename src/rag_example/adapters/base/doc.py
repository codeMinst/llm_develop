"""
문서 처리기의 기본 인터페이스와 어댑터 기본 클래스를 정의하는 모듈입니다.
"""
from abc import ABC, abstractmethod

class DocumentAdapter(ABC):
    """
    문서 처리를 위한 어댑터의 기본 클래스입니다.
    모든 문서 어댑터는 이 클래스를 상속받아 구현해야 합니다.
    """
    @abstractmethod
    def run(self) -> str:
        """
        문서를 처리합니다.
        
        Returns:
            처리된 텍스트
        """
        pass
    
    @abstractmethod
    def supports(self, content_type: str) -> bool:
        """
        해당 어댑터가 특정 문서 유형을 지원하는지 확인합니다.
        
        Args:
            content_type: 문서 유형
            
        Returns:
            지원 여부
        """
        pass
