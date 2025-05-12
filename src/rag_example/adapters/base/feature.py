"""
문서의 특정 기능을 처리하는 프로세서의 인터페이스를 정의하는 모듈입니다.
"""
from abc import ABC, abstractmethod
from typing import Any

class DocumentFeatureProcessor(ABC):
    """
    문서의 특정 기능을 처리하는 프로세서의 인터페이스입니다.
    헤더/푸터 제거, 표 처리 등의 기능별 프로세서가 이 인터페이스를 구현합니다.
    """
    
    @abstractmethod
    def get_feature_name(self) -> str:
        """
        처리하는 기능의 이름을 반환합니다.
        
        Returns:
            기능 이름
        """
        pass
    
    @abstractmethod
    def run(self, content: Any) -> Any:
        """
        문서의 특정 기능을 처리합니다.
        
        Args:
            content: 처리할 내용
            
        Returns:
            처리된 내용
        """
        pass
