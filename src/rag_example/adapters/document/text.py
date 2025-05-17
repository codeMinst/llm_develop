"""
TXT 문서를 처리하는 어댑터 모듈입니다.
"""
import logging
from typing import Optional
from rag_example.utils.runner import Runner
from rag_example.adapters.base.doc import DocumentAdapter
from rag_example.adapters.base.feature import DocumentFeatureProcessor

# 로깅 설정
logger = logging.getLogger(__name__)

class TextExtractor(DocumentFeatureProcessor):
    """
    TXT 파일에서 텍스트를 추출하는 기능을 제공합니다.
    """
    def __init__(self, mode: str = "default"):
        super().__init__()
        self.mode = mode
        
    def run(self, file_path: str) -> str:
        """
        TXT 파일에서 텍스트를 추출합니다.
        
        Returns:
            추출된 텍스트
        """
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                logger.info(f"TXT 파일에서 텍스트 추출 완료: {file_path}")
                return text
        except Exception as e:
            logger.error(f"TXT 파일 읽기 실패: {file_path}, 오류: {str(e)}")
            return ""
    
    def get_feature_name(self) -> str:
        return "text"

class TextAdapter(DocumentAdapter):
    """
    TXT 문서를 처리하는 어댑터 클래스
    """
    def __init__(self, file_path: str, 
                 text_extractor: Runner, 
                 save_processed_text: Runner, 
                 output_dir: Optional[str] = None):
        super().__init__()
        self.file_path: str = file_path
        self.extractor = text_extractor
        self.output_dir = output_dir
        self.save_processed_text = save_processed_text
        
    def run(self) -> str:
        """
        TXT 파일을 처리하여 텍스트를 반환합니다.
        
        Args:
            file_path: 처리할 TXT 파일의 경로
            
        Returns:
            처리된 텍스트
        """
        processed_text = self.extractor.run(self.file_path)
        self.save_processed_text.run(self.file_path, self.output_dir, processed_text)
        
        return processed_text

    @staticmethod
    def supports(content_type: str) -> bool:
        """
        TEXT 문서 유형을 지원하는지 확인합니다.
        
        Args:
            content_type: 문서 유형
            
        Returns:
            TEXT 유형 지원 여부
        """
        return content_type.lower() == 'txt'