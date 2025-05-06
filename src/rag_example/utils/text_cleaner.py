"""
텍스트 정제를 위한 유틸리티 모듈입니다.
"""
import re
import logging
from typing import Optional

# 로깅 설정
logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """
    텍스트를 정제하여 임베딩 모델에서 처리할 수 있는 형태로 변환합니다.
    
    Args:
        text: 정제할 텍스트
        
    Returns:
        정제된 텍스트
    """
    if not text or not isinstance(text, str):
        return "문서 내용 없음"
    
    try:
        # 제어 문자 제거
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
        
        # 깨진 문자 제거
        text = re.sub(r'[\x7f-\x9f]', '', text)
        
        # 여러 개의 공백을 하나로 축소
        text = re.sub(r'\s+', ' ', text)
        
        # 양쪽 공백 제거
        text = text.strip()
        
        # 한글과 영문 사이에 공백 추가
        text = re.sub(r'([\uac00-\ud7a3])([a-zA-Z])', r'\1 \2', text)
        text = re.sub(r'([a-zA-Z])([\uac00-\ud7a3])', r'\1 \2', text)
        
        # 빈 문자열이면 기본 텍스트 반환
        if not text:
            return "문서 내용 없음"
        
        # 텍스트가 너무 짧으면 의미 있는 문서가 아닐 수 있음
        if len(text.strip()) < 5:
            return "의미 있는 문서 내용 없음"
            
        return text
        
    except Exception as e:
        logger.error(f"텍스트 정제 오류: {str(e)}")
        # 오류 발생 시 원본 텍스트 그대로 반환
        return text if text else "문서 내용 없음"
