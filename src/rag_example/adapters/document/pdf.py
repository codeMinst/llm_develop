"""
PDF 문서를 처리하는 어댑터 모듈입니다.
"""
import logging
import os
import re
from pathlib import Path
from typing import Any, Set, List, Optional, Dict, Tuple
from abc import ABC, abstractmethod

import fitz   # PyMuPDF

from rag_example.adapters.base.base import DocumentAdapter
from rag_example.adapters.base.feature import DocumentFeatureProcessor
from rag_example.utils.runner import Runner
from rag_example.config.settings import PRE_PROC_DIR

# 로깅 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # 로깅 레벨을 DEBUG로 설정

class TextExtractor(DocumentFeatureProcessor):
    """
    PDF 페이지에서 텍스트를 추출하는 기능을 제공합니다.
    """
    def __init__(self, mode: str, header_percent: float = 0.1, footer_percent: float = 0.1):
        super().__init__()
        self.mode = mode
        self.header_percent = header_percent
        self.footer_percent = footer_percent
        
    def run(self, page: Any) -> str:
        """
        페이지에서 텍스트를 추출합니다.
        헤더/푸터 영역은 제외합니다.
        
        Args:
            page: PyMuPDF 페이지 객체
            
        Returns:
            추출된 텍스트
        """
        # blocks 모드로 텍스트 추출
        blocks = page.get_text(self.mode)
        
        # 헤더/푸터 영역 계산
        page_height = page.rect.height
        header_zone = page_height * self.header_percent
        footer_zone = page_height * (1 - self.footer_percent)
        
        # 헤더/푸터 영역 외의 블록만 수집
        page_text = ""
        
        for block in blocks:
            # block 형식: (x0, y0, x1, y1, "텍스트", block_no, block_type)
            y0 = block[1]  # 블록 상단 y 좌표
            y1 = block[3]  # 블록 하단 y 좌표
            
            # 헤더/푸터 영역에 있는 블록은 제외
            if y0 < header_zone or y1 > footer_zone:
                continue
                
            # 텍스트 블록만 처리 (block_type == 0)
            if block[6] == 0:
                page_text += block[4] + "\n"

        return page_text

    def get_feature_name(self) -> str:
        """
        처리하는 기능의 이름을 반환합니다.
        
        Returns:
            기능 이름
        """
        return "text_extractor"

class PDFAdapter(DocumentAdapter):
    """
    PDF 문서를 처리하는 어댑터입니다.
    
    이 어댑터는 다음 기능을 지원합니다:
    - 블록 단위 텍스트 추출
    - 헤더/푸터 제거
    - 텍스트 개선
    - 처리된 텍스트 파일 저장
    """
    def __init__(self, 
                 file_path: str,
                 text_extractor: Runner,
                 text_improve: Runner,
                 ollama_spacing: Runner,
                 save_processed_text: Runner,
                 output_dir: Optional[str] = None):
        super().__init__()
        self.file_path = file_path
        self.text_extractor = text_extractor
        self.text_improve = text_improve
        self.ollama_spacing = ollama_spacing
        self.save_processed_text = save_processed_text
        self.output_dir = output_dir
    
    def run(self) -> str:
        """
        PDF 파일을 처리합니다.
        
        Returns:
            처리된 텍스트
        """
        try:
            # 파일 존재 여부 확인
            if not Path(self.file_path).exists():
                logger.error(f"PDF 파일이 존재하지 않습니다: {self.file_path}")
                return "문서를 찾을 수 없습니다."
            logger.info(f"PDF 처리 시작: {self.file_path}")
            
            # PDF 파일 열기
            doc = fitz.open(self.file_path)
            total_pages = len(doc)
            logger.info(f"PDF 파일 페이지 수: {total_pages}")
            
            # 각 페이지의 텍스트를 따로 추출하여 배열에 저장
            processed_text = ""
            page_start_tag = f"--block start--"
            for page_num, page in enumerate(doc):
                logger.debug(f"페이지 {page_num+1} 텍스트 추출 시작")   
                # 블록 단위로 텍스트 추출 (헤더/푸터 제거)
                page_text = self.text_extractor.run(page).strip()
                # Ollama를 사용한 한국어 띄어쓰기 교정 적용
                logger.info("Ollama를 사용한 한국어 띄어쓰기 교정 적용 중...")
                page_text = self.ollama_spacing.run(page_text)
                logger.info("Ollama를 사용한 한국어 띄어쓰기 교정 완료")
                # 페이지 시작 태그 추가 (태그 다음에 빈 행 추가)
                page_text = re.sub(r'\n{2,}', '\n', page_text)
                page_text += "\n\n"  
                page_text = f"{page_start_tag}\n{page_text}" 
                # 처리된 페이지 텍스트 추가
                processed_text += page_text
                logger.info(f"페이지 {page_num+1} 처리 완료 (길이: {len(processed_text)} 문자)")

            processed_text = self.text_improve.run(processed_text)      
            self.save_processed_text.run(self.file_path, self.output_dir, processed_text)

            logger.info(f"PDF 처리 완료: {self.file_path} (추출된 텍스트 길이: {len(processed_text)} 문자)")
            return processed_text
            
        except Exception as e:
            logger.error(f"PDF 처리 중 오류 발생: {str(e)}")
            return f"PDF 처리 오류: {str(e)}"
    
    def supports(self, content_type: str) -> bool:
        """
        PDF 문서 유형을 지원하는지 확인합니다.
        
        Args:
            content_type: 문서 유형
            
        Returns:
            PDF 유형 지원 여부
        """
        return content_type.lower() == 'pdf'
