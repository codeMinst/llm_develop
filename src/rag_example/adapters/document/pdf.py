"""
PDF 문서를 처리하는 어댑터 모듈입니다.
"""
import logging
from pathlib import Path
from typing import Any, Optional

import fitz   # PyMuPDF

from rag_example.adapters.base.doc import DocumentAdapter
from rag_example.adapters.base.feature import DocumentFeatureProcessor
from rag_example.utils.runner import Runner

# 로깅 설정
logger = logging.getLogger(__name__)

class PDFExtractor(DocumentFeatureProcessor):
    """
    PDF 페이지에서 텍스트를 추출하는 기능을 제공합니다.
    볼드체 텍스트는 마크다운 `**굵게**` 형식으로 감쌉니다.
    """
    def __init__(self, mode: str = "dict", header_percent: float = 0.1, footer_percent: float = 0.1):
        super().__init__()
        self.mode = mode
        self.header_percent = header_percent
        self.footer_percent = footer_percent
        
    def run(self, page: Any) -> str:
        """
        페이지에서 텍스트를 추출합니다. 헤더/푸터 영역은 제외하고,
        볼드체 텍스트는 `**텍스트**`로 감쌉니다.
        
        Args:
            page: PyMuPDF 페이지 객체
            
        Returns:
            추출된 텍스트 (마크다운 볼드 포함)
        """
        text_dict = page.get_text(self.mode)
        page_height = page.rect.height
        header_zone = page_height * self.header_percent
        footer_zone = page_height * (1 - self.footer_percent)

        page_text = ""
        for block in text_dict.get("blocks", []):
            for line in block.get("lines", []):
                line_buffer = ""
                bold_buffer = ""
                is_in_bold = False

                for span in line.get("spans", []):
                    y0, y1 = span["bbox"][1], span["bbox"][3]
                    if y0 < header_zone or y1 > footer_zone:
                        continue

                    font_name = span.get("font", "").lower()
                    is_bold_span = "bold" in font_name or (span.get("flags", 0) & 2)
                    text_piece = span.get("text", "")

                    if is_bold_span:
                        is_in_bold = True
                        bold_buffer += text_piece
                    else:
                        if is_in_bold:
                            # 볼드 종료 → strip 후 감싸기
                            line_buffer += f"**{bold_buffer.strip()}**"
                            bold_buffer = ""
                            is_in_bold = False
                        line_buffer += text_piece

                # 줄 마지막에 볼드가 끝나지 않았으면 마무리
                if is_in_bold and bold_buffer:
                    line_buffer += f"**{bold_buffer.strip()}**"

                page_text += line_buffer + "\n"

        return page_text.strip()

    def get_feature_name(self) -> str:
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
                 pdf_extractor: Runner,
                 text_improve: Runner,
                 ollama_spacing: Runner,
                 save_processed_text: Runner,
                 output_dir: Optional[str] = None):
        super().__init__()
        self.file_path = file_path
        self.pdf_extractor = pdf_extractor
        self.text_improve = text_improve
        self.ollama_spacing = ollama_spacing
        self.save_processed_text = save_processed_text
        self.output_dir = output_dir
    
    def run(self) -> str:
        """
        PDF 파일을 처리하고, 페이지별로 제목과 페이지 번호를 포함한 텍스트를 생성합니다.
        
        Returns:
            처리된 전체 텍스트
        """
        
        try:
            # 파일 존재 확인
            file_path = Path(self.file_path)
            if not file_path.exists():
                logger.error(f"PDF 파일이 존재하지 않습니다: {file_path}")
                return "문서를 찾을 수 없습니다."

            logger.info(f"PDF 처리 시작: {file_path}")
            doc = fitz.open(file_path)
            total_pages = len(doc)
            logger.info(f"PDF 파일 페이지 수: {total_pages}")

            processed_text = ""
            file_stem = file_path.stem

            for page_num, page in enumerate(doc):
                logger.debug(f"페이지 {page_num+1} 텍스트 추출 시작")

                # 1. 텍스트 추출 + 전처리
                page_text = self.pdf_extractor.run(page).strip()
                page_text = self.ollama_spacing.run(page_text)

                # 2. 페이지 제목 삽입
                # header = f"### {file_stem} (페이지 {page_num + 1})\n"
                # page_text = f"{header}{page_text}\n\n"

                processed_text += page_text
                logger.info(f"페이지 {page_num+1} 처리 완료 (누적 길이: {len(processed_text)}자)")

            # 후처리 및 저장
            processed_text = self.text_improve.run(processed_text)
            self.save_processed_text.run(self.file_path, self.output_dir, processed_text)

            logger.info(f"PDF 처리 완료: {file_path} (최종 길이: {len(processed_text)}자)")
            return processed_text

        except Exception as e:
            logger.error(f"PDF 처리 중 오류 발생: {str(e)}")
            return f"PDF 처리 오류: {str(e)}"

    @staticmethod
    def supports(content_type: str) -> bool:
        """
        PDF 문서 유형을 지원하는지 확인합니다.
        
        Args:
            content_type: 문서 유형
            
        Returns:
            PDF 유형 지원 여부
        """
        return content_type.lower() == 'pdf'
