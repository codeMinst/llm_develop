"""
문서 처리 어댑터를 생성하는 팩토리 모듈입니다.
"""
from rag_example.adapters.base.doc import DocumentAdapter
from rag_example.adapters.document.pdf import PDFAdapter, PDFExtractor
from rag_example.adapters.document.text import TextAdapter, TextExtractor
from rag_example.utils.runner import Runner
from rag_example.utils.text_preproc import improve_text, ollama_spacing
from rag_example.utils.file_io import save_processed_text
from rag_example.config.settings import PRE_PROC_DIR

class DocumentAdapterError(Exception):
    """문서 어댑터 생성 중 발생하는 예외"""
    pass

def get_document_proc(file_extension: str, file_path: str) -> DocumentAdapter:
    """
    파일 확장자에 맞는 어댑터를 생성합니다.
    
    Args:
        file_extension: 파일 확장자 (예: '.pdf', '.docx', '.txt')
        file_path: 파일 경로
        
    Returns:
        문서 처리기 어댑터
        
    Raises:
        DocumentAdapterError: 지원하지 않는 파일 형식인 경우
    """
    # 파일 확장자가 점(.)으로 시작하는지 확인하고 소문자로 변환
    if not file_extension.startswith('.'):
        file_extension = '.' + file_extension
    file_extension = file_extension.lower()
    
    if file_extension == '.pdf':
        return PDFAdapter(
            file_path=file_path,
            pdf_extractor=Runner.wrap(PDFExtractor(mode="dict"), name="pdf_extractor"),
            text_improve=Runner.wrap(improve_text, name="text_improve"),
            ollama_spacing=Runner.wrap(False, name="ollama_spacing"),
            save_processed_text=Runner.wrap(save_processed_text, name="save_processed_text"),
            output_dir=PRE_PROC_DIR
        )
    if file_extension == '.txt':
        return TextAdapter(
            file_path=file_path,
            text_extractor=Runner.wrap(TextExtractor(mode="default"), name="text_extractor"),
            save_processed_text=Runner.wrap(save_processed_text, name="save_processed_text"),
            output_dir=PRE_PROC_DIR
        )    
    
    raise DocumentAdapterError(f"지원하지 않는 파일 형식입니다: {file_extension}")
    
    # 향후 다른 문서 유형 지원을 위한 확장 포인트
    # if file_extension == '.docx':
    #     return DocxAdapter()
