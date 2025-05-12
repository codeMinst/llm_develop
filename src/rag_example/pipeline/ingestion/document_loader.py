"""
문서 로딩 및 전처리 모듈

이 모듈은 다양한 형식의 문서를 로드하고 전처리하는 기능을 제공합니다.
주요 기능:
- 문서 파일 탐색 및 로드
- 텍스트 추출 및 정제
- 문서 청크

설계 철학:
- 이 모듈은 다양한 문서 형식을 처리하기 위해 어댑터 패턴을 활용합니다.
- 문서 형식별 처리 로직은 어댑터로 캡슐화하여 확장성을 확보합니다.
- DocumentLoader 클래스는 어댑터를 사용하는 클라이언트 역할을 합니다.
"""
import logging
import os
import time
from pathlib import Path
from typing import List

from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_example.adapters.factory import get_document_adapter, DocumentAdapterError
from rag_example.config.settings import PRE_PROC_DIR, CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_SEPARATORS

logger = logging.getLogger(__name__)

class DocumentLoader:
    """
    문서 로딩 및 전처리를 담당하는 클래스
    
    주요 기능:
    - 문서 파일 탐색
    - 다양한 형식의 문서 로드 (PDF, DOCX 등)
    - 텍스트 추출 및 정제
    - 문서 청크
    
    설계 의도:
    - 이 클래스는 어댑터 패턴을 활용하여 다양한 문서 형식을 처리합니다.
    - 문서 형식별 세부 처리 로직은 각 어댑터에 위임하고, 이 클래스는 고수준 오케스트레이션에 집중합니다.
    - 파이프라인 컴포넌트인 DocumentLoader는 어댑터를 사용하는 클라이언트 역할을 하며, 세부 구현은 어댑터에 위임합니다.
    """
    
    def __init__(self, document_dir: str):
        """
        DocumentLoader 초기화
        
        Args:
            document_dir: 문서 파일이 있는 디렉토리 경로
        """
        self.document_dir = Path(document_dir)
        self.pre_proc_dir = Path(PRE_PROC_DIR)
        
        # 출력 디렉토리 생성
        os.makedirs(self.pre_proc_dir, exist_ok=True)
    
    def get_document_files(self) -> List[Path]:
        """
        지정된 디렉토리에서 지원되는 문서 파일 목록을 가져옵니다.
        
        Returns:
            지원되는 문서 파일 경로 목록
        """
        # 지원되는 파일 확장자
        supported_extensions = ['.pdf', '.txt', '.docx']
        
        # 문서 디렉토리가 존재하는지 확인
        if not self.document_dir.exists():
            logger.error(f"문서 디렉토리가 존재하지 않습니다: {self.document_dir}")
            return []
        
        # 지원되는 확장자를 가진 파일만 필터링
        document_files = []
        for file_path in self.document_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                document_files.append(file_path)
        
        # 결과 로깅
        logger.info(f"처리할 문서 파일: {len(document_files)}개")
        for file_path in document_files:
            logger.info(f"  - {file_path.name}")
        
        return document_files
    
    def load_documents(self) -> List[Document]:
        """
        문서 파일을 로드하고 텍스트를 추출합니다.
        
        Returns:
            로드된 문서 리스트
            
        설계 참고:
            이 메서드는 어댑터 패턴을 활용하여 다양한 문서 형식을 처리합니다.
            파일 확장자에 따라 적절한 어댑터를 생성하고 사용하는 방식으로,
            새로운 문서 형식을 추가할 때 이 메서드를 수정할 필요 없이
            어댑터만 추가하면 됩니다.
        """
        document_files = self.get_document_files()
        documents = []
        
        pdf_count = 0
        other_count = 0
        
        for file_path in document_files:
            file_ext = file_path.suffix.lower()
            
            try:
                # 어댑터 팩토리를 통해 적절한 어댑터 생성
                logger.info(f"문서를 처리합니다: {file_path.name} (형식: {file_ext})")
                document_adapter = get_document_adapter(file_ext, str(file_path))
                
                # 어댑터를 통해 문서 처리
                processed_text = document_adapter.run()
                
                # 파일 형식에 따라 카운트 증가
                if file_ext == '.pdf':
                    pdf_count += 1
                else:
                    other_count += 1
                
                # Document 객체 생성
                metadata = {
                    "source": str(file_path),
                    "file_type": file_ext.lstrip('.'),
                    "file_name": file_path.name
                }
                
                document = Document(page_content=processed_text, metadata=metadata)
                documents.append(document)
                
            except DocumentAdapterError as e:
                logger.warning(f"문서 처리 중 오류 발생: {str(e)}")
                continue
            except Exception as e:
                logger.error(f"문서 처리 중 예기치 않은 오류 발생: {str(e)}")
                continue
        
        logger.info(f"로드된 문서: PDF {pdf_count}개, 기타 {other_count}개 (총 {len(documents)}개)")
        return documents
    
    # 이 위치에는 원래 문서 형식별 처리 메서드가 있었을 것입니다.
    # 예: process_pdf_document(), process_docx_document(), process_txt_document() 등
    # 
    # 어댑터 패턴을 도입하면서 이런 메서드들이 제거되었습니다. 
    # 대신 load_documents() 메서드에서 어댑터 패턴을 사용하여 문서를 처리합니다.
    # 이는 어댑터 패턴의 장점을 보여주는 예시로,
    # 새로운 문서 형식을 추가할 때 DocumentLoader 클래스를 수정할 필요 없이 새 어댑터만 추가하면 됩니다.
    
    def create_chunks(self, 
                     documents: List[Document], 
                     chunk_size: int = CHUNK_SIZE,
                     chunk_overlap: int = CHUNK_OVERLAP) -> List[Document]:
        """
        문서를 청크로 분할합니다.
        
        Args:
            documents: 분할할 문서 리스트
            chunk_size: 청크 크기
            chunk_overlap: 청크 간 겹침 크기
            
        Returns:
            분할된 청크 리스트
            
        설계 참고:
            이 메서드는 LangChain의 RecursiveCharacterTextSplitter를 사용하여 문서 청크를 생성합니다.
            새로운 청크 전략이 필요한 경우 이 메서드를 수정하거나 전략 패턴을 적용하여
            다양한 청크 전략을 구현할 수 있습니다.
        """
        if not documents:
            logger.warning("청크로 분할할 문서가 없습니다.")
            return []
        
        start_time = time.time()
        
        # 텍스트 분할기 초기화
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=CHUNK_SEPARATORS
        )
        
        # 문서 분할
        all_chunks = []
        for doc in documents:
            chunks = text_splitter.split_documents([doc])
            all_chunks.extend(chunks)
        
        # 청크 처리 시간 및 결과 로깅
        processing_time = time.time() - start_time
        logger.info(f"문서 청크 분할 완료: 총 {len(all_chunks)}개의 청크 생성 (처리 시간: {processing_time:.5f}초)")
        
        return all_chunks
