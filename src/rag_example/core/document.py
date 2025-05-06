"""
문서 로딩 및 처리를 담당하는 모듈입니다.
"""
import os
import logging
from typing import List, Dict, Optional
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    CSVLoader, 
    UnstructuredExcelLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader
)
from ..config.settings import CHUNK_SIZE, CHUNK_OVERLAP
from ..utils.text_cleaner import clean_text

# 로깅 설정
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """문서 처리를 위한 클래스입니다."""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        DocumentProcessor를 초기화합니다.
        
        Args:
            chunk_size: 문서 청크의 크기
            chunk_overlap: 청크 간 중복되는 문자 수
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_documents(self, file_path: str) -> List[Document]:
        """
        파일에서 문서를 로드합니다. 파일 확장자에 따라 적절한 로더를 사용합니다.
        
        Args:
            file_path: 로드할 파일 경로
            
        Returns:
            로드된 문서 리스트
        """
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()
        
        try:
            # 파일 형식에 따라 적절한 로더 선택
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension == '.csv':
                loader = CSVLoader(file_path, encoding='utf-8')
            elif file_extension in ['.xls', '.xlsx']:
                loader = UnstructuredExcelLoader(file_path)
            elif file_extension in ['.htm', '.html']:
                loader = UnstructuredHTMLLoader(file_path)
            elif file_extension in ['.md', '.markdown']:
                loader = UnstructuredMarkdownLoader(file_path)
            elif file_extension in ['.doc', '.docx']:
                loader = UnstructuredWordDocumentLoader(file_path)
            else:  # 기본값은 텍스트 파일로 간주
                loader = TextLoader(file_path, encoding='utf-8', autodetect_encoding=True)
            
            logger.info(f"파일 로드 중: {file_path} (형식: {file_extension})")
            documents = loader.load()
            
            # PDF 파일의 경우 추가 처리
            if file_extension == '.pdf':
                documents = self._post_process_pdf_documents(documents)
                
            return documents
            
        except Exception as e:
            logger.error(f"파일 로드 오류: {file_path} - {str(e)}")
            return []  # 오류 발생 시 빈 리스트 반환

    def _post_process_pdf_documents(self, documents: List[Document]) -> List[Document]:
        """
        PDF에서 추출한 문서를 추가로 처리합니다.
        
        Args:
            documents: PDF에서 추출한 문서 리스트
            
        Returns:
            처리된 문서 리스트
        """
        processed_documents = []
        
        for doc in documents:
            if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                # 텍스트 추가 처리
                processed_text = clean_text(doc.page_content)
                
                # 새 문서 생성
                processed_doc = Document(
                    page_content=processed_text,
                    metadata=doc.metadata
                )
                processed_documents.append(processed_doc)
                
        return processed_documents


        
    def clean_documents(self, documents: List[Document]) -> List[Document]:
        """
        문서 리스트의 텍스트를 정제합니다.
        
        Args:
            documents: 정제할 문서 리스트
            
        Returns:
            정제된 문서 리스트
        """
        cleaned_documents = []
        
        for doc in documents:
            if hasattr(doc, 'page_content'):
                # 텍스트 정제
                cleaned_text = clean_text(doc.page_content)
                
                # 정제된 텍스트로 문서 생성
                cleaned_doc = Document(
                    page_content=cleaned_text,
                    metadata=doc.metadata if hasattr(doc, 'metadata') else {}
                )
                cleaned_documents.append(cleaned_doc)
        
        return cleaned_documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        문서를 청크로 분할합니다.
        
        Args:
            documents: 분할할 문서 리스트
            
        Returns:
            분할된 청크 리스트
        """
        # 문서가 없으면 빈 리스트 반환
        if not documents:
            return []
            
        return self.text_splitter.split_documents(documents)
        
    def load_and_split(self, file_path: str) -> List[Document]:
        """
        파일을 로드하고 청크로 분할하는 편의 메서드입니다.
        
        Args:
            file_path: 로드할 파일 경로
            
        Returns:
            분할된 청크 리스트
        """
        # 문서 로드
        documents = self.load_documents(file_path)
        
        # 문서 정제
        cleaned_documents = self.clean_documents(documents)
        
        # 문서 분할
        return self.split_documents(cleaned_documents)
