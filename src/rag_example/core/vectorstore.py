"""
벡터 저장소 관리를 담당하는 모듈입니다.
"""
import logging
from typing import List, Dict, Optional
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from ..config.settings import VECTORSTORE_PATH, EMBEDDING_MODEL
from ..utils.text_cleaner import clean_text

# 로깅 설정
logger = logging.getLogger(__name__)

class VectorStoreManager:
    """벡터 저장소 관리를 위한 클래스입니다."""
    
    def __init__(self) -> None:
        """
        VectorStoreManager를 초기화합니다.
        """
        logger.info(f"다국어 임베딩 모델 로드 중: {EMBEDDING_MODEL}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("임베딩 모델 로드 완료")

            
    def _sanitize_documents(self, documents: List[Document]) -> List[Document]:
        """
        문서 리스트의 텍스트를 임베딩에 적합한 형태로 정제합니다.
        
        Args:
            documents: 정제할 문서 리스트
            
        Returns:
            정제된 문서 리스트
        """
        sanitized_documents = []
        
        for doc in documents:
            if hasattr(doc, 'page_content'):
                # 텍스트 정제
                sanitized_text = clean_text(doc.page_content)
                
                # 정제된 텍스트로 문서 생성
                sanitized_doc = Document(
                    page_content=sanitized_text,
                    metadata=doc.metadata if hasattr(doc, 'metadata') else {}
                )
                sanitized_documents.append(sanitized_doc)
        
        return sanitized_documents

    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """
        문서로부터 벡터 저장소를 생성합니다.
        
        Args:
            documents: 벡터화할 문서 리스트
            
        Returns:
            생성된 Chroma 벡터 저장소
        """
        if not documents:
            logger.warning("경고: 벡터화할 문서가 없습니다.")
            # 빈 문서 생성하여 최소한의 벡터 저장소 생성
            empty_doc = Document(page_content="문서 내용 없음")
            documents = [empty_doc]
        
        try:
            # 의미 없는 문서 제거 (너무 짧은 문서 등)
            filtered_documents = []
            for doc in documents:
                if hasattr(doc, 'page_content') and len(doc.page_content.strip()) > 10:
                    filtered_documents.append(doc)
                    
            if not filtered_documents:
                logger.warning("경고: 의미 있는 문서가 없습니다. 기본 문서를 생성합니다.")
                empty_doc = Document(page_content="문서 내용 없음")
                filtered_documents = [empty_doc]
            
            # 문서 추가 정제
            sanitized_documents = self._sanitize_documents(filtered_documents)
            
            logger.info(f"벡터 저장소 생성 중... (문서 수: {len(sanitized_documents)})")
            
            # 벡터 저장소 생성
            vectorstore = Chroma.from_documents(
                documents=sanitized_documents,
                embedding=self.embeddings,
                persist_directory=str(VECTORSTORE_PATH)
            )
            
            logger.info(f"벡터 저장소가 {VECTORSTORE_PATH}에 성공적으로 생성되었습니다.")
            return vectorstore
            
        except Exception as e:
            logger.error(f"벡터 저장소 생성 오류: {str(e)}")
            
            # 오류 발생 시 기존 벡터 저장소 로드 시도
            try:
                logger.info(f"기존 벡터 저장소를 로드하려고 시도합니다...")
                return self.load_vectorstore()
            except Exception as load_error:
                logger.error(f"기존 벡터 저장소 로드 오류: {str(load_error)}")
                # 아무것도 로드할 수 없을 때는 빈 벡터 저장소 생성
                empty_doc = Document(page_content="문서 내용 없음")
                return Chroma.from_documents(
                    documents=[empty_doc],
                    embedding=self.embeddings,
                    persist_directory=str(VECTORSTORE_PATH)
                )

    def load_vectorstore(self) -> Chroma:
        """
        기존 벡터 저장소를 로드합니다.
        
        Returns:
            로드된 Chroma 벡터 저장소
        """
        try:
            return Chroma(
                embedding_function=self.embeddings,
                persist_directory=str(VECTORSTORE_PATH)
            )
        except Exception as e:
            logger.error(f"벡터 저장소 로드 오류: {str(e)}")
            raise
