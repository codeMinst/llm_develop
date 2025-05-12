"""
벡터 저장소 생성 및 관리 모듈

이 모듈은 문서 청크를 벡터화하고 벡터 저장소를 생성하는 기능을 제공합니다.
주요 기능:
- 임베딩 모델 설정
- 벡터 저장소 생성 및 관리
- 벡터 검색 기능

설계 참고 사항:
- 이 모듈은 LangChain의 벡터 저장소 추상화를 활용하여 추가적인 어댑터 레이어 없이 구현했습니다.
- LangChain이 이미 다양한 벡터 저장소와 임베딩 모델을 추상화하고 있어 추가적인 추상화는 불필요한 복잡성을 초래할 수 있습니다.
"""
import logging
import os
import time
import shutil
from typing import List, Dict, Any, Optional

from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from rag_example.config.settings import (
    VECTORSTORE_PATH, 
    EMBEDDING_MODEL
)

logger = logging.getLogger(__name__)

class VectorStoreBuilder:
    """
    벡터 저장소 생성 및 관리를 담당하는 클래스
    
    주요 기능:
    - 임베딩 모델 설정
    - 벡터 저장소 생성
    - 벡터 저장소 관리 (저장, 로드, 삭제)
    
    설계 의도:
    - 이 클래스는 벡터 저장소 관리에 필요한 기능을 캡슐화하여 단순하고 일관된 인터페이스를 제공합니다.
    - LangChain이 이미 제공하는 추상화를 활용하여 추가적인 어댑터 계층 없이 직접 구현했습니다.
    - 필요한 경우 내부적으로 적절한 패턴을 적용하여 코드 복잡성을 관리합니다.
    """
    
    def __init__(self, 
                 embedding_model: str = EMBEDDING_MODEL,
                 vectorstore_dir: str = str(VECTORSTORE_PATH)):
        """
        VectorStoreBuilder 초기화
        
        Args:
            embedding_model: 사용할 임베딩 모델 이름
            vectorstore_dir: 벡터 저장소 디렉토리 경로
        """
        self.embedding_model = embedding_model
        self.vectorstore_dir = vectorstore_dir
        self.embeddings = None
        self.vectorstore = None
        
        # 벡터 저장소 디렉토리 생성
        os.makedirs(self.vectorstore_dir, exist_ok=True)
    
    def _create_embeddings(self) -> HuggingFaceEmbeddings:
        """
        임베딩 모델을 생성합니다.
        
        Returns:
            생성된 HuggingFaceEmbeddings 객체
            
        참고:
            이 구현은 LangChain의 HuggingFaceEmbeddings를 직접 사용합니다.
            다른 임베딩 모델로 전환해야 할 경우 이 메서드만 수정하면 됩니다.
            추가적인 어댑터 계층이 필요하지 않은 이유는 LangChain이 이미 임베딩 모델에 대한 추상화를 제공하기 때문입니다.
        """
        logger.info(f"임베딩 모델 생성: {self.embedding_model}")
        
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        return embeddings
    
    def _sanitize_documents(self, documents: List[Document]) -> List[Document]:
        """
        벡터화할 문서를 정제합니다.
        
        Args:
            documents: 정제할 문서 리스트
            
        Returns:
            정제된 문서 리스트
        """
        sanitized_documents = []
        
        for doc in documents:
            if not hasattr(doc, 'page_content') or not doc.page_content.strip():
                continue
                
            # 문서가 이미 정제되었다고 가정하고 최소한의 추가 처리만 수행
            # 임베딩에 영향을 줄 수 있는 불필요한 공백 제거
            page_content = doc.page_content.strip()
            
            # 지나치게 긴 문서는 임베딩 성능에 영향을 줄 수 있으므로 경고 로그 출력
            if len(page_content) > 10000:
                logger.warning(f"매우 긴 문서 발견: {len(page_content)} 문자")
            
            # 정제된 텍스트로 문서 생성
            sanitized_doc = Document(
                page_content=page_content,
                metadata=doc.metadata if hasattr(doc, 'metadata') else {}
            )
            sanitized_documents.append(sanitized_doc)
        
        return sanitized_documents
    
    def build(self, documents: List[Document], clean: bool = False) -> Chroma:
        """
        문서로부터 벡터 저장소를 생성합니다.
        
        Args:
            documents: 벡터화할 문서 리스트
            clean: 기존 벡터 저장소를 삭제하고 새로 생성할지 여부
            
        Returns:
            생성된 Chroma 벡터 저장소
            
        설계 참고:
            이 메서드는 템플릿 메서드 패턴을 적용하여 벡터 저장소 생성 과정을 정형화합니다.
            현재는 Chroma를 사용하지만, 다른 벡터 저장소로 전환해야 할 경우 이 메서드를 수정하거나 팩토리 패턴을 도입하면 됩니다.
            현재 구현은 단순성을 위해 직접 구현 방식을 사용했습니다.
        """
        start_time = time.time()
        
        # 기존 벡터 저장소 삭제 여부 확인
        if clean and os.path.exists(self.vectorstore_dir):
            logger.info("명령줄 인자 '--clean-rag'가 감지되었습니다. 기존 벡터 저장소를 삭제하고 새로 생성합니다.")
            shutil.rmtree(self.vectorstore_dir)
            os.makedirs(self.vectorstore_dir, exist_ok=True)
        
        # 임베딩 모델 생성
        if self.embeddings is None:
            self.embeddings = self._create_embeddings()
        
        # 문서 정제
        sanitized_documents = self._sanitize_documents(documents)
        
        # 벡터 저장소 생성
        logger.info("벡터 저장소를 생성합니다...")
        
        if not sanitized_documents:
            logger.warning("경고: 벡터화할 문서가 없습니다.")
            # 빈 문서 생성하여 최소한의 벡터 저장소 생성
            empty_doc = Document(page_content="문서 내용 없음")
            sanitized_documents = [empty_doc]
        
        # Chroma 벡터 저장소 생성
        self.vectorstore = Chroma.from_documents(
            documents=sanitized_documents,
            embedding=self.embeddings,
            persist_directory=self.vectorstore_dir
        )
        
        # 벡터 저장소 저장
        self.vectorstore.persist()
        
        # 벡터 저장소 생성 시간 로깅
        processing_time = time.time() - start_time
        logger.info(f"벡터 저장소 생성 완료 (시간: {processing_time:.2f}초)")
        
        return self.vectorstore
    
    def load(self) -> Optional[Chroma]:
        """
        기존 벡터 저장소를 로드합니다.
        
        Returns:
            로드된 Chroma 벡터 저장소 또는 오류 시 None
        """
        try:
            # 벡터 저장소 디렉토리 확인
            if not os.path.exists(self.vectorstore_dir):
                logger.warning(f"벡터 저장소 디렉토리가 존재하지 않습니다: {self.vectorstore_dir}")
                return None
            
            # 임베딩 모델 생성
            if self.embeddings is None:
                self.embeddings = self._create_embeddings()
            
            # 벡터 저장소 로드
            logger.info(f"기존 벡터 저장소를 로드합니다: {self.vectorstore_dir}")
            
            self.vectorstore = Chroma(
                persist_directory=self.vectorstore_dir,
                embedding_function=self.embeddings
            )
            
            return self.vectorstore
            
        except Exception as e:
            logger.error(f"벡터 저장소 로드 중 오류 발생: {str(e)}")
            return None
