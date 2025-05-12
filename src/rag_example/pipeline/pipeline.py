"""
RAG 파이프라인 모듈

이 모듈은 RAG(Retrieval-Augmented Generation) 시스템의 전체 파이프라인을 관리합니다.
문서 로딩, 벡터 저장소 생성, RAG 체인 구성 등의 단계를 오케스트레이션합니다.

설계 철학:
- 파이프라인 패턴: 데이터 흐름을 명확한 단계로 구분하여 처리합니다.
- 관심사 분리: 각 컴포넌트는 특정 기능에만 집중합니다.
- 유연한 구현: 각 컴포넌트는 내부 구현 세부사항을 캡슐화하고 필요에 따라 적절한 패턴을 사용합니다.
"""
import logging
import time
from typing import List

from rag_example.pipeline.ingestion.document_loader import DocumentLoader
from rag_example.pipeline.indexing.vectorstore_builder import VectorStoreBuilder
from rag_example.pipeline.querying.rag_chain_builder import RAGChainBuilder
from rag_example.config.settings import RAW_DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    RAG 시스템의 전체 파이프라인을 관리하는 클래스
    
    이 클래스는 다음 단계를 오케스트레이션합니다:
    1. 문서 로딩 및 전처리 (DocumentLoader)
    2. 벡터 저장소 생성 (VectorStoreBuilder)
    3. RAG 체인 구성 (RAGChainBuilder)
    
    설계 의도:
    - 파이프라인 컴포넌트들은 특정 프로세스를 캡슐화하고, 내부적으로 필요한 라이브러리를 조합해 사용합니다.
    - 각 컴포넌트는 자신의 책임 영역 내에서 필요한 경우 어댑터 패턴 등을 활용할 수 있습니다.
    - 파이프라인 자체는 높은 수준의 오케스트레이션에 집중하고, 세부 구현은 각 컴포넌트에 위임합니다.
    - 이 접근 방식은 코드의 복잡성을 최소화하면서도 필요한 유연성을 제공합니다.
    """
    
    def __init__(self, 
                 document_dir: str = RAW_DATA_DIR,
                 chunk_size: int = CHUNK_SIZE,
                 chunk_overlap: int = CHUNK_OVERLAP):
        """
        RAG 파이프라인 초기화
        
        Args:
            document_dir: 문서 파일이 있는 디렉토리 경로
            chunk_size: 문서 청크 크기
            chunk_overlap: 문서 청크 간 겹침 크기
        """
        self.document_dir = document_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 파이프라인 컴포넌트 초기화
        # 각 컴포넌트는 특정 기능에 집중하며, 내부 구현 세부사항을 캡슐화합니다.
        # 이 방식은 불필요한 추상화를 피하고 LangChain 같은 라이브러리의 기능을 효과적으로 활용합니다.
        self.document_loader = DocumentLoader(document_dir)  # 문서 수집 및 처리 담당
        self.vectorstore_builder = VectorStoreBuilder()      # 벡터 저장소 생성 및 관리 담당
        self.rag_chain_builder = RAGChainBuilder()           # 질의 처리 및 응답 생성 담당
        
        # 파이프라인 결과 저장 변수
        self.documents = None
        self.chunks = None
        self.vectorstore = None
        self.rag_chain = None
    
    def run(self, clean_vectorstore: bool = False) -> 'RAGChainBuilder':
        """
        전체 RAG 파이프라인 실행
        
        Args:
            clean_vectorstore: 기존 벡터 저장소를 삭제하고 새로 생성할지 여부
            
        Returns:
            구성된 RAG 체인 빌더
        """
        # 전체 시스템 시작 시간
        total_start_time = time.time()
        
        # 1. 문서 로딩 및 전처리
        logger.info("문서 로딩 및 전처리 단계 시작...")
        self.documents = self.document_loader.load_documents()
        self.chunks = self.document_loader.create_chunks(
            self.documents, 
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        # 2. 벡터 저장소 생성
        logger.info("벡터 저장소 생성 단계 시작...")
        self.vectorstore = self.vectorstore_builder.build(
            self.chunks,
            clean=clean_vectorstore
        )
        
        # 3. RAG 체인 구성
        logger.info("RAG 체인 구성 단계 시작...")
        self.rag_chain = self.rag_chain_builder.build(self.vectorstore)
        
        # 전체 시스템 준비 시간
        total_prep_time = time.time() - total_start_time
        logger.info(f"RAG 시스템 준비 완료 (총 준비 시간: {total_prep_time:.2f}초)")
        
        return self.rag_chain_builder

def create_rag_pipeline(document_dir: str = RAW_DATA_DIR, 
                       clean_vectorstore: bool = False) -> RAGPipeline:
    """
    RAG 파이프라인 생성 및 실행 헬퍼 함수
    
    이 함수는 파사드 패턴을 활용하여 파이프라인 생성 및 실행 과정을 단순화합니다.
    클라이언트 코드는 복잡한 파이프라인 설정 과정을 알 필요 없이 이 함수만 호출하면 됩니다.
    
    Args:
        document_dir: 문서 파일이 있는 디렉토리 경로
        clean_vectorstore: 기존 벡터 저장소를 삭제하고 새로 생성할지 여부
        
    Returns:
        구성된 RAG 파이프라인
    """
    pipeline = RAGPipeline(document_dir)
    pipeline.run(clean_vectorstore)
    return pipeline
