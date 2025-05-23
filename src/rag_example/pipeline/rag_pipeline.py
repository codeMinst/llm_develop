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
from typing import Any
from rag_example.pipeline.ingestion.document_loader import DocumentLoader
from rag_example.pipeline.indexing.vectorstore_builder import VectorStoreBuilder
from rag_example.pipeline.querying.rag_chain_builder import RAGChainBuilder
from rag_example.pipeline.querying.graph_builder import GraphRAGChainBuilder
from rag_example.config.settings import RAW_DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)    

class RAGPipeline:
    """
    RAG 시스템의 전체 파이프라인을 관리하는 클래스
    
    이 클래스는 다음 단계를 오케스트레이션합니다:
    1. 문서 로딩 및 전처리 (DocumentLoader)
    2. 벡터 저장소 생성 (VectorStoreBuilder) 
    3. RAG 체인 구성 (GraphRAGChainBuilder)
    
    설계 의도:
    - 각 파이프라인 컴포넌트는 특정 기능을 캡슐화하고, 필요한 라이브러리를 내부적으로 활용합니다.
    - 컴포넌트들은 각자의 책임 영역에서 독립적으로 동작하며 필요시 적절한 디자인 패턴을 적용합니다.
    - 파이프라인은 전체 흐름 제어에 집중하고, 세부 구현은 각 컴포넌트에 위임합니다.
    - 이 구조는 코드 복잡성을 관리하면서 확장성과 유연성을 제공합니다.
    """
    def __init__(self, 
                 document_dir: str = RAW_DATA_DIR,
                 chunk_size: int = CHUNK_SIZE,
                 chunk_overlap: int = CHUNK_OVERLAP,
                 is_clean_vectorstore: bool = False,
                 llm_type: str = "ollama"):
        """
        RAG 파이프라인 초기화
        
        Args:
            document_dir: 문서 파일이 있는 디렉토리 경로
            chunk_size: 문서 청크 크기
            chunk_overlap: 문서 청크 간 겹침 크기
            is_clean_vectorstore: 기존 벡터 저장소를 삭제하고 새로 생성할지 여부
        """
        self.document_dir = document_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.is_clean_vectorstore = is_clean_vectorstore
        
        # 파이프라인 컴포넌트 초기화
        # 각 컴포넌트는 특정 기능에 집중하며, 내부 구현 세부사항을 캡슐화합니다.
        # 이 방식은 불필요한 추상화를 피하고 LangChain 같은 라이브러리의 기능을 효과적으로 활용합니다.
        self.document_loader = DocumentLoader(document_dir)  # 문서 수집 및 처리 담당
        self.vectorstore_builder = VectorStoreBuilder()      # 벡터 저장소 생성 및 관리 담당
        self.chain_builder = GraphRAGChainBuilder(llm_type)  # 질의 처리 및 응답 생성 담당
        
        # 파이프라인 결과 저장 변수
        self.documents = None
        self.chunks = None
        self.vectorstore = None
        self.rag_chain = None
        self.llm_type = llm_type
    
    def setup_chain(self) -> Any:
        """
        전체 RAG 파이프라인을 통해 체인 생성
            
        Returns:
            RAGChainBuilder 인스턴스 - run 메서드를 통해 질의응답 가능
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
            clean=self.is_clean_vectorstore
        )
        
        # 3. RAG 체인 구성
        logger.info(f"RAG 체인 구성 단계 시작... (LLM 타입: {self.llm_type})")
        # RunnableWithMessageHistory 객체를 내부적으로 저장하고 ChainBuilder 인스턴스 반환
        self.chain_builder.build(self.vectorstore)
        
        # 전체 시스템 준비 시간
        total_prep_time = time.time() - total_start_time
        logger.info(f"RAG 시스템 준비 완료 (총 준비 시간: {total_prep_time:.2f}초)")
        
        return self.chain_builder
