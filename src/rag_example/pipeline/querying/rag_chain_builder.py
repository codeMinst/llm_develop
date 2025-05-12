"""
RAG 체인 구성 모듈

이 모듈은 RAG(Retrieval-Augmented Generation) 체인을 구성하는 기능을 제공합니다.
주요 기능:
- 프롬프트 템플릿 설정
- LLM 설정
- 검색기 구성
- RAG 체인 생성

설계 철학:
- 이 모듈은 LangChain의 ConversationalRetrievalChain을 활용하여 RAG 체인을 구성합니다.
- 빌더 패턴을 사용하여 복잡한 체인 구성 과정을 추상화하고 유연하게 구성할 수 있도록 합니다.
- 외부 의존성(프롬프트, LLM)을 캡슐화하여 관리하기 쉽게 합니다.
"""
import logging
from typing import Dict, Any, Optional

from langchain_community.vectorstores import Chroma
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

from rag_example.config.settings import OLLAMA_MODEL, SEARCH_K
from rag_example.pipeline.querying.prompts import get_condense_prompt, get_qa_prompt

logger = logging.getLogger(__name__)

class RAGChainBuilder:
    """
    RAG 체인 구성을 담당하는 클래스
    
    주요 기능:
    - 프롬프트 템플릿 설정
    - LLM 설정
    - 검색기 구성
    - RAG 체인 생성
    
    설계 의도:
    - 빌더 패턴을 사용하여 복잡한 RAG 체인 구성 과정을 추상화합니다.
    - 각 구성 요소(LLM, 메모리, 프롬프트, 검색기)를 모듈화하여 관리하기 쉽게 합니다.
    - 파이프라인의 마지막 단계로, 사용자 질의에 대한 응답 생성을 담당합니다.
    - LangChain의 추상화를 활용하면서도 필요한 경우 내부적으로 구체적인 구현을 캡슐화합니다.
    """
    
    def __init__(self, 
                 model_name: str = OLLAMA_MODEL,
                 search_k: int = SEARCH_K):
        """
        RAGChainBuilder 초기화
        
        Args:
            model_name: 사용할 LLM 모델 이름
            search_k: 검색할 문서 수
        """
        self.model_name = model_name
        self.search_k = search_k
        self.llm = None
        self.memory = None
        self.chain = None
    
    def _create_llm(self) -> Ollama:
        """
        LLM을 생성합니다.
        
        Returns:
            생성된 Ollama LLM 객체
            
        설계 참고:
            이 메서드는 팩토리 메서드 패턴을 적용하여 LLM 생성을 캡슐화합니다.
            현재는 Ollama를 사용하지만, 다른 LLM으로 전환해야 할 경우 이 메서드만 수정하면 됩니다.
            추가적인 어댑터 계층이 필요하지 않은 이유는 LangChain이 이미 LLM에 대한 추상화를 제공하기 때문입니다.
        """
        logger.info(f"LLM 생성: {self.model_name}")
        
        llm = Ollama(model=self.model_name, temperature=0.1)
        
        return llm
    
    def _create_memory(self) -> ConversationBufferMemory:
        """
        대화 메모리를 생성합니다.
        
        Returns:
            생성된 ConversationBufferMemory 객체
        """
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        return memory
    
    def _create_prompt_templates(self) -> Dict[str, PromptTemplate]:
        """
        프롬프트 템플릿을 생성합니다.
        
        Returns:
            생성된 프롬프트 템플릿 딕셔너리
        """
        # 외부 모듈에서 프롬프트 템플릿 가져오기
        condense_prompt = get_condense_prompt()
        qa_prompt = get_qa_prompt()
        
        return {
            "condense": condense_prompt,
            "qa": qa_prompt
        }
    
    def build(self, vectorstore: Chroma) -> ConversationalRetrievalChain:
        """
        RAG 체인을 구성합니다.
        
        Args:
            vectorstore: 사용할 벡터 저장소
            
        Returns:
            구성된 ConversationalRetrievalChain 객체
            
        설계 참고:
            이 메서드는 빌더 패턴의 중심 메서드로, 모든 구성 요소를 조합하여 RAG 체인을 생성합니다.
            각 구성 요소는 모듈화된 메서드로 캡슐화되어 있어 유지보수성을 향상시킵니다.
            LangChain의 ConversationalRetrievalChain을 활용하여 추가적인 추상화 없이 직접 구현했습니다.
        """
        logger.info("ConversationalRetrievalChain을 생성합니다...")
        
        # LLM 생성
        if self.llm is None:
            self.llm = self._create_llm()
        
        # 메모리 생성
        if self.memory is None:
            self.memory = self._create_memory()
        
        # 프롬프트 템플릿 생성
        prompts = self._create_prompt_templates()
        
        # 검색기 구성
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.search_k}
        )
        
        # RAG 체인 생성
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": prompts["qa"]}
        )
        
        logger.info("RAG 체인 생성 완료")
        
        return self.chain
    
    def run(self, query: str) -> Dict[str, Any]:
        """
        RAG 체인을 실행하여 질문에 답변합니다.
        
        Args:
            query: 사용자 질문
            
        Returns:
            답변 결과 딕셔너리
        """
        if self.chain is None:
            logger.error("RAG 체인이 생성되지 않았습니다. build() 메서드를 먼저 호출하세요.")
            return {"answer": "시스템이 준비되지 않았습니다."}
        
        try:
            # 체인 실행
            result = self.chain({"question": query})
            return result
            
        except Exception as e:
            logger.error(f"RAG 체인 실행 중 오류 발생: {str(e)}")
            return {"answer": f"오류가 발생했습니다: {str(e)}"}
    
    def reset_memory(self) -> None:
        """
        대화 메모리를 초기화합니다.
        """
        if self.memory is not None:
            self.memory.clear()
            logger.info("대화 메모리가 초기화되었습니다.")
        else:
            logger.warning("초기화할 대화 메모리가 없습니다.")
