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
from typing import Dict, List, Optional

from langchain_community.vectorstores import Chroma
from langchain.llms.base import BaseLLM
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory

from src.rag_example.config.settings import OLLAMA_MODEL, SEARCH_K
from src.rag_example.pipeline.querying.prompts import get_condense_prompt, get_qa_prompt, PromptTemplate
from src.rag_example.pipeline.querying.llm_factory import LLMFactory
from src.rag_example.pipeline.summarizing_memory import SummarizingMemory

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
                 llm_type: str = "ollama",
                 model_name: str = OLLAMA_MODEL,
                 search_k: int = SEARCH_K):
        """
        RAGChainBuilder 초기화
        
        Args:
            llm_type: 사용할 LLM 타입 ("ollama" 또는 "claude")
            model_name: 사용할 LLM 모델 이름
            search_k: 검색할 문서 수
        """
        self.llm_type = llm_type
        self.model_name = model_name
        self.search_k = search_k
        self.llm = None
        self.chain = None
        
        # 세션별 메세지 히스토리 저장소
        self.session_histories = {}
    
    def _create_llm(self) -> BaseLLM:
        """
        LLM을 생성합니다.
        
        Returns:
            생성된 LLM 객체
            
        설계 참고:
            이 메서드는 팩토리 패턴을 적용하여 LLM 생성을 캡슐화합니다.
            LLMFactory를 사용하여 다양한 LLM을 생성할 수 있습니다.
            현재는 Ollama와 Claude를 지원하며, 필요에 따라 LLMFactory에 다른 LLM을 추가할 수 있습니다.
        """
        logger.info(f"LLM 생성: {self.llm_type} - {self.model_name}")
        
        try:
            return LLMFactory.create_llm(self.llm_type, self.model_name, temperature=0.1)
        except ValueError as e:
            logger.error(f"LLM 생성 실패: {str(e)}")
            logger.info(f"기본 Ollama LLM으로 대체합니다.")
            return LLMFactory.create_llm("ollama", OLLAMA_MODEL, temperature=0.1)
    
    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """
        세션 ID에 해당하는 메세지 히스토리 객체를 가져오거나 생성합니다.
        
        Args:
            session_id: 세션 아이디
            
        Returns:
            해당 세션의 BaseChatMessageHistory 객체
        """
        # 세션 ID가 없으면 새로 생성
        if session_id not in self.session_histories:
            logger.info(f"새 세션 생성: {session_id}")
            # 요약을 위한 LLM 사용
            if self.llm is None:
                self.llm = self._create_llm()
            self.session_histories[session_id] = SummarizingMemory(
                session_id=session_id,
                llm=self.llm,
                max_recent_turns=4  # 최근 4턴의 대화만 유지
            )
        
        return self.session_histories[session_id]
    
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
    
    def build(self, vectorstore: Chroma) -> Optional[callable]:
        """
        RAG 체인을 구성합니다.
        
        Args:
            vectorstore: 사용할 벡터 저장소
            
        Returns:
            구성된 체인 함수
        """
        logger.info("RAG 체인 생성 시작...")
        
        # LLM 생성
        if self.llm is None:
            self.llm = self._create_llm()
        
        # 프롬프트 템플릿 생성
        prompts = self._create_prompt_templates()
        qa_prompt = prompts["qa"]
        
        # 검색기 구성
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.search_k}
        )
        
        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])
            
        # 기본 RAG 체인 구성 - 단순화
        def run_rag_chain(query_text, history=None):
            # 1. 문서 검색
            docs = retriever.invoke(query_text)
            context = format_docs(docs)
            
            # 대화 기록 가져오기
            chat_history = ""
            if history and isinstance(history, SummarizingMemory):
                # 요약된 대화 기록 가져오기
                chat_history = history.load_summary_and_recent()
                if chat_history:
                    logger.debug(f"대화 기록 사용 (길이: {len(chat_history)})")
            
            # 2. 프롬프트 구성 및 LLM 요청
            prompt_args = {"context": context, "question": query_text, "chat_history": chat_history}
            chain_response = qa_prompt.format_messages(**prompt_args)
            llm_response = self.llm.invoke(chain_response)
            
            # LLM 팩토리의 응답 처리기를 사용하여 모델별 응답 형식 차이 처리
            return LLMFactory.process_response(self.llm_type, llm_response)
        
        # 메세지 히스토리를 처리하는 함수
        def process_with_history(inputs, config=None):
            if config is None:
                config = {}
                
            session_id = config.get("configurable", {}).get("session_id", "default")
            history = self.get_session_history(session_id)
            query = ""
            
            try:
                # 질문 추출
                query = inputs.get("question", "")
                if not query:
                    return "질문이 없습니다."
                
                # 런너블 함수 실행 - 대화 기록 전달
                result = run_rag_chain(query, history)
                
                # 사용자 질문과 AI 응답 추가
                history.add_messages([
                    HumanMessage(content=query),
                    AIMessage(content=result)
                ])
                
                return result
            except Exception as e:
                logger.error(f"RAG 체인 실행 오류: {str(e)}")
                logger.error(f"질문: {query}, 세션 ID: {session_id}")
                
                # 예외 발생 시 사용자에게 유용한 오류 메시지 반환
                error_message = f"질문 처리 중 오류가 발생했습니다: {str(e)}"
                return error_message
        
        # 메세지 히스토리를 관리하는 함수 설정
        self.chain = process_with_history
        
        logger.info("RAG 체인 생성 완료")
        
        return self.chain
    
    def run(self, query: str, session_id: str = "default") -> str:
        """
        RAG 체인을 실행하여 질문에 답변합니다.
        
        Args:
            query: 사용자 질문
            session_id: 세션 ID, 기본값은 "default"
            
        Returns:
            답변 문자열
        """
        if self.chain is None:
            logger.error("RAG 체인이 생성되지 않았습니다. build() 메서드를 먼저 호출하세요.")
            return "시스템이 준비되지 않았습니다."
        
        try:
            # 세션 확인 - 존재하지 않을 때만 새로 생성
            if session_id not in self.session_histories:
                logger.info(f"새 세션 생성: {session_id}")
                self.session_histories[session_id] = SummarizingMemory(session_id=session_id)
            else:
                logger.info(f"기존 세션 사용: {session_id}, 메시지 수: {len(self.session_histories[session_id].messages)}")
            
            # 체인 실행
            logger.info(f"질문 처리 시작: '{query}', 세션 ID: {session_id}")
            result = self.chain(
                {"question": query}, 
                config={"configurable": {"session_id": session_id}}
            )
            logger.info(f"질문 처리 완료: '{query}'")
            return result
            
        except Exception as e:
            logger.error(f"RAG 체인 실행 오류: {str(e)}")
            return f"오류가 발생했습니다: {str(e)}"
    
    def reset_memory(self, session_id: str = "default") -> None:
        """
        세션의 메세지 히스토리를 초기화합니다.

        Args:
            session_id: 초기화할 세션 ID, "all" 지정 시 전체 초기화
        """
        if session_id == "all":
            self.session_histories.clear()
            logger.info("✅ 모든 세션 히스토리 초기화 완료")
            return

        if session_id in self.session_histories:
            self.session_histories[session_id].clear()
            logger.info(f"✅ 세션 '{session_id}' 히스토리 초기화 완료")
        else:
            # 세션이 없다면 새로 생성
            if self.llm is None:
                self.llm = self._create_llm()
            self.session_histories[session_id] = SummarizingMemory(
                session_id=session_id,
                llm=self.llm,
            max_recent_turns=4
        )
        logger.info(f"🆕 새 세션 '{session_id}' 생성 및 초기화 완료")
