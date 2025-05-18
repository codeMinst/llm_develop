import logging
from typing import TypedDict, Optional, Literal, Dict, Callable

from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import Chroma
from langchain.llms.base import BaseLLM

from src.rag_example.config.settings import LLM_TYPE,MODEL_NAME, SEARCH_K, MAX_RECENT_TURNS
from src.rag_example.pipeline.querying.prompts import (
    get_summary_check_prompt,
    get_summary_type_prompt,
    get_qa_prompt,
)
from src.rag_example.pipeline.querying.llm_factory import LLMFactory
from src.rag_example.pipeline.summarizing_memory import SummarizingMemory

logger = logging.getLogger(__name__)

# --------- 상태 정의 ---------
class GraphState(TypedDict):
    question: str
    session_id: str
    is_summary: Optional[bool]
    summary_type: Optional[str]
    docs: Optional[list]
    answer: Optional[str]


class GraphRAGChainBuilder:
    """
    LangGraph 기반의 RAG 체인 빌더 클래스

    이 클래스는 LangGraph를 사용하여 다음과 같은 흐름으로 질문 처리를 수행합니다:
    1. 요약 질문 여부 판별
    2. 요약 타입 분기 (resume, projects, workstyle, all)
    3. 문서 검색 (summary_type 필터 또는 일반 MMR 검색)
    4. 답변 생성 및 대화 히스토리 관리

    주요 기능:
    - LangGraph를 사용한 상태 기반 질문 처리 흐름 구성
    - 요약 질문과 일반 질문을 자동으로 구분하여 처리
    - 대화 메모리 관리를 위한 SummarizingMemory 통합
    """
    def __init__(
        self,
        llm_type: str = "ollama",
        model_name: str = "llama3.1",
        vectorstore: Chroma = None,
        search_k: int = SEARCH_K,
        max_recent_turns: int = MAX_RECENT_TURNS,
    ):
        """
        GraphRAGChainBuilder 초기화

        Args:
            llm_type: 사용할 LLM 타입 ("ollama" 또는 "claude")
            model_name: LLM 모델 이름
            vectorstore: 벡터 저장소 객체
            search_k: 검색할 문서 수
            max_recent_turns: 최근 대화 턴 수
        """
        # LLM, VectorStore, Prompt 초기화
        self.llm: BaseLLM = LLMFactory.create_llm(llm_type, model_name, temperature=0.1)
        self.llm_type = llm_type
        self.vectorstore: Optional[Chroma] = None
        self.search_k = search_k
        self.max_recent_turns = max_recent_turns

        self.check_prompt = get_summary_check_prompt()
        self.type_prompt = get_summary_type_prompt()
        self.qa_prompt = get_qa_prompt()

        self.session_histories: Dict[str, SummarizingMemory] = {}

        # 상태 그래프 컴파일
        self.runnable = None

    def _get_history(self, session_id: str) -> SummarizingMemory:
        """
        세션 ID에 해당하는 대화 히스토리 객체를 반환합니다.
        
        Args:
            session_id: 대화 세션 ID
            
        Returns:
            해당 세션의 SummarizingMemory 객체
        """
        if session_id not in self.session_histories:
            self.session_histories[session_id] = SummarizingMemory(
                session_id=session_id,
                llm=self.llm,
                max_recent_turns=self.max_recent_turns
            )
        return self.session_histories[session_id]

    def reset_memory(self, session_id: str = "default") -> None:
        """
        지정된 세션의 대화 메모리를 초기화합니다.
        
        Args:
            session_id: 초기화할 세션 ID
            
        Notes:
            - session_id가 "all"일 경우 모든 세션의 메모리가 초기화됩니다.
            - session_id가 특정 값일 경우 해당 세션만 초기화됩니다.
        """
        if session_id == "all":
            self.session_histories.clear()
            logger.info("✅ 모든 세션 히스토리 초기화 완료")
        elif session_id in self.session_histories:
            self.session_histories[session_id].clear()
            logger.info(f"✅ 세션 '{session_id}' 히스토리 초기화 완료")

    def build(self, vectorstore: Chroma) -> Callable[[dict], dict]:
        """
        LangGraph 기반 RAG 체인을 구성합니다.
        
        Args:
            vectorstore: 벡터 저장소 객체
            
        Returns:
            LangGraph로 구성된 실행 가능한 함수
            
        Notes:
            - StateGraph를 사용하여 질문 처리 흐름을 정의합니다.
            - 요약 질문과 일반 질문을 자동으로 분기 처리합니다.
            - 각 노드는 RunnableLambda로 구현되어 있습니다.
        """
        self.vectorstore = vectorstore
        # --- 노드 함수 정의 ---
        def check_summary_node(state: GraphState) -> GraphState:
            """
            질문이 요약 요청인지 확인하는 노드 함수

            Args:
                state: 현재 상태 객체
                
            Returns:
                is_summary 플래그가 업데이트된 상태 객체
            """
            prompt = self.check_prompt.format_messages(
                question=state["question"]
            )
            raw = self.llm.invoke(prompt)
            txt = LLMFactory.process_response(self.llm_type, raw).strip().lower()

            logger.info(f"[check_summary_node] LLM 응답: {txt}")

            is_sum = txt.startswith("y") or "예" in txt
            logger.info(f"[check_summary_node] is_summary: {is_sum}")

            return {**state, "is_summary": is_sum}

        def get_summary_type_node(state: GraphState) -> GraphState:
            """
            요약 타입을 분류하는 노드 함수

            Args:
                state: 현재 상태 객체
                
            Returns:
                summary_type이 업데이트된 상태 객체
            """
            prompt = self.type_prompt.format_messages(
                question=state["question"]
            )
            raw = self.llm.invoke(prompt)
            st = LLMFactory.process_response(self.llm_type, raw).strip().lower()

            logger.info(f"[get_summary_type_node] LLM 응답: {st}")

            if st not in ("resume", "projects", "workstyle", "all"):
                st = "none"
            logger.info(f"[get_summary_type_node] summary_type: {st}")
            return {**state, "summary_type": st}

        def search_summary_node(state: GraphState) -> GraphState:
            """
            요약 타입에 맞는 문서를 검색하는 노드 함수

            Args:
                state: 현재 상태 객체
                
            Returns:
                검색된 문서가 추가된 상태 객체
            """
            docs = vectorstore.similarity_search(
                query=state["question"],
                k=self.search_k,
                filter={"summary_type": state["summary_type"]},
            )
            logger.info(f"[search_summary_node] 문서 수: {len(docs)}")

            return {**state, "docs": docs}

        def search_general_node(state: GraphState) -> GraphState:
            """
            일반 MMR 검색을 수행하는 노드 함수

            Args:
                state: 현재 상태 객체
                
            Returns:
                검색된 문서가 추가된 상태 객체
            """
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": self.search_k, "fetch_k": 20, "lambda_mult": 0.75}
            )
            docs = retriever.get_relevant_documents(state["question"])
            logger.info(f"[search_general_node] 문서 수: {len(docs)}")

            return {**state, "docs": docs}

        def generate_node(state: GraphState) -> GraphState:
            """
            질문에 대한 답변을 생성하는 노드 함수

            Args:
                state: 현재 상태 객체
                
            Returns:
                답변이 추가된 상태 객체
            """
            history = self._get_history(state["session_id"])
            context = "\n\n".join(doc.page_content for doc in state["docs"])
            chat_hist = history.load_summary_and_recent()

            prompt = self.qa_prompt.format_messages(
                question=state["question"],
                chat_history=chat_hist,
                context=context
            )
            raw = self.llm.invoke(prompt)
            answer = LLMFactory.process_response(self.llm_type, raw).strip()

            history.add_messages([
                HumanMessage(content=state["question"]),
                AIMessage(content=answer)
            ])
            return {**state, "answer": answer}

        # --- 분기(edge) 정의 ---
        def check_edge(state: GraphState) -> Literal["yes", "no"]:
            return "yes" if state["is_summary"] else "no"

        def type_edge(state: GraphState) -> Literal["summary", "general"]:
            return "summary" if state["summary_type"] in ("resume", "projects", "workstyle", "all") else "general"

        # --- StateGraph 구성 ---
        builder = StateGraph(GraphState)
        builder.add_node("check_summary", RunnableLambda(check_summary_node))
        builder.add_node("get_summary_type", RunnableLambda(get_summary_type_node))
        builder.add_node("search_summary", RunnableLambda(search_summary_node))
        builder.add_node("search_general", RunnableLambda(search_general_node))
        builder.add_node("generate", RunnableLambda(generate_node))

        builder.set_entry_point("check_summary")
        builder.add_conditional_edges("check_summary", check_edge, {
            "yes": "get_summary_type",
            "no": "search_general"
        })
        builder.add_conditional_edges("get_summary_type", type_edge, {
            "summary": "search_summary",
            "general": "search_general"
        })
        builder.add_edge("search_summary", "generate")
        builder.add_edge("search_general", "generate")
        builder.set_finish_point("generate")

        self.runnable = builder.compile()
        return self.runnable

    def run(self, query: str, session_id: str = "default") -> str:
        """
        질문을 처리하고 답변을 반환합니다.
        
        Args:
            query: 사용자의 질문
            session_id: 대화 세션 ID
            
        Returns:
            질문에 대한 답변 문자열
            
        Notes:
            - 질문 처리 로그를 기록합니다.
            - 대화 히스토리를 관리합니다.
        """
        """
        질의 실행 엔트리 포인트
        """
        logger.info(f"질문 처리 시작: '{query}' (session: {session_id})")
        state = self.runnable.invoke({
            "question": query,
            "session_id": session_id
        })
        logger.info(f"질문 처리 완료: '{query}' → 답변 반환")
        return state["answer"]
