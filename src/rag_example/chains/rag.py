"""
RAG 체인 구현을 담당하는 모듈입니다.
"""
import logging
from typing import Any, Dict, List, Optional
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from ..config.settings import OLLAMA_MODEL, SEARCH_K
from ..utils.prompts import get_qa_prompt, get_condense_prompt

# 로깅 설정
logger = logging.getLogger(__name__)

class RAGChain:
    """RAG(Retrieval-Augmented Generation) 체인을 구현하는 클래스입니다."""
    
    def __init__(self, vectorstore: Chroma):
        """
        RAGChain을 초기화합니다.
        
        Args:
            vectorstore: 검색에 사용할 벡터 저장소
        """
        self.vectorstore = vectorstore
        self.llm = Ollama(model=OLLAMA_MODEL, temperature=0)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self.chain = self._create_chain()

    def _create_chain(self) -> ConversationalRetrievalChain:
        """
        ConversationalRetrievalChain을 생성합니다.
        
        Returns:
            생성된 ConversationalRetrievalChain
        """
        # 검색 설정 개선
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",  # similarity, mmr
            search_kwargs={
                "k": SEARCH_K  # 검색할 문서 수
            }
        )
        
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            condense_question_prompt=get_condense_prompt(),
            combine_docs_chain_kwargs={"prompt": get_qa_prompt()},
            return_source_documents=True,
            verbose=False
        )

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        사용자의 질문을 처리합니다.
        
        Args:
            query: 사용자 질문
            
        Returns:
            처리 결과를 담은 딕셔너리
        """
        return self.chain.invoke({"question": query})

    def format_source_documents(self, source_documents: List[Document]) -> str:
        """
        참조 문서들을 포맷팅합니다.
        
        Args:
            source_documents: 참조 문서 리스트
            
        Returns:
            포맷팅된 문서 문자열
        """
        if not source_documents:
            return "참조할 문서가 없습니다."
            
        formatted_docs = ""
        for i, doc in enumerate(source_documents):
            # 문서 내용이 없거나 짧은 경우 처리
            if not hasattr(doc, 'page_content') or not doc.page_content.strip() or doc.page_content.strip() == "문서 내용 없음":
                continue
                
            # 문서 내용 표시 - 더 많은 내용 표시 (50자로 증가)
            doc_content = doc.page_content.strip()[:50] + "..." if len(doc.page_content.strip()) > 50 else doc.page_content.strip()
            
            # 문서 출처 정보 표시 (있는 경우)
            source_info = ""
            if hasattr(doc, 'metadata'):
                if 'source' in doc.metadata:
                    source_info = f" (출처: {doc.metadata['source']})"
                elif 'page' in doc.metadata:
                    source_info = f" (Page: {doc.metadata['page']})"
            
            formatted_docs += f"문서 {i+1}{source_info}:\n{doc_content}\n{'-' * 30}\n"
            
        if not formatted_docs:
            return "참조할 유의미한 문서가 없습니다."
            
        return formatted_docs

    def run_conversation(self) -> None:
        """
        대화형 RAG 시스템을 실행합니다.
        
        사용자와 대화형으로 질의응답을 진행하며, 문서에서 참조한 내용을 함께 표시합니다.
        """
        print("RAG 대화 시스템이 시작되었습니다. 종료하려면 'exit' 또는 'quit'를 입력하세요.")
        print("-" * 50)
        print("PDF 파일과 텍스트 파일을 활용한 질의응답 시스템입니다.")
        print("예시: '인공지능 유망산업은 뭔가 있나요?', '추진배경은 뭔인가요?'")
        
        while True:
            try:
                query = input("\n질문을 입력하세요: ")
                
                if not query.strip():
                    print("질문을 입력해주세요.")
                    continue
                    
                if query.lower() in ["exit", "quit", "종료"]:
                    print("대화를 종료합니다.")
                    break
                
                # 질문 처리
                print("\n질문을 처리하고 있습니다...")
                result = self.process_query(query)
                
                print("\n응답:")
                print(result["answer"])
                
                # 참조 문서 출력
                if "source_documents" in result and result["source_documents"]:
                    formatted_docs = self.format_source_documents(result["source_documents"])
                    if formatted_docs and formatted_docs != "참조할 유의미한 문서가 없습니다.":
                        print("\n참조 문서:")
                        print(formatted_docs)
                    
            except EOFError:
                print("\n입력이 종료되었습니다. 대화를 종료합니다.")
                break
            except KeyboardInterrupt:
                print("\n사용자에 의해 종료되었습니다.")
                break
            except Exception as e:
                logger.error(f"질문 처리 오류: {str(e)}")
                print(f"\n오류 발생: {str(e)}")
                print("다시 시도해주세요.")
                continue
