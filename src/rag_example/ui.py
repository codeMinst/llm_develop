#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG 시스템 웹 인터페이스 모듈
"""
import gradio as gr
import logging
from pathlib import Path

# 절대 경로 임포트 사용
from src.rag_example.pipeline import RAGPipeline
from src.rag_example.config.settings import RAW_DATA_DIR, LLM_TYPE

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGUI:
    """
    RAG 시스템의 웹 인터페이스를 제공하는 클래스
    """
    
    def __init__(self):
        """
        RAG UI 초기화
        """
        logger.info("RAG UI 초기화 중...")
        self.pipeline = None
        self.rag_chain = None
        self.chat_history = []
        self.initialize_pipeline(clean_vectorstore=True)
        
    def initialize_pipeline(self, clean_vectorstore=True):
        """
        RAG 파이프라인 초기화
        
        Args:
            clean_vectorstore: 벡터 저장소 초기화 여부
        """
        logger.info(f"RAG 파이프라인 초기화 (clean_vectorstore={clean_vectorstore})...")
        self.pipeline = RAGPipeline(
            document_dir=RAW_DATA_DIR,
            is_clean_vectorstore=clean_vectorstore,
            llm_type=LLM_TYPE
        )
        self.rag_chain = self.pipeline.setup_chain()
        logger.info("RAG 파이프라인 초기화 완료")
        
    def process_query(self, query, history):
        """
        사용자 질의 처리
        
        Args:
            query: 사용자 질문
            history: 대화 기록
            
        Returns:
            업데이트된 대화 기록
        """
        if not query:
            return "", history
            
        logger.info(f"질문 처리 중: {query}")
        
        try:
            # 질문 처리
            result = self.rag_chain.run(query)
            response = result.get('answer', '응답을 생성할 수 없습니다.')
            
            # 대화 기록 업데이트
            history.append((query, response))
            logger.info("응답 생성 완료")
            
        except Exception as e:
            logger.error(f"질문 처리 중 오류 발생: {str(e)}")
            response = f"오류가 발생했습니다: {str(e)}"
            history.append((query, response))
            
        return "", history
        
    def reset_conversation(self, history):
        """
        대화 기록 초기화
        
        Returns:
            빈 대화 기록
        """
        logger.info("대화 기록 초기화")
        
        # RAG 체인의 메모리도 초기화
        if self.rag_chain is not None:
            self.rag_chain.reset_memory()
            logger.info("RAG 체인 메모리 초기화 완료")
        
        return [], []
        
    def rebuild_vectorstore(self):
        """
        벡터 저장소 초기화 및 재생성
        
        Returns:
            상태 메시지
        """
        logger.info("벡터 저장소 초기화 및 재생성 중...")
        try:
            self.initialize_pipeline(clean_vectorstore=True)
            return "벡터 저장소가 성공적으로 초기화되었습니다."
        except Exception as e:
            logger.error(f"벡터 저장소 초기화 중 오류 발생: {str(e)}")
            return f"벡터 저장소 초기화 중 오류 발생: {str(e)}"
    
    def get_document_info(self):
        """
        문서 정보 조회
        
        Returns:
            문서 정보 문자열
        """
        try:
            doc_path = Path(RAW_DATA_DIR)
            if not doc_path.exists():
                return "문서 디렉토리를 찾을 수 없습니다."
                
            files = list(doc_path.glob("**/*"))
            pdf_files = [f for f in files if f.suffix.lower() == '.pdf']
            txt_files = [f for f in files if f.suffix.lower() == '.txt']
            
            info = f"문서 디렉토리: {doc_path}\n"
            info += f"PDF 파일: {len(pdf_files)}개\n"
            info += f"텍스트 파일: {len(txt_files)}개\n"
            info += f"총 파일: {len(pdf_files) + len(txt_files)}개\n\n"
            
            if pdf_files or txt_files:
                info += "파일 목록:\n"
                for f in sorted(pdf_files + txt_files):
                    info += f"- {f.name}\n"
            
            return info
        except Exception as e:
            logger.error(f"문서 정보 조회 중 오류 발생: {str(e)}")
            return f"문서 정보 조회 중 오류 발생: {str(e)}"
    
    def create_ui(self):
        """
        Gradio 인터페이스 생성
        
        Returns:
            Gradio 인터페이스
        """
        with gr.Blocks(title="RAG 대화 시스템") as demo:
            gr.Markdown("# RAG 대화 시스템")
            gr.Markdown("PDF 파일과 텍스트 파일을 활용한 질의응답 시스템입니다.")
            
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        value=self.chat_history,
                        height=500,
                        show_label=False
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="질문을 입력하세요...", 
                            lines=2,
                            show_label=False
                        )
                    
                    with gr.Row():
                        submit_btn = gr.Button("질문하기", variant="primary")
                        clear_btn = gr.Button("대화 초기화")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 시스템 관리")
                    
                    with gr.Accordion("문서 정보", open=True):
                        doc_info = gr.Textbox(
                            value=self.get_document_info(),
                            lines=10,
                            label="문서 정보",
                            interactive=False
                        )
                        refresh_btn = gr.Button("새로고침")
                    
                    with gr.Accordion("벡터 저장소 관리", open=True):
                        rebuild_btn = gr.Button("벡터 저장소 초기화", variant="secondary")
                        system_msg = gr.Textbox(
                            label="시스템 메시지", 
                            interactive=False,
                            value="시스템이 준비되었습니다."
                        )
            
            # 이벤트 연결
            submit_btn.click(
                self.process_query, 
                inputs=[msg, chatbot], 
                outputs=[msg, chatbot]
            )
            
            msg.submit(
                self.process_query,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )
            
            clear_btn.click(
                self.reset_conversation, 
                inputs=[chatbot], 
                outputs=[chatbot, msg]
            )
            
            rebuild_btn.click(
                self.rebuild_vectorstore, 
                inputs=[], 
                outputs=[system_msg]
            )
            
            refresh_btn.click(
                self.get_document_info,
                inputs=[],
                outputs=[doc_info]
            )
            
        return demo

def main():
    """
    UI 실행 함수
    """
    ui = RAGUI()
    demo = ui.create_ui()
    demo.launch(share=False)

if __name__ == "__main__":
    main()
