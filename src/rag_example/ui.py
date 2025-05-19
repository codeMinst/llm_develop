#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG 시스템 웹 인터페이스 모듈
"""
import gradio as gr
import os, sys, time, uuid, json
from dotenv import load_dotenv
import logging
from collections import defaultdict
from rag_example.pipeline.rag_pipeline import RAGPipeline
from rag_example.config.settings import RAW_DATA_DIR, LLM_TYPE
from rag_example.utils.constants import WELCOME_MESSAGES


# .env 파일 로드
load_dotenv()

# 토크나이저 병렬 처리 관련 경고 해결
# 최신 버전에서는 명시적으로 환경 변수 설정 권장
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("llm_ui.log")
    ]
)
logger = logging.getLogger(__name__)

# IP 세션 추적 및 차단 리스트
ip_sessions = defaultdict(list)
banned_ips = set()


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
        # 사용자 세션 ID 관리 - UUID를 사용하여 고유한 세션 ID 생성
        self.session_id = self._generate_session_id()
        logger.info(f"새 세션 생성: {self.session_id}")

        # 자주 묻는 질문 기반 초기 가이드 메시지
        self.chat_history = WELCOME_MESSAGES
        
        # 파이프라인 초기화
        self.initialize_pipeline(clean_vectorstore=True)

    def _is_ip_blocked(self, ip):
        return ip in banned_ips

    def _register_ip(self, ip):
        if ip:
            ip_sessions[ip].append(time.time())
            if len(ip_sessions[ip]) > 10:
                banned_ips.add(ip)

    def _generate_session_id(self):
        """
        고유한 세션 ID를 생성합니다.
        
        Returns:
            세션 ID 문자열
        """
        # UUID와 타임스태프를 조합하여 고유한 세션 ID 생성
        return f"web_user_{uuid.uuid4().hex[:8]}_{int(time.time())}"
        
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
        
    def process_query(self, query, history, request: gr.Request = None):
        """
        사용자 질의 처리
        
        Args:
            query: 사용자 질문
            history: 대화 기록
            
        Returns:
            업데이트된 대화 기록
        """
        # 빈 질문 처리
        if not query or query.strip() == "":
            return "", history
        
        request_ip = None
        if request:
            request_ip = request.client.host
            self._register_ip(request_ip)
            logger.info(f"요청 IP: {request_ip}")
            if self._is_ip_blocked(request_ip):
                history.append((
                    query,
                    "🚫 질문은 IP당 10개 이하만 하실 수 있습니다."
                ))
                return "", history
    
        # 특별 명령 처리
        if query.lower() in ['reset', 'clear', '초기화', '리셋']:
            return self.reset_conversation(history)
            
        logger.info(f"질문 처리 시작: '{query}'")
        
        # RAG 체인 존재 확인
        if self.rag_chain is None:
            logger.error("RAG 체인이 초기화되지 않았습니다.")
            response = "시스템이 준비되지 않았습니다. 재시작해 주세요."
            history.append((query, response))
            return "", history
        
        try:
            # 질문 처리 - 세션 ID를 사용하여 사용자별 대화 기록 관리
            logger.info(f"세션 {self.session_id}에서 질문 처리 중")
            
            # RAG 체인 실행
            start_time = time.time()
            result = self.rag_chain.run(query, session_id=self.session_id)
            elapsed_time = time.time() - start_time
            
            # 결과가 문자열로 반환됨
            response = result
            
            # 대화 기록 업데이트
            history.append((query, response))
            logger.info(f"응답 생성 완료 (소요시간: {elapsed_time:.2f}초)")
            
        except Exception as e:
            logger.error(f"질문 처리 중 오류 발생: {str(e)}")
            response = f"질문 처리 중 오류가 발생했습니다. 다시 시도해 주세요."
            history.append((query, response))
            
        return "", history
        
    def reset_conversation(self, history):
        """
        대화 기록 초기화
        
        Args:
            history: 현재 대화 기록
            
        Returns:
            빈 대화 기록
        """
        logger.info("대화 기록 초기화 시작")
        
        try:
            # RAG 체인의 메모리도 초기화
            if self.rag_chain is not None:
                # 현재 사용 중인 세션의 메모리를 초기화
                self.rag_chain.reset_memory(session_id=self.session_id)
                logger.info(f"RAG 체인 메모리 초기화 완료 (session_id={self.session_id})")
                
                # 새 세션 ID 생성 - 새로운 대화 시작
                self.session_id = self._generate_session_id()
                logger.info(f"새 세션 생성: {self.session_id}")
            
            logger.info("대화 기록 초기화 완료")
            return [], []
            
        except Exception as e:
            logger.error(f"대화 기록 초기화 중 오류 발생: {str(e)}")
            # 오류가 발생해도 대화는 초기화
            return [], []
        
    def rebuild_vectorstore(self):
        """
        벡터 저장소 초기화 및 재생성
        
        Returns:
            상태 메시지
        """
        logger.info("벡터 저장소 초기화 및 재생성 시작")
        try:
            # 벡터 저장소 초기화
            self.initialize_pipeline(clean_vectorstore=True)
            
            # 새 세션 생성 - 새로운 대화 시작
            self.session_id = self._generate_session_id()
            logger.info(f"새 세션 생성: {self.session_id}")
            
            logger.info("벡터 저장소 초기화 및 재생성 완료")
            return "벡터 저장소가 성공적으로 초기화되었습니다."
        except Exception as e:
            logger.error(f"벡터 저장소 초기화 중 오류 발생: {str(e)}")
            return f"벡터 저장소 초기화 중 오류가 발생했습니다. 다시 시도해 주세요."
    
    def get_document_info(self):
        """
        문서 정보 조회
        
        Returns:
            문서 정보 문자열
        """
        logger.info("문서 정보 조회 시작")
        try:
            doc_path = RAW_DATA_DIR
            if not doc_path.exists():
                logger.warning(f"문서 디렉토리가 존재하지 않습니다: {doc_path}")
                return f"문서 디렉토리가 존재하지 않습니다: {doc_path}"
            
            # 파일 목록 가져오기
            files = list(doc_path.glob("**/*"))
            pdf_files = [f for f in files if f.suffix.lower() == '.pdf']
            txt_files = [f for f in files if f.suffix.lower() == '.txt']
            
            # 문서 정보 구성
            info = f"문서 디렉토리: {doc_path}\n"
            info += f"PDF 파일: {len(pdf_files)}개\n"
            info += f"텍스트 파일: {len(txt_files)}개\n"
            info += f"총 파일: {len(pdf_files) + len(txt_files)}개\n\n"
            
            # 파일 목록 추가
            if pdf_files or txt_files:
                info += "파일 목록:\n"
                for f in sorted(pdf_files + txt_files):
                    info += f"- {f.name}\n"
            else:
                info += "문서 파일이 없습니다. 문서를 추가해 주세요.\n"
            
            logger.info(f"문서 정보 조회 완료: PDF {len(pdf_files)}개, 텍스트 {len(txt_files)}개")
            return info
            
        except Exception as e:
            logger.error(f"문서 정보 조회 중 오류 발생: {str(e)}")
            return f"문서 정보를 가져오는 중 오류가 발생했습니다. 다시 시도해 주세요."
    
    def create_ui(self):
        """
        Gradio 인터페이스 생성
        
        Returns:
            Gradio 인터페이스
        """
        with gr.Blocks(title="LLMOps 개발/적용 대상 developer-info chat demo") as demo:
            with gr.Row():
                gr.Markdown("# 🚀 LLMOps 개발/적용 대상 developer-info chat demo")
            gr.Markdown("LangChain과 LangGraph를 활용한 구조적인 RAG 시스템으로, 문서를 처리하고 질의응답을 제공합니다.")
            
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
                        clear_btn = gr.Button("chat 초기화")
                        logout_btn = gr.Button("로그아웃", variant="huggingface")
                
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
                            lines=2,
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
            
            # 로그아웃 버튼 기능 연결 - Gradio 기본 로그아웃 경로 사용
            logout_btn.click(
                fn=lambda: None,
                inputs=None,
                outputs=None,
                js="""
                () => {
                    window.location.href = '/logout';
                    return [];
                }
                """)
            
            # 자동 로그아웃 기능 추가 (10분 타임아웃)
            auto_logout_html = """
            <script>
                // 자동 로그아웃 시간 (밀리초 단위)
                const TIMEOUT_DURATION = 10 * 60 * 1000; // 10분
                let timeoutId;
                
                // 타이머 재설정 함수
                function resetTimer() {
                    clearTimeout(timeoutId);
                    timeoutId = setTimeout(logout, TIMEOUT_DURATION);
                }
                
                // 로그아웃 함수
                function logout() {
                    // Gradio 기본 로그아웃 경로로 이동
                    window.location.href = "/logout";
                }
                
                // 사용자 활동 감지 이벤트 리스너 등록
                document.addEventListener('mousemove', resetTimer);
                document.addEventListener('keypress', resetTimer);
                document.addEventListener('click', resetTimer);
                
                // 초기 타이머 설정
                resetTimer();
            </script>
            """
            gr.HTML(auto_logout_html)      
            
        return demo

def main():
    """
    UI 실행 함수
    """
    ui = RAGUI()
    demo = ui.create_ui()
    # 환경 변수에서 사용자 인증 정보 가져오기
    auth = []
    
    # 환경 변수에서 사용자 정보 가져오기
    auth_users = os.getenv("AUTH_USERS")
    if auth_users:
        try:
            users_data = json.loads(auth_users)
            for user in users_data:
                if isinstance(user, dict) and "username" in user and "password" in user:
                    auth.append((user["username"], user["password"]))
        except json.JSONDecodeError:
            print("AUTH_USERS의 형식이 올바르지 않습니다.")
    
    # 서버 시작 (인증 정보 적용)
    demo.launch(
        server_name="0.0.0.0",
        server_port=53283,
        share=False, 
        auth=auth, 
        auth_message="LLM-Driven Profile System by NHS",
        quiet=True  # Gradio 자체 로그 출력 최소화
    )

if __name__ == "__main__":
    main()
