"""
RAG 시스템 메인 모듈
"""
import os
import sys
import logging
from rag_example.pipeline import RAGPipeline
from rag_example.config.settings import RAW_DATA_DIR, LLM_TYPE

# 토크나이저 병렬 처리 관련 경고 해결
# 최신 버전에서는 명시적으로 환경 변수 설정 권장
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# 메인 로거 설정
logger = logging.getLogger(__name__)



def main():
    """RAG 시스템을 설정하고 실행합니다."""
    # 명령줄 인자 파싱
    clean_rag = '--clean-rag' in sys.argv
    
    # RAG 파이프라인 생성
    pipeline = RAGPipeline(
        document_dir=RAW_DATA_DIR,
        is_clean_vectorstore=clean_rag,
        llm_type=LLM_TYPE
    )
    # RAG 체인 가져오기
    rag_chain = pipeline.setup_chain()
    
    # 대화형 인터페이스 시작
    print("RAG 대화 시스템이 시작되었습니다. 종료하려면 'exit' 또는 'quit'를 입력하세요.")
    print("--------------------------------------------------")
    print("PDF 파일과 텍스트 파일을 활용한 질의응답 시스템입니다.\n")
    
    while True:
        try:
            # 사용자 입력 받기
            query = input("질문을 입력하세요: ")
            
            # 종료 명령 확인
            if query.lower() in ['exit', 'quit', '종료', '나가기']:
                print("사용자에 의해 종료되었습니다.")
                break
            
            # 대화 초기화 명령 확인
            if query.lower() in ['reset', 'clear', '초기화', '리셋']:
                rag_chain.reset_memory()
                print("현재 대화 기록이 초기화되었습니다.")
                continue
                
            # 모든 세션 초기화 명령 확인
            if query.lower() in ['reset all', 'clear all', '모두 초기화', '전체 초기화']:
                rag_chain.reset_memory(session_id="all")
                print("모든 세션의 대화 기록이 초기화되었습니다.")
                continue
            
            # RAG 체인 실행
            result = rag_chain.run(query)
            
            # 결과 출력 (이제 항상 문자열로 반환됨)
            print("\n답변:")
            print(result)
            print("--------------------------------------------------\n")
            
        except KeyboardInterrupt:
            print("\n사용자에 의해 종료되었습니다.")
            break
            
        except Exception as e:
            logger.error(f"오류 발생: {str(e)}")
            print(f"오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    main()