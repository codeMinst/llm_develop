#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG 시스템 메인 모듈
"""
import sys
import logging
from .pipeline import create_rag_pipeline

# 로깅 설정
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from .config.settings import RAW_DATA_DIR

def main():
    """RAG 시스템을 설정하고 실행합니다."""
    # 명령줄 인자 파싱
    clean_rag = '--clean-rag' in sys.argv
    
    # RAG 파이프라인 생성 및 실행
    pipeline = create_rag_pipeline(
        document_dir=RAW_DATA_DIR,
        clean_vectorstore=clean_rag
    )
    
    # RAG 체인 가져오기
    rag_chain = pipeline.rag_chain_builder
    
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
                print("대화 기록이 초기화되었습니다.")
                continue
            
            # RAG 체인 실행
            result = rag_chain.run(query)
            
            # 결과 출력
            print("\n답변:")
            print(result['answer'])
            print("--------------------------------------------------\n")
            
        except KeyboardInterrupt:
            print("\n사용자에 의해 종료되었습니다.")
            break
            
        except Exception as e:
            logger.error(f"오류 발생: {str(e)}")
            print(f"오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    main()