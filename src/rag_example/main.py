"""
RAG 시스템의 메인 실행 파일입니다.
"""
from pathlib import Path
from typing import List
from langchain.schema import Document
from .core.document import DocumentProcessor
from .core.vectorstore import VectorStoreManager
from .chains.rag import RAGChain
from .config.settings import RAW_DATA_DIR

def get_document_files() -> List[Path]:
    """
    RAW_DATA_DIR에서 처리할 모든 문서 파일들을 가져옵니다.
    
    Returns:
        처리할 문서 파일 경로 리스트
    """
    document_files = []
    
    # raw 폴더의 모든 파일 처리
    for file in RAW_DATA_DIR.glob("*.*"):
        # 숨김 파일이나 시스템 파일은 제외
        if not file.name.startswith('.'):
            document_files.append(file)
    
    return document_files

def process_documents(files: List[Path]) -> List[Document]:
    """
    여러 문서 파일들을 처리하여 청크로 분할합니다.
    
    Args:
        files: 처리할 문서 파일 경로 리스트
        
    Returns:
        List[Document]: 분할된 청크 리스트. 각 청크는 Document 객체입니다.
        
    Raises:
        ValueError: 문서 처리 중 오류가 발생한 경우
    """
    doc_processor = DocumentProcessor()
    all_chunks = []
    
    for file_path in files:
        print(f"문서를 처리합니다: {file_path}")
        chunks = doc_processor.load_and_split(str(file_path))
        print(f"  - {len(chunks)}개의 청크가 생성되었습니다.")
        all_chunks.extend(chunks)
    
    return all_chunks

def main():
    """RAG 시스템을 설정하고 실행합니다."""
    # 문서 파일 가져오기
    document_files = get_document_files()
    
    if not document_files:
        print(f"\n경고: {RAW_DATA_DIR}에서 처리할 문서 파일(.txt 또는 .pdf)을 찾을 수 없습니다.")
        return
    
    print(f"\n처리할 문서 파일: {len(document_files)}개")
    for file in document_files:
        print(f"  - {file.name}")
    
    # 문서 처리 및 청크 분할
    print("\n문서를 처리하고 청크로 분할합니다...")
    all_chunks = process_documents(document_files)
    print(f"\n총 {len(all_chunks)}개의 청크가 생성되었습니다.")
    
    # 벡터 저장소 생성
    print("\n벡터 저장소를 생성합니다...")
    vectorstore = VectorStoreManager().create_vectorstore(all_chunks)
    
    # RAG 체인 생성 및 실행
    print("\nConversationalRetrievalChain을 생성합니다...")
    rag_chain = RAGChain(vectorstore)
    rag_chain.run_conversation()

if __name__ == "__main__":
    main()
