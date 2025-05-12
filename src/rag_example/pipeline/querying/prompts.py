"""
프롬프트 템플릿을 관리하는 모듈입니다.
"""
from langchain.prompts import PromptTemplate

def get_condense_prompt() -> PromptTemplate:
    """
    이전 대화 기록을 고려하여 현재 질문을 다시 작성하기 위한 프롬프트 템플릿을 반환합니다.
    
    Returns:
        생성된 프롬프트 템플릿
    """
    template = """다음은 사용자와 AI 비서 간의 친절한 대화입니다.
    AI 비서는 상세하고, 도움이 되며, 정확한 답변을 제공합니다.
    
    대화 기록:
    {chat_history}
    
    사용자의 질문: {question}
    """
    return PromptTemplate.from_template(template)

def get_qa_prompt() -> PromptTemplate:
    """
    문서 내용을 기반으로 질문에 답변하기 위한 프롬프트 템플릿을 반환합니다.
    
    Returns:
        생성된 프롬프트 템플릿
    """
    template = """다음은 사용자와 AI 비서 간의 친절한 대화입니다.
    AI 비서는 한글로 상세하고, 도움이 되며, 정확한 답변을 제공합니다.
    외국어나 이상한 문자는 사용하지 마세요. 반드시 한글로만 답변해야 합니다.
    
    대화 기록:
    {chat_history}
    
    질문: {question}
    
    다음 문서 내용을 참고하여 질문에 답변하세요:
    {context}
    
    답변:
    """
    return PromptTemplate.from_template(template)
