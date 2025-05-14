"""
프롬프트 템플릿을 관리하는 모듈입니다.
"""
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

def get_condense_prompt() -> PromptTemplate:
    """
    이전 대화 기록과 사용자 질문을 바탕으로,
    대화 흐름에 맞는 자연스러운 단일 질문으로 재작성하기 위한 프롬프트입니다.
    """
    template = """당신은 대화 흐름을 이해하고 질문을 자연스럽게 이어주는 AI 어시스턴트입니다.

다음은 지금까지의 대화 내용입니다:
------------------------
{chat_history}
------------------------

사용자가 아래와 같은 질문을 했습니다:
"{question}"

이전 대화를 반영하여 이 질문을 명확하고 간결하게 다시 표현해주세요.
※ 단독 질문으로도 의미가 통하는 형태로 재작성해야 합니다."""

    return PromptTemplate.from_template(template)

def get_qa_prompt() -> ChatPromptTemplate:
    """
    문서 내용을 기반으로 질문에 답변하기 위한 최적화된 프롬프트 템플릿을 반환합니다.
    """
    system_template = """당신은 사용자 질문에 친절하고 명확하게 답변하는 AI 비서입니다.
모든 응답은 한국어로 제공해야 하며, 불필요한 외국어나 이상한 문자는 포함하지 마세요.

대화 이력 (요약 + 최근):
------------------------
{chat_history}
------------------------

이전 대화를 참고해 사용자 질문에 자연스럽게 이어서 답변하세요."""
    human_template = """📌 사용자 질문: {question}

📚 참고 문서:
------------------------
{context}
------------------------

문서를 기반으로 정확하고 간결하게 답변해주세요."""

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    return chat_prompt