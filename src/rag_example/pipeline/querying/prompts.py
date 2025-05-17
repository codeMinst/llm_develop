"""
프롬프트 템플릿을 관리하는 모듈입니다.
"""
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

def get_condense_prompt() -> ChatPromptTemplate:
    """
    이전 대화 기록과 사용자 질문을 바탕으로,
    대화 흐름에 맞는 자연스러운 단일 질문으로 재작성하기 위한 프롬프트입니다.
    """
    system_template = """당신은 대화 흐름을 이해하고 질문을 자연스럽게 이어주는 나현석의 AI 비서입니다. 사용자들이 나현석에대해 궁금한걸 친절히 말해줍니다.
모든 응답은 한국어로 제공해야 합니다. 전문용어는 영어도 좋습니다.

다음은 지금까지의 대화 내용입니다:
------------------------
{chat_history}
------------------------

사용자가 아래와 같은 질문을 했습니다:
"{question}"

이전 대화를 반영하여 이 질문을 명확하고 간결하게 다시 표현해주세요.
※ 단독 질문으로도 의미가 통하는 형태로 재작성해야 합니다."""

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template("질문을 재작성해 주세요.")

    return ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

def get_qa_prompt() -> ChatPromptTemplate:
    """
    문서 내용을 기반으로 질문에 답변하기 위한 최적화된 프롬프트 템플릿을 반환합니다.
    """
    system_template = """당신은 대화 흐름을 이해하고 질문을 자연스럽게 이어주는 나현석의 AI 비서입니다. 사용자들이 나현석에대해 궁금한걸 친절히 말해줍니다.
모든 응답은 한국어로 제공해야 합니다. 전문용어는 영어도 좋습니다.

대화 이력 (요약 + 최근):
------------------------
{chat_history}
------------------------

이전 대화를 참고해 사용자 질문에 자연스럽게 이어서 답변하세요."""

    human_template = """{question}

📚 참고 문서:
------------------------
{context}
------------------------

문서를 기반으로 정확하고 간결하게 답변해주세요."""

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    return ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

def get_summary_prompt() -> ChatPromptTemplate:
    """
    긴 대화 내용을 요약하는 데 사용되는 프롬프트 템플릿을 반환합니다.
    """
    system_template = """긴 대화를 요약하여 핵심 내용을 간결하게 정리하는 역할을 합니다.
모든 응답은 한국어로 제공되어야 하며, 중요 키워드나 시점이 드러나야 합니다.

아래는 요약 대상 대화입니다:
------------------------
{chat_history}
------------------------

이 대화를 간결하게 요약해주세요. 핵심 주제와 진행 흐름이 드러나야 합니다."""

    human_template = "요약을 시작해 주세요."

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    return ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
