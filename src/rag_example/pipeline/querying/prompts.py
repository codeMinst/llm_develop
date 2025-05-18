"""
프롬프트 템플릿을 관리하는 모듈입니다.
"""
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


def get_summary_check_prompt() -> ChatPromptTemplate:
    system_template = """다음 사용자 질문이 '정리된 형태의 요약 정보'를 원하는지 판단하세요.

예를 들어 아래와 같은 질문은 모두 "YES"로 간주합니다:
- 이력/경력 요약을 보고 싶어요
- 핵심 프로젝트를 알려주세요
- 어떤 방식으로 일하시는지 설명해주세요
- 전체 경력을 간단히 정리해 주세요

단순 정보 요청이나 특정 기술 질문은 "NO"입니다.

반드시 "YES" 또는 "NO"로만 답변하세요."""
    human_template = "{question}"

    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ])



def get_summary_type_prompt() -> ChatPromptTemplate:
    """
    요약 질문인 경우, 어떤 종류의 요약을 요청하는지 분류하기 위한 프롬프트 템플릿입니다.
    """
    system_template = """아래 사용자 질문이 요청하는 요약의 종류를 판단하세요.

선택 가능한 요약 유형:
- resume: 이력서 중심 요약
- projects: 주요 프로젝트 요약
- workstyle: 업무 스타일 요약
- all: 전체 요약
- none: 어느 유형에도 해당하지 않음

반드시 위 다섯 가지 중 하나로만 정확히 출력하세요."""
    human_template = "{question}"

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    return ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
