"""
요약 기능이 있는 메모리 모듈

이 모듈은 대화 기록을 자동으로 요약하는 메모리 클래스를 제공합니다.
대화가 길어질수록 토큰 수가 증가하는 문제를 해결하기 위해,
오래된 대화는 요약하고 최근 대화만 전체 내용을 유지합니다.
"""
import logging
from typing import List, Optional

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.llms.base import BaseLLM
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SummarizingMemory(BaseChatMessageHistory, BaseModel):
    """
    요약 + 최근 메시지를 저장하는 메모리 클래스
    
    대화가 길어질 때 자동으로 오래된 대화를 요약하여 토큰 폭발 문제를 방지합니다.
    최근 대화는 그대로 유지하여 컨텍스트의 연속성을 보장합니다.d
    """
    messages: List[BaseMessage] = Field(default_factory=list)
    session_id: str = Field(default="default")
    llm: Optional[BaseLLM] = None
    max_recent_turns: int = 4
    summary: str = ""

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """메세지 목록을 히스토리에 추가합니다."""
        self.messages.extend(messages)
        self._maybe_summarize()

    def _maybe_summarize(self):
        """대화가 길어지면 자동으로 요약합니다."""
        if len(self.messages) <= self.max_recent_turns * 2:
            return  # 충분히 짧음

        # 오래된 메시지 요약
        to_summarize = self.messages[:-self.max_recent_turns * 2]
        recent = self.messages[-self.max_recent_turns * 2:]

        text = self._format_history(to_summarize)
        summary_prompt = f"다음 대화를 요약해줘:\n{text}"
        if self.llm:
            try:
                self.summary = self.llm.invoke(summary_prompt)
                logger.info(f"대화 요약 완료 (요약 길이: {len(self.summary)}, 요약된 메시지 수: {len(to_summarize)})")
            except Exception as e:
                logger.warning(f"요약 실패: {e}")

        self.messages = recent  # 최근 메시지만 유지

    def _format_history(self, msgs: List[BaseMessage]) -> str:
        """메시지 목록을 텍스트로 포맷팅합니다."""
        lines = []
        for msg in msgs:
            role = "Human" if isinstance(msg, HumanMessage) else "AI"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    def load_summary_and_recent(self) -> str:
        """요약과 최근 대화를 하나의 문자열로 반환합니다."""
        recent_text = self._format_history(self.messages)
        return f"{self.summary}\n\n{recent_text}"

    def clear(self) -> None:
        """메세지 히스토리를 초기화합니다."""
        self.messages = []
        self.summary = ""
