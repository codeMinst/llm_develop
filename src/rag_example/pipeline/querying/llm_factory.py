#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM 팩토리 모듈

이 모듈은 다양한 LLM(Large Language Model)을 생성하고 관리하기 위한 팩토리 패턴을 구현합니다.
"""
import logging
from typing import Any, Protocol

from langchain.llms.base import BaseLLM
from langchain_community.llms import Ollama
from langchain_anthropic import ChatAnthropic
from langchain.callbacks import StdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from src.rag_example.config.settings import OLLAMA_MODEL, CLAUDE_KEY, CLAUDE_MODEL
from src.rag_example.config.settings import IS_VERBOSE

logger = logging.getLogger(__name__)

class ResponseHandler(Protocol):
    """
    LLM 응답 처리를 위한 프로토콜
    """
    def process_response(self, response: Any) -> str:
        """
        LLM 응답을 처리하여 문자열로 반환합니다.
        
        Args:
            response: LLM 응답 객체
            
        Returns:
            처리된 응답 문자열
        """
        ...


class OllamaResponseHandler:
    """
    Ollama LLM 응답 처리기
    """
    @staticmethod
    def process_response(response: Any) -> str:
        """
        Ollama LLM 응답을 처리합니다.
        
        Args:
            response: Ollama LLM 응답 (문자열)
            
        Returns:
            처리된 응답 문자열
        """
        if isinstance(response, str):
            return response
        return str(response)


class ClaudeResponseHandler:
    """
    Claude LLM 응답 처리기
    """
    @staticmethod
    def process_response(response: Any) -> str:
        """
        Claude LLM 응답을 처리합니다.
        
        Args:
            response: Claude LLM 응답 (content 속성을 가진 객체)
            
        Returns:
            처리된 응답 문자열
        """
        if hasattr(response, 'content'):
            return response.content
        elif isinstance(response, str):
            return response
        return str(response)


class VerboseCallbackHandler(BaseCallbackHandler):
    """상세한 API 요청과 응답을 로깅하는 콜백 핸들러"""
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        """LLM 시작 시 호출"""
        logger.info(f"\n\n==== LLM 요청 시작 ====")
        for i, prompt in enumerate(prompts):
            logger.info(f"\n[요청 {i+1}]\n{prompt}\n")
    
    def on_llm_end(self, response, **kwargs):
        """LLM 종료 시 호출"""
        logger.debug(f"==== LLM 응답 ====\n{response}")
        
    def on_llm_error(self, error, **kwargs):
        """LLM 오류 발생 시 호출"""
        logger.error(f"==== LLM 오류 ====\n{error}")


class LLMFactory:
    """
    LLM 팩토리 클래스
    
    다양한 LLM(Large Language Model)을 생성하고 관리하기 위한 팩토리 패턴을 구현합니다.
    현재 지원하는 LLM:
    - Ollama: 로컬에서 실행되는 LLM
    - Claude: Anthropic의 Claude API
    """
    
    @staticmethod
    def create_llm(llm_type: str, model_name: str, **kwargs) -> BaseLLM:
        """
        LLM을 생성합니다.
        
        Args:
            llm_type: LLM 타입 ("ollama" 또는 "claude")
            model_name: 모델 이름
            **kwargs: 추가 매개변수
            
        Returns:
            생성된 LLM 객체
            
        Raises:
            ValueError: 지원하지 않는 LLM 타입인 경우
        """
        if llm_type.lower() == "ollama":
            return LLMFactory._create_ollama(**kwargs)
        elif llm_type.lower() == "claude":
            return LLMFactory._create_claude(**kwargs)
        else:
            raise ValueError(f"지원하지 않는 LLM 타입입니다: {llm_type}")
    
    @staticmethod
    def get_response_handler(llm_type: str) -> ResponseHandler:
        """
        LLM 타입에 맞는 응답 처리기를 가져옵니다.
        
        Args:
            llm_type: LLM 타입 ("ollama" 또는 "claude")
            
        Returns:
            응답 처리기 객체
            
        Raises:
            ValueError: 지원하지 않는 LLM 타입인 경우
        """
        if llm_type.lower() == "ollama":
            return OllamaResponseHandler()
        elif llm_type.lower() == "claude":
            return ClaudeResponseHandler()
        else:
            raise ValueError(f"지원하지 않는 LLM 타입입니다: {llm_type}")
    
    @staticmethod
    def process_response(llm_type: str, response: Any) -> str:
        """
        LLM 타입에 맞는 응답 처리기를 사용하여 응답을 처리합니다.
        
        Args:
            llm_type: LLM 타입 ("ollama" 또는 "claude")
            response: LLM 응답 객체
            
        Returns:
            처리된 응답 문자열
        """
        handler = LLMFactory.get_response_handler(llm_type)
        return handler.process_response(response)
    
    @staticmethod
    def _create_ollama(temperature: float = 0.1, **kwargs) -> Ollama:
        """
        Ollama LLM을 생성합니다.
        
        Args:
            temperature: 온도 (0.0 ~ 1.0)
            **kwargs: 추가 매개변수
            
        Returns:
            생성된 Ollama LLM 객체
        """
        logger.info(f"Ollama LLM 생성: {OLLAMA_MODEL}")
        
        # 콜백 핸들러 설정
        callbacks = []
        if IS_VERBOSE:
            callbacks.append(StdOutCallbackHandler())
            callbacks.append(VerboseCallbackHandler())
            
        return Ollama(
            model=OLLAMA_MODEL, 
            verbose=IS_VERBOSE,
            callbacks=callbacks,
            temperature=temperature,
            **kwargs)
    
    @staticmethod
    def _create_claude(temperature: float = 0.1, **kwargs) -> ChatAnthropic:
        """
        Claude LLM을 생성합니다.
        
        Args:
            temperature: 온도 (0.0 ~ 1.0)
            **kwargs: 추가 매개변수
            
        Returns:
            생성된 Claude LLM 객체
        """
        if not CLAUDE_KEY:
            raise ValueError("CLAUDE_KEY 환경 변수가 설정되지 않았습니다.")
        
        logger.info(f"Claude LLM 생성: {CLAUDE_MODEL}")
        return ChatAnthropic(
            model=CLAUDE_MODEL,
            temperature=temperature,
            verbose=IS_VERBOSE,
            anthropic_api_key=CLAUDE_KEY,
            **kwargs
        )
