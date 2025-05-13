#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM 팩토리 모듈

이 모듈은 다양한 LLM(Large Language Model)을 생성하기 위한 팩토리 패턴을 구현합니다.
"""
import logging
import os
from typing import Optional, Dict, Any, Union

from langchain.llms.base import BaseLLM
from langchain_community.llms import Ollama
from langchain_anthropic import ChatAnthropic
from src.rag_example.config.settings import OLLAMA_MODEL, CLAUDE_KEY, CLAUDE_MODEL

logger = logging.getLogger(__name__)

class LLMFactory:
    """
    LLM 팩토리 클래스
    
    다양한 LLM(Large Language Model)을 생성하기 위한 팩토리 패턴을 구현합니다.
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
            return LLMFactory._create_ollama(model_name, **kwargs)
        elif llm_type.lower() == "claude":
            return LLMFactory._create_claude(model_name, **kwargs)
        else:
            raise ValueError(f"지원하지 않는 LLM 타입입니다: {llm_type}")
    
    @staticmethod
    def _create_ollama(model_name: str, temperature: float = 0.1, **kwargs) -> Ollama:
        """
        Ollama LLM을 생성합니다.
        
        Args:
            model_name: 모델 이름 (예: "llama3")
            temperature: 온도 (0.0 ~ 1.0)
            **kwargs: 추가 매개변수
            
        Returns:
            생성된 Ollama LLM 객체
        """
        logger.info(f"Ollama LLM 생성: {model_name}")
        return Ollama(model=model_name, temperature=temperature, **kwargs)
    
    @staticmethod
    def _create_claude(model_name: str = CLAUDE_MODEL, temperature: float = 0.1, **kwargs) -> ChatAnthropic:
        """
        Claude LLM을 생성합니다.
        
        Args:
            model_name: 모델 이름 (예: "claude-3-opus-20240229")
            temperature: 온도 (0.0 ~ 1.0)
            **kwargs: 추가 매개변수
            
        Returns:
            생성된 Claude LLM 객체
        """
        if not CLAUDE_KEY:
            raise ValueError("ANTHROPIC_API_KEY 환경 변수가 설정되지 않았습니다.")
        
        logger.info(f"Claude LLM 생성: {model_name}")
        return ChatAnthropic(
            model=CLAUDE_MODEL,
            temperature=temperature,
            anthropic_api_key=CLAUDE_KEY,
            **kwargs
        )
