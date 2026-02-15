#!/usr/bin/env python3
"""
LangChain client wrapper for structured LLM calls.
"""

from __future__ import annotations

import os
from typing import Optional, Type, TypeVar

from pydantic import BaseModel

try:
    from langchain_litellm import ChatLiteLLM
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        # Fallback to old import if langchain-litellm not installed
        from langchain_community.chat_models import ChatLiteLLM
        from langchain_core.output_parsers import PydanticOutputParser
        from langchain_core.prompts import PromptTemplate
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False

T = TypeVar("T", bound=BaseModel)


class LangChainClient:
    """LangChain wrapper for consistent, structured LLM calls."""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
    ) -> None:
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain dependencies are not installed.")

        os.environ["GEMINI_API_KEY"] = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.llm = ChatLiteLLM(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def invoke(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: Optional[str] = None,
    ) -> T:
        """
        Invoke LLM with structured output parsing.
        
        Note: We bypass PromptTemplate to avoid curly brace escaping issues
        when prompts contain JSON examples.
        """
        parser = PydanticOutputParser(pydantic_object=response_model)
        
        # Build the full prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        # Append format instructions
        format_instructions = parser.get_format_instructions()
        full_prompt = f"{full_prompt}\n\n{format_instructions}"
        
        # Call LLM directly (no PromptTemplate to avoid escaping issues)
        from langchain_core.messages import HumanMessage
        response = self.llm.invoke([HumanMessage(content=full_prompt)])
        
        # Parse response
        return parser.parse(response.content)
