#!/usr/bin/env python3
"""
LLM Client for Graph Surgery Pipeline.

Provides a unified interface for LLM calls using LiteLLM and Instructor
for structured output extraction with automatic retries.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel

# LiteLLM for unified LLM API
try:
    import litellm
    from litellm import completion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

# Instructor for structured outputs
try:
    import instructor
    INSTRUCTOR_AVAILABLE = True
except ImportError:
    INSTRUCTOR_AVAILABLE = False

# Fallback to google-generativeai
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """
    Unified LLM client with structured output support.
    
    Features:
    - Structured outputs via Instructor + Pydantic
    - Automatic retries with exponential backoff
    - Fallback between LiteLLM and direct Gemini calls
    - Simple interface for graph surgery pipeline
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gemini/gemini-3-pro-preview",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        max_retries: int = 3,
        timeout: float = 60.0,
    ):
        """
        Initialize LLM client.
        
        Args:
            api_key: API key for the LLM provider (Gemini/Google)
            model: Model identifier (LiteLLM format: "provider/model")
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            max_retries: Number of retries on failure
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Set up API key for LiteLLM (Gemini)
        os.environ["GEMINI_API_KEY"] = api_key
        
        # Initialize instructor client if available
        self._instructor_client = None
        if LITELLM_AVAILABLE and INSTRUCTOR_AVAILABLE:
            self._instructor_client = instructor.from_litellm(completion)
        
        # Fallback: direct Gemini client
        self._gemini_model = None
        if GEMINI_AVAILABLE:
            genai.configure(api_key=api_key)
            # Extract model name from LiteLLM format
            model_name = model.split("/")[-1] if "/" in model else model
            self._gemini_model = genai.GenerativeModel(model_name)
    
    def call(
        self,
        prompt: str,
        response_model: Optional[Type[T]] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Union[T, str]:
        """
        Call the LLM with optional structured output.
        
        Args:
            prompt: User prompt
            response_model: Optional Pydantic model for structured output
            system_prompt: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            
        Returns:
            Parsed response model instance if response_model provided,
            otherwise raw string response
        """
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        # Try structured output with Instructor first
        if response_model and self._instructor_client:
            return self._call_with_instructor(
                prompt, response_model, system_prompt, temp, tokens
            )
        
        # Fallback to raw call + manual parsing
        raw_response = self._call_raw(prompt, system_prompt, temp, tokens)
        
        if response_model:
            return self._parse_response(raw_response, response_model)
        
        return raw_response
    
    def _call_with_instructor(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> T:
        """Call LLM with Instructor for structured output."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        for attempt in range(self.max_retries):
            try:
                response = self._instructor_client.chat.completions.create(
                    model=self.model,
                    response_model=response_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self.timeout,
                )
                return response
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                wait_time = (2 ** attempt) + 0.5
                print(f"  LLM call failed (attempt {attempt + 1}): {e}")
                print(f"  Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
        
        raise RuntimeError("All retry attempts failed")
    
    def _call_raw(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Make a raw LLM call and return string response."""
        # Try LiteLLM first
        if LITELLM_AVAILABLE:
            return self._call_litellm(prompt, system_prompt, temperature, max_tokens)
        
        # Fallback to direct Gemini
        if self._gemini_model:
            return self._call_gemini_direct(prompt, system_prompt, temperature, max_tokens)
        
        raise RuntimeError(
            "No LLM backend available. Install litellm or google-generativeai."
        )
    
    def _call_litellm(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call LLM using LiteLLM."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        for attempt in range(self.max_retries):
            try:
                response = completion(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self.timeout,
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                wait_time = (2 ** attempt) + 0.5
                print(f"  LiteLLM call failed (attempt {attempt + 1}): {e}")
                time.sleep(wait_time)
        
        raise RuntimeError("All retry attempts failed")
    
    def _call_gemini_direct(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call Gemini directly without LiteLLM."""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        for attempt in range(self.max_retries):
            try:
                response = self._gemini_model.generate_content(
                    full_prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    ),
                )
                return response.text
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                wait_time = (2 ** attempt) + 0.5
                print(f"  Gemini call failed (attempt {attempt + 1}): {e}")
                time.sleep(wait_time)
        
        raise RuntimeError("All retry attempts failed")
    
    def _parse_response(self, raw_response: str, response_model: Type[T]) -> T:
        """Parse raw response into Pydantic model."""
        # Clean up response
        text = raw_response.strip()
        
        # Extract JSON from markdown code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        
        # Find JSON object/array
        if not text.startswith("{") and not text.startswith("["):
            # Try to find JSON in response
            obj_start = text.find("{")
            arr_start = text.find("[")
            
            if obj_start >= 0 and (arr_start < 0 or obj_start < arr_start):
                text = text[obj_start:]
            elif arr_start >= 0:
                text = text[arr_start:]
        
        # Parse JSON
        try:
            data = json.loads(text)
            return response_model.model_validate(data)
        except json.JSONDecodeError as e:
            # Try to fix common JSON issues
            text = self._fix_json(text)
            data = json.loads(text)
            return response_model.model_validate(data)
    
    def _fix_json(self, text: str) -> str:
        """Attempt to fix common JSON formatting issues."""
        # Remove trailing commas before } or ]
        import re
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)
        
        # Ensure proper closing
        open_braces = text.count('{') - text.count('}')
        open_brackets = text.count('[') - text.count(']')
        
        text = text + ('}' * open_braces) + (']' * open_brackets)
        
        return text


# Convenience function for simple calls
def call_llm(
    prompt: str,
    api_key: str,
    response_model: Optional[Type[T]] = None,
    model: str = "gemini/gemini-3-pro-preview",
    temperature: float = 0.1,
    system_prompt: Optional[str] = None,
) -> Union[T, str]:
    """
    Convenience function for one-off LLM calls.
    
    For repeated calls, create an LLMClient instance instead.
    """
    client = LLMClient(api_key=api_key, model=model, temperature=temperature)
    return client.call(prompt, response_model, system_prompt)


# Export availability flags
__all__ = [
    "LLMClient",
    "call_llm",
    "LITELLM_AVAILABLE",
    "INSTRUCTOR_AVAILABLE",
    "GEMINI_AVAILABLE",
]
