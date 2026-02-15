#!/usr/bin/env python3
"""
Base agent for the fully agentic strategic pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

from agents.langchain_client import LangChainClient
from agents.config import PipelineConfig
from knowledge_base.rag_retriever import RAGRetriever
from knowledge_base.surgery_database import SurgeryDatabase
from knowledge_base.strategy_database import StrategyDatabase
from agents.base.unified_retriever import UnifiedRetriever, RetrievalProfile, CombinedContext

T = TypeVar("T", bound=BaseModel)


class BaseAgent:
    """
    Base class for specialized agents.

    Provides:
    - LLM access with consistent settings
    - RAG retrieval for contextual examples
    - Surgery/strategy database access
    """

    def __init__(
        self,
        api_key: str,
        pipeline_config: PipelineConfig,
        agent_model_name: str,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self.api_key = api_key
        self.pipeline_config = pipeline_config

        # LLM client (LangChain with structured output)
        self.llm = LangChainClient(
            api_key=api_key,
            model_name=agent_model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # RAG retriever
        self.rag_retriever: Optional[RAGRetriever] = None
        try:
            self.rag_retriever = RAGRetriever(
                kb_path=pipeline_config.kb_path,
                api_key=api_key,
            )
        except Exception:
            self.rag_retriever = None

        # Surgery database
        self.surgery_db: Optional[SurgeryDatabase] = None
        try:
            self.surgery_db = SurgeryDatabase.load(pipeline_config.surgery_db_path)
        except Exception:
            self.surgery_db = None

        # Strategy database
        self.strategy_db: Optional[StrategyDatabase] = None
        try:
            self.strategy_db = StrategyDatabase.load(pipeline_config.strategy_db_path)
        except Exception:
            self.strategy_db = StrategyDatabase.create_with_defaults()

        # Unified retriever
        self.unified_retriever = UnifiedRetriever(
            rag_retriever=self.rag_retriever,
            surgery_db=self.surgery_db,
            strategy_db=self.strategy_db,
        )

    def retrieve_combined(
        self,
        profile: RetrievalProfile,
        query: str,
        op_types: Optional[List[str]] = None,
        model_category: Optional[str] = None,
        architecture: Optional[str] = None,
        detected_patterns: Optional[List[str]] = None,
        blocker_ops: Optional[List[str]] = None,
    ) -> CombinedContext:
        """Retrieve combined context from KB + Surgery DB."""
        return self.unified_retriever.retrieve(
            profile=profile,
            query=query,
            op_types=op_types,
            model_category=model_category,
            architecture=architecture,
            detected_patterns=detected_patterns,
            blocker_ops=blocker_ops,
        )

    def call_llm(self, prompt: str, response_model: Type[T], system_prompt: Optional[str] = None) -> T:
        """Call the LLM with optional structured output."""
        return self.llm.invoke(
            prompt=prompt,
            response_model=response_model,
            system_prompt=system_prompt,
        )
