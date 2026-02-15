#!/usr/bin/env python3
"""
Unified retrieval interface for KB + Surgery DB.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from knowledge_base.rag_retriever import RAGRetriever
from knowledge_base.surgery_database import SurgeryDatabase, NodeTransformation
from knowledge_base.strategy_database import StrategyDatabase


@dataclass
class RetrievalProfile:
    name: str
    kb_max_chunks: int = 5
    kb_max_chars: int = 3000
    surgery_top_k: int = 5
    include_kb: bool = True
    include_surgery_examples: bool = True
    include_strategy_candidate: bool = False

    @classmethod
    def diagnosis(cls) -> "RetrievalProfile":
        return cls(name="diagnosis", kb_max_chunks=6, kb_max_chars=3500, surgery_top_k=3)

    @classmethod
    def strategy(cls) -> "RetrievalProfile":
        return cls(
            name="strategy",
            kb_max_chunks=6,
            kb_max_chars=4000,
            surgery_top_k=2,
            include_strategy_candidate=True,
        )

    @classmethod
    def surgery(cls) -> "RetrievalProfile":
        return cls(name="surgery", kb_max_chunks=3, kb_max_chars=2000, surgery_top_k=6)


@dataclass
class CombinedContext:
    kb_text: str = ""
    surgery_examples: List[Dict[str, Any]] = field(default_factory=list)
    strategy_candidate: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_text(self) -> str:
        parts = []
        if self.kb_text:
            parts.append(f"[KB]\n{self.kb_text}")
        if self.surgery_examples:
            parts.append("[SurgeryDB]\n" + json.dumps(self.surgery_examples, indent=2))
        if self.strategy_candidate:
            parts.append("[StrategyDB]\n" + self.strategy_candidate)
        return "\n\n".join(parts).strip()


class UnifiedRetriever:
    """Unified retrieval over KB + SurgeryDB + StrategyDB."""

    def __init__(
        self,
        rag_retriever: Optional[RAGRetriever],
        surgery_db: Optional[SurgeryDatabase],
        strategy_db: Optional[StrategyDatabase],
    ) -> None:
        self.rag_retriever = rag_retriever
        self.surgery_db = surgery_db
        self.strategy_db = strategy_db

    def retrieve(
        self,
        profile: RetrievalProfile,
        query: str,
        op_types: Optional[List[str]] = None,
        model_category: Optional[str] = None,
        architecture: Optional[str] = None,
        detected_patterns: Optional[List[str]] = None,
        blocker_ops: Optional[List[str]] = None,
    ) -> CombinedContext:
        context = CombinedContext()

        # KB retrieval (hybrid mode by default when embeddings are available)
        if profile.include_kb and self.rag_retriever:
            try:
                rag = self.rag_retriever.retrieve(query, mode="hybrid")
                context.kb_text = rag.get_text(
                    max_chunks=profile.kb_max_chunks,
                    max_chars=profile.kb_max_chars,
                )
            except Exception:
                context.kb_text = ""

        # Surgery DB retrieval
        if profile.include_surgery_examples and self.surgery_db:
            examples: List[Dict[str, Any]] = []
            op_list = list(op_types) if op_types else []
            for op_type in op_list[: profile.surgery_top_k]:
                try:
                    matches = self.surgery_db.find_similar_blocker(
                        op_type=op_type,
                        model_category=model_category,
                        top_k=1,
                    )
                    for t in matches:
                        examples.append(self._transformation_to_example(t))
                except Exception:
                    continue
            context.surgery_examples = examples[: profile.surgery_top_k]

        # Strategy candidate
        if profile.include_strategy_candidate and self.strategy_db and architecture:
            try:
                strategy = self.strategy_db.find_best_strategy(
                    architecture=architecture,
                    detected_patterns=detected_patterns or [],
                    blocker_ops=blocker_ops or [],
                )
                if strategy:
                    context.strategy_candidate = json.dumps(strategy.to_dict(), indent=2)
            except Exception:
                context.strategy_candidate = None

        context.metadata = {
            "profile": profile.name,
            "kb_chars": len(context.kb_text or ""),
            "surgery_examples": len(context.surgery_examples),
            "has_strategy_candidate": bool(context.strategy_candidate),
        }
        return context

    @staticmethod
    def _transformation_to_example(t: NodeTransformation) -> Dict[str, Any]:
        return {
            "op_type": t.original_op_type,
            "action": t.action,
            "confidence": t.confidence,
            "source_model": t.source_model,
            "surgery_steps": t.surgery_steps,
            "code_snippet": t.code_snippet,
        }
