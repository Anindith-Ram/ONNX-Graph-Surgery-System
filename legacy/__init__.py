"""
DEPRECATED: Legacy modules have been removed.

This package is deprecated. All functionality has been migrated to:

- knowledge_base/surgery_database.py - Unified data structures
- core_analysis/transformation_extractor.py - Precise transformation extraction
- knowledge_base/llm_context_generator.py - Rich LLM context
- suggestion_pipeline/rag_suggestion_generator.py - RAG-enhanced suggestions
- suggestion_pipeline/suggestion_applicator.py - Suggestion application

Migration guide:
- RuleParser -> SurgeryTemplate from surgery_database
- RuleApplicator -> SuggestionApplicator
- GeminiModelModifier -> RAGSuggestionGenerator + LLMContextGenerator
- EnhancedFeatureExtractor -> TransformationExtractor
"""

import warnings
warnings.warn(
    "The 'legacy' package is deprecated and empty. All modules have been migrated.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = []
