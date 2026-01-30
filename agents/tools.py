#!/usr/bin/env python3
"""
Tool Functions for Graph Surgery.

Provides plain Python functions for graph surgery operations:
- analyze_model: Deep ONNX model analysis
- retrieve_patterns: RAG retrieval from knowledge base
- apply_suggestion: Apply single graph surgery suggestion
- validate_model: Validate model structure
- compare_with_gt: Compare with ground truth

These are direct functions (not LangChain tools) used by the executor.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from agents.diagnostics import FeedbackCollector, TransformationResult


# =============================================================================
# Tool Context
# =============================================================================

class ToolContext:
    """
    Context for tool functions.
    
    Maintains references to the model, applicator, and other components.
    Replaces the old global AgentContext pattern.
    """
    
    def __init__(
        self,
        model_path: str,
        api_key: Optional[str] = None,
        kb_path: str = "rag_data/knowledge_base.pkl",
    ):
        self.model_path = model_path
        self.api_key = api_key
        self.kb_path = kb_path
        
        # Model state
        self.current_model = None
        self.original_model = None
        
        # Suggestions
        self.suggestions: Dict[int, Dict] = {}  # id -> suggestion
        
        # Components (lazy initialization)
        self._analyzer = None
        self._applicator = None
        self._retriever = None
        self._comparator = None
        
        # Feedback
        self.feedback_collector = FeedbackCollector()
    
    @property
    def analyzer(self):
        """Lazy-load analyzer."""
        if self._analyzer is None:
            try:
                from core_analysis.onnx_analyzer import ONNXAnalyzer
                self._analyzer = ONNXAnalyzer()
            except ImportError:
                pass
        return self._analyzer
    
    @property
    def applicator(self):
        """Lazy-load applicator."""
        if self._applicator is None:
            try:
                from suggestion_pipeline.suggestion_applicator import SuggestionApplicator
                model_name = Path(self.model_path).stem
                self._applicator = SuggestionApplicator(model_name=model_name)
            except ImportError:
                pass
        return self._applicator
    
    @property
    def retriever(self):
        """Lazy-load retriever."""
        if self._retriever is None and self.api_key:
            try:
                from knowledge_base.rag_retriever import RAGRetriever
                self._retriever = RAGRetriever(self.kb_path, api_key=self.api_key)
            except (ImportError, Exception):
                pass
        return self._retriever
    
    @property
    def comparator(self):
        """Lazy-load comparator."""
        if self._comparator is None:
            try:
                from evaluation.model_comparator import ModelComparator
                self._comparator = ModelComparator()
            except ImportError:
                pass
        return self._comparator
    
    def load_model(self):
        """Load ONNX model."""
        import copy
        self.original_model = onnx.load(self.model_path)
        self.current_model = copy.deepcopy(self.original_model)
    
    def set_suggestions(self, suggestions: List[Dict]):
        """Set suggestions for the session."""
        self.suggestions = {s.get('id', i): s for i, s in enumerate(suggestions)}
    
    def get_suggestion(self, suggestion_id: int) -> Optional[Dict]:
        """Get suggestion by ID."""
        return self.suggestions.get(suggestion_id)


# Backward compatibility alias
AgentContext = ToolContext


# =============================================================================
# Tool Functions
# =============================================================================

def analyze_model(
    context: ToolContext,
    model_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Deep analysis of ONNX model.
    
    Returns analysis including:
    - Compilation blockers
    - Non-4D tensors
    - Dynamic shapes
    - Operation statistics
    """
    path = model_path or context.model_path
    
    if context.analyzer is None:
        return {"error": "Analyzer not available"}
    
    try:
        analysis = context.analyzer.analyze(path)
        
        return {
            "model_name": analysis.model_name,
            "node_count": len(analysis.nodes) if hasattr(analysis, 'nodes') else 0,
            "compilation_blockers": len(analysis.compilation_blockers) if hasattr(analysis, 'compilation_blockers') else 0,
            "dynamic_dimensions": len(analysis.dynamic_dimensions) if hasattr(analysis, 'dynamic_dimensions') else 0,
            "blockers": [
                {
                    "op_type": b.get("op_type"),
                    "node_name": b.get("node_name"),
                    "reason": b.get("reason"),
                }
                for b in (analysis.compilation_blockers if hasattr(analysis, 'compilation_blockers') else [])[:10]
            ],
            "op_types": dict(analysis.op_type_counts) if hasattr(analysis, "op_type_counts") else {},
        }
    except Exception as e:
        return {"error": f"Analysis failed: {e}"}


def retrieve_patterns(
    context: ToolContext,
    query: str,
    op_type: Optional[str] = None,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Retrieve similar transformation patterns from knowledge base.
    """
    if context.retriever is None:
        return {"error": "Retriever not available (missing API key or KB)"}
    
    try:
        result = context.retriever.retrieve(
            query=query,
            op_type=op_type,
            top_k=top_k
        )
        
        if not result.chunks:
            return {"patterns": [], "message": f"No patterns found for: {query}"}
        
        patterns = []
        for i, chunk in enumerate(result.chunks[:top_k]):
            pattern_text = chunk.text[:500] if hasattr(chunk, 'text') else str(chunk)[:500]
            metadata = chunk.metadata if hasattr(chunk, 'metadata') else {}
            patterns.append({
                "index": i + 1,
                "source": metadata.get('source', 'unknown'),
                "category": metadata.get('model_category', 'unknown'),
                "content": pattern_text,
            })
        
        return {"patterns": patterns, "count": len(result.chunks)}
        
    except Exception as e:
        return {"error": f"Pattern retrieval failed: {e}"}


def apply_suggestion(
    context: ToolContext,
    suggestion_id: int
) -> TransformationResult:
    """
    Apply a single graph surgery suggestion.
    
    Returns detailed transformation result.
    """
    suggestion = context.get_suggestion(suggestion_id)
    
    if suggestion is None:
        from agents.diagnostics import ErrorCategory
        return TransformationResult(
            suggestion_id=suggestion_id,
            node_name="unknown",
            op_type="unknown",
            action_type="unknown",
            success=False,
            was_transformed=False,
            error_category=ErrorCategory.NODE_NOT_FOUND,
            error_message=f"Suggestion {suggestion_id} not found",
        )
    
    if context.current_model is None:
        context.load_model()
    
    if context.applicator is None:
        from agents.diagnostics import ErrorCategory
        return TransformationResult(
            suggestion_id=suggestion_id,
            node_name=suggestion.get('location', {}).get('node_name', 'unknown'),
            op_type=suggestion.get('location', {}).get('op_type', 'unknown'),
            action_type='unknown',
            success=False,
            was_transformed=False,
            error_category=ErrorCategory.HANDLER_NOT_IMPLEMENTED,
            error_message="Applicator not available",
        )
    
    result = context.applicator.apply_single(context.current_model, suggestion)
    context.feedback_collector.add(result)
    
    return result


def validate_model(context: ToolContext) -> Dict[str, Any]:
    """
    Validate current model structure.
    """
    if context.current_model is None:
        return {"valid": False, "error": "No model loaded"}
    
    try:
        onnx.checker.check_model(context.current_model)
        
        return {
            "valid": True,
            "node_count": len(context.current_model.graph.node),
            "input_count": len(context.current_model.graph.input),
            "output_count": len(context.current_model.graph.output),
        }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
        }


def compare_with_ground_truth(
    context: ToolContext,
    ground_truth_path: str
) -> Dict[str, Any]:
    """
    Compare modified model against ground truth.
    """
    if context.current_model is None:
        return {"error": "No model loaded"}
    
    if context.comparator is None:
        return {"error": "Comparator not available"}
    
    try:
        ground_truth = onnx.load(ground_truth_path)
        
        comparison = context.comparator.compare_models(
            context.current_model,
            ground_truth,
            context.original_model
        )
        
        return {
            "overall_similarity": comparison.get('overall_similarity', 0),
            "operation_similarity": comparison.get('operation_similarity', 0),
            "transformation_score": comparison.get('transformation_accuracy', {}).get('transformation_score', 0),
            "critical_areas_match": comparison.get('transformation_accuracy', {}).get('critical_areas_match', 0),
            "details": comparison.get('transformation_accuracy', {}),
        }
    except Exception as e:
        return {"error": f"Comparison failed: {e}"}


def get_feedback(context: ToolContext, detailed: bool = False) -> Dict[str, Any]:
    """
    Get aggregated feedback from all transformations.
    """
    if detailed:
        return context.feedback_collector.to_dict()
    else:
        return context.feedback_collector.get_summary()


# =============================================================================
# Backward Compatibility
# =============================================================================

# Global context for backward compatibility (deprecated)
_agent_context: Optional[ToolContext] = None


def set_agent_context(context: ToolContext):
    """Set the global agent context (deprecated)."""
    global _agent_context
    _agent_context = context


def get_agent_context() -> Optional[ToolContext]:
    """Get the global agent context (deprecated)."""
    return _agent_context


# Wrapper functions that use global context (deprecated)
def _analyze_model_impl(model_path: Optional[str] = None) -> str:
    """Backward compatible wrapper."""
    context = get_agent_context()
    if context is None:
        return "Error: Agent context not set"
    result = analyze_model(context, model_path)
    return json.dumps(result, indent=2)


def _retrieve_patterns_impl(query: str, op_type: Optional[str] = None, top_k: int = 5) -> str:
    """Backward compatible wrapper."""
    context = get_agent_context()
    if context is None:
        return "Error: Agent context not set"
    result = retrieve_patterns(context, query, op_type, top_k)
    return json.dumps(result, indent=2)


def _apply_suggestion_impl(suggestion_id: int) -> str:
    """Backward compatible wrapper."""
    context = get_agent_context()
    if context is None:
        return "Error: Agent context not set"
    result = apply_suggestion(context, suggestion_id)
    return result.to_observation_string()


def _validate_model_impl() -> str:
    """Backward compatible wrapper."""
    context = get_agent_context()
    if context is None:
        return "Error: Agent context not set"
    result = validate_model(context)
    if result.get("valid"):
        return f"Model Validation: PASSED\n- Nodes: {result['node_count']}\n- Inputs: {result['input_count']}\n- Outputs: {result['output_count']}"
    else:
        return f"Model Validation: FAILED\nError: {result.get('error', 'unknown')}"


def _compare_with_gt_impl(ground_truth_path: str) -> str:
    """Backward compatible wrapper."""
    context = get_agent_context()
    if context is None:
        return "Error: Agent context not set"
    result = compare_with_ground_truth(context, ground_truth_path)
    return json.dumps(result, indent=2)


def _get_feedback_impl(detailed: bool = False) -> str:
    """Backward compatible wrapper."""
    context = get_agent_context()
    if context is None:
        return "Error: Agent context not set"
    result = get_feedback(context, detailed)
    return json.dumps(result, indent=2)


def _request_strategy_change_impl(reason: str) -> str:
    """Backward compatible wrapper (no-op without callback)."""
    return "Strategy change not available (use executor with callback instead)"


def create_graph_surgery_tools(context: Optional[ToolContext] = None) -> List:
    """
    Create tool list (backward compatible - returns empty list).
    
    The new executor doesn't use LangChain tools, so this returns
    an empty list for backward compatibility.
    """
    if context:
        set_agent_context(context)
    return []


def create_simple_tools(context: Optional[ToolContext] = None) -> List:
    """
    Create simple tool list (backward compatible - returns empty list).
    """
    if context:
        set_agent_context(context)
    return []


# Export
__all__ = [
    # New API
    "ToolContext",
    "analyze_model",
    "retrieve_patterns",
    "apply_suggestion",
    "validate_model",
    "compare_with_ground_truth",
    "get_feedback",
    # Backward compatibility
    "AgentContext",
    "set_agent_context",
    "get_agent_context",
    "create_graph_surgery_tools",
    "create_simple_tools",
]
