#!/usr/bin/env python3
"""
Strategy Planner for Graph Surgery.

Simplified replacement for Tree of Thoughts (ToT) planner that:
- Generates transformation strategies with a single LLM call
- Uses pattern database for confidence scoring
- Provides structured outputs via Pydantic + Instructor
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.config import StrategyConfig
from agents.llm_client import LLMClient


# =============================================================================
# Pydantic Models for Structured Output
# =============================================================================

class TransformationStrategy(BaseModel):
    """
    A transformation strategy for graph surgery.
    
    Guides the executor on how to approach transformations.
    """
    
    name: str = Field(description="Human-readable strategy name")
    priority_order: List[str] = Field(
        description="Order of transformation types to apply",
        default_factory=lambda: ["critical_blockers", "shape_issues", "optimizations"]
    )
    approach: str = Field(
        description="Strategy approach: aggressive_removal, conservative, or hybrid",
        default="hybrid"
    )
    estimated_success: float = Field(
        description="Estimated probability of success (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
        default=0.5
    )
    rationale: str = Field(
        description="Explanation of why this strategy was chosen",
        default=""
    )
    key_transformations: List[str] = Field(
        description="Key transformations to apply",
        default_factory=list
    )
    risk_level: str = Field(
        description="Risk level: low, medium, or high",
        default="medium"
    )
    
    # Metadata
    model_category: str = Field(default="")
    generated_at: float = Field(default_factory=time.time)
    pattern_confidence: float = Field(
        description="Confidence from pattern database (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
        default=0.0
    )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict) -> "TransformationStrategy":
        """Create from dictionary."""
        return cls.model_validate(data)


class StrategyList(BaseModel):
    """List of strategies from LLM response."""
    strategies: List[TransformationStrategy] = Field(
        description="List of transformation strategies"
    )


class StrategyEvaluation(BaseModel):
    """Evaluation result for a strategy."""
    
    likelihood_of_success: float = Field(ge=0.0, le=1.0, default=0.5)
    risk_of_breaking_model: float = Field(ge=0.0, le=1.0, default=0.3)
    alignment_with_patterns: float = Field(ge=0.0, le=1.0, default=0.5)
    estimated_complexity: float = Field(ge=0.0, le=1.0, default=0.5)
    overall_score: float = Field(ge=0.0, le=1.0, default=0.5)
    reasoning: str = Field(default="")
    
    def to_dict(self) -> Dict:
        return self.model_dump()


# =============================================================================
# Prompts
# =============================================================================

STRATEGY_SYSTEM_PROMPT = """You are an expert ONNX graph surgery strategist specializing in making neural network models compilable on hardware accelerators (MLA).

Your task is to generate transformation strategies based on model analysis and historical patterns from successfully modified models.

Key principles:
1. Prioritize critical blockers that prevent compilation
2. Consider the model category (YOLO, ViT, Transformer, CNN) for category-specific patterns
3. Balance between aggressive removal and conservative transformation
4. Leverage patterns that have worked on similar models"""

STRATEGY_GENERATION_PROMPT = """Generate {num_strategies} transformation strategies for this model:

MODEL ANALYSIS:
- Model Name: {model_name}
- Model Category: {model_category}
- Total Nodes: {node_count}
- Compilation Blockers: {blocker_count}
- Non-4D Tensors: {non_4d_count}
- Dynamic Dimensions: {dynamic_count}

BLOCKER DETAILS:
{blocker_details}

SIMILAR PATTERNS FROM SUCCESSFUL MODELS:
{pattern_context}

Generate {num_strategies} diverse strategies with different approaches (aggressive, conservative, hybrid).
Each strategy should specify:
- name: descriptive name
- approach: aggressive_removal, conservative, or hybrid
- priority_order: list of transformation types in order
- rationale: why this approach
- estimated_success: 0.0-1.0 probability
- risk_level: low, medium, or high
- key_transformations: specific operations to perform"""


# =============================================================================
# Strategy Planner
# =============================================================================

class StrategyPlanner:
    """
    Simplified strategy planner for graph surgery.
    
    Replaces complex ToT tree traversal with:
    - Single LLM call for strategy generation
    - Pattern database for confidence boosting
    - Structured Pydantic outputs
    """
    
    def __init__(
        self,
        api_key: str,
        config: Optional[StrategyConfig] = None,
        kb_path: str = "rag_data/knowledge_base.pkl",
        pattern_db: Optional[Any] = None,  # PatternDatabase when available
    ):
        """
        Initialize strategy planner.
        
        Args:
            api_key: API key for LLM provider
            config: Strategy configuration
            kb_path: Path to knowledge base for historical patterns
            pattern_db: Optional pattern database for confidence scoring
        """
        self.api_key = api_key
        self.config = config or StrategyConfig()
        self.kb_path = kb_path
        self.pattern_db = pattern_db
        
        # Initialize LLM client
        self.llm = LLMClient(
            api_key=api_key,
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        
        # Cache for strategies
        self._strategy_cache: Dict[str, List[TransformationStrategy]] = {}
    
    def generate_strategy(
        self,
        analysis: Any,
        model_category: str = "Other",
        pattern_context: Optional[str] = None,
    ) -> TransformationStrategy:
        """
        Generate a single best strategy for the model.
        
        This is the simplified interface - generates strategies and selects best.
        
        Args:
            analysis: Model analysis (from ONNXAnalyzer or suggestion report)
            model_category: Detected model category
            pattern_context: Optional context from pattern database
            
        Returns:
            Best TransformationStrategy
        """
        strategies = self.generate_strategies(analysis, model_category, pattern_context)
        return self.select_strategy(strategies, analysis, model_category)
    
    def generate_strategies(
        self,
        analysis: Any,
        model_category: str = "Other",
        pattern_context: Optional[str] = None,
    ) -> List[TransformationStrategy]:
        """
        Generate multiple transformation strategies.
        
        Args:
            analysis: Model analysis (from ONNXAnalyzer or suggestion report)
            model_category: Detected model category
            pattern_context: Optional context from pattern database
            
        Returns:
            List of TransformationStrategy objects
        """
        # Extract analysis info
        model_info = self._extract_model_info(analysis)
        model_name = model_info["model_name"]
        
        # Check cache
        cache_key = f"{model_name}_{model_category}"
        if self.config.cache_strategies and cache_key in self._strategy_cache:
            return self._strategy_cache[cache_key]
        
        # Get pattern context if not provided
        if pattern_context is None:
            pattern_context = self._get_pattern_context(model_category, model_info)
        
        # Build prompt
        prompt = STRATEGY_GENERATION_PROMPT.format(
            num_strategies=self.config.num_strategies,
            model_name=model_info["model_name"],
            model_category=model_category,
            node_count=model_info["node_count"],
            blocker_count=model_info["blocker_count"],
            non_4d_count=model_info["non_4d_count"],
            dynamic_count=model_info["dynamic_count"],
            blocker_details=model_info["blocker_details"],
            pattern_context=pattern_context,
        )
        
        try:
            # Try structured output first
            response = self.llm.call(
                prompt=prompt,
                response_model=StrategyList,
                system_prompt=STRATEGY_SYSTEM_PROMPT,
            )
            strategies = response.strategies
            
        except Exception as e:
            if self.config.verbose:
                print(f"  Structured output failed, falling back to parsing: {e}")
            
            # Fallback to raw call + parsing
            try:
                raw_response = self.llm.call(
                    prompt=prompt,
                    system_prompt=STRATEGY_SYSTEM_PROMPT,
                )
                strategies = self._parse_strategies_from_text(raw_response, model_category)
            except Exception as e2:
                if self.config.verbose:
                    print(f"  Strategy generation failed: {e2}")
                strategies = self._get_default_strategies(model_category)
        
        # Add model category and compute pattern confidence
        for strategy in strategies:
            strategy.model_category = model_category
            strategy.pattern_confidence = self._compute_pattern_confidence(
                strategy, model_category
            )
        
        # Cache strategies
        if self.config.cache_strategies:
            self._strategy_cache[cache_key] = strategies
        
        if self.config.log_strategies:
            print(f"  Generated {len(strategies)} strategies:")
            for s in strategies:
                print(f"    - {s.name}: {s.estimated_success:.0%} (pattern: {s.pattern_confidence:.0%})")
        
        return strategies
    
    def select_strategy(
        self,
        strategies: List[TransformationStrategy],
        analysis: Optional[Any] = None,
        model_category: str = "Other",
    ) -> TransformationStrategy:
        """
        Select the best strategy from candidates.
        
        Args:
            strategies: List of candidate strategies
            analysis: Optional model analysis
            model_category: Model category
            
        Returns:
            Selected TransformationStrategy
        """
        if not strategies:
            return self._get_default_strategies(model_category)[0]
        
        if len(strategies) == 1:
            return strategies[0]
        
        # Score strategies combining LLM estimate and pattern confidence
        def score(s: TransformationStrategy) -> float:
            pattern_weight = self.config.pattern_weight
            return (
                (1 - pattern_weight) * s.estimated_success +
                pattern_weight * s.pattern_confidence
            )
        
        # Filter by confidence threshold
        valid = [s for s in strategies if score(s) >= self.config.min_confidence_threshold]
        
        if valid:
            return max(valid, key=score)
        
        # Fallback to best available
        return max(strategies, key=score)
    
    def generate_new_strategy(
        self,
        reason: str,
        analysis: Any,
        model_category: str = "Other",
        previous_strategies: Optional[List[TransformationStrategy]] = None,
    ) -> Optional[TransformationStrategy]:
        """
        Generate a new strategy when current approach is failing.
        
        Args:
            reason: Why a new strategy is needed
            analysis: Current model analysis
            model_category: Model category
            previous_strategies: Previously tried strategies
            
        Returns:
            New TransformationStrategy or None if generation fails
        """
        # Add context about failure
        pattern_context = self._get_pattern_context(model_category, self._extract_model_info(analysis))
        pattern_context += f"\n\nPREVIOUS ATTEMPT FAILED: {reason}"
        
        if previous_strategies:
            prev_names = [s.name for s in previous_strategies]
            pattern_context += f"\nAvoid these approaches: {', '.join(prev_names)}"
        
        # Generate with higher temperature for diversity
        old_temp = self.llm.temperature
        self.llm.temperature = min(0.9, old_temp + 0.3)
        
        try:
            strategies = self.generate_strategies(analysis, model_category, pattern_context)
            
            # Filter out previously tried strategies
            if previous_strategies:
                prev_names = {s.name for s in previous_strategies}
                strategies = [s for s in strategies if s.name not in prev_names]
            
            if strategies:
                return self.select_strategy(strategies, analysis, model_category)
                
        finally:
            self.llm.temperature = old_temp
        
        return None
    
    def _extract_model_info(self, analysis: Any) -> Dict[str, Any]:
        """Extract model info from analysis object or dict."""
        if hasattr(analysis, 'model_name'):
            model_name = analysis.model_name
            node_count = len(analysis.nodes) if hasattr(analysis, 'nodes') else 0
            blockers = analysis.compilation_blockers if hasattr(analysis, 'compilation_blockers') else []
            blocker_count = len(blockers)
            dynamic_count = len(analysis.dynamic_dimensions) if hasattr(analysis, 'dynamic_dimensions') else 0
            non_4d_count = 0
        else:
            model_name = analysis.get('model_name', 'unknown')
            node_count = analysis.get('node_count', 0)
            blocker_count = analysis.get('critical_count', 0) + analysis.get('high_count', 0)
            blockers = analysis.get('blockers', [])
            dynamic_count = analysis.get('dynamic_count', 0)
            non_4d_count = analysis.get('non_4d_count', 0)
        
        return {
            "model_name": model_name,
            "node_count": node_count,
            "blocker_count": blocker_count,
            "blockers": blockers,
            "blocker_details": self._format_blockers(blockers),
            "dynamic_count": dynamic_count,
            "non_4d_count": non_4d_count,
        }
    
    def _format_blockers(self, blockers: List[Any], max_items: int = 10) -> str:
        """Format blockers for prompt."""
        if not blockers:
            return "No blockers identified."
        
        lines = []
        for b in blockers[:max_items]:
            if isinstance(b, dict):
                op_type = b.get('op_type', 'unknown')
                reason = b.get('reason', 'unknown')
                lines.append(f"- {op_type}: {reason}")
            else:
                lines.append(f"- {b}")
        
        if len(blockers) > max_items:
            lines.append(f"... and {len(blockers) - max_items} more")
        
        return "\n".join(lines)
    
    def _get_pattern_context(self, model_category: str, model_info: Dict) -> str:
        """Get pattern context from knowledge base and pattern database."""
        context_parts = []
        
        # Try pattern database first
        if self.pattern_db:
            try:
                patterns = self.pattern_db.get_patterns_for_category(model_category)
                if patterns:
                    context_parts.append(f"Learned patterns for {model_category} models:")
                    for p in patterns[:10]:
                        context_parts.append(f"- {p.action} {p.op_type}: {p.frequency} times, {p.confidence:.0%} success")
            except Exception:
                pass
        
        # Fallback to hardcoded patterns
        if not context_parts:
            context_parts.append(f"Common patterns for {model_category} models:")
            context_parts.append("- Einsum operations are commonly replaced with MatMul + Reshape")
            context_parts.append("- Non-4D tensors require Reshape operations")
            context_parts.append("- Dropout and Identity nodes can be safely removed")
            if model_category == "YOLO":
                context_parts.append("- YOLO models often need Sigmoid at output heads")
            context_parts.append("- Dynamic shapes should be resolved to static values")
        
        return "\n".join(context_parts)
    
    def _compute_pattern_confidence(
        self,
        strategy: TransformationStrategy,
        model_category: str,
    ) -> float:
        """Compute confidence based on pattern database matches."""
        if not self.pattern_db:
            return 0.5  # Default when no pattern DB
        
        try:
            # Get matching patterns from database
            confidence_scores = []
            for transform in strategy.key_transformations:
                patterns = self.pattern_db.find_similar(transform, model_category)
                if patterns:
                    avg_confidence = sum(p.confidence for p in patterns) / len(patterns)
                    confidence_scores.append(avg_confidence)
            
            if confidence_scores:
                return sum(confidence_scores) / len(confidence_scores)
                
        except Exception:
            pass
        
        return 0.5
    
    def _parse_strategies_from_text(
        self,
        response_text: str,
        model_category: str,
    ) -> List[TransformationStrategy]:
        """Parse strategies from raw LLM response text."""
        try:
            # Clean response
            text = response_text.strip()
            if "```json" in text:
                start = text.find("```json") + 7
                end = text.find("```", start)
                text = text[start:end].strip()
            elif "```" in text:
                start = text.find("```") + 3
                end = text.find("```", start)
                if end > start:
                    text = text[start:end].strip()
            
            # Find JSON array
            if not text.startswith("["):
                start = text.find("[")
                if start != -1:
                    text = text[start:]
            if not text.endswith("]"):
                end = text.rfind("]")
                if end != -1:
                    text = text[:end + 1]
            
            data = json.loads(text)
            
            strategies = []
            for item in data:
                strategy = TransformationStrategy(
                    name=item.get("name", "Unnamed Strategy"),
                    priority_order=item.get("priority_order", ["critical_blockers"]),
                    approach=item.get("approach", "hybrid"),
                    estimated_success=float(item.get("estimated_success", 0.5)),
                    rationale=item.get("rationale", ""),
                    key_transformations=item.get("key_transformations", []),
                    risk_level=item.get("risk_level", "medium"),
                    model_category=model_category,
                )
                strategies.append(strategy)
            
            return strategies
            
        except Exception as e:
            if self.config.verbose:
                print(f"Error parsing strategies: {e}")
            return self._get_default_strategies(model_category)
    
    def _get_default_strategies(self, model_category: str) -> List[TransformationStrategy]:
        """Get default strategies when generation fails."""
        return [
            TransformationStrategy(
                name="Priority-Based Removal",
                priority_order=["critical_blockers", "shape_issues", "dynamic_shapes", "optimizations"],
                approach="aggressive_removal",
                estimated_success=0.6,
                rationale="Remove blockers in order of criticality",
                key_transformations=["remove_einsum", "remove_identity", "remove_dropout"],
                risk_level="medium",
                model_category=model_category,
            ),
            TransformationStrategy(
                name="Conservative Transformation",
                priority_order=["optimizations", "shape_issues", "critical_blockers"],
                approach="conservative",
                estimated_success=0.5,
                rationale="Start with safe transformations, then address blockers",
                key_transformations=["remove_identity", "remove_dropout", "fix_shapes"],
                risk_level="low",
                model_category=model_category,
            ),
            TransformationStrategy(
                name="Hybrid Approach",
                priority_order=["critical_blockers", "optimizations", "shape_issues"],
                approach="hybrid",
                estimated_success=0.55,
                rationale="Balance aggressive removal with conservative optimization",
                key_transformations=["remove_einsum", "fix_shapes", "remove_identity"],
                risk_level="medium",
                model_category=model_category,
            ),
        ]


# =============================================================================
# Backward Compatibility
# =============================================================================

# Alias for backward compatibility
ToTPlanner = StrategyPlanner


# Export
__all__ = [
    "StrategyPlanner",
    "TransformationStrategy",
    "StrategyEvaluation",
    "StrategyList",
    "ToTPlanner",  # Backward compatibility
]
