#!/usr/bin/env python3
"""
Unified Pipeline for Graph Surgery.

The GraphSurgeryPipeline provides:
- Strategic planning for complex models
- State machine execution with feedback loops
- Adaptive strategy changes
- Pattern database integration
- Comprehensive evaluation and reporting
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.config import AgentConfig, StrategyConfig, PipelineConfig
from agents.diagnostics import FeedbackCollector
from agents.executor import GraphSurgeryExecutor, ExecutionResult, AgentResult
from agents.strategy_planner import StrategyPlanner, TransformationStrategy

# Import existing components
try:
    from core_analysis.onnx_analyzer import ONNXAnalyzer
    from suggestion_pipeline.rag_suggestion_generator import RAGSuggestionGenerator
    from suggestion_pipeline.suggestion_generator import SuggestionGenerator
    from evaluation.model_comparator import ModelComparator
    from knowledge_base.rag_retriever import detect_model_category
    COMPONENTS_AVAILABLE = True
except ImportError:
    COMPONENTS_AVAILABLE = False

# Try to import pattern database
try:
    from knowledge_base.knowledge_base import PatternDatabase
    PATTERN_DB_AVAILABLE = True
except ImportError:
    PATTERN_DB_AVAILABLE = False


class PipelineResult(BaseModel):
    """Result from pipeline execution."""
    
    success: bool
    model_path: str
    model_name: str
    
    # Phase results
    analysis: Optional[Dict] = None
    suggestions_count: int = 0
    strategy: Optional[Dict] = None  # Serialized strategy
    execution_result: Optional[Dict] = None  # From executor
    
    # Evaluation
    evaluation: Optional[Dict] = None
    
    # Timing
    total_time_seconds: float = 0.0
    phase_times: Dict[str, float] = Field(default_factory=dict)
    
    # State history (for debugging)
    state_history: List[str] = Field(default_factory=list)
    
    # Output paths
    modified_model_path: Optional[str] = None
    report_path: Optional[str] = None
    
    model_config = {"extra": "allow"}
    
    # Backward compatibility
    @property
    def agent_result(self) -> Optional[AgentResult]:
        """Backward compatible agent_result property."""
        if self.execution_result:
            result = AgentResult(
                success=self.success,
                message=self.execution_result.get('message', ''),
                iterations=self.execution_result.get('iterations', 0),
                evaluation=self.evaluation,
            )
            # Create feedback from summary
            if 'feedback_summary' in self.execution_result:
                result._feedback = FeedbackCollector()
            return result
        return None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return self.model_dump(exclude_none=True)
    
    def save(self, output_path: str):
        """Save result to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class GraphSurgeryPipeline:
    """
    Unified pipeline for ONNX graph surgery.
    
    Workflow:
    1. Analysis: Deep ONNX model analysis
    2. Suggestion Generation: RAG-enhanced suggestions
    3. Strategy Planning: Generate and select transformation strategy (if complex)
    4. Execution: Apply transformations with state machine
    5. Evaluation: Compare with ground truth
    """
    
    def __init__(
        self,
        api_key: str,
        config: Optional[PipelineConfig] = None,
    ):
        """
        Initialize pipeline.
        
        Args:
            api_key: API key for LLM provider
            config: Pipeline configuration
        """
        if not COMPONENTS_AVAILABLE:
            raise ImportError(
                "Core components not available. Ensure all modules are installed."
            )
        
        self.api_key = api_key
        self.config = config or PipelineConfig()
        
        # Initialize components
        self.analyzer = ONNXAnalyzer()
        self.comparator = ModelComparator()
        
        # Initialize suggestion generator
        if self.config.use_rag:
            self.suggestion_generator = RAGSuggestionGenerator(
                kb_path=self.config.kb_path,
                api_key=api_key,
                use_rag=True,
            )
        else:
            self.suggestion_generator = SuggestionGenerator(api_key=api_key)
        
        # Load pattern database if available
        self.pattern_db = None
        if self.config.use_pattern_db and PATTERN_DB_AVAILABLE:
            try:
                pattern_db_path = Path(self.config.pattern_db_path)
                if pattern_db_path.exists():
                    self.pattern_db = PatternDatabase.load(str(pattern_db_path))
                    print(f"  Loaded pattern database: {len(self.pattern_db.patterns)} patterns")
            except Exception as e:
                print(f"  Warning: Could not load pattern database: {e}")
        
        # Initialize strategy planner
        self.strategy_planner = StrategyPlanner(
            api_key=api_key,
            config=self.config.strategy_config,
            kb_path=self.config.kb_path,
            pattern_db=self.pattern_db,
        )
        
        # Initialize executor
        self.executor = GraphSurgeryExecutor(
            api_key=api_key,
            config=self.config.agent_config,
            strategy_change_callback=self._handle_strategy_change,
        )
        
        # Track current state for strategy change callback
        self._current_analysis: Optional[Any] = None
        self._current_model_category: str = "Other"
        self._previous_strategies: List[TransformationStrategy] = []
    
    def process(
        self,
        model_path: str,
        ground_truth_path: Optional[str] = None,
    ) -> PipelineResult:
        """
        Process model through full pipeline.
        
        Args:
            model_path: Path to ONNX model
            ground_truth_path: Optional path to ground truth model
            
        Returns:
            PipelineResult with all outcomes
        """
        start_time = time.time()
        model_name = Path(model_path).stem
        phase_times: Dict[str, float] = {}
        
        print(f"\n{'='*80}")
        print(f"Graph Surgery Pipeline: {model_name}")
        print(f"{'='*80}")
        
        # Phase 1: Analysis
        print("\n[Phase 1] Model Analysis...")
        phase_start = time.time()
        analysis = self._analyze_model(model_path)
        phase_times["analysis"] = time.time() - phase_start
        print(f"  Completed in {phase_times['analysis']:.1f}s")
        
        # Detect model category
        model_category = detect_model_category(model_path)
        self._current_model_category = model_category
        print(f"  Model category: {model_category}")
        
        # Phase 2: Suggestion Generation
        print("\n[Phase 2] Generating Suggestions...")
        phase_start = time.time()
        suggestions_report = self.suggestion_generator.analyze_and_suggest(model_path)
        suggestions = [s.to_dict() for s in suggestions_report.suggestions]
        phase_times["suggestions"] = time.time() - phase_start
        print(f"  Generated {len(suggestions)} suggestions in {phase_times['suggestions']:.1f}s")
        print(f"  Critical: {suggestions_report.critical_count}, High: {suggestions_report.high_count}")
        
        # Phase 3: Strategy Planning (if needed)
        strategy = None
        if self._needs_strategic_planning(suggestions_report):
            print("\n[Phase 3] Strategy Planning...")
            phase_start = time.time()
            strategy = self._plan_strategy(analysis, model_category)
            phase_times["planning"] = time.time() - phase_start
            print(f"  Selected strategy: {strategy.name}")
            print(f"  Approach: {strategy.approach}")
            print(f"  Pattern confidence: {strategy.pattern_confidence:.0%}")
            print(f"  Completed in {phase_times['planning']:.1f}s")
        else:
            print("\n[Phase 3] Skipping strategic planning (simple case)")
            phase_times["planning"] = 0.0
        
        # Store analysis for strategy change callback
        self._current_analysis = analysis
        
        # Phase 4: Execution
        print("\n[Phase 4] Executing Transformations...")
        phase_start = time.time()
        execution_result = self.executor.run(
            model_path=model_path,
            suggestions=suggestions,
            strategy=strategy,
            ground_truth_path=ground_truth_path,
        )
        phase_times["execution"] = time.time() - phase_start
        print(f"  Completed {execution_result.iterations} iterations in {phase_times['execution']:.1f}s")
        print(f"  Final state: {execution_result.final_state}")
        print(f"  Success rate: {execution_result.feedback_summary.get('success_rate', 0):.1%}")
        
        # Phase 5: Evaluation
        evaluation = execution_result.evaluation
        if evaluation:
            print("\n[Phase 5] Evaluation Results:")
            print(f"  Overall similarity: {evaluation.get('overall_similarity', 0):.1%}")
            ta = evaluation.get('transformation_accuracy', {})
            print(f"  Transformation accuracy: {ta.get('transformation_score', 0):.1%}")
        
        # Save outputs
        modified_model_path = None
        report_path = None
        if self.config.save_intermediate:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save modified model
            if self.executor.current_model:
                import onnx
                modified_model_path = str(output_dir / f"{model_name}_modified.onnx")
                onnx.save(self.executor.current_model, modified_model_path)
            
            # Save report
            report_path = str(output_dir / f"{model_name}_report.json")
        
        total_time = time.time() - start_time
        
        result = PipelineResult(
            success=execution_result.success,
            model_path=model_path,
            model_name=model_name,
            analysis=self._format_analysis(analysis),
            suggestions_count=len(suggestions),
            strategy=strategy.to_dict() if strategy else None,
            execution_result=execution_result.to_dict(),
            evaluation=evaluation,
            total_time_seconds=total_time,
            phase_times=phase_times,
            state_history=execution_result.state_history,
            modified_model_path=modified_model_path,
            report_path=report_path,
        )
        
        if report_path:
            result.save(report_path)
        
        print(f"\n{'='*80}")
        print(f"Pipeline Complete: {'SUCCESS' if result.success else 'PARTIAL'}")
        print(f"Total time: {total_time:.1f}s")
        print(f"{'='*80}")
        
        return result
    
    def _analyze_model(self, model_path: str) -> Any:
        """Analyze ONNX model."""
        return self.analyzer.analyze(model_path)
    
    def _format_analysis(self, analysis: Any) -> Dict:
        """Format analysis for result."""
        if hasattr(analysis, 'model_name'):
            return {
                "model_name": analysis.model_name,
                "node_count": len(analysis.nodes) if hasattr(analysis, 'nodes') else 0,
                "blocker_count": len(analysis.compilation_blockers) if hasattr(analysis, 'compilation_blockers') else 0,
                "dynamic_count": len(analysis.dynamic_dimensions) if hasattr(analysis, 'dynamic_dimensions') else 0,
            }
        return {}
    
    def _needs_strategic_planning(self, report: Any) -> bool:
        """Determine if strategic planning is needed."""
        return self.config.agent_config.should_use_strategy(
            critical_count=report.critical_count,
            total_issues=report.total_issues,
            status=report.compilation_status,
        )
    
    def _plan_strategy(
        self,
        analysis: Any,
        model_category: str,
    ) -> TransformationStrategy:
        """Generate and select transformation strategy."""
        strategy = self.strategy_planner.generate_strategy(analysis, model_category)
        self._previous_strategies.append(strategy)
        return strategy
    
    def _handle_strategy_change(self, reason: str) -> Optional[TransformationStrategy]:
        """Handle strategy change request from executor."""
        if self._current_analysis is None:
            return None
        
        new_strategy = self.strategy_planner.generate_new_strategy(
            reason=reason,
            analysis=self._current_analysis,
            model_category=self._current_model_category,
            previous_strategies=self._previous_strategies,
        )
        
        if new_strategy:
            self._previous_strategies.append(new_strategy)
        
        return new_strategy
    
    def process_batch(
        self,
        model_paths: List[str],
        ground_truth_dir: Optional[str] = None,
    ) -> List[PipelineResult]:
        """
        Process multiple models.
        
        Args:
            model_paths: List of model paths
            ground_truth_dir: Optional directory containing ground truth models
            
        Returns:
            List of PipelineResult for each model
        """
        results = []
        
        for i, model_path in enumerate(model_paths):
            print(f"\n[{i+1}/{len(model_paths)}] Processing {Path(model_path).stem}")
            
            # Find ground truth if available
            ground_truth_path = None
            if ground_truth_dir:
                model_name = Path(model_path).stem
                gt_path = Path(ground_truth_dir) / f"{model_name}_modified.onnx"
                if gt_path.exists():
                    ground_truth_path = str(gt_path)
            
            try:
                result = self.process(model_path, ground_truth_path)
                results.append(result)
            except Exception as e:
                print(f"  Error: {e}")
                results.append(PipelineResult(
                    success=False,
                    model_path=model_path,
                    model_name=Path(model_path).stem,
                ))
        
        # Summary
        print(f"\n{'='*80}")
        print("Batch Summary")
        print(f"{'='*80}")
        success_count = sum(1 for r in results if r.success)
        print(f"Processed: {len(results)}")
        print(f"Successful: {success_count}")
        print(f"Success rate: {success_count/len(results):.1%}" if results else "N/A")
        
        return results


# =============================================================================
# Backward Compatibility
# =============================================================================

# Alias for backward compatibility
ReActToTPipeline = GraphSurgeryPipeline

# Note: StateManager is now defined in agents.state module only
# Use: from agents.state import StateManager


def run_pipeline(
    model_path: str,
    api_key: str,
    ground_truth_path: Optional[str] = None,
    use_strategy: bool = True,
    max_iterations: int = 15,
    verbose: bool = False,
    output_dir: str = "inference_results",
) -> PipelineResult:
    """
    Convenience function to run the pipeline.
    
    Args:
        model_path: Path to ONNX model
        api_key: API key for LLM provider
        ground_truth_path: Optional ground truth path
        use_strategy: Whether to use strategic planning
        max_iterations: Max execution iterations
        verbose: Verbose output
        output_dir: Output directory
        
    Returns:
        PipelineResult
    """
    config = PipelineConfig(
        agent_config=AgentConfig(
            max_iterations=max_iterations,
            verbose=verbose,
            use_strategy_planning=use_strategy,
        ),
        output_dir=output_dir,
    )
    
    pipeline = GraphSurgeryPipeline(api_key=api_key, config=config)
    return pipeline.process(model_path, ground_truth_path)


# Export
__all__ = [
    "PipelineResult",
    "GraphSurgeryPipeline",
    "ReActToTPipeline",  # Backward compatibility
    "StateManager",
    "run_pipeline",
]
