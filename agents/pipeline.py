#!/usr/bin/env python3
"""
Unified Pipeline for Graph Surgery (LangGraph).

Builds and invokes the LangGraph ``StateGraph`` compiled by
:func:`agents.orchestrator.build_graph`.  The ``PipelineResult``
data-class captures every output needed for reporting and downstream use.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.config import PipelineConfig
from agents.evaluation import PipelineEvaluation
from agents.orchestrator import build_graph


class PipelineResult(BaseModel):
    """Result from a single pipeline execution."""

    success: bool
    model_path: str
    model_name: str

    # Phase results
    analysis: Optional[Dict] = None
    suggestions_count: int = 0
    strategy: Optional[Dict] = None
    execution_result: Optional[Dict] = None

    # Evaluation
    evaluation: Optional[Dict] = None

    # Timing
    total_time_seconds: float = 0.0
    phase_times: Dict[str, float] = Field(default_factory=dict)

    # Output paths
    modified_model_path: Optional[str] = None
    report_path: Optional[str] = None

    model_config = {"extra": "allow"}

    def to_dict(self) -> Dict:
        """Convert to dictionary (excludes bytes)."""
        return self.model_dump(exclude_none=True)

    def save(self, output_path: str):
        """Save result to JSON file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class GraphSurgeryPipeline:
    """LangGraph-powered pipeline for ONNX graph surgery."""

    def __init__(self, api_key: str, config: Optional[PipelineConfig] = None):
        self.api_key = api_key
        self.config = config or PipelineConfig()
        self.graph = build_graph(api_key=self.api_key, config=self.config)

    def process(
        self,
        model_path: str,
        ground_truth_path: Optional[str] = None,
    ) -> PipelineResult:
        """
        Process a model through the full LangGraph pipeline.

        Args:
            model_path: Path to the ONNX model.
            ground_truth_path: Optional path to the ground-truth model.

        Returns:
            PipelineResult with all outcomes.
        """
        start_time = time.time()
        model_name = Path(model_path).stem

        # Build the initial state for the graph
        initial_state = {
            "model_path": model_path,
            "ground_truth_path": ground_truth_path,
            "api_key": self.api_key,
            "config": self.config.to_dict(),
            "max_iterations": self.config.max_surgery_retries,
            "iteration": 0,
            "surgery_history": [],
            "remaining_blockers": [],
            "transformations_applied": [],
            "kb_records_added": 0,
            "phase_times": {},
        }

        # Invoke the compiled LangGraph
        final_state = self.graph.invoke(initial_state)

        total_time = time.time() - start_time

        # Save modified model to disk if surgery produced bytes
        modified_model_path: Optional[str] = None
        model_bytes = final_state.get("current_model_bytes")
        if model_bytes:
            out_dir = Path(self.config.output_dir) / model_name
            out_dir.mkdir(parents=True, exist_ok=True)
            modified_model_path = str(out_dir / "modified_model.onnx")
            with open(modified_model_path, "wb") as f:
                f.write(model_bytes)

        # Determine success from evaluation
        evaluation = final_state.get("evaluation", {})
        remaining = final_state.get("remaining_blockers", [])
        success = len(remaining) == 0 or evaluation.get("compilation_passes", False)

        pipeline_result = PipelineResult(
            success=success,
            model_path=model_path,
            model_name=model_name,
            analysis=final_state.get("diagnosis"),
            suggestions_count=len(
                final_state.get("plan", {}).get("phases", [])
            ),
            strategy=final_state.get("plan"),
            execution_result={
                "surgery_history": final_state.get("surgery_history", []),
                "iterations": final_state.get("iteration", 0),
            },
            evaluation=evaluation,
            total_time_seconds=total_time,
            phase_times=final_state.get("phase_times", {}),
            modified_model_path=modified_model_path,
            report_path=str(
                Path(self.config.output_dir)
                / model_name
                / f"{model_name}_report.json"
            ),
        )

        # Print evaluation summary
        if self.config.verbose and evaluation:
            try:
                ev = PipelineEvaluation.from_state(evaluation)
                print(f"  [result] {ev.summary_line()}")
            except Exception:
                pass

        # Persist the report
        if pipeline_result.report_path:
            pipeline_result.save(pipeline_result.report_path)

        return pipeline_result

    def process_batch(
        self,
        model_paths: List[str],
        ground_truth_dir: Optional[str] = None,
    ) -> List[PipelineResult]:
        """Process multiple models sequentially."""
        results: List[PipelineResult] = []

        for i, model_path in enumerate(model_paths):
            print(f"\n[{i + 1}/{len(model_paths)}] Processing {Path(model_path).stem}")

            ground_truth_path = None
            if ground_truth_dir:
                gt_path = Path(ground_truth_dir) / f"{Path(model_path).stem}_modified.onnx"
                if gt_path.exists():
                    ground_truth_path = str(gt_path)

            try:
                result = self.process(model_path, ground_truth_path)
                results.append(result)
            except Exception as e:
                print(f"  Error: {e}")
                results.append(
                    PipelineResult(
                        success=False,
                        model_path=model_path,
                        model_name=Path(model_path).stem,
                    )
                )

        # Summary
        print(f"\n{'=' * 80}")
        print("Batch Summary")
        print(f"{'=' * 80}")
        success_count = sum(1 for r in results if r.success)
        print(f"Processed: {len(results)}")
        print(f"Successful: {success_count}")
        if results:
            print(f"Success rate: {success_count / len(results):.1%}")

        return results


# Backward compatibility
ReActToTPipeline = GraphSurgeryPipeline


def run_pipeline(
    model_path: str,
    api_key: str,
    ground_truth_path: Optional[str] = None,
    verbose: bool = False,
    output_dir: str = "inference_results",
) -> PipelineResult:
    """Convenience function to run the pipeline."""
    config = PipelineConfig(output_dir=output_dir, verbose=verbose)
    pipeline = GraphSurgeryPipeline(api_key=api_key, config=config)
    return pipeline.process(model_path, ground_truth_path)


__all__ = [
    "PipelineResult",
    "GraphSurgeryPipeline",
    "ReActToTPipeline",
    "run_pipeline",
]
