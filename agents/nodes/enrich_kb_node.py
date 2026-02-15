#!/usr/bin/env python3
"""
Enrich-KB node -- writes successful transformations back to the knowledge base.

After surgery completes (success or max retries) this node persists
learnings into the SurgeryDatabase and StrategyDatabase so future runs
benefit from accumulated experience.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict

from agents.config import PipelineConfig
from agents.state import PipelineState


def make_enrich_kb_node(
    config: PipelineConfig,
) -> Callable[[PipelineState], Dict[str, Any]]:
    """Return a LangGraph node function for KB enrichment."""

    from knowledge_base.strategy_database import StrategyDatabase
    from knowledge_base.surgery_database import SurgeryDatabase

    def enrich_kb_node(state: PipelineState) -> Dict[str, Any]:
        applied = state.get("transformations_applied", [])
        history = state.get("surgery_history", [])
        diagnosis = state.get("diagnosis", {})
        remaining = state.get("remaining_blockers", [])
        plan_dict = state.get("plan", {})

        model_name = diagnosis.get("model_name", "unknown")
        model_category = diagnosis.get("architecture_type", "Unknown")
        success = len(remaining) == 0

        start = time.time()
        records_added = 0

        # -- Surgery Database write-back -----------------------------------
        try:
            surgery_db = SurgeryDatabase.load(config.surgery_db_path)
        except Exception:
            from knowledge_base.surgery_database import create_database_with_defaults
            surgery_db = create_database_with_defaults()

        if hasattr(surgery_db, "add_from_pipeline_result"):
            records_added = surgery_db.add_from_pipeline_result(
                surgery_history=history,
                model_name=model_name,
                model_category=model_category,
            )
        else:
            # Fallback: add individual records manually
            from knowledge_base.surgery_database import (
                NodeTransformation,
                TransformationRecord,
            )

            transformations = []
            for entry in applied:
                t = NodeTransformation(
                    original_node_id=-1,
                    original_node_name=entry.get("suggestion_id", ""),
                    original_op_type=", ".join(entry.get("target_ops", [])),
                    graph_position=0.5,
                    total_nodes_in_graph=0,
                    action="replace",
                    code_snippet=entry.get("code", ""),
                    is_compilation_blocker=True,
                    confidence=0.7 if success else 0.3,
                    source_model=model_name,
                )
                transformations.append(t)
                records_added += 1

            if transformations:
                record = TransformationRecord(
                    model_name=model_name,
                    model_category=model_category,
                    compilation_success=success,
                    transformations=transformations,
                )
                surgery_db.add_transformation_record(record)

        try:
            surgery_db.save(config.surgery_db_path)
        except Exception:
            pass  # non-fatal

        # -- Strategy Database write-back ----------------------------------
        strategy_id = plan_dict.get("strategy_id", "")
        if strategy_id:
            try:
                strategy_db = StrategyDatabase.load(config.strategy_db_path)
                total_time = sum(state.get("phase_times", {}).values())
                strategy_db.record_execution(
                    strategy_id=strategy_id,
                    success=success,
                    execution_time=total_time,
                )
                strategy_db.save(config.strategy_db_path)
            except Exception:
                pass  # non-fatal

        elapsed = time.time() - start
        phase_times = dict(state.get("phase_times", {}))
        phase_times["enrich_kb"] = elapsed

        return {
            "kb_records_added": records_added,
            "phase_times": phase_times,
        }

    return enrich_kb_node
