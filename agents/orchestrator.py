#!/usr/bin/env python3
"""
LangGraph-based orchestrator for the graph surgery pipeline.

Builds a cyclic StateGraph:
    START -> diagnose -> plan -> execute -> validate
        validate --[blockers remain]--> refine_plan -> execute  (loop)
        validate --[all clear / max retries]--> enrich_kb -> evaluate -> END
"""

from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.config import PipelineConfig
from agents.state import PipelineState


def _make_node_factories(api_key: str, config: PipelineConfig) -> Dict[str, Any]:
    """
    Return a dict of node functions, each closing over *api_key* and *config*.

    Every node has the signature ``(state: PipelineState) -> dict`` and returns
    a partial state update.
    """
    from agents.nodes.diagnose_node import make_diagnose_node
    from agents.nodes.enrich_kb_node import make_enrich_kb_node
    from agents.nodes.evaluate_node import make_evaluate_node
    from agents.nodes.execute_node import make_execute_node
    from agents.nodes.plan_node import make_plan_node
    from agents.nodes.refine_plan_node import make_refine_plan_node
    from agents.nodes.validate_node import make_validate_node

    return {
        "diagnose": make_diagnose_node(api_key, config),
        "plan": make_plan_node(api_key, config),
        "execute": make_execute_node(api_key, config),
        "validate": make_validate_node(config),
        "refine_plan": make_refine_plan_node(api_key, config),
        "enrich_kb": make_enrich_kb_node(config),
        "evaluate": make_evaluate_node(config),
    }


def should_retry(state: PipelineState) -> str:
    """
    Conditional edge after *validate*:
      - ``"retry"``  if blockers remain **and** we haven't hit max iterations
      - ``"done"``   otherwise (success **or** max retries exhausted)
    """
    remaining = state.get("remaining_blockers", [])
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 3)

    if remaining and iteration < max_iter:
        return "retry"
    return "done"


def build_graph(api_key: str, config: PipelineConfig) -> CompiledStateGraph:
    """
    Construct and compile the LangGraph pipeline.

    Returns a ``CompiledStateGraph`` that can be invoked with
    ``graph.invoke(initial_state)``.
    """
    nodes = _make_node_factories(api_key, config)

    graph = StateGraph(PipelineState)

    # Register nodes
    for name, fn in nodes.items():
        graph.add_node(name, fn)

    # Fixed edges
    graph.add_edge(START, "diagnose")
    graph.add_edge("diagnose", "plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "validate")

    # Conditional retry loop
    graph.add_conditional_edges(
        "validate",
        should_retry,
        {
            "retry": "refine_plan",
            "done": "enrich_kb",
        },
    )
    graph.add_edge("refine_plan", "execute")

    # Post-loop
    graph.add_edge("enrich_kb", "evaluate")
    graph.add_edge("evaluate", END)

    return graph.compile()
