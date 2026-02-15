"""
LangGraph node functions for the graph surgery pipeline.

Each module exposes a ``make_*_node(...)`` factory that returns a callable
with the signature ``(state: PipelineState) -> dict``.
"""
