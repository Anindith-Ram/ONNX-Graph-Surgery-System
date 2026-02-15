"""
Prompt templates for the LangGraph agentic pipeline.

All prompts follow a consistent structure:
    CONTEXT  ->  TASKS  ->  OUTPUT FORMAT

Placeholders use Python str.format() syntax.  Prompts that accept
optional dynamic few-shot examples include a ``{few_shot_examples}``
placeholder that can be filled at call-time or left as an empty string.
"""

# =====================================================================
# Diagnosis
# =====================================================================

DIAGNOSIS_SYSTEM_PROMPT = (
    "You are a senior ONNX model compilation engineer specialising in "
    "SiMa MLA hardware targets. You analyse model architecture, identify "
    "compilation blockers, and explain precisely why each blocker fails. "
    "Always ground your reasoning in hardware constraints."
)

DIAGNOSIS_ANALYSIS_PROMPT = """
Analyse the following model for MLA compilation readiness.

## Model Summary
{model_summary}

## Architecture Analyser Output
{architecture_summary}

## Compilation Simulation
{compilation_summary}

## Related Context (KB + Surgery DB)
{rag_context}

## Dynamic Examples (similar models from the knowledge base)
{few_shot_examples}

## Tasks
1. Identify the model architecture type and justify your reasoning.
2. Map every blocker to an architectural region (attention, FFN, norm, head, embedding, etc.).
3. For each blocker explain (a) the hardware constraint violated, and (b) the root cause in the graph.
4. Recommend a high-level transformation approach: in-place surgery vs divide-and-conquer.
5. Return a prioritised blocker list ordered by severity (critical first).

## Output Format
Return a JSON object matching the **DiagnosisReport** schema.
Fields: model_name, architecture_type, architecture_reasoning, detected_patterns,
blockers (list of {{node_name, op_type, reason, severity, region}}),
recommended_approach, confidence, analysis_payload.
"""

# =====================================================================
# Strategy Planning
# =====================================================================

STRATEGY_SYSTEM_PROMPT = (
    "You are a strategic planner for ONNX graph surgery. "
    "You design precise, dependency-aware, multi-phase transformation "
    "plans.  Each phase must specify target ops, transformation type, "
    "validation criteria, and a fallback strategy for failure."
)

STRATEGY_PLANNING_PROMPT = """
Design a transformation strategy for the diagnosed model.

## Diagnosis Report
{diagnosis_report}

## Best Strategy Candidate (from Strategy DB, if any)
{strategy_candidate}

## Related Context (KB + Surgery DB + Strategy DB)
{rag_context}

## Tasks
1. Select or refine the best strategy for this model.
2. Create a phased plan with explicit dependencies between phases.
3. For each phase specify: phase_id, name, objective, target_op_types, transformation_type, validation, fallback.
4. Estimate overall success probability.
5. If the model requires divide-and-conquer, specify split points.

## Output Format
Return a JSON object matching the **TransformationPlan** schema.
Fields: strategy_id, strategy_reasoning, phases (list), risk_assessment,
expected_success_rate, divide_and_conquer.
"""

# =====================================================================
# Surgery Code Generation (executable)
# =====================================================================

SURGERY_SYSTEM_PROMPT = (
    "You are an expert ONNX graph surgery developer. "
    "Generate precise, executable Python code that mutates an onnx.ModelProto "
    "in-place. The variable `model` is available in the execution namespace "
    "along with `onnx`, `np` (numpy), `helper` (onnx.helper), "
    "`numpy_helper` (onnx.numpy_helper), and `TensorProto`. "
    "Do NOT call onnx.load() or onnx.save() -- work directly on the `model` object."
)

SURGERY_CODE_PROMPT = """
Generate executable surgery code for the following transformation phase.

## Phase
{phase}

## Region Context (target nodes in the model)
{region_context}

## Similar Transformations (from knowledge base)
{transformation_examples}

## Related Context (KB + Surgery DB)
{rag_context}

## Requirements
1. Each `code_snippet` MUST be valid, self-contained Python that mutates the `model` variable in-place.
2. Available globals: `model` (onnx.ModelProto), `onnx`, `np`, `numpy`, `helper`, `numpy_helper`, `TensorProto`.
3. Do NOT call `onnx.load()` or `onnx.save()`. Work directly on `model.graph`.
4. Preserve model semantics within numerical tolerance.
5. Use static integer dimensions -- avoid dynamic shapes.
6. Include inline comments explaining each transformation step.
7. In `validation_steps` describe how to verify the change succeeded.

## Output Format
Return a JSON object matching the **SurgerySuggestionSet** schema.
Fields: phase_id, phase_name, suggestions (list of {{suggestion_id, summary,
rationale, target_ops, code_snippet, expected_effect, validation_steps,
manual_checks, risk_level, confidence}}), overall_risk, notes.
"""

# =====================================================================
# Refinement Prompts (feedback loop)
# =====================================================================

REFINE_STRATEGY_PROMPT = """
You are REFINING a transformation strategy after a previous attempt partially failed.

## Original Diagnosis
{diagnosis_report}

## Remaining Blockers (still present after surgery)
{remaining_blockers}

## Blocker Details (op_type -> count)
{blocker_details}

## Previous Surgery History (what was tried and what happened)
{surgery_history}

## Current Iteration
{iteration}

## Best Strategy Candidate (if any)
{strategy_candidate}

## Related Context (KB + Surgery DB + Strategy DB)
{rag_context}

## Tasks
1. Analyse why previous attempts did not resolve all blockers.
2. Design a NEW phased plan targeting ONLY the remaining blockers listed above.
3. Do NOT repeat strategies that already failed -- try alternative approaches.
4. For each phase specify: target_op_types, transformation_type, validation, fallback.
5. Estimate success probability for this revised plan.

## Output Format
Return a JSON object matching the **TransformationPlan** schema.
"""

SURGERY_FIX_PROMPT = """
The previous surgery attempt FAILED. The compilation simulator still reports blockers.

## Error Context
{error_context}

## Previous Code That Was Tried
```python
{previous_code}
```

## Phase
{phase}

## Region Context
{region_context}

## Similar Transformations
{transformation_examples}

## Related Context (KB + Surgery DB)
{rag_context}

## Requirements
1. Analyse why the previous code did not resolve the blockers.
2. Generate CORRECTED executable Python code that fixes the remaining issues.
3. The `code_snippet` MUST be valid Python that mutates `model` in-place.
4. Available globals: model, onnx, np, numpy, helper, numpy_helper, TensorProto.
5. Do NOT repeat the same approach if it already failed -- try an alternative.

## Output Format
Return a JSON object matching the **SurgerySuggestionSet** schema.
"""
