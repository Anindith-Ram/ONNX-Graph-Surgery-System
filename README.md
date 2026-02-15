# Automated Model Surgery

An agentic pipeline that transforms ONNX models for hardware accelerator compilation (SiMa MLA) using LLM-driven graph surgery with automated validation and self-improving knowledge.

## How It Works

The pipeline uses a **LangGraph cyclic state graph** to iteratively diagnose, plan, execute, and validate ONNX model transformations until the model compiles cleanly -- or a retry budget is exhausted.

```
START ──▶ Diagnose ──▶ Plan ──▶ Execute ──▶ Validate ──┐
                                               │        │
                                    ┌──────────┘        │
                                    ▼                   │
                              Blockers remain?          │
                              & retries left?           │
                                    │                   │
                              Yes ──▶ Refine Plan ──▶ Execute ...
                                    │
                              No ───▶ Enrich KB ──▶ Evaluate ──▶ END
```

### Pipeline Nodes

| Node | What it does |
|------|-------------|
| **Diagnose** | Analyses the ONNX graph, identifies compilation blockers, classifies architecture |
| **Plan** | Generates a multi-phase transformation strategy with dependencies and fallbacks |
| **Execute** | Runs LLM-generated GraphSurgeon code against the model in a sandboxed `exec()` |
| **Validate** | Runs `CompilationSimulator` to check for remaining blockers |
| **Refine Plan** | Feeds compilation errors back to the strategy agent for a revised plan |
| **Enrich KB** | Writes successful transformations back to the Surgery & Strategy databases |
| **Evaluate** | Computes blocker resolution rate, model validity, ground-truth similarity, and more |

### Key Capabilities

- **Closed-loop execution** -- surgery code is actually applied and validated, not just suggested
- **Automatic retry loop** -- failed validations trigger plan refinement and re-execution
- **Knowledge base enrichment** -- every successful transformation is persisted for future runs
- **Hybrid RAG retrieval** -- reciprocal rank fusion of semantic + keyword search
- **Structured evaluation** -- systematic metrics (resolution rate, GT similarity, compilation pass) after every run
- **Sandboxed code execution** -- generated Python runs in a restricted namespace with safe builtins

## Quick Start

### 1. Install

```bash
git clone https://github.com/yourusername/automated-model-surgery.git
cd automated-model-surgery
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

Set your Gemini API key via environment variable or `config.py`:

```bash
export GEMINI_API_KEY="your-key-here"
```

Or create `config.py`:

```python
import os
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-key-here")
```

### 3. Prepare Data

Place ONNX models in the `dataset/` directory:

```
dataset/
├── Model_Name/
│   ├── original/
│   │   └── model.onnx
│   └── modified/          # ground truth (optional)
│       └── model.onnx
```

### 4. Build the Knowledge Base

```bash
python main.py build-kb
```

### 5. Run the Pipeline

```bash
# Single model
python main.py react --model dataset/T5_Small/original/model.onnx --verbose

# Full test set with 3 retry iterations
python main.py react --test-set --max-iterations 3

# Generate ONNX model maps
python main.py generate-maps
```

## Project Structure

```
agents/
  __init__.py                # Public exports
  config.py                  # Pydantic configs (PipelineConfig, agent configs)
  state.py                   # PipelineState TypedDict (shared LangGraph state)
  orchestrator.py            # build_graph() -- LangGraph StateGraph builder
  pipeline.py                # GraphSurgeryPipeline (entry point)
  evaluation.py              # PipelineEvaluation Pydantic model
  diagnostics.py             # GraphSnapshot, TransformationDelta, FeedbackCollector
  llm_client.py              # LiteLLM + Instructor structured output
  langchain_client.py        # LangChain ChatLiteLLM + PydanticOutputParser
  base/
    agent_base.py            # BaseAgent (LLM access + RAG retrieval)
    unified_retriever.py     # Combines KB, SurgeryDB, StrategyDB retrieval
    prompts.py               # All prompt templates (diagnosis, strategy, surgery, refinement)
  specialized/
    diagnosis_agent.py       # Architecture analysis + blocker identification
    strategy_agent.py        # Multi-phase plan generation + refinement
    surgery_agent.py         # Executable code generation + syntax validation
  nodes/
    diagnose_node.py         # Wraps DiagnosisAgent
    plan_node.py             # Wraps StrategyAgent + loads model bytes
    execute_node.py          # Sandboxed exec() with GraphSnapshot deltas
    validate_node.py         # CompilationSimulator check
    refine_plan_node.py      # Error-driven strategy refinement
    enrich_kb_node.py        # Write-back to SurgeryDB + StrategyDB
    evaluate_node.py         # Metrics computation

core_analysis/
  onnx_analyzer.py           # Deep ONNX graph analysis
  architecture_analyzer.py   # Architecture classification
  compilation_simulator.py   # MLA compilation simulation
  dataset_analyzer.py        # Dataset-wide pattern extraction

knowledge_base/
  knowledge_base.py          # KnowledgeBase + KnowledgeBaseBuilder
  rag_retriever.py           # Semantic, keyword, and hybrid retrieval (RRF)
  surgery_database.py        # Node-level transformation records + templates
  strategy_database.py       # High-level strategy records
  llm_context_generator.py   # Rich context generation for LLM prompts
  response_cache.py          # Gemini response caching
  transformation_regions.py  # Region mapping utilities

scripts/
  generate_all_maps.py       # Batch ONNX map generation

utilities/
  train_test_split.py        # Dataset splitting
  api_quota_manager.py       # API quota management
  checkpoint_manager.py      # Checkpoint management

tests/
  test_agents.py             # Agent system tests

main.py                      # CLI entry point
config.py                    # API keys (not tracked)
requirements.txt             # Dependencies
```

## Architecture Details

### LangGraph State

All nodes share a `PipelineState` TypedDict that flows through the graph:

```python
class PipelineState(TypedDict, total=False):
    model_path: str
    ground_truth_path: Optional[str]
    api_key: str
    config: Dict[str, Any]
    diagnosis: Optional[Dict]
    plan: Optional[Dict]
    current_model_bytes: Optional[bytes]
    iteration: int
    max_iterations: int
    surgery_history: List[Dict]
    compilation_report: Optional[Dict]
    remaining_blockers: List[str]
    transformations_applied: List[Dict]
    kb_records_added: int
    evaluation: Optional[Dict]
    phase_times: Dict[str, float]
```

### Specialized Agents

| Agent | Role | Key Methods |
|-------|------|-------------|
| `DiagnosisAgent` | Analyse model, identify blockers | `analyze()` |
| `StrategyAgent` | Plan transformations, refine on failure | `plan()`, `refine_plan()` |
| `SurgeryAgent` | Generate executable code, fix errors | `generate_suggestions()`, `generate_fix_for_error()`, `validate_code()` |

### Retrieval System

The `UnifiedRetriever` combines three sources:
1. **RAG Knowledge Base** -- PDF docs + dataset patterns (hybrid semantic+keyword via RRF)
2. **Surgery Database** -- 1100+ node-level transformation records with code snippets
3. **Strategy Database** -- high-level strategy records with execution history

### Evaluation Metrics

Every run produces a `PipelineEvaluation`:

| Metric | Description |
|--------|-------------|
| `blocker_resolution_rate` | Fraction of original blockers resolved |
| `iterations_used` | Execute-validate cycles performed |
| `model_valid` | Passes `onnx.checker.check_model` |
| `compilation_passes` | Zero remaining blockers |
| `gt_similarity` | Jaccard similarity with ground-truth op-types |
| `gt_op_match_rate` | Fraction of GT op counts matched |
| `kb_records_added` | Transformations written back to knowledge base |

## Dependencies

| Package | Purpose |
|---------|---------|
| `onnx` | ONNX model loading and manipulation |
| `numpy` | Numerical operations |
| `langgraph` | Stateful cyclic agent orchestration |
| `langchain-core` | LLM abstractions and output parsing |
| `litellm` | Unified LLM API (Gemini, OpenAI, etc.) |
| `instructor` | Structured LLM outputs with Pydantic |
| `google-generativeai` | Gemini API + embeddings |
| `pydantic` | Type-safe data models and validation |

## Troubleshooting

**"Knowledge base not found"**
```bash
python main.py build-kb
```

**"API key required"**
```bash
export GEMINI_API_KEY="your-key"
# or
python main.py react --api-key YOUR_KEY --model path/to/model.onnx
```

**Pipeline runs but resolution rate is low**
- Rebuild the knowledge base with more training data
- Increase `--max-iterations` to allow more retry cycles
- Check that `rag_data/surgery_database.json` has relevant transformation patterns

## License

See [LICENSE](LICENSE) for details.
