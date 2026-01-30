# Automated Model Surgery: RAG Pipeline for ONNX Compilation

A Retrieval-Augmented Generation (RAG) system that learns from successful ONNX model transformations and automatically generates actionable suggestions to make models compilable on hardware accelerators (MLA/CVU/APU backends).

## Overview

This system uses a **true RAG architecture** where an LLM (Gemini) not only generates transformation rules but can also **apply suggestions to modify ONNX models** to make them compilable on edge devices.

### Key Capabilities

- **Intelligent Analysis**: Deep ONNX graph analysis to identify compilation blockers
- **RAG-Enhanced Suggestions**: Context-aware suggestions using knowledge from successful transformations
- **Automated Graph Surgery**: Apply suggestions to modify ONNX models programmatically
- **Agentic Pipeline**: ReAct-style agents with strategic planning, state management, and adaptive execution
- **Structural Comparison**: Compare suggested modifications against ground truth
- **Comprehensive Evaluation**: Precision, recall, transformation accuracy metrics
- **Automated Testing**: Test suite for agent system validation

## Recent Updates

### Agentic Pipeline System
- **ReAct-style agents** with multi-step reasoning and tool usage
- **Strategic planning** for complex model transformations
- **State management** and adaptive execution
- **Feedback loops** for continuous improvement
- **Pattern database integration** for historical learning

### Project Reorganization
- Scripts consolidated in `scripts/` directory
- New `agents/` package for agentic pipeline
- Test suite added in `tests/` directory
- Enhanced dependency management with modern LLM libraries

### Enhanced LLM Integration
- Replaced LangChain with `litellm` and `instructor` for better control
- Type-safe outputs with Pydantic models
- Improved error handling and retry logic

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AUTOMATED MODEL SURGERY                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        AGENTIC PIPELINE (NEW)                         │  │
│  │  ┌──────────────┐   ┌──────────────┐   ┌────────────────────────┐     │  │
│  │  │   Strategy   │──▶│    State     │──▶│  Executor & Feedback   │     │  │
│  │  │   Planner    │   │  Management  │   │      Collection        │     │  │
│  │  └──────────────┘   └──────────────┘   └────────────────────────┘     │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────┐    ┌──────────────────┐    ┌────────────────────────┐  │
│  │  Knowledge Base │    │   ONNX Analyzer  │    │  Suggestion Generator  │  │
│  │  ───────────────│    │  ──────────────  │    │  ────────────────────  │  │
│  │  • PDF docs     │───▶│  • Graph parsing │───▶│  • RAG retrieval       │  │
│  │  • Dataset      │    │  • Blocker ID    │    │  • Gemini generation   │  │
│  │  • Patterns     │    │  • Shape analysis│    │  • Priority scoring    │  │
│  └─────────────────┘    └──────────────────┘    └────────────────────────┘  │
│           │                                               │                 │
│           ▼                                               ▼                 │
│  ┌─────────────────┐                          ┌────────────────────────┐    │
│  │  RAG Retriever  │                          │ Suggestion Applicator  │    │
│  │  ─────────────  │                          │ ────────────────────   │    │
│  │  • Similarity   │                          │ • Graph surgery        │    │
│  │  • Multi-factor │                          │ • Node replacement     │    │
│  │  • Ranking      │                          │ • Shape transformation │    │
│  └─────────────────┘                          └────────────────────────┘    │
│                                                          │                  │
│                                                          ▼                  │
│                                               ┌────────────────────────┐    │
│                                               │   Model Comparator     │    │
│                                               │   ────────────────     │    │
│                                               │   • Structural diff    │    │
│                                               │   • Operation match    │    │
│                                               │   • Accuracy metrics   │    │
│                                               └────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Pipeline Flow

```
Training Phase (Build Knowledge Base):
  PDF Documentation + Dataset (Original → Modified Models)
    ↓
  Extract Patterns → Build Knowledge Chunks → Create Vector Store

Inference Phase (Standard RAG):
  Unseen Model
    ↓
  [1] ANALYZE: Deep ONNX analysis to identify blockers
    ↓
  [2] RETRIEVE: Find similar transformation patterns (RAG)
    ↓
  [3] GENERATE: Create prioritized suggestions with Gemini
    ↓
  [4] APPLY: Graph surgery to modify ONNX model
    ↓
  [5] COMPARE: Structural comparison with ground truth
    ↓
  Modified Model (Compilable) + Evaluation Metrics

Agentic Pipeline (NEW - ReAct Style):
  Unseen Model
    ↓
  [1] PLAN: Strategic analysis and transformation planning
    ↓
  [2] EXECUTE: Multi-step reasoning with state management
      ↓ (feedback loop)
      ├─▶ THOUGHT: Analyze current state and plan next action
      ├─▶ ACTION: Use tools (analyzer, RAG, applicator)
      ├─▶ OBSERVE: Evaluate results and collect feedback
      └─▶ ADAPT: Adjust strategy based on outcomes
    ↓
  [3] EVALUATE: Comprehensive evaluation with diagnostics
    ↓
  Modified Model + Detailed Report + Performance Metrics
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/automated-model-surgery.git
cd automated-model-surgery

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `config.py` file with your Gemini API key:

```python
import os

# Gemini API Key
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'your-api-key-here')

# Optional: Hybrid mode settings
USE_HYBRID_MODE = False
DAILY_API_LIMIT = 250
```

Or set the environment variable:

```bash
export GEMINI_API_KEY="your-key-here"
```

### 3. Prepare Data

Place your ONNX models in the `dataset/` directory:

```
dataset/
├── Model_Name_1/
│   ├── original/
│   │   └── model.onnx
│   └── modified/
│       └── model_modified.onnx
├── Model_Name_2/
│   ├── original/
│   │   └── model.onnx
│   └── modified/
│       └── model_modified.onnx
...
```

### 4. Run Complete Workflow

```bash
# Complete workflow: train → test → evaluate
python main.py workflow --api-key YOUR_KEY
```

## Usage

### Main Entry Point

The `main.py` script provides a unified interface for all operations:

```bash
# Generate model maps (text representations of ONNX graphs)
python main.py generate-maps

# Train RAG pipeline (build knowledge base)
python main.py train --rebuild-kb

# Run inference on test set
python main.py inference --test-set

# Generate rules without applying them
python main.py generate-rules --test-only

# Evaluate pipeline performance
python main.py evaluate --test-only

# Complete workflow
python main.py workflow --api-key YOUR_KEY
```

### Agentic Pipeline (Recommended for Complex Models)

The new agentic pipeline provides intelligent, multi-step reasoning for complex models:

```bash
# Run the agentic pipeline on a model
python -m agents.pipeline --model path/to/model.onnx

# With custom configuration
python -m agents.pipeline --model path/to/model.onnx --max-iterations 10 --strategy adaptive
```

### Advisory Pipeline (Production)

The production pipeline provides an advisory system for analyzing models:

```bash
# Analyze a single model
python production_pipeline.py model.onnx --output analysis_output --format markdown

# Analyze all models in a directory
python production_pipeline.py ./models/ --batch --recursive
```

### Comprehensive Evaluation

The evaluation script tests the full pipeline on a test set:

```bash
# Full evaluation (builds KB + evaluates)
python evaluate_rag_pipeline.py

# Skip KB build (use existing)
python evaluate_rag_pipeline.py --skip-kb-build

# Disable enhanced structural comparison
python evaluate_rag_pipeline.py --no-enhanced
```

### Running Tests

Run the test suite to validate the agent system:

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_agents.py

# Run with verbose output
python -m pytest tests/ -v
```

## Project Structure

```
automated-model-surgery/
│
├── agents/                           # Agentic Pipeline System (NEW)
│   ├── __init__.py
│   ├── pipeline.py                   # ReAct-style agentic pipeline
│   ├── strategy_planner.py           # Strategic planning for transformations
│   ├── executor.py                   # State machine execution
│   ├── state.py                      # Pipeline state management
│   ├── diagnostics.py                # Feedback collection & analysis
│   ├── llm_client.py                 # LLM integration
│   └── tools.py                      # Agent tools & actions
│
├── core_analysis/                    # Core ONNX Analysis
│   ├── __init__.py
│   ├── onnx_analyzer.py              # Deep ONNX model analysis
│   ├── dataset_analyzer.py           # Dataset pattern extraction
│   ├── difference_extractor.py       # Model difference extraction
│   └── feature_extractor.py          # Feature extraction (base)
│
├── knowledge_base/                   # Knowledge Base & RAG
│   ├── __init__.py
│   ├── knowledge_base.py             # KB builder and storage
│   ├── rag_retriever.py              # RAG retrieval logic
│   └── response_cache.py             # Gemini response caching
│
├── suggestion_pipeline/              # Suggestion Generation
│   ├── __init__.py
│   ├── suggestion_generator.py       # Base suggestion generation
│   ├── rag_suggestion_generator.py   # RAG-enhanced suggestions
│   ├── suggestion_scorer.py          # Multi-factor scoring
│   └── suggestion_applicator.py      # Apply suggestions (graph surgery)
│
├── rag_pipeline/                     # RAG Pipeline
│   ├── __init__.py
│   ├── rag_pipeline.py               # Core RAG pipeline
│   ├── run_rag_pipeline.py           # Training script
│   └── inference_pipeline.py         # Inference workflow
│
├── evaluation/                       # Evaluation & Comparison
│   ├── __init__.py
│   ├── model_comparator.py           # Structural model comparison
│   ├── evaluator.py                  # Rule evaluation metrics
│   └── evaluate_rag_pipeline.py      # Comprehensive evaluation
│
├── production/                       # Production Pipeline
│   ├── __init__.py
│   ├── production_pipeline.py        # Advisory pipeline
│   └── report_generator.py           # Report generation (MD/JSON/HTML)
│
├── tests/                            # Test Suite (NEW)
│   ├── __init__.py
│   ├── test_agents.py                # Agent system tests
│   └── test_diagnostics.py           # Diagnostics tests
│
├── utilities/                        # Utility Modules
│   ├── __init__.py
│   ├── train_test_split.py           # Dataset splitting
│   ├── api_quota_manager.py          # API quota management
│   └── checkpoint_manager.py         # Checkpoint management
│
├── scripts/                          # Analysis & Workflow Scripts
│   ├── __init__.py
│   ├── complete_workflow.py          # All-in-one workflow
│   ├── generate_all_maps.py          # Generate all model maps
│   ├── print_onnx_graph.py           # ONNX graph visualization
│   ├── pretty_print_cache.py         # Cache inspection
│   ├── analyze_skipped_suggestions.py    # Debug skipped suggestions
│   └── analyze_transformation_issues.py  # Debug transformations
│
├── legacy/                           # Legacy/Compatibility
│   ├── __init__.py
│   ├── rule_parser.py                # Legacy rule parsing
│   ├── rule_applicator.py            # Legacy rule application
│   ├── gemini_model_modifier.py      # Gemini-based modification
│   └── enhanced_feature_extractor.py # Gemini-enhanced features
│
├── react_results_test/               # Test Results & Reports
│   └── mt5_small_encoder_report.json # Example test output
│
├── main.py                           # Main entry point
├── config.py                         # Configuration (API keys) - NOT TRACKED
├── config.py.example                 # Example configuration template
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Key Components

### 1. ONNX Analyzer (`onnx_analyzer.py`)

Deep analysis of ONNX models to identify:
- Compilation blockers (unsupported operations)
- Shape issues (dynamic dimensions, non-4D tensors)
- Graph structure problems
- Optimization opportunities

### 2. Knowledge Base (`knowledge_base.py`)

Builds a knowledge base from:
- PDF documentation (ONNX Graph Surgery best practices)
- Dataset transformation patterns (original → modified models)
- Template-based knowledge chunks

### 3. RAG Suggestion Generator (`rag_suggestion_generator.py`)

Generates context-aware suggestions using:
- Vector similarity search for relevant patterns
- Multi-factor matching (operation type, context, severity)
- Gemini LLM for intelligent suggestion generation
- Priority-based ranking

### 4. Suggestion Applicator (`suggestion_applicator.py`)

Applies suggestions via graph surgery:
- Node replacement (e.g., Einsum → MatMul)
- Shape transformations (e.g., reshape to 4D)
- Graph restructuring
- Validation and rollback

### 5. Model Comparator (`model_comparator.py`)

Compares models structurally:
- Operation-level comparison
- Shape matching
- Transformation accuracy metrics
- Critical area analysis

### 6. Agentic Pipeline (`agents/pipeline.py`)

**NEW**: ReAct-style agentic system for intelligent graph surgery:
- **Strategic Planning**: Analyzes model complexity and plans transformation strategy
- **State Management**: Tracks pipeline state across execution steps
- **Adaptive Execution**: Adjusts strategy based on feedback and results
- **Tool Integration**: Uses analyzer, suggestion generator, and applicator as tools
- **Feedback Loop**: Learns from failures and adjusts approach
- **Pattern Database**: Leverages historical transformation patterns

The agentic pipeline provides:
- Multi-step reasoning for complex models
- Automatic error recovery and retry logic
- Progress tracking and checkpointing
- Comprehensive evaluation and reporting

## Evaluation Metrics

### Simple Comparison Metrics
- **Precision**: Correctly identified blockers / Total detected
- **Recall**: Correctly identified blockers / Total ground truth blockers
- **F1 Score**: Harmonic mean of precision and recall

### Enhanced Comparison Metrics
- **Structural Similarity**: Overall model structure match
- **Operation Similarity**: Jaccard similarity of operations
- **Transformation Accuracy**: How well suggestions match ground truth changes
- **Critical Areas Match**: Match in critical transformation regions

### Transformation Metrics
- **Applied Suggestions**: Successfully applied suggestions
- **Transformed Suggestions**: Suggestions that changed the model
- **Transformation Effectiveness**: Transformed / Attempted ratio

## Configuration Options

### Environment Variables

```bash
export GEMINI_API_KEY="your-api-key"
export USE_HYBRID_MODE="true"          # Enable hybrid mode
export DAILY_API_LIMIT="250"           # Daily API call limit
export CHECKPOINT_DIR="checkpoints"    # Checkpoint directory
export RESUME_FROM_CHECKPOINT="true"   # Resume from checkpoint
```

### Command Line Options

```bash
# Training options
--rebuild-kb          # Rebuild knowledge base from scratch
--use-enhanced        # Use Gemini-enhanced features

# Inference options
--test-set           # Process test set
--model PATH         # Process single model
--output-dir DIR     # Output directory

# Evaluation options
--skip-kb-build      # Skip KB building
--no-enhanced        # Disable enhanced comparison
--no-rag             # Baseline comparison (no RAG)
```

## Output Files

### Analysis Output
```
analysis_output/
├── model_name_report.md      # Human-readable report
├── model_name_analysis.json  # Structured analysis data
└── batch_summary.json        # Batch processing summary
```

### Evaluation Output
```
evaluation_output/
├── evaluation_report.json    # Comprehensive metrics
├── suggested_models/         # Models with applied suggestions
│   └── model_suggested.onnx
└── model_test_analysis.json  # Per-model analysis
```

## ONNX Graph Surgery Principles

Based on the ONNX Graph Surgery documentation, this system applies:

1. **4D Tensor Requirement**: Reshape non-4D tensors throughout the model for MLA compatibility
2. **Operator Replacement**: Replace unsupported operators (Einsum, Complex, etc.) with supported equivalents
3. **Shape Resolution**: Resolve dynamic/unknown shapes to concrete dimensions
4. **Pattern Optimization**: Replace Reshape/Transpose with Slice/Concat patterns when beneficial
5. **Divide and Conquer**: Split complex models, modify parts, then recombine

## Dependencies

See `requirements.txt`:
- `onnx>=1.20.0` - ONNX model handling
- `google-generativeai>=0.8.0` - Gemini API
- `litellm>=1.0.0` - Unified LLM API interface
- `instructor>=1.0.0` - Structured LLM outputs
- `pydantic>=2.0.0` - Type-safe data models
- `numpy>=1.23.2` - Numerical operations
- `onnxruntime>=1.16.0` - (Optional) Model execution for comparison
- `pytest` - (Optional) For running test suite

## Troubleshooting

### "Knowledge base not found"
```bash
python main.py train --rebuild-kb
# or
python run_rag_pipeline.py --rebuild-kb
```

### "API key required"
```bash
export GEMINI_API_KEY="your-key"
# or use --api-key flag
python main.py workflow --api-key YOUR_KEY
```

### "Model not found" (Gemini)
The system automatically tries multiple Gemini models:
1. gemini-3-pro-preview
2. gemini-1.5-pro
3. gemini-pro
4. gemini-2.0-flash-exp

### Poor quality suggestions
- Rebuild knowledge base: `--rebuild-kb`
- Check training data quality
- Ensure diverse transformation patterns in dataset

### API quota exceeded
- Use `--skip-kb-build` to reuse existing KB
- Enable hybrid mode with checkpointing
- Run `evaluate_rag_pipeline.py --no-rag` for baseline

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and test
4. Submit a pull request
