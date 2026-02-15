#!/usr/bin/env python3
"""
Main entry point for Automated Model Surgery Pipeline (LangGraph).

Commands:
- react:          Run the LangGraph graph surgery pipeline
                  (diagnose -> plan -> execute -> validate [-> refine] -> enrich -> evaluate)
- build-kb:       Build or rebuild the knowledge base from dataset + PDF
- generate-maps:  Generate ONNX model maps for all models
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))


def get_api_key(args_api_key: Optional[str] = None) -> str:
    """Get API key from args, env, or config."""
    api_key = args_api_key
    if not api_key:
        api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        try:
            from config import GEMINI_API_KEY
            api_key = GEMINI_API_KEY
        except ImportError:
            pass

    if not api_key:
        print("Error: Gemini API key required.")
        print("  Options:")
        print("  1. Use --api-key flag")
        print("  2. Set GEMINI_API_KEY environment variable")
        print("  3. Add GEMINI_API_KEY to config.py")
        sys.exit(1)

    return api_key


def cmd_generate_maps(args):
    """Generate ONNX model maps for all models."""
    import subprocess

    print("=" * 80)
    print("Generating ONNX Model Maps")
    print("=" * 80)

    script_path = os.path.join(os.path.dirname(__file__), "scripts", "generate_all_maps.py")
    subprocess.run([sys.executable, script_path], check=True)

    print("\n✓ Model maps generated in map_dataset/")


def cmd_build_kb(args):
    """Build or rebuild the knowledge base."""
    from knowledge_base.knowledge_base import KnowledgeBaseBuilder

    print("=" * 80)
    print("Building Knowledge Base")
    print("=" * 80)

    api_key = get_api_key(args.api_key) if not args.no_gemini else None

    builder = KnowledgeBaseBuilder(
        api_key=api_key,
        use_gemini_enhancement=not args.no_gemini,
    )

    kb = builder.build(
        pdf_path=args.pdf,
        dataset_dir=args.dataset,
        train_test_split_path=args.split_file,
        use_train_only=not args.use_all,
    )

    output_path = args.output
    kb.save(output_path)

    print(f"\n✓ Knowledge base saved to {output_path}")
    print(f"  Total chunks: {len(kb.chunks)}")


def cmd_react(args):
    """Run the agentic graph surgery pipeline."""
    from agents.pipeline import GraphSurgeryPipeline
    from agents.config import PipelineConfig, AgentConfig, StrategyConfig

    print("=" * 80)
    print("LangGraph Graph Surgery Pipeline")
    print("=" * 80)

    api_key = get_api_key(args.api_key)

    agent_config = AgentConfig(
        max_iterations=args.max_iterations,
        verbose=args.verbose,
        use_strategy_planning=args.use_strategy,
        use_pattern_db=getattr(args, 'use_pattern_db', True),
    )

    strategy_config = StrategyConfig(
        verbose=args.verbose,
        use_pattern_db=getattr(args, 'use_pattern_db', True),
    )

    config = PipelineConfig(
        agent_config=agent_config,
        strategy_config=strategy_config,
        output_dir=args.output_dir,
        use_pattern_db=getattr(args, 'use_pattern_db', True),
        max_surgery_retries=args.max_iterations,
        verbose=args.verbose,
    )

    pipeline = GraphSurgeryPipeline(api_key=api_key, config=config)

    if args.test_set:
        from utilities.train_test_split import load_train_test_split
        _, test_models = load_train_test_split(args.split_file)

        print(f"\nProcessing {len(test_models)} test models...")

        base_dir = Path(__file__).parent
        dataset_dir = base_dir / "dataset"

        model_paths = []
        ground_truth_paths = []

        for model_name in test_models:
            model_dir = dataset_dir / model_name

            original_path = None
            for orig_dir in ["original", "Original"]:
                orig_full = model_dir / orig_dir
                if orig_full.exists():
                    for file in orig_full.iterdir():
                        if file.suffix == '.onnx':
                            original_path = str(file)
                            break
                    if original_path:
                        break

            gt_path = None
            for mod_dir in ["modified", "Modified"]:
                mod_full = model_dir / mod_dir
                if mod_full.exists():
                    for file in mod_full.iterdir():
                        if file.suffix == '.onnx':
                            gt_path = str(file)
                            break
                    if gt_path:
                        break

            if original_path:
                model_paths.append(original_path)
                ground_truth_paths.append(gt_path)

        results = []
        for i, (model_path, gt_path) in enumerate(zip(model_paths, ground_truth_paths)):
            print(f"\n[{i+1}/{len(model_paths)}] {Path(model_path).parent.parent.name}")
            try:
                result = pipeline.process(model_path, gt_path)
                results.append(result)
            except Exception as e:
                print(f"  Error: {e}")

        success_count = sum(1 for r in results if r.success)
        print(f"\n{'='*80}")
        print("Summary")
        print(f"{'='*80}")
        print(f"Processed: {len(results)}")
        print(f"Successful: {success_count}")
        print(f"Success rate: {success_count/len(results):.1%}" if results else "N/A")
        print(f"Results saved to: {args.output_dir}/")

    elif args.model:
        print(f"\nProcessing: {args.model}")

        model_path = Path(args.model)
        gt_path = None

        if model_path.parent.name.lower() == 'original':
            model_dir = model_path.parent.parent
            for mod_dir in ["modified", "Modified"]:
                mod_full = model_dir / mod_dir
                if mod_full.exists():
                    for file in mod_full.iterdir():
                        if file.suffix == '.onnx':
                            gt_path = str(file)
                            break
                    if gt_path:
                        break

        result = pipeline.process(args.model, gt_path)

        print(f"\n{'='*80}")
        print(f"Result: {'SUCCESS' if result.success else 'PARTIAL'}")
        print(f"{'='*80}")
        print(f"Phases: {result.suggestions_count}")
        if result.execution_result:
            print(f"Iterations: {result.execution_result.get('iterations', 0)}")
        if result.evaluation:
            ev = result.evaluation
            rate = ev.get("blocker_resolution_rate", 0)
            print(f"Blocker resolution: {rate:.0%}")
            print(f"Compiles: {ev.get('compilation_passes', False)}")
            gt_sim = ev.get("gt_similarity")
            if gt_sim is not None:
                print(f"GT similarity: {gt_sim:.1%}")
            print(f"KB records added: {ev.get('kb_records_added', 0)}")
        print(f"Time: {result.total_time_seconds:.1f}s")
        if result.modified_model_path:
            print(f"Modified model: {result.modified_model_path}")

    else:
        print("Error: Specify --model or --test-set")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Automated Model Surgery - LangGraph Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run pipeline on a single model (diagnose -> plan -> execute -> validate -> evaluate)
  python main.py react --model dataset/T5_Small/original/model.onnx --verbose

  # Run pipeline on test set with 3 retry iterations
  python main.py react --test-set --max-iterations 3

  # Build/rebuild knowledge base
  python main.py build-kb --rebuild

  # Generate model maps
  python main.py generate-maps
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Agentic Pipeline
    parser_react = subparsers.add_parser('react', help='Run agentic graph surgery pipeline')
    parser_react.add_argument('--api-key', help='API key (Gemini)')
    parser_react.add_argument('--model', help='Single model to process')
    parser_react.add_argument('--test-set', action='store_true', help='Process all test models')
    parser_react.add_argument('--split-file', default='rag_data/train_test_split.json', help='Train/test split file')
    parser_react.add_argument('--output-dir', default='react_results', help='Output directory')
    parser_react.add_argument('--use-strategy', action='store_true', default=True, help='Enable strategic planning (default: True)')
    parser_react.add_argument('--no-strategy', dest='use_strategy', action='store_false', help='Disable strategic planning')
    parser_react.add_argument('--use-pattern-db', action='store_true', default=True, help='Use pattern database (default: True)')
    parser_react.add_argument('--no-pattern-db', dest='use_pattern_db', action='store_false', help='Disable pattern database')
    parser_react.add_argument('--max-iterations', type=int, default=15, help='Max iterations (default: 15)')
    parser_react.add_argument('--verbose', action='store_true', help='Verbose output')

    # Build Knowledge Base
    parser_kb = subparsers.add_parser('build-kb', help='Build or rebuild the knowledge base')
    parser_kb.add_argument('--api-key', help='Gemini API key')
    parser_kb.add_argument('--pdf', default='ONNX Graph Surgery for Model SDK.pdf', help='Path to SDK PDF')
    parser_kb.add_argument('--dataset', default='dataset', help='Dataset directory')
    parser_kb.add_argument('--output', default='rag_data/knowledge_base.json', help='Output KB path')
    parser_kb.add_argument('--split-file', default='rag_data/train_test_split.json', help='Train/test split file')
    parser_kb.add_argument('--use-all', action='store_true', help='Use all models (not just training set)')
    parser_kb.add_argument('--no-gemini', action='store_true', help='Disable Gemini enhancement')

    # Generate Maps
    parser_maps = subparsers.add_parser('generate-maps', help='Generate ONNX model maps')
    parser_maps.add_argument('--dataset-dir', default='dataset', help='Dataset directory')
    parser_maps.add_argument('--output-dir', default='map_dataset', help='Output directory')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        'react': cmd_react,
        'build-kb': cmd_build_kb,
        'generate-maps': cmd_generate_maps,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
