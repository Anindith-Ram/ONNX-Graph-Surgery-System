#!/usr/bin/env python3
"""
Main entry point for Automated Model Surgery RAG Pipeline.

This script consolidates all workflows:
- Generate model maps
- Train RAG pipeline (build knowledge base)
- Run inference (generate rules and modify models)
- Complete workflow (train + test + evaluate)
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional


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
    import os
    
    print("=" * 80)
    print("Generating ONNX Model Maps")
    print("=" * 80)
    
    # Run generate_all_maps.py as a script
    script_path = os.path.join(os.path.dirname(__file__), "generate_all_maps.py")
    subprocess.run([sys.executable, script_path], check=True)
    
    print("\n✓ Model maps generated in map_dataset/")


def cmd_train(args):
    """Train RAG pipeline: build knowledge base from training models."""
    from run_rag_pipeline import setup_pipeline
    
    print("=" * 80)
    print("Training RAG Pipeline (Building Knowledge Base)")
    print("=" * 80)
    
    api_key = get_api_key(args.api_key)
    
    pipeline, rag_data_dir, train_models, test_models = setup_pipeline(
        api_key,
        rebuild_kb=args.rebuild_kb,
        use_enhanced=args.use_enhanced
    )
    
    print(f"\n✓ Knowledge base built from {len(train_models)} training models")
    print(f"  - Features: {rag_data_dir}/features_train.json")
    print(f"  - Knowledge base: {rag_data_dir}/knowledge_base.pkl")


def cmd_inference(args):
    """Run inference: generate rules and modify models."""
    from inference_pipeline import InferencePipeline
    from rag_pipeline import RAGPipeline
    from train_test_split import load_train_test_split
    from pathlib import Path
    
    print("=" * 80)
    print("Running Inference Pipeline")
    print("=" * 80)
    
    api_key = get_api_key(args.api_key)
    
    # Initialize RAG pipeline
    print("Initializing RAG pipeline...")
    rag_pipeline = RAGPipeline(api_key)
    
    # Load knowledge base
    kb_file = Path("rag_data/knowledge_base.pkl")
    if not kb_file.exists():
        print("Error: Knowledge base not found. Run training first:")
        print("  python main.py train --rebuild-kb")
        sys.exit(1)
    
    import pickle
    with open(kb_file, 'rb') as f:
        rag_pipeline.vector_store = pickle.load(f)
    print("✓ Knowledge base loaded")
    
    # Initialize inference pipeline
    inference_pipeline = InferencePipeline(rag_pipeline, gemini_api_key=api_key)
    
    # Process models
    if args.test_set:
        split_file = args.split_file or "rag_data/train_test_split.json"
        _, test_models = load_train_test_split(split_file)
        print(f"\nProcessing {len(test_models)} test models...")
        results = inference_pipeline.process_test_set(
            test_models,
            output_dir=args.output_dir or "inference_results"
        )
        print(f"\n✓ Processed {len(results)} models")
        print(f"  Results: {args.output_dir or 'inference_results'}/all_results.json")
    elif args.model:
        print(f"\nProcessing model: {args.model}")
        result = inference_pipeline.process_model(
            args.model,
            output_dir=args.output_dir or "inference_results"
        )
        print(f"\n✓ Model processed")
        print(f"  Results: {result.get('results_path', 'N/A')}")
    else:
        print("Error: Specify --test-set or --model")
        sys.exit(1)


def cmd_generate_rules(args):
    """Generate rules for models (without modifying them)."""
    from run_rag_pipeline import setup_pipeline, generate_rules_for_models
    from pathlib import Path
    
    print("=" * 80)
    print("Generating Rules")
    print("=" * 80)
    
    api_key = get_api_key(args.api_key)
    
    pipeline, rag_data_dir, train_models, test_models = setup_pipeline(
        api_key,
        rebuild_kb=False,
        use_enhanced=args.use_enhanced
    )
    
    if args.test_only:
        models = test_models
        split_name = "test"
    elif args.train_only:
        models = train_models
        split_name = "train"
    elif args.all:
        models = train_models + test_models
        split_name = "all"
    else:
        models = test_models
        split_name = "test"
    
    print(f"\nGenerating rules for {len(models)} models...")
    generate_rules_for_models(pipeline, rag_data_dir, models, split_name)
    
    print(f"\n✓ Rules generated")
    print(f"  Rules saved to: {rag_data_dir}/rules_*.json")


def cmd_evaluate(args):
    """Evaluate generated rules against ground truth."""
    from run_rag_pipeline import setup_pipeline
    from evaluator import RuleEvaluator
    
    print("=" * 80)
    print("Evaluating Rules")
    print("=" * 80)
    
    api_key = get_api_key(args.api_key)
    
    pipeline, rag_data_dir, train_models, test_models = setup_pipeline(
        api_key,
        rebuild_kb=False,
        use_enhanced=args.use_enhanced
    )
    
    evaluator = RuleEvaluator()
    
    if args.test_only:
        models = test_models
    else:
        models = train_models + test_models
    
    print(f"\nEvaluating rules for {len(models)} models...")
    results = evaluator.evaluate_all(
        rag_data_dir,
        models,
        output_file=str(rag_data_dir / "evaluation_results.json")
    )
    
    print(f"\n✓ Evaluation complete")
    print(f"  Results: {rag_data_dir}/evaluation_results.json")
    print(f"  Report: {rag_data_dir}/evaluation_report.txt")


def cmd_complete_workflow(args):
    """Complete workflow: train -> test -> evaluate."""
    from train_test_split import get_all_models, create_train_test_split
    from pathlib import Path
    
    print("=" * 80)
    print("Complete Workflow: Train -> Test -> Evaluate")
    print("=" * 80)
    
    api_key = get_api_key(args.api_key)
    base_dir = Path(__file__).parent
    rag_data_dir = base_dir / "rag_data"
    rag_data_dir.mkdir(exist_ok=True)
    
    split_file = rag_data_dir / "train_test_split.json"
    
    # Step 1: Create train/test split
    if not args.skip_split and (not split_file.exists() or args.rebuild_split):
        print("\n[1/4] Creating train/test split...")
        map_dataset_dir = base_dir / "map_dataset"
        models = get_all_models(str(map_dataset_dir))
        train_models, test_models = create_train_test_split(
            models,
            train_ratio=0.8,
            seed=42,
            output_file=str(split_file)
        )
        print(f"✓ Split created: {len(train_models)} train, {len(test_models)} test")
    else:
        print("[1/4] Using existing train/test split")
    
    # Step 2: Train (build knowledge base)
    if not args.skip_train:
        print("\n[2/4] Training RAG pipeline (building knowledge base)...")
        train_args = argparse.Namespace(
            api_key=api_key,
            rebuild_kb=args.rebuild_kb,
            use_enhanced=args.use_enhanced
        )
        cmd_train(train_args)
    else:
        print("[2/4] Skipping training (using existing knowledge base)")
    
    # Step 3: Generate rules for test set
    print("\n[3/4] Generating rules for test set...")
    rules_args = argparse.Namespace(
        api_key=api_key,
        test_only=True,
        use_enhanced=args.use_enhanced
    )
    cmd_generate_rules(rules_args)
    
    # Step 4: Run inference (modify models)
    print("\n[4/4] Running inference (modifying models with Gemini)...")
    inference_args = argparse.Namespace(
        api_key=api_key,
        test_set=True,
        split_file=str(split_file),
        output_dir="inference_results"
    )
    cmd_inference(inference_args)
    
    # Step 5: Evaluate
    print("\n[5/5] Evaluating results...")
    eval_args = argparse.Namespace(
        api_key=api_key,
        test_only=True,
        use_enhanced=args.use_enhanced
    )
    cmd_evaluate(eval_args)
    
    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETE")
    print("=" * 80)
    print("\nResults:")
    print(f"  - Train/test split: {split_file}")
    print(f"  - Knowledge base: {rag_data_dir}/knowledge_base.pkl")
    print(f"  - Test rules: {rag_data_dir}/rules_*_test.json")
    print(f"  - Modified models: inference_results/")
    print(f"  - Evaluation: {rag_data_dir}/evaluation_results.json")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Automated Model Surgery RAG Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate model maps
  python main.py generate-maps

  # Train RAG pipeline
  python main.py train --rebuild-kb

  # Run inference on test set
  python main.py inference --test-set

  # Complete workflow
  python main.py workflow
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Generate maps
    parser_generate = subparsers.add_parser('generate-maps', help='Generate ONNX model maps')
    parser_generate.add_argument('--dataset-dir', default='dataset', help='Dataset directory')
    parser_generate.add_argument('--output-dir', default='map_dataset', help='Output directory')
    
    # Train
    parser_train = subparsers.add_parser('train', help='Train RAG pipeline (build knowledge base)')
    parser_train.add_argument('--api-key', help='Gemini API key')
    parser_train.add_argument('--rebuild-kb', action='store_true', help='Rebuild knowledge base')
    parser_train.add_argument('--use-enhanced', action='store_true', default=False, help='Use Gemini-enhanced features (default: False, uses rule-based features)')
    parser_train.add_argument('--no-use-enhanced', dest='use_enhanced', action='store_false', help='Disable Gemini-enhanced features (use rule-based features)')
    
    # Inference
    parser_inference = subparsers.add_parser('inference', help='Run inference (generate rules and modify models)')
    parser_inference.add_argument('--api-key', help='Gemini API key')
    parser_inference.add_argument('--test-set', action='store_true', help='Process all test models')
    parser_inference.add_argument('--model', help='Process single model')
    parser_inference.add_argument('--split-file', default='rag_data/train_test_split.json')
    parser_inference.add_argument('--output-dir', default='inference_results')
    
    # Generate rules
    parser_rules = subparsers.add_parser('generate-rules', help='Generate rules (without modifying models)')
    parser_rules.add_argument('--api-key', help='Gemini API key')
    parser_rules.add_argument('--test-only', action='store_true', help='Generate for test set only')
    parser_rules.add_argument('--train-only', action='store_true', help='Generate for train set only')
    parser_rules.add_argument('--all', action='store_true', help='Generate for all models')
    parser_rules.add_argument('--use-enhanced', action='store_true', default=False, help='Use Gemini-enhanced features')
    parser_rules.add_argument('--no-use-enhanced', dest='use_enhanced', action='store_false', help='Disable Gemini-enhanced features')
    
    # Evaluate
    parser_eval = subparsers.add_parser('evaluate', help='Evaluate generated rules')
    parser_eval.add_argument('--api-key', help='Gemini API key')
    parser_eval.add_argument('--test-only', action='store_true', help='Evaluate test set only')
    parser_eval.add_argument('--use-enhanced', action='store_true', default=False, help='Use Gemini-enhanced features')
    parser_eval.add_argument('--no-use-enhanced', dest='use_enhanced', action='store_false', help='Disable Gemini-enhanced features')
    
    # Complete workflow
    parser_workflow = subparsers.add_parser('workflow', help='Complete workflow: train -> test -> evaluate')
    parser_workflow.add_argument('--api-key', help='Gemini API key')
    parser_workflow.add_argument('--skip-split', action='store_true', help='Skip train/test split')
    parser_workflow.add_argument('--skip-train', action='store_true', help='Skip training')
    parser_workflow.add_argument('--rebuild-split', action='store_true', help='Rebuild train/test split')
    parser_workflow.add_argument('--rebuild-kb', action='store_true', help='Rebuild knowledge base')
    parser_workflow.add_argument('--use-enhanced', action='store_true', default=False, help='Use Gemini-enhanced features')
    parser_workflow.add_argument('--no-use-enhanced', dest='use_enhanced', action='store_false', help='Disable Gemini-enhanced features')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Route to appropriate command
    commands = {
        'generate-maps': cmd_generate_maps,
        'train': cmd_train,
        'inference': cmd_inference,
        'generate-rules': cmd_generate_rules,
        'evaluate': cmd_evaluate,
        'workflow': cmd_complete_workflow
    }
    
    commands[args.command](args)


if __name__ == "__main__":
    main()

