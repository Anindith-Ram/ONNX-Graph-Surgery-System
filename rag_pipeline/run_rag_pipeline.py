#!/usr/bin/env python3
"""
Main script to run the complete RAG pipeline for ONNX model compilation rules.
"""

import os
import sys
import json
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag_pipeline.rag_pipeline import RAGPipeline
from core_analysis.difference_extractor import extract_differences, extract_all_differences
from core_analysis.feature_extractor import extract_all_features


def setup_pipeline(api_key: str, rebuild_kb: bool = False, use_enhanced: bool = False):
    """Setup and initialize the RAG pipeline with train/test split."""
    from utilities.train_test_split import load_train_test_split, get_all_models, create_train_test_split
    
    base_dir = Path(__file__).parent.parent
    rag_data_dir = base_dir / "rag_data"
    rag_data_dir.mkdir(exist_ok=True)
    
    split_file = rag_data_dir / "train_test_split.json"
    features_file = rag_data_dir / "features_train.json" if use_enhanced else rag_data_dir / "features.json"
    kb_file = rag_data_dir / "knowledge_base.pkl"
    
    # Create train/test split if it doesn't exist
    if not split_file.exists():
        print("Creating train/test split...")
        map_dataset_dir = base_dir / "map_dataset"
        models = get_all_models(str(map_dataset_dir))
        train_models, test_models = create_train_test_split(
            models,
            train_ratio=0.8,
            seed=42,
            output_file=str(split_file)
        )
    else:
        train_models, test_models = load_train_test_split(str(split_file))
    
    print(f"Train set: {len(train_models)} models")
    print(f"Test set: {len(test_models)} models")
    
    # Initialize pipeline
    print("Initializing RAG pipeline...")
    pipeline = RAGPipeline(api_key)
    
    # Build or load knowledge base (training models only)
    if rebuild_kb or not kb_file.exists():
        print("\n=== Building Knowledge Base (Training Models Only) ===")
        map_dataset_dir = base_dir / "map_dataset"
        
        if not features_file.exists() or rebuild_kb:
            if use_enhanced:
                print("Extracting enhanced features with Gemini...")
                from enhanced_feature_extractor import extract_all_features_enhanced
                features = extract_all_features_enhanced(
                    str(map_dataset_dir),
                    train_models,
                    api_key,
                    str(features_file)
                )
            else:
                print("Extracting features from model maps...")
                from feature_extractor import extract_all_features
                all_features = extract_all_features(str(map_dataset_dir), None)
                # Filter to training models
                features = [f for f in all_features if f['model_name'] in train_models]
                with open(features_file, 'w') as f:
                    json.dump(features, f, indent=2)
            print(f"Extracted features for {len(features)} training models")
        else:
            print(f"Loading existing features from {features_file}...")
            with open(features_file, 'r') as f:
                features = json.load(f)
        
        print("Building vector store from training models...")
        pipeline.build_knowledge_base(str(features_file), train_models=train_models)
        
        # Save knowledge base
        import pickle
        with open(kb_file, 'wb') as f:
            pickle.dump(pipeline.vector_store, f)
        print(f"Knowledge base saved to {kb_file}")
    else:
        print(f"Loading knowledge base from {kb_file}...")
        import pickle
        with open(kb_file, 'rb') as f:
            pipeline.vector_store = pickle.load(f)
        print("Knowledge base loaded")
    
    return pipeline, rag_data_dir, train_models, test_models


def generate_rules_for_models(pipeline: RAGPipeline, rag_data_dir: Path, model_names: list[str], split_name: str = ""):
    """Generate rules for specified models."""
    base_dir = Path(__file__).parent
    map_dataset_dir = base_dir / "map_dataset"
    
    results = []
    
    for model_name in model_names:
        map_file = map_dataset_dir / f"{model_name}.txt"
        
        if not map_file.exists():
            print(f"Warning: {map_file} not found, skipping")
            continue
        
        print(f"\n=== Processing {model_name} ===")
        
        try:
            # Extract differences
            diff = extract_differences(str(map_file))
            
            # Generate rules
            result = pipeline.generate_rules(diff)
            results.append(result)
            
            # Save individual result
            output_file = rag_data_dir / f"rules_{diff.model_name.replace('/', '_')}{split_name}.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"✓ Generated {len(result['generated_rules'])} rules")
            print(f"  Saved to {output_file}")
            
        except Exception as e:
            print(f"✗ Error processing {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save consolidated results
    if split_name:
        consolidated_file = rag_data_dir / f"all_rules_{split_name}.json"
    else:
        consolidated_file = rag_data_dir / "all_rules.json"
    
    with open(consolidated_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Summary ===")
    print(f"Processed {len(results)} models")
    print(f"Consolidated rules saved to {consolidated_file}")
    
    return results


def generate_rules_for_single_model(pipeline: RAGPipeline, model_map_path: str, rag_data_dir: Path):
    """Generate rules for a single model."""
    print(f"\n=== Processing {model_map_path} ===")
    
    # Extract differences
    diff = extract_differences(model_map_path)
    
    # Generate rules
    result = pipeline.generate_rules(diff)
    
    # Save result
    output_file = rag_data_dir / f"rules_{diff.model_name.replace('/', '_')}.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n=== Generated Rules for {diff.model_name} ===")
    for i, rule in enumerate(result['generated_rules'], 1):
        print(f"\n{i}. {rule['name']}")
        print(f"   Condition: {rule['condition']}")
        print(f"   Transformation: {rule['transformation']}")
        print(f"   Steps: {rule['steps']}")
        print(f"   Benefit: {rule['benefit']}")
    
    print(f"\nFull results saved to {output_file}")
    
    return result


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='RAG Pipeline for ONNX Model Compilation Rules',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build knowledge base and generate rules for all models
  python run_rag_pipeline.py --api-key YOUR_KEY --rebuild-kb --all
  
  # Generate rules for a specific model
  python run_rag_pipeline.py --api-key YOUR_KEY --model-map map_dataset/YOLO_11n_1.txt
  
  # Just rebuild knowledge base
  python run_rag_pipeline.py --api-key YOUR_KEY --rebuild-kb
        """
    )
    
    parser.add_argument(
        '--api-key',
        help='Gemini API key (or set GEMINI_API_KEY env var, or use config.py)',
        default=None
    )
    parser.add_argument(
        '--rebuild-kb',
        action='store_true',
        help='Rebuild the knowledge base from scratch'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Generate rules for all models in map_dataset'
    )
    parser.add_argument(
        '--model-map',
        help='Path to a specific model map file to analyze'
    )
    parser.add_argument(
        '--test-only',
        action='store_true',
        help='Generate rules only for test set models'
    )
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate RAG rules against ground truth (use with --test-only)'
    )
    parser.add_argument(
        '--use-enhanced',
        action='store_true',
        default=True,
        help='Use Gemini-enhanced feature extraction (default: True)'
    )
    parser.add_argument(
        '--no-enhanced',
        dest='use_enhanced',
        action='store_false',
        help='Disable Gemini-enhanced feature extraction'
    )
    
    args = parser.parse_args()
    
    # Get API key from: command line > environment variable > config file
    api_key = args.api_key
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
    
    # Use the resolved API key
    args.api_key = api_key
    
    base_dir = Path(__file__).parent
    
    # Setup pipeline
    pipeline, rag_data_dir, train_models, test_models = setup_pipeline(
        args.api_key,
        args.rebuild_kb,
        use_enhanced=args.use_enhanced
    )
    
    # Generate rules
    if args.test_only:
        print("\n=== Generating Rules for TEST Set ===")
        generate_rules_for_models(pipeline, rag_data_dir, test_models, split_name="_test")
        
        # Evaluate
        if args.evaluate:
            print("\n=== Evaluating RAG Pipeline ===")
            from evaluator import RuleEvaluator
            evaluator = RuleEvaluator()
            results = evaluator.evaluate_all(
                test_models,
                str(rag_data_dir),
                str(base_dir / "map_dataset")
            )
            
            eval_output = rag_data_dir / "evaluation_results.json"
            with open(eval_output, 'w') as f:
                json.dump(results, f, indent=2)
            
            report_file = rag_data_dir / "evaluation_report.txt"
            evaluator.generate_report(results, str(report_file))
            
    elif args.all:
        print("\n=== Generating Rules for ALL Models ===")
        all_models = train_models + test_models
        generate_rules_for_models(pipeline, rag_data_dir, all_models)
    elif args.model_map:
        generate_rules_for_single_model(pipeline, args.model_map, rag_data_dir)
    else:
        print("\nNo action specified. Use --test-only, --all, or --model-map")
        print("Use --help for usage information")


if __name__ == "__main__":
    main()

