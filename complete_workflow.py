#!/usr/bin/env python3
"""
Complete workflow script for train/test split RAG pipeline.
Runs the entire pipeline: split -> train -> test -> evaluate
"""

import os
import sys
from pathlib import Path
import subprocess


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: {description} failed")
        print(result.stderr)
        return False
    
    print(result.stdout)
    return True


def main():
    """Run the complete workflow."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Complete workflow: train/test split -> train -> test -> evaluate'
    )
    parser.add_argument(
        '--api-key',
        help='Gemini API key (or set GEMINI_API_KEY env var, or use config.py)',
        default=None
    )
    parser.add_argument(
        '--skip-split',
        action='store_true',
        help='Skip train/test split creation (use existing)'
    )
    parser.add_argument(
        '--skip-train',
        action='store_true',
        help='Skip knowledge base building (use existing)'
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
    
    args.api_key = api_key
    
    base_dir = Path(__file__).parent
    
    # Step 1: Create train/test split
    if not args.skip_split:
        if not run_command(
            ['python3', 'train_test_split.py'],
            'Step 1: Creating train/test split (17 train, 5 test)'
        ):
            print("Failed to create train/test split")
            sys.exit(1)
    
    # Step 2: Build knowledge base from training models
    if not args.skip_train:
        if not run_command(
            ['python3', 'run_rag_pipeline.py', '--api-key', args.api_key, '--rebuild-kb', '--use-enhanced'],
            'Step 2: Building knowledge base from 17 training models (with Gemini-enhanced features)'
        ):
            print("Failed to build knowledge base")
            sys.exit(1)
    
    # Step 3: Generate rules for test set
    if not run_command(
        ['python3', 'run_rag_pipeline.py', '--api-key', args.api_key, '--test-only'],
        'Step 3: Generating rules for 5 test models'
    ):
        print("Failed to generate rules for test set")
        sys.exit(1)
    
    # Step 4: Evaluate
    if not run_command(
        ['python3', 'run_rag_pipeline.py', '--api-key', args.api_key, '--test-only', '--evaluate'],
        'Step 4: Evaluating RAG rules against ground truth'
    ):
        print("Failed to evaluate")
        sys.exit(1)
    
    # Summary
    print(f"\n{'='*80}")
    print("WORKFLOW COMPLETE")
    print(f"{'='*80}")
    print("\nResults:")
    print(f"  - Train/test split: rag_data/train_test_split.json")
    print(f"  - Training features: rag_data/features_train.json")
    print(f"  - Knowledge base: rag_data/knowledge_base.pkl")
    print(f"  - Test rules: rag_data/rules_*_test.json")
    print(f"  - Evaluation: rag_data/evaluation_results.json")
    print(f"  - Report: rag_data/evaluation_report.txt")
    print("\nView evaluation report:")
    print(f"  cat rag_data/evaluation_report.txt")


if __name__ == "__main__":
    main()

