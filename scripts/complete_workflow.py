#!/usr/bin/env python3
"""
Complete workflow script for train/test split RAG pipeline.
Runs the entire pipeline: split -> train -> test -> evaluate

NOTE: This script is deprecated. Use 'python main.py workflow' instead.
"""

import os
import sys
from pathlib import Path
import subprocess
import warnings

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent


def run_command(cmd, description, cwd=None):
    """Run a command and handle errors."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd or PROJECT_ROOT)
    
    if result.returncode != 0:
        print(f"ERROR: {description} failed")
        print(result.stderr)
        return False
    
    print(result.stdout)
    return True


def main():
    """Run the complete workflow."""
    warnings.warn(
        "This script is deprecated. Use 'python main.py workflow' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    
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
            sys.path.insert(0, str(PROJECT_ROOT))
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
    
    main_script = PROJECT_ROOT / "main.py"
    
    # Run the complete workflow via main.py
    cmd = [sys.executable, str(main_script), "workflow", "--api-key", args.api_key]
    if args.skip_split:
        cmd.append("--skip-split")
    if args.skip_train:
        cmd.append("--skip-train")
    
    if not run_command(cmd, 'Running complete workflow'):
        print("Workflow failed")
        sys.exit(1)
    
    print("\nWorkflow completed successfully!")


if __name__ == "__main__":
    main()
