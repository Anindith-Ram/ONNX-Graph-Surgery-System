#!/usr/bin/env python3
"""
Train/Test split for ONNX model compilation rules.
Splits 22 models into 17 training (80%) and 5 test (20%) models.
"""

import os
import json
import random
from pathlib import Path
from typing import List, Tuple


def get_all_models(map_dataset_dir: str) -> List[str]:
    """Get all model names from map_dataset directory."""
    models = []
    for filename in os.listdir(map_dataset_dir):
        if filename.endswith('.txt'):
            model_name = filename.replace('.txt', '')
            models.append(model_name)
    return sorted(models)


def create_train_test_split(
    models: List[str],
    train_ratio: float = 0.8,
    seed: int = 42,
    output_file: str = None
) -> Tuple[List[str], List[str]]:
    """
    Create train/test split.
    
    Args:
        models: List of model names
        train_ratio: Ratio for training set (default 0.8)
        seed: Random seed for reproducibility
        output_file: Optional file to save split
    
    Returns:
        (train_models, test_models) tuple
    """
    random.seed(seed)
    
    # Shuffle models
    shuffled = models.copy()
    random.shuffle(shuffled)
    
    # Calculate split point
    n_train = int(len(models) * train_ratio)
    
    train_models = sorted(shuffled[:n_train])
    test_models = sorted(shuffled[n_train:])
    
    split_info = {
        'train_models': train_models,
        'test_models': test_models,
        'train_count': len(train_models),
        'test_count': len(test_models),
        'train_ratio': len(train_models) / len(models),
        'test_ratio': len(test_models) / len(models),
        'seed': seed
    }
    
    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(split_info, f, indent=2)
        print(f"Train/test split saved to {output_file}")
    
    return train_models, test_models


def load_train_test_split(split_file: str) -> Tuple[List[str], List[str]]:
    """Load train/test split from file."""
    with open(split_file, 'r') as f:
        split_info = json.load(f)
    
    return split_info['train_models'], split_info['test_models']


def main():
    """Create and save train/test split."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create train/test split for models')
    parser.add_argument('--map-dataset-dir', default='map_dataset', help='Directory with model maps')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Training set ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', default='rag_data/train_test_split.json', help='Output file')
    
    args = parser.parse_args()
    
    # Get all models
    models = get_all_models(args.map_dataset_dir)
    print(f"Found {len(models)} models")
    
    # Create split
    train_models, test_models = create_train_test_split(
        models,
        train_ratio=args.train_ratio,
        seed=args.seed,
        output_file=args.output
    )
    
    print(f"\nTrain set ({len(train_models)} models):")
    for model in train_models:
        print(f"  - {model}")
    
    print(f"\nTest set ({len(test_models)} models):")
    for model in test_models:
        print(f"  - {model}")
    
    print(f"\nSplit saved to {args.output}")


if __name__ == "__main__":
    main()

