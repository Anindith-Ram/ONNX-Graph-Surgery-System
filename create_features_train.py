#!/usr/bin/env python3
"""
Create features_train.json by filtering features.json to training models only.
"""

import json
from pathlib import Path
from train_test_split import load_train_test_split

def create_features_train():
    """Filter features.json to create features_train.json with training models only."""
    
    # Load train/test split
    split_file = Path("rag_data/train_test_split.json")
    train_models, test_models = load_train_test_split(str(split_file))
    
    print(f"Training models: {len(train_models)}")
    print(f"Test models: {len(test_models)}")
    
    # Load all features
    features_file = Path("rag_data/features.json")
    if not features_file.exists():
        print(f"Error: {features_file} not found")
        return
    
    with open(features_file, 'r') as f:
        all_features = json.load(f)
    
    print(f"\nLoaded {len(all_features)} models from features.json")
    
    # Filter to training models
    train_features = [f for f in all_features if f['model_name'] in train_models]
    
    print(f"Filtered to {len(train_features)} training models")
    
    # Save to features_train.json
    output_file = Path("rag_data/features_train.json")
    with open(output_file, 'w') as f:
        json.dump(train_features, f, indent=2)
    
    print(f"\n✓ Saved {len(train_features)} training models to {output_file}")
    
    # Verify
    train_model_names = {f['model_name'] for f in train_features}
    missing = set(train_models) - train_model_names
    
    if missing:
        print(f"\n⚠️  Warning: Missing {len(missing)} training models in features.json:")
        for model in missing:
            print(f"  - {model}")
    else:
        print(f"\n✅ All {len(train_models)} training models found in features.json")

if __name__ == "__main__":
    create_features_train()

