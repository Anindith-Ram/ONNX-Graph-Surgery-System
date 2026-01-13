#!/usr/bin/env python3
"""
Generate ONNX graph maps for all models in the dataset.
For each model, creates a text file with both original and modified maps.
"""

import os
import subprocess
from pathlib import Path

def find_onnx_files(model_dir):
    """Find original and modified ONNX files in a model directory."""
    original_path = None
    modified_path = None
    
    # Try lowercase first
    orig_dir = os.path.join(model_dir, "original")
    mod_dir = os.path.join(model_dir, "modified")
    
    # Try capitalized if lowercase doesn't exist
    if not os.path.exists(orig_dir):
        orig_dir = os.path.join(model_dir, "Original")
    if not os.path.exists(mod_dir):
        mod_dir = os.path.join(model_dir, "Modified")
    
    # Find ONNX files
    if os.path.exists(orig_dir):
        for file in os.listdir(orig_dir):
            if file.endswith('.onnx'):
                original_path = os.path.join(orig_dir, file)
                break
    
    if os.path.exists(mod_dir):
        for file in os.listdir(mod_dir):
            if file.endswith('.onnx'):
                modified_path = os.path.join(mod_dir, file)
                break
    
    return original_path, modified_path

def run_print_onnx_graph(onnx_path):
    """Run the print_onnx_graph script and capture output."""
    script_path = os.path.join(os.path.dirname(__file__), "print_onnx_graph.py")
    # Use venv Python if available, otherwise system Python
    venv_python = os.path.join(os.path.dirname(__file__), "venv", "bin", "python3")
    python_cmd = venv_python if os.path.exists(venv_python) else "python3"
    try:
        result = subprocess.run(
            [python_cmd, script_path, onnx_path],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"[ERROR] Failed to process {onnx_path}:\n{e.stderr}\n"

def main():
    dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")
    map_dataset_dir = os.path.join(os.path.dirname(__file__), "map_dataset")
    
    # Create map_dataset directory if it doesn't exist
    os.makedirs(map_dataset_dir, exist_ok=True)
    
    # Get all model directories
    model_dirs = [d for d in os.listdir(dataset_dir) 
                  if os.path.isdir(os.path.join(dataset_dir, d))]
    
    print(f"Found {len(model_dirs)} model directories")
    
    for model_name in sorted(model_dirs):
        model_dir = os.path.join(dataset_dir, model_name)
        original_path, modified_path = find_onnx_files(model_dir)
        
        if not original_path:
            print(f"[WARN] No original ONNX file found for {model_name}")
            continue
        if not modified_path:
            print(f"[WARN] No modified ONNX file found for {model_name}")
            continue
        
        print(f"Processing {model_name}...")
        
        # Generate maps
        print("  Generating original map...")
        original_map = run_print_onnx_graph(original_path)
        
        print("  Generating modified map...")
        modified_map = run_print_onnx_graph(modified_path)
        
        # Combine and save
        output_file = os.path.join(map_dataset_dir, f"{model_name}.txt")
        with open(output_file, 'w') as f:
            f.write(f"# Model: {model_name}\n")
            f.write(f"# Original ONNX: {os.path.basename(original_path)}\n")
            f.write("=" * 80 + "\n")
            f.write("ORIGINAL MODEL MAP\n")
            f.write("=" * 80 + "\n\n")
            f.write(original_map)
            f.write("\n\n")
            f.write("=" * 80 + "\n")
            f.write("MODIFIED MODEL MAP\n")
            f.write("=" * 80 + "\n\n")
            f.write(modified_map)
        
        print(f"  Saved to {output_file}")
    
    print("\nDone! All maps generated in map_dataset/")

if __name__ == "__main__":
    main()

