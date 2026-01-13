#!/usr/bin/env python3
"""
Complete inference pipeline: Generate rules → Apply rules → Compare with ground truth.
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional, Any
import onnx
import numpy as np

from rag_pipeline import RAGPipeline
from difference_extractor import extract_differences
from rule_parser import RuleParser
from rule_applicator import RuleApplicator
from model_comparator import ModelComparator


class InferencePipeline:
    """Complete pipeline for inference on unseen models."""
    
    def __init__(self, rag_pipeline: RAGPipeline, gemini_api_key: Optional[str] = None):
        """
        Initialize inference pipeline.
        
        Args:
            rag_pipeline: RAG pipeline for rule generation
            gemini_api_key: API key for Gemini-based model modification
        """
        self.rag_pipeline = rag_pipeline
        self.rule_parser = RuleParser()
        self.rule_applicator = RuleApplicator(gemini_api_key)
        self.comparator = ModelComparator()
    
    def process_model(
        self,
        original_model_path: str,
        ground_truth_model_path: Optional[str] = None,
        test_inputs: Optional[Dict[str, np.ndarray]] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete inference pipeline for a single model.
        
        Args:
            original_model_path: Path to original (non-compilable) ONNX model
            ground_truth_model_path: Optional path to ground truth modified model
            test_inputs: Optional test inputs for output comparison
            output_dir: Optional directory to save results
        
        Returns:
            Dictionary with complete results
        """
        results = {
            'model_name': Path(original_model_path).stem,
            'original_model_path': original_model_path,
            'ground_truth_path': ground_truth_model_path
        }
        
        # Step 1: Load original model
        print(f"\n{'='*80}")
        print(f"Processing: {results['model_name']}")
        print(f"{'='*80}")
        
        original_model = onnx.load(original_model_path)
        results['original_node_count'] = len(original_model.graph.node)
        
        # Step 2: Extract differences (for feature extraction)
        # We need the map file for this - generate it if needed
        map_file = self._get_or_create_map_file(original_model_path)
        if not map_file:
            results['error'] = "Could not create model map file"
            return results
        
        model_diff = extract_differences(map_file)
        
        # Step 3: Generate rules using RAG
        print("\n--- Step 1: Generating Rules ---")
        rag_result = self.rag_pipeline.generate_rules(model_diff)
        results['generated_rules'] = rag_result['generated_rules']
        results['rules_count'] = len(rag_result['generated_rules'])
        print(f"Generated {results['rules_count']} rules")
        
        # Step 4: Apply rules using Gemini (RAG application)
        print("\n--- Step 2: Applying Rules with Gemini (RAG Graph Surgery) ---")
        pipeline_modified_model = self.rule_applicator.apply_rules(
            original_model,
            rag_result['generated_rules'],  # Pass rules directly to Gemini
            retrieved_examples=rag_result.get('retrieved_context', '')
        )
        results['rule_application_summary'] = self.rule_applicator.get_summary()
        results['pipeline_node_count'] = len(pipeline_modified_model.graph.node)
        print(f"Applied {results['rule_application_summary']['applied']} rules")
        
        # Step 5: Compare with ground truth (if available)
        if ground_truth_model_path and os.path.exists(ground_truth_model_path):
            print("\n--- Step 3: Comparing with Ground Truth ---")
            ground_truth_model = onnx.load(ground_truth_model_path)
            results['ground_truth_node_count'] = len(ground_truth_model.graph.node)
            
            # Structural comparison
            comparison = self.comparator.compare_models(
                pipeline_modified_model,
                ground_truth_model,
                original_model
            )
            results['structural_comparison'] = comparison
            
            # Output comparison (if test inputs provided)
            if test_inputs:
                output_comparison = self.comparator.compare_outputs(
                    pipeline_modified_model,
                    ground_truth_model,
                    test_inputs
                )
                results['output_comparison'] = output_comparison
                print(f"Output similarity: {output_comparison.get('all_outputs_match', False)}")
            
            print(f"Overall similarity: {comparison['overall_similarity']:.3f}")
        
        # Step 7: Save results
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save modified model
            pipeline_output_path = os.path.join(output_dir, f"{results['model_name']}_pipeline_modified.onnx")
            onnx.save(pipeline_modified_model, pipeline_output_path)
            results['pipeline_modified_path'] = pipeline_output_path
            
            # Save results JSON
            results_path = os.path.join(output_dir, f"{results['model_name']}_results.json")
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            results['results_path'] = results_path
        
        return results
    
    def _get_or_create_map_file(self, model_path: str) -> Optional[str]:
        """Get or create map file for a model."""
        # Check if map file exists
        model_name = Path(model_path).stem
        map_file = f"map_dataset/{model_name}.txt"
        
        if os.path.exists(map_file):
            return map_file
        
        # Try to generate it
        try:
            from print_onnx_graph import print_onnx_graph
            import subprocess
            
            # Generate map
            script_path = os.path.join(os.path.dirname(__file__), "print_onnx_graph.py")
            result = subprocess.run(
                ["python3", script_path, model_path],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Save to map file
                os.makedirs("map_dataset", exist_ok=True)
                with open(map_file, 'w') as f:
                    f.write(result.stdout)
                return map_file
        except Exception as e:
            print(f"Warning: Could not generate map file: {e}")
        
        return None
    
    def process_test_set(
        self,
        test_models: list,
        dataset_dir: str = "dataset",
        output_dir: str = "inference_results"
    ) -> Dict[str, Any]:
        """
        Process all test set models.
        
        Args:
            test_models: List of model names to process
            dataset_dir: Directory containing model datasets
            output_dir: Directory to save results
        
        Returns:
            Dictionary with results for all models
        """
        all_results = {}
        
        for model_name in test_models:
            # Find original and modified models
            model_dir = os.path.join(dataset_dir, model_name)
            original_path = None
            ground_truth_path = None
            
            # Try different directory structures
            for orig_dir in ["original", "Original"]:
                orig_full = os.path.join(model_dir, orig_dir)
                if os.path.exists(orig_full):
                    for file in os.listdir(orig_full):
                        if file.endswith('.onnx'):
                            original_path = os.path.join(orig_full, file)
                            break
                    if original_path:
                        break
            
            for mod_dir in ["modified", "Modified"]:
                mod_full = os.path.join(model_dir, mod_dir)
                if os.path.exists(mod_full):
                    for file in os.listdir(mod_full):
                        if file.endswith('.onnx'):
                            ground_truth_path = os.path.join(mod_full, file)
                            break
                    if ground_truth_path:
                        break
            
            if not original_path:
                print(f"Warning: Could not find original model for {model_name}")
                continue
            
            # Process model
            try:
                result = self.process_model(
                    original_path,
                    ground_truth_path,
                    output_dir=output_dir
                )
                all_results[model_name] = result
            except Exception as e:
                print(f"Error processing {model_name}: {e}")
                all_results[model_name] = {'error': str(e)}
        
        # Save consolidated results
        consolidated_path = os.path.join(output_dir, "all_results.json")
        with open(consolidated_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        return all_results


def main():
    """Main function for inference pipeline."""
    import argparse
    from train_test_split import load_train_test_split
    
    parser = argparse.ArgumentParser(description='Inference pipeline for unseen models')
    parser.add_argument('--api-key', help='Gemini API key', default=None)
    parser.add_argument('--model', help='Single model to process')
    parser.add_argument('--test-set', action='store_true', help='Process all test set models')
    parser.add_argument('--split-file', default='rag_data/train_test_split.json')
    parser.add_argument('--output-dir', default='inference_results')
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key
    if not api_key:
        api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        try:
            from config import GEMINI_API_KEY
            api_key = GEMINI_API_KEY
        except:
            pass
    
    if not api_key:
        print("Error: API key required")
        return
    
    # Initialize RAG pipeline
    print("Initializing RAG pipeline...")
    from rag_pipeline import RAGPipeline
    rag_pipeline = RAGPipeline(api_key)
    
    # Load knowledge base
    kb_file = Path("rag_data/knowledge_base.pkl")
    if kb_file.exists():
        import pickle
        with open(kb_file, 'rb') as f:
            rag_pipeline.vector_store = pickle.load(f)
        print("Knowledge base loaded")
    else:
        print("Error: Knowledge base not found. Run training first.")
        return
    
    # Initialize inference pipeline with Gemini API key
    inference_pipeline = InferencePipeline(rag_pipeline, gemini_api_key=api_key)
    
    # Process models
    if args.test_set:
        _, test_models = load_train_test_split(args.split_file)
        print(f"Processing {len(test_models)} test models...")
        results = inference_pipeline.process_test_set(test_models, output_dir=args.output_dir)
        print(f"\nProcessed {len(results)} models")
        print(f"Results saved to {args.output_dir}/all_results.json")
    elif args.model:
        # Process single model
        result = inference_pipeline.process_model(args.model, output_dir=args.output_dir)
        print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()

