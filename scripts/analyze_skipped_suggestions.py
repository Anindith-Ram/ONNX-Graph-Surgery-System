#!/usr/bin/env python3
"""
Analyze which suggestions are being skipped and why.

This script helps identify:
1. Which operation types are being skipped
2. Why handlers aren't matching
3. What needs to be implemented
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List


def analyze_evaluation_report(report_path: str = "evaluation_output/evaluation_report.json"):
    """Analyze the evaluation report to identify skipped suggestions."""
    
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    print("=" * 80)
    print("SKIPPED SUGGESTIONS ANALYSIS")
    print("=" * 80)
    
    summary = report['summary']
    print(f"\nOverall Statistics:")
    print(f"  Total Suggestions: {summary['total_suggestions']}")
    print(f"  Attempted: {summary['total_attempted_suggestions']}")
    print(f"  Skipped: {summary['total_skipped_suggestions']} ({summary['total_skipped_suggestions']/summary['total_attempted_suggestions']*100:.1f}%)")
    print(f"  Transformed: {summary['total_transformed_suggestions']} ({summary['total_transformed_suggestions']/summary['total_attempted_suggestions']*100:.1f}%)")
    
    # Analyze per-model
    print("\n" + "=" * 80)
    print("PER-MODEL ANALYSIS")
    print("=" * 80)
    
    skipped_by_op_type = Counter()
    skipped_by_category = Counter()
    skipped_suggestions = []
    
    for model_result in report['per_model_results']:
        model_name = model_result['model_name']
        skipped = model_result.get('skipped_suggestions_count', 0)
        attempted = model_result.get('attempted_suggestions_count', 0)
        transformed = model_result.get('transformed_suggestions_count', 0)
        
        print(f"\n{model_name}:")
        print(f"  Attempted: {attempted}")
        print(f"  Skipped: {skipped} ({skipped/attempted*100:.1f}%)")
        print(f"  Transformed: {transformed} ({transformed/attempted*100:.1f}%)")
        
        # Get transformation accuracy details
        if model_result.get('structural_comparison') and 'transformation_accuracy' in model_result['structural_comparison']:
            ta = model_result['structural_comparison']['transformation_accuracy']
            print(f"  Transformation Score: {ta.get('transformation_score', 0):.3f}")
            print(f"  Critical Areas Match: {ta.get('critical_areas_match', 0):.3f}")
            
            # Show mismatches
            removal_mismatches_gt = ta.get('removal_mismatches_gt', [])
            addition_mismatches_gt = ta.get('addition_mismatches_gt', [])
            
            if removal_mismatches_gt:
                print(f"  ❌ GT Removed (we didn't): {', '.join(removal_mismatches_gt)}")
            if addition_mismatches_gt:
                print(f"  ❌ GT Added (we didn't): {', '.join(addition_mismatches_gt)}")
    
    # Analyze transformation accuracy gaps
    print("\n" + "=" * 80)
    print("TRANSFORMATION ACCURACY GAPS")
    print("=" * 80)
    
    all_removal_mismatches_gt = Counter()
    all_addition_mismatches_gt = Counter()
    
    for model_result in report['per_model_results']:
        if model_result.get('structural_comparison') and 'transformation_accuracy' in model_result['structural_comparison']:
            ta = model_result['structural_comparison']['transformation_accuracy']
            for op in ta.get('removal_mismatches_gt', []):
                all_removal_mismatches_gt[op] += 1
            for op in ta.get('addition_mismatches_gt', []):
                all_addition_mismatches_gt[op] += 1
    
    if all_removal_mismatches_gt:
        print("\nOperations GT Removed (we didn't) - Priority Order:")
        for op, count in all_removal_mismatches_gt.most_common():
            print(f"  {op}: {count} model(s)")
    
    if all_addition_mismatches_gt:
        print("\nOperations GT Added (we didn't) - Priority Order:")
        for op, count in all_addition_mismatches_gt.most_common():
            print(f"  {op}: {count} model(s)")
    
    # Load actual suggestions to see what's being skipped
    print("\n" + "=" * 80)
    print("SUGGESTION TYPE ANALYSIS")
    print("=" * 80)
    
    suggestion_types = Counter()
    suggestion_categories = Counter()
    
    for model_result in report['per_model_results']:
        model_name = model_result['model_name']
        analysis_path = f"evaluation_output/{model_name}_test_analysis.json"
        
        if Path(analysis_path).exists():
            with open(analysis_path, 'r') as f:
                analysis = json.load(f)
            
            suggestions = analysis.get('suggestions', [])
            for sug in suggestions:
                op_type = sug.get('op_type', 'Unknown')
                category = sug.get('category', 'Unknown')
                suggestion_types[op_type] += 1
                suggestion_categories[category] += 1
    
    print("\nMost Common Operation Types in Suggestions:")
    for op_type, count in suggestion_types.most_common(20):
        print(f"  {op_type}: {count}")
    
    print("\nMost Common Categories in Suggestions:")
    for category, count in suggestion_categories.most_common(10):
        print(f"  {category}: {count}")
    
    return {
        'skipped_by_op_type': dict(skipped_by_op_type),
        'skipped_by_category': dict(skipped_by_category),
        'removal_mismatches_gt': dict(all_removal_mismatches_gt),
        'addition_mismatches_gt': dict(all_addition_mismatches_gt),
        'suggestion_types': dict(suggestion_types)
    }


if __name__ == '__main__':
    analyze_evaluation_report()

