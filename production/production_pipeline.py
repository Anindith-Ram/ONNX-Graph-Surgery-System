#!/usr/bin/env python3
"""
Production Pipeline for ONNX Compilation Advisory.

This is an ADVISORY system - it analyzes models and suggests changes,
but does NOT modify models directly. Engineers review suggestions
and implement changes manually.

Pipeline:
1. ANALYZE: Deep ONNX analysis to identify issues
2. SUGGEST: Generate prioritized suggestions with confidence scores
3. REPORT: Output actionable reports for engineers

Output:
- Detailed suggestions with implementation steps
- Confidence scores for each recommendation
- Priority-based ordering
- Multiple output formats (Markdown, JSON, HTML)
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Force unbuffered output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core_analysis.onnx_analyzer import ONNXAnalyzer, ModelAnalysis
from suggestion_pipeline.suggestion_generator import SuggestionGenerator, SuggestionReport
from suggestion_pipeline.rag_suggestion_generator import RAGSuggestionGenerator
from production.report_generator import ReportGenerator


@dataclass
class AdvisoryConfig:
    """Configuration for advisory pipeline."""
    # Output settings
    output_dir: str = "analysis_output"
    report_format: str = "markdown"  # markdown, json, html, text
    
    # Analysis settings
    include_optimizations: bool = True
    include_info: bool = False
    min_confidence: float = 0.0  # Filter suggestions below this confidence
    
    # RAG settings
    use_rag: bool = True  # Use RAG-enhanced suggestions
    kb_path: str = "knowledge_base.json"  # Path to knowledge base
    
    # API settings (optional, for enhanced suggestions)
    use_gemini_enhancement: bool = False


@dataclass
class AdvisoryResult:
    """Result from advisory pipeline."""
    model_name: str
    model_path: str
    success: bool
    
    # Analysis results
    compilation_status: str
    total_issues: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    
    # Output paths
    report_path: Optional[str] = None
    json_path: Optional[str] = None
    
    # Timing
    analysis_time: float = 0.0
    
    # Errors
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'model_name': self.model_name,
            'model_path': self.model_path,
            'success': self.success,
            'compilation_status': self.compilation_status,
            'total_issues': self.total_issues,
            'issue_counts': {
                'critical': self.critical_count,
                'high': self.high_count,
                'medium': self.medium_count,
                'low': self.low_count
            },
            'report_path': self.report_path,
            'json_path': self.json_path,
            'analysis_time': self.analysis_time,
            'errors': self.errors
        }


class AdvisoryPipeline:
    """
    Advisory pipeline for ONNX model compilation analysis.
    
    This system ANALYZES and SUGGESTS - it does NOT modify models.
    Engineers review the generated reports and implement changes.
    
    Usage:
        pipeline = AdvisoryPipeline()
        result = pipeline.analyze_model("model.onnx")
        # Result includes report_path pointing to generated suggestions
    """
    
    def __init__(
        self,
        config: Optional[AdvisoryConfig] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize advisory pipeline.
        
        Args:
            config: Pipeline configuration
            api_key: Optional Gemini API key for enhanced analysis
        """
        self.config = config or AdvisoryConfig()
        self.api_key = api_key
        
        # Initialize components
        self.analyzer = ONNXAnalyzer()
        
        # Use RAG-enhanced generator if enabled
        if self.config.use_rag:
            self.suggestion_generator = RAGSuggestionGenerator(
                kb_path=self.config.kb_path,
                api_key=api_key,
                use_rag=True
            )
        else:
            self.suggestion_generator = SuggestionGenerator(api_key=api_key)
        
        self.report_generator = ReportGenerator()
        
        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def analyze_model(
        self,
        model_path: str,
        output_name: Optional[str] = None
    ) -> AdvisoryResult:
        """
        Analyze a single model and generate suggestions.
        
        Args:
            model_path: Path to ONNX model
            output_name: Optional custom name for output files
            
        Returns:
            AdvisoryResult with paths to generated reports
        """
        start_time = time.time()
        
        # Determine output name
        model_name = Path(model_path).stem
        output_name = output_name or model_name
        
        result = AdvisoryResult(
            model_name=model_name,
            model_path=model_path,
            success=False,
            compilation_status="unknown",
            total_issues=0,
            critical_count=0,
            high_count=0,
            medium_count=0,
            low_count=0
        )
        
        print(f"\n{'='*70}")
        print(f"ANALYZING: {model_name}")
        print(f"{'='*70}")
        
        try:
            # Step 1: Generate suggestions
            print("\n[1/2] Analyzing model and generating suggestions...")
            suggestion_report = self.suggestion_generator.analyze_and_suggest(model_path)
            
            # Update result
            result.compilation_status = suggestion_report.compilation_status
            result.total_issues = suggestion_report.total_issues
            result.critical_count = suggestion_report.critical_count
            result.high_count = suggestion_report.high_count
            result.medium_count = suggestion_report.medium_count
            result.low_count = suggestion_report.low_count
            
            print(f"  Status: {suggestion_report.compilation_status.upper()}")
            print(f"  Issues: {suggestion_report.total_issues} total")
            print(f"    - Critical: {suggestion_report.critical_count}")
            print(f"    - High: {suggestion_report.high_count}")
            print(f"    - Medium: {suggestion_report.medium_count}")
            print(f"    - Low: {suggestion_report.low_count}")
            
            # Step 2: Generate reports
            print("\n[2/2] Generating reports...")
            
            # Generate main report (user-specified format)
            report_ext = {
                'markdown': '.md',
                'md': '.md',
                'json': '.json',
                'html': '.html',
                'text': '.txt',
                'txt': '.txt'
            }.get(self.config.report_format, '.md')
            
            # Add timestamp to filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Generate main report with timestamp
            report_path = Path(self.config.output_dir) / f"{output_name}_report_{timestamp}{report_ext}"
            self.report_generator.generate(
                suggestion_report,
                format=self.config.report_format,
                output_path=str(report_path)
            )
            result.report_path = str(report_path)
            print(f"  Report: {report_path}")
            
            # Also save without timestamp for compatibility
            report_path_compat = Path(self.config.output_dir) / f"{output_name}_report{report_ext}"
            self.report_generator.generate(
                suggestion_report,
                format=self.config.report_format,
                output_path=str(report_path_compat)
            )
            
            # Always save JSON for programmatic access (with timestamp)
            json_path = Path(self.config.output_dir) / f"{output_name}_analysis_{timestamp}.json"
            suggestion_dict = suggestion_report.to_dict()
            suggestion_dict['analysis_timestamp'] = datetime.now().isoformat()
            suggestion_dict['analysis_timestamp_readable'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(json_path, 'w') as f:
                json.dump(suggestion_dict, f, indent=2)
            result.json_path = str(json_path)
            print(f"  JSON: {json_path}")
            
            # Also save without timestamp for compatibility
            json_path_compat = Path(self.config.output_dir) / f"{output_name}_analysis.json"
            with open(json_path_compat, 'w') as f:
                json.dump(suggestion_dict, f, indent=2)
            
            result.success = True
            
        except Exception as e:
            result.errors.append(str(e))
            print(f"\n  ERROR: {e}")
        
        result.analysis_time = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"COMPLETE: {model_name} ({result.analysis_time:.1f}s)")
        print(f"{'='*70}")
        
        return result
    
    def analyze_batch(
        self,
        model_paths: List[str],
        save_summary: bool = True
    ) -> List[AdvisoryResult]:
        """
        Analyze multiple models.
        
        Args:
            model_paths: List of paths to ONNX models
            save_summary: Whether to save a summary report
            
        Returns:
            List of AdvisoryResults
        """
        results = []
        
        print(f"\n{'#'*70}")
        print(f"BATCH ANALYSIS: {len(model_paths)} models")
        print(f"{'#'*70}")
        
        for i, model_path in enumerate(model_paths):
            print(f"\n[{i+1}/{len(model_paths)}] Processing {Path(model_path).name}...")
            result = self.analyze_model(model_path)
            results.append(result)
        
        # Generate summary
        if save_summary:
            summary_path = Path(self.config.output_dir) / "batch_summary.json"
            summary = {
                'total_models': len(results),
                'successful': sum(1 for r in results if r.success),
                'failed': sum(1 for r in results if not r.success),
                'by_status': {},
                'total_issues': sum(r.total_issues for r in results),
                'models': [r.to_dict() for r in results]
            }
            
            # Count by status
            for result in results:
                status = result.compilation_status
                summary['by_status'][status] = summary['by_status'].get(status, 0) + 1
            
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\nBatch summary saved to: {summary_path}")
        
        # Print summary
        print(f"\n{'#'*70}")
        print("BATCH SUMMARY")
        print(f"{'#'*70}")
        print(f"  Total models: {len(results)}")
        print(f"  Successful: {sum(1 for r in results if r.success)}")
        print(f"  Failed: {sum(1 for r in results if not r.success)}")
        print(f"  Total issues found: {sum(r.total_issues for r in results)}")
        
        return results
    
    def analyze_directory(
        self,
        directory: str,
        recursive: bool = False
    ) -> List[AdvisoryResult]:
        """
        Analyze all ONNX models in a directory.
        
        Args:
            directory: Path to directory containing models
            recursive: Whether to search subdirectories
            
        Returns:
            List of AdvisoryResults
        """
        dir_path = Path(directory)
        
        if recursive:
            model_paths = list(dir_path.rglob("*.onnx"))
        else:
            model_paths = list(dir_path.glob("*.onnx"))
        
        if not model_paths:
            print(f"No ONNX files found in {directory}")
            return []
        
        return self.analyze_batch([str(p) for p in model_paths])


def analyze_model(
    model_path: str,
    output_dir: str = "analysis_output",
    format: str = "markdown"
) -> AdvisoryResult:
    """
    Convenience function to analyze a single model.
    
    Args:
        model_path: Path to ONNX model
        output_dir: Directory for output reports
        format: Report format (markdown, json, html, text)
        
    Returns:
        AdvisoryResult
    """
    config = AdvisoryConfig(output_dir=output_dir, report_format=format)
    pipeline = AdvisoryPipeline(config=config)
    return pipeline.analyze_model(model_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='ONNX Compilation Advisory - Analyze models and suggest fixes'
    )
    parser.add_argument('model', help='Path to ONNX model or directory')
    parser.add_argument('--output', '-o', default='analysis_output',
                       help='Output directory for reports')
    parser.add_argument('--format', '-f', default='markdown',
                       choices=['markdown', 'json', 'html', 'text'],
                       help='Report format')
    parser.add_argument('--batch', '-b', action='store_true',
                       help='Process all .onnx files in directory')
    parser.add_argument('--recursive', '-r', action='store_true',
                       help='Search subdirectories (with --batch)')
    
    args = parser.parse_args()
    
    config = AdvisoryConfig(
        output_dir=args.output,
        report_format=args.format
    )
    
    pipeline = AdvisoryPipeline(config=config)
    
    if args.batch or Path(args.model).is_dir():
        pipeline.analyze_directory(args.model, recursive=args.recursive)
    else:
        pipeline.analyze_model(args.model)
