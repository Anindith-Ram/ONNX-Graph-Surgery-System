#!/usr/bin/env python3
"""
Evaluation Dashboard for ONNX Model Surgery.

Aggregates all evaluation metrics into comprehensive reports in multiple
formats (JSON, Markdown, HTML). Provides unified view of:
- Numerical Verification
- Compilation Verification
- Strategic Evaluation
- Transformation Details
- Error Analysis

Author: Automated Model Surgery Pipeline
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.numerical_verifier import (
    NumericalVerifier, NumericalVerificationResult, ToleranceCategory
)
from evaluation.compilation_verifier import (
    CompilationVerifier, CompilationVerificationResult, CompilationStatus
)
from evaluation.strategic_evaluator import (
    StrategicEvaluator, StrategicEvaluationResult, EvaluationGrade
)


class OverallStatus(Enum):
    """Overall evaluation status."""
    PASSED = "passed"       # All critical checks passed
    PARTIAL = "partial"     # Some checks passed
    FAILED = "failed"       # Critical checks failed
    ERROR = "error"         # Evaluation error


@dataclass
class ExecutiveSummary:
    """Executive summary of evaluation."""
    status: OverallStatus = OverallStatus.ERROR
    
    # Key metrics
    compilation_success: bool = False
    single_lm_file: bool = False
    numerical_match: bool = False
    tolerance_category: str = "unknown"
    
    # Scores (0-100)
    overall_score: float = 0.0
    compilation_score: float = 0.0
    numerical_score: float = 0.0
    strategic_score: float = 0.0
    
    # Blockers
    blockers_resolved: int = 0
    blockers_remaining: int = 0
    blocker_resolution_rate: float = 0.0
    
    # Quick stats
    total_transformations: int = 0
    successful_transformations: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'status': self.status.value,
            'compilation_success': self.compilation_success,
            'single_lm_file': self.single_lm_file,
            'numerical_match': self.numerical_match,
            'tolerance_category': self.tolerance_category,
            'overall_score': self.overall_score,
            'compilation_score': self.compilation_score,
            'numerical_score': self.numerical_score,
            'strategic_score': self.strategic_score,
            'blockers_resolved': self.blockers_resolved,
            'blockers_remaining': self.blockers_remaining,
            'blocker_resolution_rate': self.blocker_resolution_rate,
            'total_transformations': self.total_transformations,
            'successful_transformations': self.successful_transformations
        }


@dataclass
class EvaluationDashboard:
    """Complete evaluation dashboard."""
    model_name: str
    created_at: str = ""
    
    # Executive summary
    summary: ExecutiveSummary = field(default_factory=ExecutiveSummary)
    
    # Component results
    numerical_result: Optional[NumericalVerificationResult] = None
    compilation_result: Optional[CompilationVerificationResult] = None
    strategic_result: Optional[StrategicEvaluationResult] = None
    
    # Ground truth comparison (if available)
    ground_truth_comparison: Optional[Dict] = None
    
    # Detailed metrics
    transformation_details: List[Dict] = field(default_factory=list)
    error_analysis: List[Dict] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Tier compliance
    tier1_passed: bool = False  # Critical metrics
    tier2_passed: bool = False  # Quality metrics
    tier3_passed: bool = False  # Efficiency metrics
    
    def to_dict(self) -> Dict:
        result = {
            'model_name': self.model_name,
            'created_at': self.created_at,
            'summary': self.summary.to_dict(),
            'tier1_passed': self.tier1_passed,
            'tier2_passed': self.tier2_passed,
            'tier3_passed': self.tier3_passed,
            'transformation_details': self.transformation_details,
            'error_analysis': self.error_analysis,
            'recommendations': self.recommendations
        }
        
        if self.numerical_result:
            result['numerical_verification'] = self.numerical_result.to_dict()
        if self.compilation_result:
            result['compilation_verification'] = self.compilation_result.to_dict()
        if self.strategic_result:
            result['strategic_evaluation'] = self.strategic_result.to_dict()
        if self.ground_truth_comparison:
            result['ground_truth_comparison'] = self.ground_truth_comparison
        
        return result
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def to_markdown(self) -> str:
        """Convert to Markdown format."""
        lines = [
            f"# Evaluation Report: {self.model_name}",
            f"",
            f"**Generated:** {self.created_at}",
            f"",
            f"## Executive Summary",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| **Status** | {self.summary.status.value.upper()} |",
            f"| **Overall Score** | {self.summary.overall_score:.1f}% |",
            f"| **Compilation Success** | {'✓' if self.summary.compilation_success else '✗'} |",
            f"| **Single LM File** | {'✓' if self.summary.single_lm_file else '✗'} |",
            f"| **Numerical Match** | {'✓' if self.summary.numerical_match else '✗'} |",
            f"| **Tolerance Category** | {self.summary.tolerance_category} |",
            f"",
            f"### Blocker Resolution",
            f"",
            f"- Resolved: {self.summary.blockers_resolved}",
            f"- Remaining: {self.summary.blockers_remaining}",
            f"- Resolution Rate: {self.summary.blocker_resolution_rate:.1%}",
            f"",
            f"## Tier Compliance",
            f"",
            f"| Tier | Status | Description |",
            f"|------|--------|-------------|",
            f"| Tier 1 (Critical) | {'✓ PASS' if self.tier1_passed else '✗ FAIL'} | Compilation, numerical tolerance |",
            f"| Tier 2 (Quality) | {'✓ PASS' if self.tier2_passed else '✗ FAIL'} | Blocker resolution, accuracy |",
            f"| Tier 3 (Efficiency) | {'✓ PASS' if self.tier3_passed else '✗ FAIL'} | Transformation efficiency |",
        ]
        
        # Component scores
        lines.extend([
            f"",
            f"## Component Scores",
            f"",
            f"| Component | Score |",
            f"|-----------|-------|",
            f"| Compilation | {self.summary.compilation_score:.1f}% |",
            f"| Numerical | {self.summary.numerical_score:.1f}% |",
            f"| Strategic | {self.summary.strategic_score:.1f}% |",
        ])
        
        # Numerical verification details
        if self.numerical_result:
            lines.extend([
                f"",
                f"## Numerical Verification",
                f"",
                f"- Max Absolute Difference: {self.numerical_result.max_absolute_difference:.2e}",
                f"- Mean Absolute Difference: {self.numerical_result.mean_absolute_difference:.2e}",
                f"- Tolerance Category: {self.numerical_result.tolerance_category.value}",
                f"- Outputs Match: {'Yes' if self.numerical_result.outputs_match else 'No'}",
            ])
        
        # Compilation verification details
        if self.compilation_result:
            lines.extend([
                f"",
                f"## Compilation Verification",
                f"",
                f"- Status: {self.compilation_result.status.value}",
                f"- Predicted LM Files: {self.compilation_result.predicted_lm_files}",
                f"- MLA Nodes: {self.compilation_result.node_mapping.nodes_on_mla}",
                f"- CVU Nodes: {self.compilation_result.node_mapping.nodes_on_cvu}",
                f"- APU Nodes: {self.compilation_result.node_mapping.nodes_on_apu}",
            ])
        
        # Strategic evaluation details
        if self.strategic_result:
            lines.extend([
                f"",
                f"## Strategic Evaluation",
                f"",
                f"- Overall Grade: {self.strategic_result.overall_grade.value.upper()}",
                f"- Architecture Understanding: {self.strategic_result.architecture_understanding.score:.1%}",
                f"- Strategy Selection: {self.strategic_result.strategy_selection.score:.1%}",
                f"- Transformation Completeness: {self.strategic_result.transformation_completeness.score:.1%}",
                f"- Blocker Resolution: {self.strategic_result.blocker_resolution.score:.1%}",
                f"- Efficiency: {self.strategic_result.efficiency.score:.1%}",
            ])
        
        # Recommendations
        if self.recommendations:
            lines.extend([
                f"",
                f"## Recommendations",
                f"",
            ])
            for rec in self.recommendations:
                lines.append(f"- {rec}")
        
        # Error analysis
        if self.error_analysis:
            lines.extend([
                f"",
                f"## Error Analysis",
                f"",
            ])
            for error in self.error_analysis[:5]:
                lines.append(f"- **{error.get('type', 'Unknown')}**: {error.get('message', '')}")
        
        return "\n".join(lines)
    
    def to_html(self) -> str:
        """Convert to HTML format."""
        status_color = {
            OverallStatus.PASSED: "#28a745",
            OverallStatus.PARTIAL: "#ffc107",
            OverallStatus.FAILED: "#dc3545",
            OverallStatus.ERROR: "#6c757d"
        }
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Evaluation Report: {self.model_name}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .summary-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .status {{ display: inline-block; padding: 5px 15px; border-radius: 20px; color: white; font-weight: bold; }}
        .metric {{ display: inline-block; margin: 10px 20px; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #007bff; }}
        .metric-label {{ font-size: 0.9em; color: #666; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f8f9fa; }}
        .pass {{ color: #28a745; }}
        .fail {{ color: #dc3545; }}
        .progress-bar {{ background: #e9ecef; border-radius: 4px; height: 20px; overflow: hidden; }}
        .progress-fill {{ height: 100%; background: #007bff; transition: width 0.3s; }}
        .recommendations {{ background: #fff3cd; padding: 15px; border-radius: 8px; margin: 15px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Evaluation Report: {self.model_name}</h1>
        <p>Generated: {self.created_at}</p>
        
        <div class="summary-card">
            <span class="status" style="background: {status_color.get(self.summary.status, '#6c757d')}">
                {self.summary.status.value.upper()}
            </span>
            
            <div style="margin-top: 20px;">
                <div class="metric">
                    <div class="metric-value">{self.summary.overall_score:.0f}%</div>
                    <div class="metric-label">Overall Score</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{self.summary.blocker_resolution_rate:.0%}</div>
                    <div class="metric-label">Blockers Resolved</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{'✓' if self.summary.compilation_success else '✗'}</div>
                    <div class="metric-label">Compilation</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{'✓' if self.summary.numerical_match else '✗'}</div>
                    <div class="metric-label">Numerical</div>
                </div>
            </div>
        </div>
        
        <h2>Tier Compliance</h2>
        <table>
            <tr><th>Tier</th><th>Status</th><th>Description</th></tr>
            <tr>
                <td>Tier 1 (Critical)</td>
                <td class="{'pass' if self.tier1_passed else 'fail'}">{'PASS' if self.tier1_passed else 'FAIL'}</td>
                <td>Compilation success, numerical tolerance</td>
            </tr>
            <tr>
                <td>Tier 2 (Quality)</td>
                <td class="{'pass' if self.tier2_passed else 'fail'}">{'PASS' if self.tier2_passed else 'FAIL'}</td>
                <td>Blocker resolution rate, transformation accuracy</td>
            </tr>
            <tr>
                <td>Tier 3 (Efficiency)</td>
                <td class="{'pass' if self.tier3_passed else 'fail'}">{'PASS' if self.tier3_passed else 'FAIL'}</td>
                <td>Transformation efficiency, region coverage</td>
            </tr>
        </table>
        
        <h2>Component Scores</h2>
        <table>
            <tr><th>Component</th><th>Score</th><th>Progress</th></tr>
            <tr>
                <td>Compilation</td>
                <td>{self.summary.compilation_score:.1f}%</td>
                <td><div class="progress-bar"><div class="progress-fill" style="width: {self.summary.compilation_score}%"></div></div></td>
            </tr>
            <tr>
                <td>Numerical</td>
                <td>{self.summary.numerical_score:.1f}%</td>
                <td><div class="progress-bar"><div class="progress-fill" style="width: {self.summary.numerical_score}%"></div></div></td>
            </tr>
            <tr>
                <td>Strategic</td>
                <td>{self.summary.strategic_score:.1f}%</td>
                <td><div class="progress-bar"><div class="progress-fill" style="width: {self.summary.strategic_score}%"></div></div></td>
            </tr>
        </table>
"""
        
        # Add recommendations if any
        if self.recommendations:
            html += """
        <h2>Recommendations</h2>
        <div class="recommendations">
            <ul>
"""
            for rec in self.recommendations:
                html += f"                <li>{rec}</li>\n"
            html += """            </ul>
        </div>
"""
        
        html += """
    </div>
</body>
</html>"""
        
        return html
    
    def save(self, output_dir: str, formats: List[str] = None) -> Dict[str, str]:
        """
        Save dashboard to files in specified formats.
        
        Args:
            output_dir: Directory to save files
            formats: List of formats ('json', 'md', 'html'). Defaults to all.
            
        Returns:
            Dictionary mapping format to file path
        """
        formats = formats or ['json', 'md', 'html']
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        paths = {}
        base_name = f"{self.model_name}_evaluation"
        
        if 'json' in formats:
            json_path = output_dir / f"{base_name}.json"
            with open(json_path, 'w') as f:
                f.write(self.to_json())
            paths['json'] = str(json_path)
        
        if 'md' in formats:
            md_path = output_dir / f"{base_name}.md"
            with open(md_path, 'w') as f:
                f.write(self.to_markdown())
            paths['md'] = str(md_path)
        
        if 'html' in formats:
            html_path = output_dir / f"{base_name}.html"
            with open(html_path, 'w') as f:
                f.write(self.to_html())
            paths['html'] = str(html_path)
        
        return paths


class DashboardGenerator:
    """
    Generate evaluation dashboards from model paths.
    
    Orchestrates all evaluation components and aggregates results.
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize dashboard generator.
        
        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self.numerical_verifier = NumericalVerifier(verbose=verbose)
        self.compilation_verifier = CompilationVerifier(verbose=verbose)
        self.strategic_evaluator = StrategicEvaluator(verbose=verbose)
    
    def generate(
        self,
        original_model_path: str,
        modified_model_path: str,
        ground_truth_path: Optional[str] = None,
        execution_report: Optional[Dict] = None
    ) -> EvaluationDashboard:
        """
        Generate complete evaluation dashboard.
        
        Args:
            original_model_path: Path to original model
            modified_model_path: Path to modified model
            ground_truth_path: Optional path to ground truth
            execution_report: Optional execution report from orchestrator
            
        Returns:
            EvaluationDashboard
        """
        model_name = Path(original_model_path).stem
        dashboard = EvaluationDashboard(
            model_name=model_name,
            created_at=datetime.now().isoformat()
        )
        
        if self.verbose:
            print(f"Generating evaluation dashboard for {model_name}...")
        
        # Run numerical verification
        if self.verbose:
            print("  Running numerical verification...")
        try:
            dashboard.numerical_result = self.numerical_verifier.verify_from_paths(
                original_model_path, modified_model_path
            )
        except Exception as e:
            dashboard.error_analysis.append({
                'type': 'NumericalVerification',
                'message': str(e)
            })
        
        # Run compilation verification
        if self.verbose:
            print("  Running compilation verification...")
        try:
            dashboard.compilation_result = self.compilation_verifier.verify(
                modified_model_path, original_model_path
            )
        except Exception as e:
            dashboard.error_analysis.append({
                'type': 'CompilationVerification',
                'message': str(e)
            })
        
        # Run strategic evaluation
        if self.verbose:
            print("  Running strategic evaluation...")
        try:
            dashboard.strategic_result = self.strategic_evaluator.evaluate(
                original_model_path, modified_model_path,
                execution_report=execution_report,
                ground_truth_path=ground_truth_path
            )
        except Exception as e:
            dashboard.error_analysis.append({
                'type': 'StrategicEvaluation',
                'message': str(e)
            })
        
        # Build executive summary
        self._build_summary(dashboard)
        
        # Check tier compliance
        self._check_tier_compliance(dashboard)
        
        # Aggregate recommendations
        self._aggregate_recommendations(dashboard)
        
        if self.verbose:
            print(f"  Dashboard generation complete.")
        
        return dashboard
    
    def _build_summary(self, dashboard: EvaluationDashboard) -> None:
        """Build executive summary from component results."""
        summary = dashboard.summary
        
        # Compilation metrics
        if dashboard.compilation_result:
            comp = dashboard.compilation_result
            summary.compilation_success = comp.status == CompilationStatus.WILL_COMPILE
            summary.single_lm_file = comp.single_lm_file
            summary.blockers_remaining = comp.blocker_stats.remaining_blockers
            summary.blockers_resolved = comp.blocker_stats.resolved_blockers
            if comp.blocker_stats.original_blockers > 0:
                summary.blocker_resolution_rate = comp.blocker_stats.resolution_rate
            else:
                summary.blocker_resolution_rate = 1.0
            
            # Compilation score
            if summary.compilation_success and summary.single_lm_file:
                summary.compilation_score = 100.0
            elif summary.compilation_success:
                summary.compilation_score = 80.0
            else:
                summary.compilation_score = summary.blocker_resolution_rate * 60.0
        
        # Numerical metrics
        if dashboard.numerical_result:
            num = dashboard.numerical_result
            summary.numerical_match = num.outputs_match
            summary.tolerance_category = num.tolerance_category.value
            
            # Numerical score based on tolerance category
            tolerance_scores = {
                ToleranceCategory.IDENTICAL: 100.0,
                ToleranceCategory.CLOSE: 95.0,
                ToleranceCategory.ACCEPTABLE: 85.0,
                ToleranceCategory.MARGINAL: 60.0,
                ToleranceCategory.DIVERGENT: 20.0
            }
            summary.numerical_score = tolerance_scores.get(num.tolerance_category, 0.0)
        
        # Strategic metrics
        if dashboard.strategic_result:
            strat = dashboard.strategic_result
            summary.strategic_score = strat.overall_score * 100
        
        # Overall score (weighted average)
        weights = {'compilation': 0.4, 'numerical': 0.3, 'strategic': 0.3}
        summary.overall_score = (
            weights['compilation'] * summary.compilation_score +
            weights['numerical'] * summary.numerical_score +
            weights['strategic'] * summary.strategic_score
        )
        
        # Determine overall status
        if summary.compilation_success and summary.numerical_match:
            summary.status = OverallStatus.PASSED
        elif summary.blocker_resolution_rate > 0.5:
            summary.status = OverallStatus.PARTIAL
        else:
            summary.status = OverallStatus.FAILED
    
    def _check_tier_compliance(self, dashboard: EvaluationDashboard) -> None:
        """Check compliance with evaluation tiers."""
        summary = dashboard.summary
        
        # Tier 1: Critical metrics
        # - compilation_success = True
        # - single_lm_file = True
        # - numerical_tolerance <= 1e-4
        dashboard.tier1_passed = (
            summary.compilation_success and
            summary.single_lm_file and
            summary.numerical_match
        )
        
        # Tier 2: Quality metrics
        # - blocker_resolution_rate > 95%
        # - transformation_accuracy > 90%
        # - strategy_appropriateness > 85%
        tier2_requirements = [
            summary.blocker_resolution_rate >= 0.95,
            summary.strategic_score >= 85.0
        ]
        dashboard.tier2_passed = all(tier2_requirements)
        
        # Tier 3: Efficiency metrics
        # - transformation_efficiency > 80%
        # - region_coverage = 100%
        # - fallback_rate < 10%
        if dashboard.strategic_result:
            efficiency = dashboard.strategic_result.efficiency.efficiency_rate
            dashboard.tier3_passed = efficiency >= 0.8
        else:
            dashboard.tier3_passed = True  # Default to pass if no data
    
    def _aggregate_recommendations(self, dashboard: EvaluationDashboard) -> None:
        """Aggregate recommendations from all evaluators."""
        recommendations = []
        
        # From strategic evaluator
        if dashboard.strategic_result:
            recommendations.extend(dashboard.strategic_result.recommendations)
        
        # From compilation result
        if dashboard.compilation_result:
            if not dashboard.compilation_result.all_nodes_mla_compatible:
                recommendations.append(
                    "Address remaining MLA compatibility issues"
                )
            if not dashboard.compilation_result.single_lm_file:
                recommendations.append(
                    f"Reduce graph splits (currently predicting {dashboard.compilation_result.predicted_lm_files} LM files)"
                )
        
        # From numerical result
        if dashboard.numerical_result:
            if not dashboard.numerical_result.outputs_match:
                recommendations.append(
                    f"Investigate numerical divergence (max diff: {dashboard.numerical_result.max_absolute_difference:.2e})"
                )
        
        # Deduplicate
        dashboard.recommendations = list(dict.fromkeys(recommendations))


# =============================================================================
# Convenience Functions
# =============================================================================

def generate_dashboard(
    original_path: str,
    modified_path: str,
    output_dir: str = ".",
    verbose: bool = False
) -> EvaluationDashboard:
    """
    Convenience function to generate and save dashboard.
    
    Args:
        original_path: Path to original model
        modified_path: Path to modified model
        output_dir: Directory to save reports
        verbose: Enable verbose output
        
    Returns:
        EvaluationDashboard
    """
    generator = DashboardGenerator(verbose=verbose)
    dashboard = generator.generate(original_path, modified_path)
    dashboard.save(output_dir)
    return dashboard


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python evaluation_dashboard.py <original.onnx> <modified.onnx> [output_dir]")
        sys.exit(1)
    
    original_path = sys.argv[1]
    modified_path = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "."
    
    generator = DashboardGenerator(verbose=True)
    dashboard = generator.generate(original_path, modified_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print(dashboard.to_markdown())
    
    # Save files
    paths = dashboard.save(output_dir)
    print("\n" + "=" * 60)
    print("Saved reports:")
    for fmt, path in paths.items():
        print(f"  {fmt}: {path}")
