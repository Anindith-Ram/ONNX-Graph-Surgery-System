#!/usr/bin/env python3
"""
Report Generator for ONNX Model Compilation Suggestions.

Generates human-readable reports in multiple formats:
- Markdown: For documentation and code reviews
- JSON: For programmatic consumption
- HTML: For web display and sharing
- Text: For terminal output

Reports include:
- Model compilation status
- Prioritized suggestions with confidence scores
- Implementation steps for each suggestion
- Summary statistics
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from suggestion_generator import SuggestionReport, Suggestion, Priority


class ReportGenerator:
    """
    Generate reports in multiple formats from suggestion data.
    """
    
    def __init__(self):
        self.priority_icons = {
            Priority.CRITICAL: "🔴",
            Priority.HIGH: "🟠",
            Priority.MEDIUM: "🟡",
            Priority.LOW: "🟢",
            Priority.INFO: "ℹ️"
        }
        
        self.priority_labels = {
            Priority.CRITICAL: "CRITICAL",
            Priority.HIGH: "HIGH",
            Priority.MEDIUM: "MEDIUM",
            Priority.LOW: "LOW",
            Priority.INFO: "INFO"
        }
    
    def generate(
        self,
        report: SuggestionReport,
        format: str = "markdown",
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate report in specified format.
        
        Args:
            report: SuggestionReport to format
            format: Output format ("markdown", "json", "html", "text")
            output_path: Optional path to save report
            
        Returns:
            Formatted report string
        """
        generators = {
            'markdown': self.generate_markdown,
            'md': self.generate_markdown,
            'json': self.generate_json,
            'html': self.generate_html,
            'text': self.generate_text,
            'txt': self.generate_text
        }
        
        if format.lower() not in generators:
            raise ValueError(f"Unknown format: {format}. Use: {list(generators.keys())}")
        
        output = generators[format.lower()](report)
        
        if output_path:
            Path(output_path).write_text(output)
            print(f"Report saved to: {output_path}")
        
        return output
    
    def generate_markdown(self, report: SuggestionReport) -> str:
        """Generate Markdown formatted report."""
        lines = [
            "# ONNX Model Compilation Analysis Report",
            "",
            f"**Generated:** {report.analysis_timestamp or datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Analyzer Version:** {report.analyzer_version}",
            "",
            "---",
            "",
            "## Model Information",
            "",
            f"| Property | Value |",
            f"|----------|-------|",
            f"| **Model Name** | `{report.model_name}` |",
            f"| **Model Path** | `{report.model_path}` |",
            f"| **Compilation Status** | {self._format_status_badge(report.compilation_status)} |",
            f"| **Total Issues** | {report.total_issues} |",
            "",
            "### Issue Summary",
            "",
            f"| Priority | Count |",
            f"|----------|-------|",
            f"| 🔴 Critical | {report.critical_count} |",
            f"| 🟠 High | {report.high_count} |",
            f"| 🟡 Medium | {report.medium_count} |",
            f"| 🟢 Low | {report.low_count} |",
            "",
            "---",
            "",
            "## Summary",
            "",
            report.summary,
            "",
            "---",
            "",
        ]
        
        if not report.suggestions:
            lines.extend([
                "## ✅ No Issues Found",
                "",
                "This model appears to be compatible with MLA compilation.",
                ""
            ])
        else:
            # Group by priority
            by_priority = {
                Priority.CRITICAL: [],
                Priority.HIGH: [],
                Priority.MEDIUM: [],
                Priority.LOW: [],
                Priority.INFO: []
            }
            
            for suggestion in report.suggestions:
                by_priority[suggestion.priority].append(suggestion)
            
            # Critical issues first
            if by_priority[Priority.CRITICAL]:
                lines.extend([
                    "## 🔴 Critical Issues",
                    "",
                    "> These issues **must** be resolved for compilation.",
                    ""
                ])
                for s in by_priority[Priority.CRITICAL]:
                    lines.extend(self._format_suggestion_markdown(s))
            
            # High priority
            if by_priority[Priority.HIGH]:
                lines.extend([
                    "## 🟠 High Priority Issues",
                    "",
                    "> These issues significantly impact compilation.",
                    ""
                ])
                for s in by_priority[Priority.HIGH]:
                    lines.extend(self._format_suggestion_markdown(s))
            
            # Medium priority
            if by_priority[Priority.MEDIUM]:
                lines.extend([
                    "## 🟡 Medium Priority Issues",
                    "",
                    "> These issues should be addressed for best results.",
                    ""
                ])
                for s in by_priority[Priority.MEDIUM]:
                    lines.extend(self._format_suggestion_markdown(s))
            
            # Low priority
            if by_priority[Priority.LOW]:
                lines.extend([
                    "## 🟢 Low Priority / Optimizations",
                    "",
                    "> Optional improvements for better performance.",
                    ""
                ])
                for s in by_priority[Priority.LOW]:
                    lines.extend(self._format_suggestion_markdown(s))
            
            # Info
            if by_priority[Priority.INFO]:
                lines.extend([
                    "## ℹ️ Informational",
                    "",
                ])
                for s in by_priority[Priority.INFO]:
                    lines.extend(self._format_suggestion_markdown(s))
        
        # Footer
        lines.extend([
            "---",
            "",
            "## Next Steps",
            "",
            "1. Address **Critical** issues first - these block compilation entirely",
            "2. Review **High** priority issues - these may cause partial compilation failure",
            "3. Consider **Medium** and **Low** priority suggestions for optimization",
            "4. Re-analyze model after changes to verify fixes",
            "",
            "---",
            "",
            "*Report generated by ONNX Compilation Advisor*"
        ])
        
        return "\n".join(lines)
    
    def _format_suggestion_markdown(self, s: Suggestion) -> List[str]:
        """Format a single suggestion in Markdown (engineer-friendly)."""
        confidence_bar = self._confidence_bar(s.confidence)
        
        # Build location string for easy reference
        location_str = f"`{s.location.node_name}` (node_id: {s.location.node_id}, op_type: {s.location.op_type})"
        
        lines = [
            f"### {s.id}. {s.issue}",
            "",
            f"**📍 Location:** {location_str}",
            "",
            f"**Confidence:** {s.confidence:.0%} {confidence_bar} | **Category:** {s.category} | **Effort:** {s.estimated_effort}",
            "",
            f"**Suggested Fix:**",
            "",
            f"> {s.suggestion}",
            "",
            "**Implementation Steps:**",
            ""
        ]
        
        for step in s.implementation_steps:
            lines.append(f"- {step}")
        
        lines.extend([
            "",
            f"**Impact:** {s.impact}",
            ""
        ])
        
        if s.reference:
            lines.append(f"**Reference:** {s.reference}")
            lines.append("")
        
        lines.append("---")
        lines.append("")
        
        return lines
    
    def _confidence_bar(self, confidence: float) -> str:
        """Create visual confidence bar."""
        filled = int(confidence * 10)
        empty = 10 - filled
        return f"[{'█' * filled}{'░' * empty}]"
    
    def _format_status_badge(self, status: str) -> str:
        """Format compilation status as badge."""
        badges = {
            'blocked': '🔴 **BLOCKED**',
            'partially_blocked': '🟠 **PARTIALLY BLOCKED**',
            'likely_compilable': '🟢 **LIKELY COMPILABLE**',
            'compilable': '✅ **COMPILABLE**'
        }
        return badges.get(status, status)
    
    def generate_json(self, report: SuggestionReport) -> str:
        """Generate JSON formatted report."""
        return json.dumps(report.to_dict(), indent=2)
    
    def generate_html(self, report: SuggestionReport) -> str:
        """Generate HTML formatted report."""
        status_colors = {
            'blocked': '#dc3545',
            'partially_blocked': '#fd7e14',
            'likely_compilable': '#28a745',
            'compilable': '#28a745'
        }
        
        status_color = status_colors.get(report.compilation_status, '#6c757d')
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ONNX Compilation Report - {report.model_name}</title>
    <style>
        :root {{
            --bg-color: #1a1a2e;
            --card-bg: #16213e;
            --text-color: #eee;
            --text-muted: #aaa;
            --border-color: #0f3460;
            --critical: #dc3545;
            --high: #fd7e14;
            --medium: #ffc107;
            --low: #28a745;
        }}
        
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1000px;
            margin: 0 auto;
        }}
        
        h1 {{
            color: #e94560;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 10px;
        }}
        
        h2 {{
            color: #0f3460;
            background: linear-gradient(90deg, #e94560, transparent);
            padding: 10px 15px;
            border-radius: 5px;
            margin-top: 30px;
        }}
        
        .summary-card {{
            background: var(--card-bg);
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            border-left: 4px solid {status_color};
        }}
        
        .status-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            background: {status_color};
            color: white;
            font-weight: bold;
            text-transform: uppercase;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin: 20px 0;
        }}
        
        .stat-card {{
            background: var(--card-bg);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        
        .stat-card.critical {{ border-top: 3px solid var(--critical); }}
        .stat-card.high {{ border-top: 3px solid var(--high); }}
        .stat-card.medium {{ border-top: 3px solid var(--medium); }}
        .stat-card.low {{ border-top: 3px solid var(--low); }}
        
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
        }}
        
        .suggestion {{
            background: var(--card-bg);
            border-radius: 10px;
            padding: 20px;
            margin: 15px 0;
            border-left: 4px solid var(--border-color);
        }}
        
        .suggestion.critical {{ border-left-color: var(--critical); }}
        .suggestion.high {{ border-left-color: var(--high); }}
        .suggestion.medium {{ border-left-color: var(--medium); }}
        .suggestion.low {{ border-left-color: var(--low); }}
        
        .suggestion-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 15px;
        }}
        
        .suggestion-title {{
            font-size: 1.2em;
            font-weight: bold;
            margin: 0;
        }}
        
        .confidence {{
            background: var(--border-color);
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.9em;
        }}
        
        .confidence-bar {{
            display: inline-block;
            width: 100px;
            height: 8px;
            background: #333;
            border-radius: 4px;
            overflow: hidden;
            vertical-align: middle;
            margin-left: 5px;
        }}
        
        .confidence-fill {{
            height: 100%;
            background: linear-gradient(90deg, #e94560, #0f3460);
        }}
        
        .location {{
            background: rgba(0,0,0,0.2);
            padding: 10px;
            border-radius: 5px;
            font-family: monospace;
            margin: 10px 0;
        }}
        
        .steps {{
            background: rgba(0,0,0,0.2);
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        
        .steps ol {{
            margin: 0;
            padding-left: 20px;
        }}
        
        .impact {{
            color: var(--text-muted);
            font-style: italic;
        }}
        
        .reference {{
            color: #e94560;
            font-size: 0.9em;
        }}
        
        footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid var(--border-color);
            color: var(--text-muted);
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔧 ONNX Compilation Analysis Report</h1>
        
        <div class="summary-card">
            <h3>Model: <code>{report.model_name}</code></h3>
            <p>Path: <code>{report.model_path}</code></p>
            <p>Status: <span class="status-badge">{report.compilation_status.replace('_', ' ')}</span></p>
            <p>{report.summary}</p>
        </div>
        
        <div class="stats">
            <div class="stat-card critical">
                <div class="stat-number">{report.critical_count}</div>
                <div>Critical</div>
            </div>
            <div class="stat-card high">
                <div class="stat-number">{report.high_count}</div>
                <div>High</div>
            </div>
            <div class="stat-card medium">
                <div class="stat-number">{report.medium_count}</div>
                <div>Medium</div>
            </div>
            <div class="stat-card low">
                <div class="stat-number">{report.low_count}</div>
                <div>Low</div>
            </div>
        </div>
        
        <h2>Suggestions</h2>
"""
        
        for s in report.suggestions:
            priority_class = s.priority.value
            confidence_pct = int(s.confidence * 100)
            
            steps_html = "\n".join([f"<li>{step}</li>" for step in s.implementation_steps])
            
            html += f"""
        <div class="suggestion {priority_class}">
            <div class="suggestion-header">
                <h3 class="suggestion-title">{s.id}. {s.issue}</h3>
                <span class="confidence">
                    {confidence_pct}%
                    <span class="confidence-bar">
                        <span class="confidence-fill" style="width: {confidence_pct}%"></span>
                    </span>
                </span>
            </div>
            
            <div class="location">
                <strong>Location:</strong> Node {s.location.node_id} ({s.location.node_name}) - {s.location.op_type}
            </div>
            
            <p><strong>Suggestion:</strong> {s.suggestion}</p>
            
            <div class="steps">
                <strong>Implementation Steps:</strong>
                <ol>
                    {steps_html}
                </ol>
            </div>
            
            <p class="impact"><strong>Impact:</strong> {s.impact}</p>
            
            {f'<p class="reference">📚 {s.reference}</p>' if s.reference else ''}
        </div>
"""
        
        html += f"""
        <footer>
            <p>Generated: {report.analysis_timestamp or datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>ONNX Compilation Advisor v{report.analyzer_version}</p>
        </footer>
    </div>
</body>
</html>
"""
        
        return html
    
    def generate_text(self, report: SuggestionReport) -> str:
        """Generate plain text formatted report."""
        lines = [
            "=" * 70,
            "ONNX MODEL COMPILATION ANALYSIS REPORT",
            "=" * 70,
            "",
            f"Model: {report.model_name}",
            f"Path: {report.model_path}",
            f"Status: {report.compilation_status.upper()}",
            f"Total Issues: {report.total_issues}",
            "",
            f"  Critical: {report.critical_count}",
            f"  High:     {report.high_count}",
            f"  Medium:   {report.medium_count}",
            f"  Low:      {report.low_count}",
            "",
            "-" * 70,
            "SUMMARY",
            "-" * 70,
            "",
            report.summary,
            "",
            "-" * 70,
            "SUGGESTIONS",
            "-" * 70,
        ]
        
        for s in report.suggestions:
            priority_label = self.priority_labels.get(s.priority, "?")
            
            lines.extend([
                "",
                f"[{s.id}] [{priority_label}] {s.issue}",
                f"    Confidence: {s.confidence:.0%}",
                f"    Location: Node {s.location.node_id} ({s.location.node_name})",
                f"    Operation: {s.location.op_type}",
                "",
                f"    SUGGESTION: {s.suggestion}",
                "",
                "    STEPS:"
            ])
            
            for step in s.implementation_steps:
                lines.append(f"      {step}")
            
            lines.extend([
                "",
                f"    IMPACT: {s.impact}"
            ])
            
            if s.reference:
                lines.append(f"    REF: {s.reference}")
            
            lines.append("")
            lines.append("-" * 40)
        
        lines.extend([
            "",
            "=" * 70,
            f"Generated: {report.analysis_timestamp or datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70
        ])
        
        return "\n".join(lines)


def generate_report(
    model_path: str,
    output_path: Optional[str] = None,
    format: str = "markdown"
) -> str:
    """
    Convenience function to analyze model and generate report.
    
    Args:
        model_path: Path to ONNX model
        output_path: Optional path to save report
        format: Output format (markdown, json, html, text)
        
    Returns:
        Generated report string
    """
    from suggestion_generator import SuggestionGenerator
    
    generator = SuggestionGenerator()
    report = generator.analyze_and_suggest(model_path)
    
    reporter = ReportGenerator()
    return reporter.generate(report, format=format, output_path=output_path)


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate ONNX compilation report')
    parser.add_argument('model', help='Path to ONNX model')
    parser.add_argument('--output', '-o', help='Output file path')
    parser.add_argument('--format', '-f', default='markdown',
                       choices=['markdown', 'md', 'json', 'html', 'text', 'txt'],
                       help='Output format')
    
    args = parser.parse_args()
    
    output = generate_report(
        args.model,
        output_path=args.output,
        format=args.format
    )
    
    if not args.output:
        print(output)

