#!/usr/bin/env python3
"""
LLM Context Generator for Surgery Recommendations.

Generates rich, structured context for LLM prompts that explains:
- WHY: The reason for compilation issues (blocker context)
- HOW: Step-by-step surgery recommendations

This module bridges the SurgeryDatabase with the RAG pipeline,
transforming raw database entries into actionable LLM context.

Author: Automated Model Surgery Pipeline
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge_base.surgery_database import (
    SurgeryDatabase,
    NodeTransformation,
    TransformationRecord,
    SurgeryTemplate,
    CompilationBlocker
)


@dataclass
class ContextSection:
    """A section of generated LLM context."""
    title: str
    content: str
    priority: int = 1  # 1 = highest priority
    
    def format(self, include_title: bool = True) -> str:
        """Format the section for output."""
        if include_title:
            return f"## {self.title}\n{self.content}"
        return self.content


@dataclass
class GeneratedContext:
    """Complete generated context for LLM."""
    sections: List[ContextSection] = field(default_factory=list)
    total_tokens_estimate: int = 0
    relevance_score: float = 0.0
    source_models: List[str] = field(default_factory=list)
    
    def to_string(self, max_tokens: int = 4000) -> str:
        """
        Convert to string, respecting token limit.
        
        Args:
            max_tokens: Maximum estimated tokens (4 chars ~= 1 token)
        """
        # Sort by priority
        sorted_sections = sorted(self.sections, key=lambda s: s.priority)
        
        result_parts = []
        current_chars = 0
        max_chars = max_tokens * 4  # Rough estimate
        
        for section in sorted_sections:
            formatted = section.format()
            if current_chars + len(formatted) <= max_chars:
                result_parts.append(formatted)
                current_chars += len(formatted)
            else:
                # Truncate this section if possible
                remaining = max_chars - current_chars - 50  # Buffer
                if remaining > 200:  # Minimum useful content
                    truncated = formatted[:remaining] + "\n... (truncated)"
                    result_parts.append(truncated)
                break
        
        return "\n\n".join(result_parts)


class LLMContextGenerator:
    """
    Generate rich, structured context for LLM prompts.
    
    This class takes queries about compilation issues and generates
    comprehensive context that helps LLMs understand and solve problems.
    
    Key features:
    - WHY context: Explains why something is a blocker
    - HOW context: Provides step-by-step solutions
    - Code snippets: Ready-to-use GraphSurgeon code
    - Examples: Similar transformations from the database
    """
    
    def __init__(self, database: SurgeryDatabase):
        """
        Initialize with a surgery database.
        
        Args:
            database: Populated SurgeryDatabase instance
        """
        self.db = database
    
    def generate_blocker_context(
        self,
        op_type: str,
        model_category: str = "Other",
        node_context: Optional[Dict] = None,
        max_examples: int = 3
    ) -> GeneratedContext:
        """
        Generate context for a compilation blocker.
        
        Args:
            op_type: The blocking operation type (e.g., "Einsum")
            model_category: Model category (e.g., "Transformer")
            node_context: Optional additional context about the node
            max_examples: Maximum number of similar examples to include
            
        Returns:
            GeneratedContext with WHY/HOW sections
        """
        sections = []
        source_models = []
        
        # =======================================================================
        # Section 1: Blocker Explanation (WHY)
        # =======================================================================
        blocker = self.db.get_blocker_info(op_type)
        
        why_content = self._generate_why_section(op_type, blocker, model_category)
        sections.append(ContextSection(
            title="WHY THIS IS BLOCKING",
            content=why_content,
            priority=1
        ))
        
        # =======================================================================
        # Section 2: Similar Transformations (Examples)
        # =======================================================================
        similar = self.db.find_similar_blocker(op_type, model_category, top_k=max_examples)
        
        if similar:
            examples_content = self._generate_examples_section(similar, max_examples)
            sections.append(ContextSection(
                title="SIMILAR FIXES FROM TRAINING DATA",
                content=examples_content,
                priority=2
            ))
            source_models = list(set(t.source_model for t in similar))
        
        # =======================================================================
        # Section 3: Recommended Template (HOW)
        # =======================================================================
        template = self.db.get_surgery_recommendation(op_type, model_category, node_context)
        
        if template:
            how_content = self._generate_how_section(template)
            sections.append(ContextSection(
                title="RECOMMENDED SURGERY PROCEDURE",
                content=how_content,
                priority=3
            ))
        
        # =======================================================================
        # Section 4: Code Snippet
        # =======================================================================
        code = self._get_best_code_snippet(op_type, similar, template)
        
        if code:
            sections.append(ContextSection(
                title="GRAPHSURGEON CODE EXAMPLE",
                content=f"```python\n{code}\n```",
                priority=4
            ))
        
        # =======================================================================
        # Section 5: Warnings and Edge Cases
        # =======================================================================
        warnings = self._generate_warnings_section(op_type, template, similar)
        
        if warnings:
            sections.append(ContextSection(
                title="WARNINGS AND EDGE CASES",
                content=warnings,
                priority=5
            ))
        
        # Calculate estimates
        total_content = "\n\n".join(s.content for s in sections)
        token_estimate = len(total_content) // 4
        
        # Calculate relevance score
        relevance = 0.5
        if blocker:
            relevance += 0.2
        if similar:
            relevance += 0.2 * min(len(similar) / max_examples, 1.0)
        if template:
            relevance += 0.1
        
        return GeneratedContext(
            sections=sections,
            total_tokens_estimate=token_estimate,
            relevance_score=min(relevance, 1.0),
            source_models=source_models
        )
    
    def generate_node_context(
        self,
        node_analysis: Dict,
        model_category: str = "Other"
    ) -> GeneratedContext:
        """
        Generate context for a specific node from model analysis.
        
        Args:
            node_analysis: Node analysis dict from ONNXAnalyzer.get_node_context_for_surgery()
            model_category: Model category
            
        Returns:
            GeneratedContext with node-specific information
        """
        sections = []
        
        op_type = node_analysis.get('op_type', '')
        is_blocker = node_analysis.get('is_compilation_blocker', False)
        
        # =======================================================================
        # Section 1: Node Overview
        # =======================================================================
        overview = self._generate_node_overview(node_analysis)
        sections.append(ContextSection(
            title="NODE OVERVIEW",
            content=overview,
            priority=1
        ))
        
        # =======================================================================
        # Section 2: If blocker, add blocker context
        # =======================================================================
        if is_blocker:
            blocker_context = self.generate_blocker_context(
                op_type, model_category, node_analysis
            )
            # Merge sections with adjusted priorities
            for section in blocker_context.sections:
                section.priority += 1  # Lower priority than node overview
                sections.append(section)
            
            return GeneratedContext(
                sections=sections,
                total_tokens_estimate=blocker_context.total_tokens_estimate + 500,
                relevance_score=blocker_context.relevance_score,
                source_models=blocker_context.source_models
            )
        
        # =======================================================================
        # Section 3: For non-blockers, provide general context
        # =======================================================================
        transformations = self.db.find_transformations_by_op_type(op_type, model_category=model_category)
        
        if transformations:
            trans_content = self._generate_transformation_summary(transformations[:3])
            sections.append(ContextSection(
                title="RELATED TRANSFORMATIONS",
                content=trans_content,
                priority=2
            ))
        
        total_content = "\n\n".join(s.content for s in sections)
        
        return GeneratedContext(
            sections=sections,
            total_tokens_estimate=len(total_content) // 4,
            relevance_score=0.5 if not transformations else 0.7,
            source_models=[t.source_model for t in transformations[:3]]
        )
    
    def generate_model_context(
        self,
        model_analysis: Dict,
        focus_on_blockers: bool = True
    ) -> GeneratedContext:
        """
        Generate context for an entire model's transformation needs.
        
        Args:
            model_analysis: Model analysis dict from ONNXAnalyzer
            focus_on_blockers: Whether to focus on compilation blockers
            
        Returns:
            GeneratedContext with model-level recommendations
        """
        sections = []
        source_models = set()
        
        # Extract model info
        model_name = model_analysis.get('model_name', 'Unknown')
        blockers = model_analysis.get('compilation_blockers', [])
        node_count = model_analysis.get('node_count', 0)
        
        # =======================================================================
        # Section 1: Model Overview
        # =======================================================================
        overview = f"""Model: {model_name}
Total Nodes: {node_count}
Compilation Blockers: {len(blockers)}
"""
        if blockers:
            blocker_types = list(set(b.get('op_type', '') for b in blockers))
            overview += f"Blocking Operations: {', '.join(blocker_types)}"
        
        sections.append(ContextSection(
            title="MODEL OVERVIEW",
            content=overview,
            priority=1
        ))
        
        # =======================================================================
        # Section 2: Blocker-specific contexts
        # =======================================================================
        if focus_on_blockers and blockers:
            # Group blockers by type
            blockers_by_type = {}
            for b in blockers:
                op_type = b.get('op_type', 'Unknown')
                if op_type not in blockers_by_type:
                    blockers_by_type[op_type] = []
                blockers_by_type[op_type].append(b)
            
            # Generate context for each blocker type
            for op_type, type_blockers in list(blockers_by_type.items())[:5]:
                blocker_ctx = self.generate_blocker_context(
                    op_type,
                    model_category=self._detect_category(model_name),
                    max_examples=2
                )
                
                # Add abbreviated version
                blocker_summary = f"**{op_type}** ({len(type_blockers)} nodes)\n\n"
                for section in blocker_ctx.sections[:2]:  # Only first 2 sections
                    blocker_summary += section.format() + "\n\n"
                
                sections.append(ContextSection(
                    title=f"BLOCKER: {op_type}",
                    content=blocker_summary,
                    priority=2
                ))
                
                source_models.update(blocker_ctx.source_models)
        
        # =======================================================================
        # Section 3: Recommended Transformation Order
        # =======================================================================
        if blockers:
            order_content = self._generate_transformation_order(blockers)
            sections.append(ContextSection(
                title="RECOMMENDED TRANSFORMATION ORDER",
                content=order_content,
                priority=3
            ))
        
        total_content = "\n\n".join(s.content for s in sections)
        
        return GeneratedContext(
            sections=sections,
            total_tokens_estimate=len(total_content) // 4,
            relevance_score=0.8 if blockers else 0.5,
            source_models=list(source_models)
        )
    
    # =========================================================================
    # Private Helper Methods
    # =========================================================================
    
    def _generate_why_section(
        self,
        op_type: str,
        blocker: Optional[CompilationBlocker],
        model_category: str
    ) -> str:
        """Generate the WHY explanation section."""
        content = []
        
        content.append(f"**Operation:** {op_type}")
        
        if blocker:
            content.append(f"**Reason:** {blocker.reason}")
            if blocker.hardware_limitation:
                content.append(f"**Hardware Limitation:** {blocker.hardware_limitation}")
            content.append(f"**Severity:** {blocker.severity}")
            if blocker.occurrence_count > 0:
                content.append(f"**Frequency:** Found in {blocker.occurrence_count} model transformations")
        else:
            # Generate generic reason based on op type
            content.append(f"**Reason:** {op_type} operations may not be supported by target hardware")
            content.append("**Note:** No specific blocker information available in database")
        
        return "\n".join(content)
    
    def _generate_examples_section(
        self,
        similar: List[NodeTransformation],
        max_examples: int
    ) -> str:
        """Generate the examples section from similar transformations."""
        content = []
        
        content.append(f"Found {len(similar)} similar transformations:")
        content.append("")
        
        for i, t in enumerate(similar[:max_examples], 1):
            content.append(f"### Example {i}: {t.source_model}")
            content.append(f"- **Node:** {t.original_node_name} ({t.original_op_type})")
            content.append(f"- **Position:** {t.graph_position:.1%} through graph")
            content.append(f"- **Action:** {t.action}")
            
            if t.replacement_ops:
                content.append(f"- **Replaced with:** {' → '.join(t.replacement_ops)}")
            
            if t.input_tensors:
                shapes = [str(tensor.get('shape', '?')) for tensor in t.input_tensors[:3]]
                content.append(f"- **Input shapes:** {', '.join(shapes)}")
            
            if t.attributes:
                attr_items = list(t.attributes.items())[:3]
                attr_str = ', '.join(f"{k}={v}" for k, v in attr_items)
                content.append(f"- **Attributes:** {attr_str}")
            
            if t.surgery_steps:
                content.append("- **Steps:**")
                for step in t.surgery_steps[:4]:
                    content.append(f"  - {step}")
            
            content.append("")
        
        return "\n".join(content)
    
    def _generate_how_section(self, template: SurgeryTemplate) -> str:
        """Generate the HOW section from a surgery template."""
        content = []
        
        content.append(f"**Template:** {template.name}")
        content.append(f"**Confidence:** {template.confidence:.0%} ({template.success_count}/{template.success_count + template.failure_count} successful)")
        content.append(f"**Description:** {template.description}")
        content.append("")
        
        if template.steps:
            content.append("**Steps:**")
            for step in template.steps:
                content.append(f"{step.step_number}. **[{step.action}]** {step.description}")
                if step.target_pattern:
                    content.append(f"   - Target: `{step.target_pattern}`")
                if step.operation:
                    content.append(f"   - Operation: {step.operation}")
                if step.validation:
                    content.append(f"   - Validate: {step.validation}")
            content.append("")
        
        if template.example_models:
            content.append(f"**Successfully applied to:** {', '.join(template.example_models[:5])}")
        
        return "\n".join(content)
    
    def _get_best_code_snippet(
        self,
        op_type: str,
        similar: List[NodeTransformation],
        template: Optional[SurgeryTemplate]
    ) -> Optional[str]:
        """Get the best available code snippet."""
        # Prefer template code
        if template and template.graphsurgeon_code:
            return template.graphsurgeon_code
        
        # Fall back to similar transformation code
        for t in similar:
            if t.code_snippet:
                return t.code_snippet
        
        return None
    
    def _generate_warnings_section(
        self,
        op_type: str,
        template: Optional[SurgeryTemplate],
        similar: List[NodeTransformation]
    ) -> Optional[str]:
        """Generate warnings and edge cases section."""
        warnings = []
        
        # Get template warnings
        if template:
            warnings.extend(template.warnings)
            if template.contraindications:
                warnings.append("**Do NOT apply if:**")
                warnings.extend(f"- {c}" for c in template.contraindications)
        
        # Add common warnings based on op type
        if op_type == "Einsum":
            if "Verify equation pattern matches expected format" not in warnings:
                warnings.append("Verify equation pattern matches expected format")
        elif op_type in ("Reshape", "Transpose"):
            if "Validate output shapes after transformation" not in warnings:
                warnings.append("Validate output shapes after transformation")
        
        # Add dynamic shape warning if any similar transformations had dynamic shapes
        has_dynamic = any(
            any(tensor.get('is_dynamic', False) for tensor in t.input_tensors)
            for t in similar
        )
        if has_dynamic:
            warnings.append("Some instances had dynamic shapes - ensure static dimensions")
        
        if not warnings:
            return None
        
        return "\n".join(f"- {w}" if not w.startswith("**") and not w.startswith("-") else w for w in warnings)
    
    def _generate_node_overview(self, node_analysis: Dict) -> str:
        """Generate a node overview from analysis dict."""
        content = []
        
        content.append(f"**Name:** {node_analysis.get('name', 'Unknown')}")
        content.append(f"**Operation:** {node_analysis.get('op_type', 'Unknown')}")
        content.append(f"**Position:** {node_analysis.get('graph_position', 0):.1%} through graph")
        
        input_shapes = node_analysis.get('input_shapes', [])
        if input_shapes:
            content.append(f"**Input shapes:** {', '.join(str(s) for s in input_shapes[:3])}")
        
        output_shapes = node_analysis.get('output_shapes', [])
        if output_shapes:
            content.append(f"**Output shapes:** {', '.join(str(s) for s in output_shapes[:3])}")
        
        attributes = node_analysis.get('attributes', {})
        if attributes:
            attr_items = list(attributes.items())[:5]
            content.append("**Attributes:**")
            for k, v in attr_items:
                content.append(f"  - {k}: {v}")
        
        predecessors = node_analysis.get('predecessors', [])
        if predecessors:
            pred_ops = [p.get('op_type', '?') for p in predecessors[:3]]
            content.append(f"**Predecessors:** {', '.join(pred_ops)}")
        
        successors = node_analysis.get('successors', [])
        if successors:
            succ_ops = [s.get('op_type', '?') for s in successors[:3]]
            content.append(f"**Successors:** {', '.join(succ_ops)}")
        
        if node_analysis.get('is_compilation_blocker'):
            content.append(f"**⚠️ COMPILATION BLOCKER:** {node_analysis.get('blocker_reason', 'Unknown reason')}")
        
        return "\n".join(content)
    
    def _generate_transformation_summary(
        self,
        transformations: List[NodeTransformation]
    ) -> str:
        """Generate a brief summary of transformations."""
        content = []
        
        for t in transformations[:3]:
            content.append(f"- **{t.source_model}:** {t.action} {t.original_op_type}")
            if t.replacement_ops:
                content.append(f"  → Replaced with {' → '.join(t.replacement_ops)}")
        
        return "\n".join(content)
    
    def _generate_transformation_order(self, blockers: List[Dict]) -> str:
        """Generate recommended transformation order."""
        content = []
        
        # Group by severity/type
        priority_map = {
            'Loop': 1,
            'If': 1,
            'Scan': 1,
            'Einsum': 2,
            'Where': 2,
            'NonZero': 2,
            'NonMaxSuppression': 3,
            'GatherND': 4,
            'ScatterND': 4,
        }
        
        # Sort blockers by priority
        sorted_blockers = sorted(
            blockers,
            key=lambda b: priority_map.get(b.get('op_type', ''), 5)
        )
        
        content.append("Transform in this order for best results:")
        content.append("")
        
        seen_types = set()
        for i, b in enumerate(sorted_blockers, 1):
            op_type = b.get('op_type', 'Unknown')
            if op_type not in seen_types:
                content.append(f"{i}. **{op_type}** - {b.get('reason', '')[:50]}...")
                seen_types.add(op_type)
        
        return "\n".join(content)
    
    def _detect_category(self, model_name: str) -> str:
        """Detect model category from name."""
        name_lower = model_name.lower()
        
        if any(p in name_lower for p in ['yolo', 'yolov']):
            return 'YOLO'
        elif any(p in name_lower for p in ['t5', 'bert', 'gpt', 'transformer', 'marian']):
            return 'Transformer'
        elif any(p in name_lower for p in ['vit', 'deit', 'vision_transformer']):
            return 'ViT'
        elif any(p in name_lower for p in ['resnet', 'mobilenet', 'efficientnet', 'midas']):
            return 'CNN'
        
        return 'Other'


# =============================================================================
# Convenience Functions
# =============================================================================

def create_context_generator(db_path: str) -> LLMContextGenerator:
    """
    Create an LLMContextGenerator from a database file.
    
    Args:
        db_path: Path to surgery_database.json
        
    Returns:
        Configured LLMContextGenerator
    """
    db = SurgeryDatabase.load(db_path)
    return LLMContextGenerator(db)


def generate_quick_context(
    op_type: str,
    model_category: str = "Other",
    db_path: str = "rag_data/surgery_database.json"
) -> str:
    """
    Quick helper to generate context for an operation type.
    
    Args:
        op_type: Operation type (e.g., "Einsum")
        model_category: Model category
        db_path: Path to database
        
    Returns:
        Formatted context string
    """
    try:
        generator = create_context_generator(db_path)
        context = generator.generate_blocker_context(op_type, model_category)
        return context.to_string()
    except FileNotFoundError:
        return f"Database not found at {db_path}. Run build_surgery_db.py first."
    except Exception as e:
        return f"Error generating context: {e}"


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate LLM context for surgery recommendations')
    parser.add_argument('--db', default='rag_data/surgery_database.json', help='Database path')
    parser.add_argument('--op', default='Einsum', help='Operation type')
    parser.add_argument('--category', default='Transformer', help='Model category')
    
    args = parser.parse_args()
    
    # Try to load database
    try:
        db = SurgeryDatabase.load(args.db)
    except FileNotFoundError:
        # Create a test database
        from knowledge_base.surgery_database import create_database_with_defaults
        db = create_database_with_defaults()
        print("Using default database (no trained data)")
    
    generator = LLMContextGenerator(db)
    
    print(f"\n{'='*70}")
    print(f"CONTEXT FOR: {args.op} in {args.category} models")
    print(f"{'='*70}\n")
    
    context = generator.generate_blocker_context(args.op, args.category)
    
    print(context.to_string())
    
    print(f"\n{'='*70}")
    print(f"Relevance Score: {context.relevance_score:.2f}")
    print(f"Estimated Tokens: {context.total_tokens_estimate}")
    print(f"Source Models: {context.source_models}")
    print(f"{'='*70}")
