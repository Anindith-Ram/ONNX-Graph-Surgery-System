#!/usr/bin/env python3
"""
Knowledge Base for RAG-Enhanced Advisory Pipeline.

Extracts and stores knowledge from:
1. PDF documentation (ONNX Graph Surgery for Model SDK.pdf)
2. Dataset transformation patterns
3. Model-specific best practices

This knowledge is used to provide context-aware suggestions.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import onnx

# Initialize PDF library flags
PDF_AVAILABLE = False
USE_PDFPLUMBER = False

try:
    import PyPDF2
    PDF_AVAILABLE = True
    USE_PDFPLUMBER = False
except ImportError:
    try:
        import pdfplumber
        PDF_AVAILABLE = True
        USE_PDFPLUMBER = True
    except ImportError:
        PDF_AVAILABLE = False
        USE_PDFPLUMBER = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from dataset_analyzer import DatasetAnalyzer, AnalysisReport
from response_cache import cached_gemini_call


@dataclass
class KnowledgeChunk:
    """A chunk of knowledge from the knowledge base."""
    id: str
    source: str  # "pdf", "dataset", "template"
    content: str
    metadata: Dict = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'source': self.source,
            'content': self.content,
            'metadata': self.metadata
        }


@dataclass
class KnowledgeBase:
    """Complete knowledge base with all chunks."""
    chunks: List[KnowledgeChunk] = field(default_factory=list)
    version: str = "1.0.0"
    
    def to_dict(self) -> Dict:
        return {
            'version': self.version,
            'chunks': [chunk.to_dict() for chunk in self.chunks]
        }
    
    def save(self, path: str):
        """Save knowledge base to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'KnowledgeBase':
        """Load knowledge base from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        chunks = [KnowledgeChunk(**chunk) for chunk in data['chunks']]
        kb = cls(chunks=chunks, version=data.get('version', '1.0.0'))
        return kb


class KnowledgeBaseBuilder:
    """
    Builds knowledge base from PDF and dataset.
    
    Usage:
        builder = KnowledgeBaseBuilder()
        kb = builder.build(
            pdf_path="ONNX Graph Surgery for Model SDK.pdf",
            dataset_dir="dataset"
        )
        kb.save("knowledge_base.json")
    """
    
    def __init__(self, api_key: Optional[str] = None, use_gemini_enhancement: bool = True):
        self.chunks = []
        self.chunk_id_counter = 0
        self.api_key = api_key
        self.use_gemini_enhancement = use_gemini_enhancement and GEMINI_AVAILABLE and api_key
        
        if self.use_gemini_enhancement:
            try:
                genai.configure(api_key=api_key)
                print("  Gemini enhancement enabled for knowledge base")
            except Exception as e:
                print(f"  Warning: Could not initialize Gemini: {e}")
                self.use_gemini_enhancement = False
    
    def build(
        self,
        pdf_path: Optional[str] = None,
        dataset_dir: Optional[str] = None,
        existing_kb_path: Optional[str] = None,
        train_test_split_path: Optional[str] = None,
        use_train_only: bool = True
    ) -> KnowledgeBase:
        """
        Build knowledge base from sources.
        
        Args:
            pdf_path: Path to PDF documentation
            dataset_dir: Directory with model pairs
            existing_kb_path: Path to existing KB to merge with
            train_test_split_path: Path to train_test_split.json
            use_train_only: If True, only use training models for KB
            
        Returns:
            KnowledgeBase object
        """
        print("Building Knowledge Base...")
        
        # Load train/test split if provided
        train_models = None
        if train_test_split_path and Path(train_test_split_path).exists():
            try:
                with open(train_test_split_path) as f:
                    split_data = json.load(f)
                train_models = set(split_data.get('train_models', []))
                print(f"  Loaded train/test split: {len(train_models)} training models")
            except Exception as e:
                print(f"  Warning: Could not load train/test split: {e}")
        
        # Load existing if provided
        if existing_kb_path and Path(existing_kb_path).exists():
            print(f"  Loading existing KB from {existing_kb_path}")
            existing_kb = KnowledgeBase.load(existing_kb_path)
            self.chunks.extend(existing_kb.chunks)
            self.chunk_id_counter = len(self.chunks)
        
        # Extract from PDF
        if pdf_path and Path(pdf_path).exists():
            print(f"  Extracting from PDF: {pdf_path}")
            pdf_chunks = self._extract_from_pdf(pdf_path)
            
            # Skip PDF enhancement - use raw chunks for better reliability
            # Raw PDF chunks work just as well for RAG semantic search
            # Enhancement adds little value and causes blocking issues
            # if self.use_gemini_enhancement:
            #     print(f"    Enhancing {len(pdf_chunks)} PDF chunks with Gemini...")
            #     pdf_chunks = self._enhance_chunks(pdf_chunks, chunk_type="pdf")
            
            self.chunks.extend(pdf_chunks)
            print(f"    Extracted {len(pdf_chunks)} chunks from PDF")
        elif pdf_path:
            print(f"  Warning: PDF not found at {pdf_path}")
        
        # Extract from dataset
        if dataset_dir and Path(dataset_dir).exists():
            print(f"  Extracting patterns from dataset: {dataset_dir}")
            dataset_chunks = self._extract_from_dataset(
                dataset_dir,
                train_models=train_models if use_train_only else None
            )
            
            # Enhance dataset chunks with Gemini if enabled
            if self.use_gemini_enhancement:
                print(f"    Enhancing {len(dataset_chunks)} dataset chunks with Gemini...")
                dataset_chunks = self._enhance_chunks(dataset_chunks, chunk_type="dataset")
            
            self.chunks.extend(dataset_chunks)
            print(f"    Extracted {len(dataset_chunks)} chunks from dataset")
        elif dataset_dir:
            print(f"  Warning: Dataset directory not found: {dataset_dir}")
        
        # Add template knowledge (templates don't need enhancement)
        template_chunks = self._extract_template_knowledge()
        self.chunks.extend(template_chunks)
        print(f"    Added {len(template_chunks)} template chunks")
        
        print(f"  Total chunks: {len(self.chunks)}")
        
        return KnowledgeBase(chunks=self.chunks)
    
    def _extract_from_pdf(self, pdf_path: str) -> List[KnowledgeChunk]:
        """Extract text chunks from PDF."""
        if not PDF_AVAILABLE:
            print("    Warning: PDF libraries not available. Install PyPDF2 or pdfplumber.")
            return []
        
        chunks = []
        
        try:
            if USE_PDFPLUMBER:
                import pdfplumber
                with pdfplumber.open(pdf_path) as pdf:
                    for page_num, page in enumerate(pdf.pages, 1):
                        text = page.extract_text()
                        if text and text.strip():
                            # Split into paragraphs
                            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                            for para in paragraphs:
                                if len(para) > 50:  # Skip very short paragraphs
                                    chunk = KnowledgeChunk(
                                        id=f"pdf_{page_num}_{self.chunk_id_counter}",
                                        source="pdf",
                                        content=para,
                                        metadata={
                                            'page': page_num,
                                            'type': 'paragraph'
                                        }
                                    )
                                    chunks.append(chunk)
                                    self.chunk_id_counter += 1
            else:
                import PyPDF2
                with open(pdf_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page_num, page in enumerate(pdf_reader.pages, 1):
                        text = page.extract_text()
                        if text and text.strip():
                            # Split into paragraphs
                            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                            for para in paragraphs:
                                if len(para) > 50:
                                    chunk = KnowledgeChunk(
                                        id=f"pdf_{page_num}_{self.chunk_id_counter}",
                                        source="pdf",
                                        content=para,
                                        metadata={
                                            'page': page_num,
                                            'type': 'paragraph'
                                        }
                                    )
                                    chunks.append(chunk)
                                    self.chunk_id_counter += 1
        except Exception as e:
            print(f"    Error extracting PDF: {e}")
        
        return chunks
    
    def _extract_from_dataset(
        self,
        dataset_dir: str,
        train_models: Optional[set] = None
    ) -> List[KnowledgeChunk]:
        """Extract transformation patterns from dataset."""
        chunks = []
        
        try:
            analyzer = DatasetAnalyzer()
            
            # Filter dataset directory to only training models if specified
            if train_models:
                # Analyze only training models
                dataset_path = Path(dataset_dir)
                model_diffs = []
                
                for model_dir in sorted(dataset_path.iterdir()):
                    if not model_dir.is_dir():
                        continue
                    
                    # Check if this model is in training set
                    if model_dir.name not in train_models:
                        continue
                    
                    original_dir = model_dir / "original"
                    modified_dir = model_dir / "modified"
                    
                    # Handle case variations
                    if not original_dir.exists():
                        original_dir = model_dir / "Original"
                    if not modified_dir.exists():
                        modified_dir = model_dir / "Modified"
                    
                    if not original_dir.exists() or not modified_dir.exists():
                        continue
                    
                    original_files = list(original_dir.glob("*.onnx"))
                    modified_files = list(modified_dir.glob("*.onnx"))
                    
                    if not original_files or not modified_files:
                        continue
                    
                    try:
                        diff = analyzer._compute_diff(
                            str(original_files[0]),
                            str(modified_files[0]),
                            model_dir.name
                        )
                        model_diffs.append(diff)
                    except Exception as e:
                        print(f"    Error analyzing {model_dir.name}: {e}")
                
                # Find patterns from training models only
                patterns = analyzer._find_patterns(model_diffs)
                op_stats = analyzer._compute_op_statistics(model_diffs)
                recommendations = analyzer._generate_recommendations(patterns, op_stats)
                
                # Create a minimal report for KB extraction
                from dataset_analyzer import AnalysisReport
                report = AnalysisReport(
                    total_models=len(model_diffs),
                    model_diffs=model_diffs,
                    patterns=patterns,
                    op_type_statistics=op_stats,
                    recommendations=recommendations
                )
            else:
                # Use all models (original behavior)
                report = analyzer.analyze_dataset(dataset_dir)
            
            # Extract pattern information
            for pattern in report.patterns:
                content = f"Pattern: {pattern.name}\n"
                content += f"Description: {pattern.description}\n"
                content += f"Model Category: {pattern.model_category}\n"
                content += f"Frequency: {pattern.frequency} models\n"
                content += f"Operations involved: {', '.join(pattern.op_types_involved)}\n"
                if pattern.context:
                    if pattern.context.get('position'):
                        content += f"Position: {pattern.context['position']}\n"
                    if pattern.context.get('surrounding_ops'):
                        content += f"Surrounding ops: {', '.join(pattern.context['surrounding_ops'])}\n"
                if pattern.implementation_hint:
                    content += f"Implementation: {pattern.implementation_hint}\n"
                if pattern.example_models:
                    content += f"Examples: {', '.join(pattern.example_models[:3])}"
                
                chunk = KnowledgeChunk(
                    id=f"dataset_pattern_{self.chunk_id_counter}",
                    source="dataset",
                    content=content,
                    metadata={
                        'pattern_name': pattern.name,
                        'model_category': pattern.model_category,
                        'frequency': pattern.frequency,
                        'op_types': pattern.op_types_involved,
                        'context': pattern.context,
                        'template_candidate': pattern.template_candidate,
                        'pattern_type': pattern.context.get('pattern_type', 'unknown') if pattern.context else 'unknown'
                    }
                )
                chunks.append(chunk)
                self.chunk_id_counter += 1
            
            # Extract common transformations
            for op_change, count in sorted(
                report.op_type_statistics.get('common_replacements', {}).items(),
                key=lambda x: x[1],
                reverse=True
            )[:20]:  # Top 20
                content = f"Common transformation: {op_change}\n"
                content += f"Occurred in {count} models\n"
                content += "This is a frequently applied transformation pattern."
                
                chunk = KnowledgeChunk(
                    id=f"dataset_transformation_{self.chunk_id_counter}",
                    source="dataset",
                    content=content,
                    metadata={
                        'transformation': op_change,
                        'frequency': count,
                        'type': 'replacement'
                    }
                )
                chunks.append(chunk)
                self.chunk_id_counter += 1
            
            # Extract model-specific insights
            for diff in report.model_diffs[:10]:  # Top 10 models
                if diff.removed_nodes or diff.added_nodes:
                    content = f"Model: {diff.model_name}\n"
                    content += f"Node count: {diff.original_node_count} -> {diff.modified_node_count}\n"
                    if diff.removed_nodes:
                        removed_ops = [n.op_type for n in diff.removed_nodes[:5]]
                        content += f"Removed operations: {', '.join(removed_ops)}\n"
                    if diff.added_nodes:
                        added_ops = [n.op_type for n in diff.added_nodes[:5]]
                        content += f"Added operations: {', '.join(added_ops)}"
                    
                    chunk = KnowledgeChunk(
                        id=f"dataset_model_{self.chunk_id_counter}",
                        source="dataset",
                        content=content,
                        metadata={
                            'model_name': diff.model_name,
                            'type': 'model_example'
                        }
                    )
                    chunks.append(chunk)
                    self.chunk_id_counter += 1
        
        except Exception as e:
            print(f"    Error extracting from dataset: {e}")
        
        return chunks
    
    def _extract_template_knowledge(self) -> List[KnowledgeChunk]:
        """Extract knowledge from deterministic templates."""
        chunks = []
        
        template_knowledge = [
            {
                'name': 'Einsum Replacement',
                'content': """Einsum operations should be replaced with MatMul + Reshape/Transpose sequences.
For batch matmul patterns (e.g., 'bhid,bhjd->bhij'), use MatMul directly as it handles batched inputs.
For complex patterns, decompose into Transpose + MatMul + Reshape operations.
Always verify output shape matches original Einsum output.""",
                'ops': ['Einsum']
            },
            {
                'name': 'Identity Removal',
                'content': """Identity nodes are pass-through operations that can be safely removed.
To remove: Note the Identity node's input tensor name, find all nodes consuming the Identity's output,
update those nodes to consume the input directly, then remove the Identity node.
This is a trivial operation with no mathematical impact.""",
                'ops': ['Identity']
            },
            {
                'name': 'Dropout Removal',
                'content': """Dropout is only needed during training. For inference, remove Dropout nodes entirely.
Connect Dropout's input directly to its output consumers. This is a no-op removal with no mathematical impact.""",
                'ops': ['Dropout']
            },
            {
                'name': '4D Tensor Requirement',
                'content': """MLA hardware requires 4D tensors. For non-4D tensors, add Reshape operations.
For 1D: [1, N, 1, 1]
For 2D: [B, N, 1, 1]
For 3D: [B, C, H, 1]
For 5D: Flatten some dimensions to create 4D.
However, avoid Reshape near outputs when possible - use split paths, slicing, and layout-aware packing instead.""",
                'ops': ['Reshape', 'Shape']
            },
            {
                'name': 'YOLO Output Handling',
                'content': """YOLO models often have non-4D outputs. Instead of adding Reshape everywhere:
1. Use split paths to separate different output branches
2. Use slicing operations to extract needed dimensions
3. Use layout-aware packing to maintain hardware alignment
4. Avoid Reshape/Transpose near outputs when possible
This approach is more hardware-aligned than naive "reshape to 4D everywhere".""",
                'ops': ['Reshape', 'Transpose', 'Slice', 'Split']
            },
            {
                'name': 'Dynamic Shape Fixing',
                'content': """Dynamic shapes must be replaced with static concrete values.
Identify the source of dynamic dimension (batch, sequence, etc.).
Determine appropriate static value for deployment.
Update graph input shape with concrete dimension.
Run shape inference to propagate changes.
Fix any downstream shape mismatches.""",
                'ops': ['Shape', 'Reshape']
            },
            {
                'name': 'LayerNorm Decomposition',
                'content': """LayerNorm can be decomposed into primitive operations:
LayerNorm = (x - mean) / sqrt(var + eps) * scale + bias
Replace with: ReduceMean -> Sub -> Pow -> ReduceMean -> Add -> Sqrt -> Div -> Mul -> Add
Create constants for epsilon and exponent (2).
Preserve scale and bias initializers.
Verify numerical equivalence.""",
                'ops': ['LayerNormalization']
            },
            {
                'name': 'GELU Approximation',
                'content': """GELU can be replaced with sigmoid approximation:
GELU(x) ≈ x * sigmoid(1.702 * x)
Create constant tensor with value 1.702.
Add Mul node: scaled = x * 1.702
Add Sigmoid node: sig = sigmoid(scaled)
Add Mul node: output = x * sig
Remove original GELU node.""",
                'ops': ['Gelu']
            }
        ]
        
        for template in template_knowledge:
            chunk = KnowledgeChunk(
                id=f"template_{self.chunk_id_counter}",
                source="template",
                content=f"Template: {template['name']}\n\n{template['content']}",
                metadata={
                    'template_name': template['name'],
                    'related_ops': template['ops'],
                    'type': 'template'
                }
            )
            chunks.append(chunk)
            self.chunk_id_counter += 1
        
        return chunks
    
    def _enhance_chunks(
        self,
        chunks: List[KnowledgeChunk],
        chunk_type: str = "general"
    ) -> List[KnowledgeChunk]:
        """
        Enhance knowledge chunks using Gemini.
        
        Skips PDF chunks (uses raw content for better reliability).
        Enhances dataset chunks (transformation patterns benefit from structuring).
        
        Args:
            chunks: List of chunks to enhance
            chunk_type: Type of chunks ("pdf", "dataset", "general")
            
        Returns:
            List of enhanced chunks (or original chunks for PDF)
        """
        if not self.use_gemini_enhancement:
            return chunks
        
        # Skip PDF enhancement - use raw chunks
        # Raw PDF chunks work perfectly for semantic search
        if chunk_type == "pdf":
            return chunks
        
        # Live enhancement (original logic)
        enhanced_chunks = []
        
        for i, chunk in enumerate(chunks):
            try:
                enhanced = self._enhance_single_chunk(chunk, chunk_type)
                enhanced_chunks.append(enhanced)
                
                if (i + 1) % 10 == 0:
                    print(f"      Enhanced {i + 1}/{len(chunks)} chunks...")
            
            except Exception as e:
                # Silently use original chunk if enhancement fails
                enhanced_chunks.append(chunk)
        
        return enhanced_chunks
    
    def _enhance_single_chunk(
        self,
        chunk: KnowledgeChunk,
        chunk_type: str
    ) -> KnowledgeChunk:
        """Enhance a single knowledge chunk using Gemini."""
        
        # Build enhancement prompt based on chunk type
        if chunk_type == "pdf":
            prompt = self._build_pdf_enhancement_prompt(chunk)
        elif chunk_type == "dataset":
            prompt = self._build_dataset_enhancement_prompt(chunk)
        else:
            prompt = self._build_general_enhancement_prompt(chunk)
        
        # Call Gemini with caching
        try:
            if not self.api_key:
                return chunk
            
            # #region agent log
            import json
            with open('/Users/anindithram/Documents/Automated Model Surgery/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"F","location":"knowledge_base.py:559","message":"Calling cached_gemini_call","data":{"chunk_id":chunk.id,"chunk_type":chunk_type,"prompt_length":len(prompt),"chunk_content_length":len(chunk.content),"chunk_content_preview":chunk.content[:150]},"timestamp":int(__import__('time').time()*1000)}) + '\n')
            # #endregion
            
            response = cached_gemini_call(
                prompt=prompt,
                api_key=self.api_key,
                model_name="models/gemini-3-pro-preview",
                temperature=0.2,  # Lower temperature for more consistent extraction
                max_tokens=1500
            )
            
            # #region agent log
            with open('/Users/anindithram/Documents/Automated Model Surgery/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"F","location":"knowledge_base.py:570","message":"Response received","data":{"response_is_none":response is None,"response_length":len(response) if response else 0},"timestamp":int(__import__('time').time()*1000)}) + '\n')
            # #endregion
            
            if not response:
                return chunk
            
            # Parse enhancement
            enhancement = self._parse_enhancement_response(response)
            
            if enhancement:
                # Apply enhancement
                chunk = self._apply_enhancement(chunk, enhancement)
        
        except Exception as e:
            print(f"        Error enhancing chunk {chunk.id}: {e}")
        
        return chunk
    
    def _build_pdf_enhancement_prompt(self, chunk: KnowledgeChunk) -> str:
        """Build prompt for enhancing PDF chunks."""
        return f"""You are an expert ONNX neural network graph modification engineer specializing in edge hardware compilation (MLA/CVU/APU).

This is technical documentation about modifying neural network computational graphs for hardware deployment. "Graph surgery" is a technical term referring to the process of modifying neural network graph structures.

Analyze this technical documentation excerpt and extract structured knowledge:

CONTENT:
{chunk.content}

TASK:
1. Extract key technical terms and ONNX operation types mentioned
2. Identify model categories if mentioned (YOLO, ViT, Transformer, CNN, etc.)
3. Summarize the main technique, best practice, or insight
4. Extract any implementation hints or steps
5. Normalize terminology to standard ONNX terms
6. Identify hardware-specific considerations (4D tensors, static shapes, etc.)

OUTPUT FORMAT (JSON only, no markdown):
{{
    "enhanced_content": "Refined, clear description of the technique/insight",
    "key_terms": ["term1", "term2", ...],
    "operation_types": ["Op1", "Op2", ...],
    "model_categories": ["YOLO", "Transformer", ...],
    "implementation_hints": ["hint1", "hint2", ...],
    "summary": "One-sentence summary of the key insight",
    "hardware_considerations": ["consideration1", ...],
    "related_techniques": ["technique1", ...]
}}"""
    
    def _build_dataset_enhancement_prompt(self, chunk: KnowledgeChunk) -> str:
        """Build prompt for enhancing dataset pattern chunks."""
        return f"""You are an expert ONNX graph surgery engineer. Analyze this transformation pattern from a dataset of successfully modified models:

CONTENT:
{chunk.content}

METADATA:
{json.dumps(chunk.metadata, indent=2)}

TASK:
1. Extract the core transformation pattern
2. Identify which operations are involved
3. Determine model categories this pattern applies to
4. Create a clear, actionable description
5. Suggest implementation approach
6. Identify when this pattern should be used

OUTPUT FORMAT (JSON only, no markdown):
{{
    "enhanced_content": "Clear description of the transformation pattern and when to use it",
    "key_terms": ["term1", "term2", ...],
    "operation_types": ["Op1", "Op2", ...],
    "model_categories": ["YOLO", "Transformer", ...],
    "implementation_hints": ["step1", "step2", ...],
    "summary": "One-sentence summary",
    "pattern_type": "removal|replacement|addition|modification",
    "applicability": "When this pattern should be applied"
}}"""
    
    def _build_general_enhancement_prompt(self, chunk: KnowledgeChunk) -> str:
        """Build prompt for general chunk enhancement."""
        return f"""You are an expert ONNX graph surgery engineer. Enhance this knowledge chunk:

CONTENT:
{chunk.content}

SOURCE: {chunk.source}

TASK:
Extract structured knowledge: key terms, operations, model categories, implementation hints.

OUTPUT FORMAT (JSON only, no markdown):
{{
    "enhanced_content": "Refined description",
    "key_terms": ["term1", ...],
    "operation_types": ["Op1", ...],
    "model_categories": ["Category1", ...],
    "implementation_hints": ["hint1", ...],
    "summary": "One-sentence summary"
}}"""
    
    def _parse_enhancement_response(self, response: str) -> Optional[Dict]:
        """Parse Gemini response to extract enhancement."""
        try:
            response = response.strip()
            
            # Find JSON block
            if '```json' in response:
                start = response.find('```json') + 7
                end = response.find('```', start)
                response = response[start:end].strip()
            elif '```' in response:
                start = response.find('```') + 3
                end = response.find('```', start)
                if end > start:
                    response = response[start:end].strip()
            
            # Parse JSON
            enhancement = json.loads(response)
            return enhancement
        
        except Exception as e:
            # Try to extract JSON from partial response
            try:
                # Look for JSON object
                start = response.find('{')
                end = response.rfind('}') + 1
                if start >= 0 and end > start:
                    enhancement = json.loads(response[start:end])
                    return enhancement
            except:
                pass
            
            return None
    
    def _apply_enhancement(
        self,
        chunk: KnowledgeChunk,
        enhancement: Dict
    ) -> KnowledgeChunk:
        """Apply enhancement to a chunk."""
        # Update content if enhanced version provided
        if 'enhanced_content' in enhancement and enhancement['enhanced_content']:
            chunk.content = enhancement['enhanced_content']
        
        # Update metadata with extracted information
        if 'key_terms' in enhancement:
            chunk.metadata['key_terms'] = enhancement['key_terms']
        
        if 'operation_types' in enhancement:
            # Merge with existing op_types if present
            existing_ops = chunk.metadata.get('op_types', [])
            new_ops = enhancement['operation_types']
            chunk.metadata['op_types'] = list(set(existing_ops + new_ops))
        
        if 'model_categories' in enhancement:
            chunk.metadata['model_categories'] = enhancement['model_categories']
        
        if 'implementation_hints' in enhancement:
            chunk.metadata['implementation_hints'] = enhancement['implementation_hints']
        
        if 'summary' in enhancement:
            chunk.metadata['summary'] = enhancement['summary']
        
        if 'hardware_considerations' in enhancement:
            chunk.metadata['hardware_considerations'] = enhancement['hardware_considerations']
        
        if 'related_techniques' in enhancement:
            chunk.metadata['related_techniques'] = enhancement['related_techniques']
        
        if 'pattern_type' in enhancement:
            chunk.metadata['pattern_type'] = enhancement['pattern_type']
        
        if 'applicability' in enhancement:
            chunk.metadata['applicability'] = enhancement['applicability']
        
        # Mark as enhanced
        chunk.metadata['gemini_enhanced'] = True
        
        return chunk


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Build knowledge base from PDF and dataset')
    parser.add_argument('--pdf', default='ONNX Graph Surgery for Model SDK.pdf',
                       help='Path to PDF documentation')
    parser.add_argument('--dataset', default='dataset',
                       help='Path to dataset directory')
    parser.add_argument('--output', default='knowledge_base.json',
                       help='Output path for knowledge base')
    parser.add_argument('--existing', help='Path to existing KB to merge')
    parser.add_argument('--train-test-split', help='Path to train_test_split.json')
    parser.add_argument('--use-train-only', action='store_true', default=True,
                       help='Only use training models for KB (default: True)')
    parser.add_argument('--api-key', help='Gemini API key for enhancement (optional)')
    parser.add_argument('--no-gemini-enhancement', action='store_true',
                       help='Disable Gemini enhancement (faster, but lower quality)')
    
    args = parser.parse_args()
    
    # Get API key from args, env, or config
    api_key = args.api_key
    if not api_key:
        api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        try:
            from config import GEMINI_API_KEY
            if GEMINI_API_KEY and GEMINI_API_KEY != "your-api-key-here":
                api_key = GEMINI_API_KEY
        except ImportError:
            pass
    
    builder = KnowledgeBaseBuilder(
        api_key=api_key,
        use_gemini_enhancement=not args.no_gemini_enhancement
    )
    kb = builder.build(
        pdf_path=args.pdf,
        dataset_dir=args.dataset,
        existing_kb_path=args.existing,
        train_test_split_path=args.train_test_split,
        use_train_only=args.use_train_only
    )
    
    kb.save(args.output)
    print(f"\nKnowledge base saved to: {args.output}")
    print(f"Total chunks: {len(kb.chunks)}")

