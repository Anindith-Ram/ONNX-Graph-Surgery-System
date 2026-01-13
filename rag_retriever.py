#!/usr/bin/env python3
"""
RAG Retriever for Context-Aware Suggestions.

Retrieves relevant knowledge chunks from the knowledge base
based on semantic similarity to the query.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from knowledge_base import KnowledgeBase, KnowledgeChunk


@dataclass
class RetrievedContext:
    """Retrieved context for a query."""
    query: str
    chunks: List[KnowledgeChunk]
    scores: List[float]
    
    def get_text(self, max_chunks: int = 5, max_chars: int = 3000) -> str:
        """Get formatted text from top chunks with aggressive cleaning."""
        import re
        selected = self.chunks[:max_chunks]
        texts = []
        total_chars = 0
        
        for i, chunk in enumerate(selected):
            if total_chars >= max_chars:
                break
                
            # Aggressively clean PDF extraction artifacts
            content = chunk.content
            
            # Join words split across lines (common PDF artifact)
            content = re.sub(r'(\w)\n\s*(\w)', r'\1 \2', content)
            
            # Collapse all whitespace to single spaces
            content = re.sub(r'\s+', ' ', content)
            
            # Remove repeated punctuation/artifacts
            content = re.sub(r'\.{2,}', '.', content)
            content = re.sub(r'-{2,}', '-', content)
            
            content = content.strip()
            
            # Truncate individual chunks to prevent one huge chunk
            max_chunk_len = min(800, max_chars - total_chars)
            if len(content) > max_chunk_len:
                content = content[:max_chunk_len] + "..."
            
            if content:
                texts.append(f"[{chunk.source}] {content}")
                total_chars += len(content)
        
        return "\n\n".join(texts)
    
    def get_metadata_summary(self) -> Dict:
        """Get summary of retrieved chunks."""
        sources = {}
        for chunk in self.chunks:
            source = chunk.source
            sources[source] = sources.get(source, 0) + 1
        
        return {
            'total_chunks': len(self.chunks),
            'sources': sources,
            'avg_score': sum(self.scores) / len(self.scores) if self.scores else 0.0
        }


class RAGRetriever:
    """
    Retrieves relevant knowledge chunks using semantic similarity.
    
    Uses Gemini embeddings for semantic search when available,
    falls back to keyword matching otherwise.
    
    Usage:
        retriever = RAGRetriever(kb_path="knowledge_base.json", api_key="...")
        context = retriever.retrieve("How to fix Einsum operations?")
    """
    
    def __init__(
        self,
        kb_path: str,
        api_key: Optional[str] = None,
        top_k: int = 5
    ):
        """
        Initialize RAG retriever.
        
        Args:
            kb_path: Path to knowledge base JSON file
            api_key: Gemini API key for embeddings (optional)
            top_k: Number of top chunks to retrieve
        """
        self.kb_path = Path(kb_path)
        self.api_key = api_key
        self.top_k = top_k
        
        # Load knowledge base
        if self.kb_path.exists():
            self.kb = KnowledgeBase.load(str(self.kb_path))
            print(f"Loaded knowledge base: {len(self.kb.chunks)} chunks")
        else:
            print(f"Warning: Knowledge base not found at {kb_path}")
            self.kb = KnowledgeBase(chunks=[])
        
        # Initialize Gemini if available
        self.use_embeddings = False
        if GEMINI_AVAILABLE and api_key:
            try:
                genai.configure(api_key=api_key)
                self.use_embeddings = True
                print("Using Gemini embeddings for semantic search")
            except Exception as e:
                print(f"Warning: Could not initialize Gemini: {e}")
                self.use_embeddings = False
        
        # Cache for query embeddings (to avoid repeated API calls)
        self._query_embedding_cache: Dict[str, List[float]] = {}
    
    def retrieve(
        self,
        query: str,
        model_category: Optional[str] = None,
        op_type: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> RetrievedContext:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Search query
            model_category: Optional model category (YOLO, ViT, Transformer, etc.)
            op_type: Optional operation type to filter by
            top_k: Number of chunks to retrieve (overrides default)
            
        Returns:
            RetrievedContext with relevant chunks
        """
        top_k = top_k or self.top_k
        
        if self.use_embeddings:
            return self._retrieve_semantic(query, model_category, op_type, top_k)
        else:
            return self._retrieve_keyword(query, model_category, op_type, top_k)
    
    def _retrieve_semantic(
        self,
        query: str,
        model_category: Optional[str],
        op_type: Optional[str],
        top_k: int
    ) -> RetrievedContext:
        """Retrieve using semantic similarity with embeddings."""
        try:
            # Check cache for query embedding first
            query_cache_key = f"{query}_{model_category}_{op_type}"
            if query_cache_key in self._query_embedding_cache:
                query_embedding = self._query_embedding_cache[query_cache_key]
                embedding_model = None  # Will be set when needed for chunks
            else:
                # Get query embedding using embedding model
                # Note: Gemini embedding API may vary - adjust model name as needed
                try:
                    # Try newer embedding API
                    embedding_model = genai.GenerativeModel('models/embedding-001')
                    query_embedding = embedding_model.embed_content(query)['embedding']
                    # Cache the query embedding
                    self._query_embedding_cache[query_cache_key] = query_embedding
                except:
                    # Fallback: use text-embedding-004 or similar
                    try:
                        embedding_model = genai.GenerativeModel('models/text-embedding-004')
                        query_embedding = embedding_model.embed_content(query)['embedding']
                        # Cache the query embedding
                        self._query_embedding_cache[query_cache_key] = query_embedding
                    except:
                        # If embeddings not available, fall back to keyword matching
                        return self._retrieve_keyword(query, model_category, op_type, top_k)
            
            # If embedding_model wasn't set (from cache), create it for chunk embeddings
            if embedding_model is None:
                try:
                    embedding_model = genai.GenerativeModel('models/embedding-001')
                except:
                    try:
                        embedding_model = genai.GenerativeModel('models/text-embedding-004')
                    except:
                        # If can't create model, fall back to keyword
                        return self._retrieve_keyword(query, model_category, op_type, top_k)
            
            # Compute similarities
            scores = []
            for chunk in self.kb.chunks:
                # Filter by metadata if provided
                if model_category:
                    if 'model_category' in chunk.metadata:
                        if chunk.metadata['model_category'] != model_category:
                            continue
                
                if op_type:
                    if 'related_ops' in chunk.metadata:
                        if op_type not in chunk.metadata['related_ops']:
                            continue
                    elif 'op_types' in chunk.metadata:
                        if op_type not in chunk.metadata['op_types']:
                            continue
                
                # Get chunk embedding (compute if not cached)
                if chunk.embedding is None:
                    chunk_embedding = embedding_model.embed_content(chunk.content)['embedding']
                    chunk.embedding = chunk_embedding
                else:
                    chunk_embedding = chunk.embedding
                
                # Cosine similarity
                similarity = self._cosine_similarity(query_embedding, chunk_embedding)
                scores.append((similarity, chunk))
            
            # Sort by score
            scores.sort(key=lambda x: x[0], reverse=True)
            
            # Return top K
            top_scores = scores[:top_k]
            chunks = [chunk for _, chunk in top_scores]
            score_values = [score for score, _ in top_scores]
            
            return RetrievedContext(
                query=query,
                chunks=chunks,
                scores=score_values
            )
        
        except Exception as e:
            print(f"Error in semantic retrieval: {e}, falling back to keyword matching")
            return self._retrieve_keyword(query, model_category, op_type, top_k)
    
    def _retrieve_keyword(
        self,
        query: str,
        model_category: Optional[str],
        op_type: Optional[str],
        top_k: int
    ) -> RetrievedContext:
        """Retrieve using keyword matching (fallback)."""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scores = []
        for chunk in self.kb.chunks:
            # Filter by metadata if provided
            if model_category:
                if 'model_category' in chunk.metadata:
                    if chunk.metadata['model_category'] != model_category:
                        continue
            
            if op_type:
                if 'related_ops' in chunk.metadata:
                    if op_type not in chunk.metadata['related_ops']:
                        continue
                elif 'op_types' in chunk.metadata:
                    if op_type not in chunk.metadata['op_types']:
                        continue
            
            # Keyword matching score
            content_lower = chunk.content.lower()
            content_words = set(content_lower.split())
            
            # Jaccard similarity
            intersection = query_words & content_words
            union = query_words | content_words
            
            if len(union) == 0:
                score = 0.0
            else:
                score = len(intersection) / len(union)
            
            # Boost score if query words appear in content
            for word in query_words:
                if word in content_lower:
                    score += 0.1
            
            scores.append((score, chunk))
        
        # Sort by score
        scores.sort(key=lambda x: x[0], reverse=True)
        
        # Return top K
        top_scores = scores[:top_k]
        chunks = [chunk for _, chunk in top_scores]
        score_values = [score for score, _ in top_scores]
        
        return RetrievedContext(
            query=query,
            chunks=chunks,
            scores=score_values
        )
    
    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def get_model_category_hints(self, model_category: str) -> List[KnowledgeChunk]:
        """Get category-specific hints."""
        hints = []
        for chunk in self.kb.chunks:
            if chunk.metadata.get('model_category') == model_category:
                hints.append(chunk)
            elif chunk.metadata.get('model_name', '').lower().startswith(model_category.lower()):
                hints.append(chunk)
        return hints[:3]  # Top 3 hints


def detect_model_category(model_path: str) -> str:
    """
    Detect model category from model path or structure.
    
    Returns:
        Category string: "YOLO", "ViT", "Transformer", "CNN", "Other"
    """
    path_lower = model_path.lower()
    
    # Check path for keywords
    if 'yolo' in path_lower:
        return "YOLO"
    elif 'vit' in path_lower or 'vision_transformer' in path_lower:
        return "ViT"
    elif 't5' in path_lower or 'transformer' in path_lower or 'bert' in path_lower:
        return "Transformer"
    elif 'cnn' in path_lower or 'resnet' in path_lower or 'mobilenet' in path_lower:
        return "CNN"
    else:
        # Try to infer from model structure
        try:
            import onnx
            model = onnx.load(model_path)
            ops = [node.op_type for node in model.graph.node]
            
            # Check for YOLO patterns
            if any('yolo' in op.lower() for op in ops) or 'NonMaxSuppression' in ops:
                return "YOLO"
            
            # Check for Transformer patterns
            if 'Attention' in ops or 'LayerNormalization' in ops:
                return "Transformer"
            
            # Check for CNN patterns
            if 'Conv' in ops and 'Pool' in ops:
                return "CNN"
        except:
            pass
    
    return "Other"


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test RAG retriever')
    parser.add_argument('--kb', default='knowledge_base.json',
                       help='Path to knowledge base')
    parser.add_argument('--query', required=True,
                       help='Search query')
    parser.add_argument('--api-key', help='Gemini API key')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of results')
    
    args = parser.parse_args()
    
    retriever = RAGRetriever(args.kb, api_key=args.api_key, top_k=args.top_k)
    context = retriever.retrieve(args.query)
    
    print(f"\nQuery: {args.query}")
    print(f"Retrieved {len(context.chunks)} chunks")
    print(f"\nTop results:\n")
    print(context.get_text(max_chunks=args.top_k))

