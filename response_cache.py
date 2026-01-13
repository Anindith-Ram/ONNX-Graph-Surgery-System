#!/usr/bin/env python3
"""
Response Cache for Gemini API calls.

Caches Gemini responses to avoid repeat API calls for similar prompts.
This significantly reduces API costs and improves reliability when
re-running the pipeline on similar models.

Features:
- MD5-based prompt hashing for fast lookup
- JSON file storage for persistence
- TTL (time-to-live) support for cache expiration
- Statistics tracking
- Thread-safe operations
"""

import os
import json
import time
import hashlib
import threading
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


@dataclass
class CacheEntry:
    """A single cache entry."""
    prompt_hash: str
    response: str
    timestamp: float  # Unix timestamp for TTL calculations
    prompt_preview: str  # First 200 chars for debugging
    model_name: str
    hits: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dict with human-readable timestamp (primary) and formatted response."""
        d = asdict(self)
        # Reorder for readability: put human-readable timestamp FIRST
        # Note: JSON files escape newlines as \n (standard JSON behavior)
        # When parsed with json.loads(), \n becomes actual newlines
        result = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp)),  # Human-readable as primary
            'timestamp_unix': d['timestamp'],  # Keep Unix timestamp for TTL calculations (renamed for clarity)
            'prompt_hash': d['prompt_hash'],
            'model_name': d['model_name'],
            'hits': d['hits'],
            'prompt_preview': d['prompt_preview'],
            'response': d['response']  # Response: \n in JSON file becomes actual newlines when parsed
        }
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CacheEntry':
        """Load from dict, handling both old and new timestamp formats."""
        # Handle new format: timestamp is readable string, timestamp_unix is Unix float
        # Old format: timestamp is Unix float
        entry_data = data.copy()
        
        if isinstance(data.get('timestamp'), str):
            # New format: timestamp is readable, use timestamp_unix for internal
            if 'timestamp_unix' in data:
                entry_data['timestamp'] = data['timestamp_unix']
            else:
                # Convert readable timestamp to Unix
                try:
                    entry_data['timestamp'] = time.mktime(time.strptime(data['timestamp'], "%Y-%m-%d %H:%M:%S"))
                except:
                    entry_data['timestamp'] = time.time()
        
        return cls(**entry_data)


@dataclass 
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    total_entries: int = 0
    total_size_bytes: int = 0
    oldest_entry: Optional[float] = None
    newest_entry: Optional[float] = None
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['hit_rate'] = f"{self.hit_rate:.1%}"
        return d


class ResponseCache:
    """
    Cache for Gemini API responses.
    
    Usage:
        cache = ResponseCache()
        
        # Check cache before API call
        cached = cache.get(prompt)
        if cached:
            return cached
        
        # Make API call
        response = gemini.generate(prompt)
        
        # Store in cache
        cache.set(prompt, response, model_name="gemini-3-pro-preview")
    """
    
    def __init__(
        self,
        cache_dir: str = "cache",
        ttl_hours: float = 168,  # 7 days default
        max_entries: int = 10000,
        enabled: bool = True
    ):
        """
        Initialize the response cache.
        
        Args:
            cache_dir: Directory to store cache files
            ttl_hours: Time-to-live in hours (0 = no expiration)
            max_entries: Maximum number of entries (LRU eviction)
            enabled: Whether caching is enabled
        """
        self.cache_dir = Path(cache_dir)
        self.ttl_seconds = ttl_hours * 3600 if ttl_hours > 0 else 0
        self.max_entries = max_entries
        self.enabled = enabled
        self._lock = threading.Lock()
        self._stats = CacheStats()
        
        if enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_stats()
    
    def get_cache_key(self, prompt: str) -> str:
        """Generate cache key from prompt using MD5 hash."""
        return hashlib.md5(prompt.encode('utf-8')).hexdigest()
    
    def get(self, prompt: str, cache_key: Optional[str] = None) -> Optional[str]:
        """
        Get cached response for a prompt.
        
        Args:
            prompt: The prompt to look up (used if cache_key not provided)
            cache_key: Optional custom cache key (will be hashed to avoid filename length issues)
            
        Returns:
            Cached response string, or None if not found/expired
        """
        if not self.enabled:
            return None
        
        # Hash the key if it's a custom cache_key (to avoid filename length issues)
        if cache_key:
            key = self.get_cache_key(cache_key)
        else:
            key = self.get_cache_key(prompt)
        cache_file = self.cache_dir / f"{key}.json"
        
        with self._lock:
            if not cache_file.exists():
                self._stats.misses += 1
                return None
            
            try:
                data = json.loads(cache_file.read_text())
                entry = CacheEntry.from_dict(data)
                
                # Check TTL (handle both old and new timestamp formats)
                # New format: timestamp is readable string, timestamp_unix is Unix float
                # Old format: timestamp is Unix float
                entry_timestamp = entry.timestamp
                if isinstance(data.get('timestamp'), str) and 'timestamp_unix' in data:
                    entry_timestamp = data['timestamp_unix']
                
                if self.ttl_seconds > 0:
                    age = time.time() - entry_timestamp
                    if age > self.ttl_seconds:
                        # Expired
                        cache_file.unlink()
                        self._stats.misses += 1
                        return None
                
                # Update hit count
                entry.hits += 1
                # Save with readable format (timestamp_readable first, ensure_ascii=False for better formatting)
                cache_data = entry.to_dict()
                cache_file.write_text(json.dumps(cache_data, indent=2, ensure_ascii=False))
                
                self._stats.hits += 1
                return entry.response
                
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                # Corrupted cache entry
                try:
                    cache_file.unlink()
                except:
                    pass
                self._stats.misses += 1
                return None
    
    def _is_response_valid(self, response: str) -> bool:
        """Check if response is valid (not truncated or empty)."""
        if not response:
            return False
        
        response = response.strip()
        
        # Very short responses are likely invalid unless they're complete JSON
        if len(response) < 10 and not (response.startswith('[') or response.startswith('{')):
            return False
        
        # Check for truncated JSON
        if response.startswith('['):
            # Array should end with ]
            if not response.endswith(']'):
                return False
        elif response.startswith('{'):
            # Object should end with }
            if not response.endswith('}'):
                return False
        
        # Check for obviously incomplete responses
        if response.endswith(('...', '","', '": "', '": [', '": {', ': "')):
            return False
        
        return True
    
    def set(
        self,
        prompt: str,
        response: str,
        model_name: str = "unknown",
        force: bool = False,
        cache_key: Optional[str] = None
    ) -> None:
        """
        Store response in cache.
        
        Args:
            prompt: The prompt (used for preview if cache_key not provided)
            response: The response to cache
            model_name: Name of the model used
            force: If True, cache even if response appears invalid
            cache_key: Optional custom cache key (will be hashed to avoid filename length issues)
        """
        if not self.enabled:
            return
        
        # Skip caching invalid/truncated responses unless forced
        if not force and not self._is_response_valid(response):
            return
        
        # Hash the key if it's a custom cache_key (to avoid filename length issues)
        if cache_key:
            key = self.get_cache_key(cache_key)
        else:
            key = self.get_cache_key(prompt)
        cache_file = self.cache_dir / f"{key}.json"
        
        entry = CacheEntry(
            prompt_hash=key,
            response=response,
            timestamp=time.time(),
            prompt_preview=prompt[:200].replace('\n', ' ') if prompt else (cache_key[:200] if cache_key else ""),
            model_name=model_name,
            hits=0
        )
        
        with self._lock:
            try:
                # Save with readable format (timestamp_readable will be included)
                cache_data = entry.to_dict()
                cache_file.write_text(json.dumps(cache_data, indent=2, ensure_ascii=False))
                self._stats.total_entries += 1
                
                # Check if we need to evict old entries
                if self._stats.total_entries > self.max_entries:
                    self._evict_oldest()
                    
            except Exception as e:
                print(f"Warning: Failed to cache response: {e}")
    
    def invalidate(self, prompt: str) -> bool:
        """
        Invalidate a cached entry.
        
        Args:
            prompt: The prompt to invalidate
            
        Returns:
            True if entry was found and removed
        """
        if not self.enabled:
            return False
        
        key = self.get_cache_key(prompt)
        cache_file = self.cache_dir / f"{key}.json"
        
        with self._lock:
            if cache_file.exists():
                try:
                    cache_file.unlink()
                    self._stats.total_entries -= 1
                    return True
                except:
                    pass
            return False
    
    def clear(self) -> int:
        """
        Clear all cached entries.
        
        Returns:
            Number of entries cleared
        """
        if not self.enabled:
            return 0
        
        count = 0
        with self._lock:
            for cache_file in self.cache_dir.glob("*.json"):
                if cache_file.name != "_stats.json":
                    try:
                        cache_file.unlink()
                        count += 1
                    except:
                        pass
            
            self._stats = CacheStats()
            self._save_stats()
        
        return count
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            # Recalculate some stats
            self._stats.total_entries = 0
            self._stats.total_size_bytes = 0
            self._stats.oldest_entry = None
            self._stats.newest_entry = None
            
            for cache_file in self.cache_dir.glob("*.json"):
                if cache_file.name == "_stats.json":
                    continue
                    
                try:
                    stat = cache_file.stat()
                    self._stats.total_entries += 1
                    self._stats.total_size_bytes += stat.st_size
                    
                    data = json.loads(cache_file.read_text())
                    ts = data.get('timestamp', 0)
                    
                    if self._stats.oldest_entry is None or ts < self._stats.oldest_entry:
                        self._stats.oldest_entry = ts
                    if self._stats.newest_entry is None or ts > self._stats.newest_entry:
                        self._stats.newest_entry = ts
                        
                except:
                    pass
            
            return self._stats
    
    def _evict_oldest(self, count: int = 100) -> int:
        """Evict oldest entries to make room."""
        entries = []
        
        for cache_file in self.cache_dir.glob("*.json"):
            if cache_file.name == "_stats.json":
                continue
            try:
                data = json.loads(cache_file.read_text())
                entries.append((data.get('timestamp', 0), cache_file))
            except:
                pass
        
        # Sort by timestamp (oldest first)
        entries.sort(key=lambda x: x[0])
        
        evicted = 0
        for _, cache_file in entries[:count]:
            try:
                cache_file.unlink()
                evicted += 1
            except:
                pass
        
        self._stats.total_entries -= evicted
        return evicted
    
    def _load_stats(self) -> None:
        """Load stats from disk."""
        stats_file = self.cache_dir / "_stats.json"
        if stats_file.exists():
            try:
                data = json.loads(stats_file.read_text())
                self._stats.hits = data.get('hits', 0)
                self._stats.misses = data.get('misses', 0)
            except:
                pass
    
    def _save_stats(self) -> None:
        """Save stats to disk."""
        stats_file = self.cache_dir / "_stats.json"
        try:
            stats_file.write_text(json.dumps({
                'hits': self._stats.hits,
                'misses': self._stats.misses,
                'last_updated': time.time()
            }, indent=2))
        except:
            pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._save_stats()
        return False


# Global cache instance
_global_cache: Optional[ResponseCache] = None


def get_cache(cache_dir: str = "cache") -> ResponseCache:
    """Get or create global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = ResponseCache(cache_dir=cache_dir)
    return _global_cache


def cached_gemini_call(
    prompt: str,
    api_key: Optional[str] = None,
    model_name: str = "models/gemini-3-pro-preview",
    temperature: float = 0.3,
    max_tokens: int = 2000,
    cache: Optional[ResponseCache] = None,
    generate_fn = None  # Legacy parameter for backward compatibility
) -> Optional[str]:
    """
    Wrapper for Gemini calls with caching.
    
    Args:
        prompt: The prompt to send
        api_key: Gemini API key (required if generate_fn not provided)
        model_name: Model name (default: "models/gemini-3-pro-preview")
        temperature: Temperature for generation (default: 0.3)
        max_tokens: Maximum tokens (default: 2000)
        cache: Optional cache instance (uses global if None)
        generate_fn: Optional function that takes prompt and returns response (legacy)
        
    Returns:
        Response string
    """
    if cache is None:
        cache = get_cache()
    
    # Check cache
    cached = cache.get(prompt)
    # #region agent log
    import json
    with open('/Users/anindithram/Documents/Automated Model Surgery/.cursor/debug.log', 'a') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"response_cache.py:393","message":"Cache check","data":{"cache_hit":cached is not None,"cached_length":len(cached) if cached else 0},"timestamp":int(__import__('time').time()*1000)}) + '\n')
    # #endregion
    if cached is not None:
        return cached
    
    # Make actual API call
    try:
        if generate_fn:
            # Legacy mode: use provided function
            response = generate_fn(prompt)
        else:
            # New mode: use API key directly
            if not api_key:
                # Try to get from environment or config
                api_key = os.getenv('GEMINI_API_KEY')
                if not api_key:
                    try:
                        from config import GEMINI_API_KEY
                        if GEMINI_API_KEY and GEMINI_API_KEY != "your-api-key-here":
                            api_key = GEMINI_API_KEY
                    except ImportError:
                        pass
            
            if not api_key:
                raise ValueError("API key required for Gemini calls")
            
            # Configure and call Gemini
            genai.configure(api_key=api_key)
            
            # Use only gemini-3-pro-preview
            model_names_to_try = [
                "models/gemini-3-pro-preview"  # Only model to use
            ]
            
            response = None
            last_error = None
            
            # #region agent log
            import json
            with open('/Users/anindithram/Documents/Automated Model Surgery/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"response_cache.py:428","message":"About to create model","data":{"model_name":model_names_to_try[0],"prompt_length":len(prompt),"prompt_preview":prompt[:100]},"timestamp":int(__import__('time').time()*1000)}) + '\n')
            # #endregion
            
            # Configure safety settings to allow technical content
            try:
                from google.generativeai.types import HarmCategory, HarmBlockThreshold
                safety_settings = {
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            except ImportError:
                safety_settings = {
                    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
                }
            
            for try_model_name in model_names_to_try:
                try:
                    # #region agent log
                    with open('/Users/anindithram/Documents/Automated Model Surgery/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run2","hypothesisId":"A","location":"response_cache.py:432","message":"Creating GenerativeModel","data":{"model_name":try_model_name,"has_safety_settings":True,"safety_settings_keys":list(safety_settings.keys())},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                    # #endregion
                    
                    # Set safety settings at MODEL level
                    model = genai.GenerativeModel(
                        try_model_name,
                        safety_settings=safety_settings
                    )
                    
                    generation_config = {
                        'temperature': temperature,
                        'max_output_tokens': max_tokens,
                    }
                    
                    # #region agent log
                    with open('/Users/anindithram/Documents/Automated Model Surgery/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run2","hypothesisId":"B","location":"response_cache.py:443","message":"Calling generate_content","data":{"has_safety_in_config":True,"safety_settings_applied":True,"prompt_preview":prompt[:200]},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                    # #endregion
                    
                    response_obj = model.generate_content(
                        prompt,
                        generation_config=generation_config
                    )
                    
                    # Extract text - handle finish_reason=2 gracefully (even with BLOCK_NONE, .text may fail)
                    response = None
                    finish_reason_val = None
                    
                    # #region agent log
                    has_candidates = False
                    candidate_count = 0
                    has_text_attr = False
                    if hasattr(response_obj, 'candidates') and response_obj.candidates:
                        has_candidates = True
                        candidate_count = len(response_obj.candidates)
                        if candidate_count > 0 and hasattr(response_obj.candidates[0], 'finish_reason'):
                            finish_reason_val = response_obj.candidates[0].finish_reason
                    # Check for .text attribute safely (don't use hasattr as it may trigger the accessor)
                    try:
                        # Just check if attribute exists without accessing it
                        has_text_attr = 'text' in dir(response_obj)
                    except:
                        has_text_attr = False
                    with open('/Users/anindithram/Documents/Automated Model Surgery/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run2","hypothesisId":"C","location":"response_cache.py:485","message":"Response received","data":{"finish_reason":finish_reason_val,"has_candidates":has_candidates,"candidate_count":candidate_count,"has_text_attr":has_text_attr},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                    # #endregion
                    
                    # Try .text accessor first
                    try:
                        response = response_obj.text
                    except Exception as text_error:
                        # #region agent log
                        with open('/Users/anindithram/Documents/Automated Model Surgery/.cursor/debug.log', 'a') as f:
                            f.write(json.dumps({"sessionId":"debug-session","runId":"run2","hypothesisId":"C","location":"response_cache.py:504","message":".text accessor failed, trying parts","data":{"error_type":type(text_error).__name__,"error_msg":str(text_error)[:200],"has_candidates":hasattr(response_obj,'candidates') and bool(response_obj.candidates)},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                        # #endregion
                        # If .text fails (common with finish_reason=2 even with BLOCK_NONE), extract from candidates/parts
                        if hasattr(response_obj, 'candidates') and response_obj.candidates:
                            candidate = response_obj.candidates[0]
                            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                                parts_text = []
                                parts_count = len(candidate.content.parts)
                                parts_info = []
                                for i, part in enumerate(candidate.content.parts):
                                    part_type = type(part).__name__
                                    has_text_attr = hasattr(part, 'text')
                                    part_text_len = 0
                                    if has_text_attr:
                                        try:
                                            part_text = part.text
                                            part_text_len = len(part_text) if part_text else 0
                                            parts_text.append(part_text)
                                        except Exception as part_error:
                                            part_text_len = -1  # Error accessing text
                                    parts_info.append({"index": i, "type": part_type, "has_text": has_text_attr, "text_len": part_text_len})
                                
                                if parts_text:
                                    response = ''.join(parts_text)
                                    # #region agent log
                                    with open('/Users/anindithram/Documents/Automated Model Surgery/.cursor/debug.log', 'a') as f:
                                        f.write(json.dumps({"sessionId":"debug-session","runId":"run2","hypothesisId":"C","location":"response_cache.py:530","message":"Extracted from parts","data":{"parts_count":len(parts_text),"response_length":len(response)},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                                    # #endregion
                                else:
                                    # #region agent log
                                    with open('/Users/anindithram/Documents/Automated Model Surgery/.cursor/debug.log', 'a') as f:
                                        f.write(json.dumps({"sessionId":"debug-session","runId":"run2","hypothesisId":"C","location":"response_cache.py:535","message":"No parts with text found","data":{"parts_count":parts_count,"parts_info":parts_info},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                                    # #endregion
                            else:
                                # #region agent log
                                with open('/Users/anindithram/Documents/Automated Model Surgery/.cursor/debug.log', 'a') as f:
                                    f.write(json.dumps({"sessionId":"debug-session","runId":"run2","hypothesisId":"C","location":"response_cache.py:525","message":"No content/parts in candidate","data":{"has_content":hasattr(candidate,'content') if hasattr(response_obj,'candidates') and response_obj.candidates else False},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                                # #endregion
                        else:
                            # #region agent log
                            with open('/Users/anindithram/Documents/Automated Model Surgery/.cursor/debug.log', 'a') as f:
                                f.write(json.dumps({"sessionId":"debug-session","runId":"run2","hypothesisId":"C","location":"response_cache.py:530","message":"No candidates in response","data":{},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                            # #endregion
                    
                    # #region agent log
                    extraction_method = "text_attr" if response and hasattr(response_obj, 'text') and response == getattr(response_obj, 'text', None) else ("parts" if response else "failed")
                    with open('/Users/anindithram/Documents/Automated Model Surgery/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run2","hypothesisId":"C","location":"response_cache.py:545","message":"Text extracted","data":{"response_length":len(response) if response else 0,"response_preview":response[:100] if response else None,"extraction_method":extraction_method},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                    # #endregion
                    
                    # If we still don't have a response after all attempts, this is an error
                    if not response:
                        raise ValueError(f"Could not extract text from response (finish_reason={finish_reason_val}). Even with BLOCK_NONE safety settings, no text content was available.")
                    
                    # If successful, update model_name for caching
                    if try_model_name != model_name:
                        model_name = try_model_name
                    break
                except Exception as e:
                    # #region agent log
                    import traceback
                    tb_lines = traceback.format_exc().split('\n')
                    actual_location = [l for l in tb_lines if 'response_cache.py' in l and 'line' in l]
                    loc_str = actual_location[0] if actual_location else "unknown"
                    with open('/Users/anindithram/Documents/Automated Model Surgery/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run2","hypothesisId":"E","location":"response_cache.py:outer_exception","message":"Exception in model call (outer handler)","data":{"error_type":type(e).__name__,"error_msg":str(e)[:200],"traceback_location":loc_str[:100]},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                    # #endregion
                    last_error = e
                    continue
            
            if response is None:
                raise last_error or ValueError(f"Failed to call Gemini with any model name")
        
        if response:
            cache.set(prompt, response, model_name=model_name)
        return response
    except Exception as e:
        print(f"API call failed: {e}")
        return None


if __name__ == "__main__":
    # Demo/test
    cache = ResponseCache(cache_dir="test_cache")
    
    # Test set/get
    test_prompt = "What is 2+2?"
    test_response = "The answer is 4."
    
    cache.set(test_prompt, test_response, model_name="test")
    
    retrieved = cache.get(test_prompt)
    assert retrieved == test_response, "Cache retrieval failed"
    
    # Test stats
    stats = cache.get_stats()
    print("Cache Stats:")
    print(json.dumps(stats.to_dict(), indent=2))
    
    # Cleanup
    cache.clear()
    print("\nCache test passed!")

