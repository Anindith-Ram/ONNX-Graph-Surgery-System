#!/usr/bin/env python3
"""
Pretty-print cache files for human readability.

Usage:
    python pretty_print_cache.py cache/file.json
    python pretty_print_cache.py cache/  # Print all cache files
"""

import json
import sys
from pathlib import Path
from typing import Optional


def pretty_print_cache_file(cache_file: Path):
    """Pretty-print a single cache file."""
    try:
        with open(cache_file) as f:
            data = json.load(f)
        
        print(f"\n{'='*80}")
        print(f"Cache File: {cache_file.name}")
        print(f"{'='*80}\n")
        
        # Timestamp (human-readable)
        timestamp = data.get('timestamp', 'Unknown')
        if isinstance(timestamp, str):
            print(f"Timestamp: {timestamp}")
        else:
            # Convert Unix timestamp to readable
            import time
            print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}")
        
        if 'timestamp_unix' in data:
            print(f"   (Unix: {data['timestamp_unix']})")
        
        print(f"\nModel: {data.get('model_name', 'Unknown')}")
        print(f"Hash: {data.get('prompt_hash', 'Unknown')}")
        print(f"Hits: {data.get('hits', 0)}")
        print(f"\nPrompt Preview:\n   {data.get('prompt_preview', 'N/A')[:200]}")
        
        # Response (parse and pretty-print if it's JSON)
        response = data.get('response', '')
        print(f"\nResponse:")
        print("-" * 80)
        
        # Try to parse response as JSON and pretty-print it
        try:
            # Remove markdown code blocks if present
            response_clean = response.strip()
            if response_clean.startswith('```json'):
                response_clean = response_clean[7:].strip()
            if response_clean.startswith('```'):
                response_clean = response_clean[3:].strip()
            if response_clean.endswith('```'):
                response_clean = response_clean[:-3].strip()
            
            # Parse and pretty-print
            response_json = json.loads(response_clean)
            print(json.dumps(response_json, indent=2, ensure_ascii=False))
        except:
            # Not JSON, just print as-is (newlines will be actual newlines after json.loads)
            print(response)
        
        print("-" * 80)
        print()
        
    except Exception as e:
        print(f"Error reading {cache_file}: {e}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python pretty_print_cache.py <cache_file_or_directory>")
        sys.exit(1)
    
    target = Path(sys.argv[1])
    
    if target.is_file():
        pretty_print_cache_file(target)
    elif target.is_dir():
        cache_files = sorted(target.glob("*.json"))
        if not cache_files:
            print(f"No JSON files found in {target}")
            return
        
        print(f"Found {len(cache_files)} cache files\n")
        for cache_file in cache_files:
            pretty_print_cache_file(cache_file)
    else:
        print(f"Error: {target} is not a file or directory")
        sys.exit(1)


if __name__ == "__main__":
    main()
