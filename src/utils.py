"""
Utility Functions Module

Helper functions for data loading, evaluation, and argument parsing.
"""

import os
import json
import pandas as pd
from typing import List, Optional


def get_prompt_column(df: pd.DataFrame) -> str:
    """
    Auto-detect the prompt/query column in a DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Column name containing prompts
        
    Raises:
        ValueError: If no prompt column is found
    """
    possible_names = ['jailbreak_query', 'query', 'prompt', 'question', 'text', 'content']
    
    for name in possible_names:
        if name in df.columns:
            return name
    
    for col in df.columns:
        if 'query' in col.lower() or 'prompt' in col.lower():
            return col
    
    raise ValueError(f"Cannot find prompt column. Available columns: {df.columns.tolist()}")


def get_image_column(df: pd.DataFrame) -> str:
    """
    Auto-detect the image path column in a DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Column name containing image paths
        
    Raises:
        ValueError: If no image column is found
    """
    possible_names = ['image_path', 'image', 'img_path', 'img', 'image_file', 'filename']
    
    for name in possible_names:
        if name in df.columns:
            return name
    
    for col in df.columns:
        if 'image' in col.lower() or 'img' in col.lower():
            return col
    
    raise ValueError(f"Cannot find image column. Available columns: {df.columns.tolist()}")


def parse_layers(s: str) -> Optional[List[int]]:
    """
    Parse layer range string into list of integers.
    
    Supports formats:
    - "7-31": Range from 7 to 31 inclusive
    - "5,10,15": Comma-separated list
    - "none" or empty: Returns None
    
    Args:
        s: Layer specification string
        
    Returns:
        List of layer indices or None
    """
    if not s or s.lower() == 'none':
        return None
    if '-' in s:
        start, end = map(int, s.split('-'))
        return list(range(start, end + 1))
    return [int(x) for x in s.split(',')]


def save_results(results: dict, output_path: str):
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Results dictionary
        output_path: Output file path
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_path}")


def load_results(input_path: str) -> dict:
    """
    Load evaluation results from JSON file.
    
    Args:
        input_path: Input file path
        
    Returns:
        Results dictionary
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)
