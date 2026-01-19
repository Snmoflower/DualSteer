"""
DualSteer Configuration Module

This module defines configuration classes and enums for the DualSteer framework.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional


class SteeringMode(Enum):
    """Steering mode for decode phase."""
    NONE = "none"
    FIRST_ONLY = "first_only"
    FIRST_N = "first_n"
    ALL = "all"


@dataclass
class DualSteerConfig:
    """
    Configuration for DualSteer Framework.
    
    Stage 1: Attention Steering
        - Redirects attention distribution to counter jailbreak distractors
        - Prefill phase: per-layer head-specific baseline
        - Decode phase: last-layer baseline
    
    Stage 2: Hidden Representation Steering
        - Steers hidden states toward refusal manifold
        - Uses contrastive SVD to extract steering vectors
    """
    
    # Stage 1: Attention Steering Parameters
    attn_c: float = 0.5  # Scaling coefficient for attention bias
    
    # Prefill phase configuration
    attn_prefill_layers: Optional[List[int]] = None  # Layers for prefill steering
    
    # Decode phase configuration
    attn_decode_layers: Optional[List[int]] = None  # Layers for decode steering
    attn_decode_mode: SteeringMode = SteeringMode.FIRST_N
    attn_decode_n: int = 10  # Number of tokens to steer in first_n mode
    
    # Stage 2: Hidden Representation Steering Parameters
    act_method: str = "svd"  # Method: mean_diff, pca, svd, contrastive
    act_strength: float = 1.0  # Steering strength
    act_layers: Optional[List[int]] = None  # Layers to apply activation steering
    act_normalize: bool = True  # Whether to normalize steering vectors
    act_mode: SteeringMode = SteeringMode.ALL  # Decode phase behavior
    act_n: int = 10  # Number of tokens for first_n mode
    
    # General
    num_img_tokens: int = 576  # Number of image tokens (LLaVA default)
    
    def get_all_attn_layers(self) -> List[int]:
        """Get all layers that need attention baseline training."""
        layers = set()
        if self.attn_prefill_layers:
            layers.update(self.attn_prefill_layers)
        if self.attn_decode_layers:
            layers.update(self.attn_decode_layers)
        return sorted(layers)


# Dataset configurations for evaluation
DATASETS = {
    "figstep": {
        "csv_path": "data/figstep.csv",
        "image_dir": "data/figstep/images",
    },
    "qrel": {
        "csv_path": "data/qrel.csv",
        "image_dir": "data/qrel/images",
    },
    "jood": {
        "csv_path": "data/jood.csv",
        "image_dir": "data/jood/images",
    },
    "bap": {
        "csv_path": "data/bap.csv",
        "image_dir": "data/bap/images",
    },
    "umk": {
        "csv_path": "data/umk.csv",
        "image_dir": "data/umk/images",
    },
    "hades": {
        "csv_path": "data/hades.csv",
        "image_dir": "data/hades/images",
    },
    "mm-vet": {
        "csv_path": "data/mm-vet.csv",
        "image_dir": "data/mm-vet/images",
    },
}
