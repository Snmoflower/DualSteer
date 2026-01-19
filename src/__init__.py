"""
DualSteer: Dual-Space Steering for Robust Jailbreak Mitigation of LVLMs

A lightweight and training-free defense framework that restores safe reasoning
via dual-space steering:
- Stage 1: Attention Steering - recalibrates attention to counter jailbreak distractors
- Stage 2: Activation Steering - steers hidden representations toward refusal manifold
"""

from .config import DualSteerConfig, SteeringMode, DATASETS
from .attention_extractor import AttentionFeatureExtractor
from .attention_steering import AttentionPatternModeler, AttentionSteering
from .activation_extractor import ActivationExtractor
from .activation_steering import ActivationPatternAnalyzer, ActivationSteering
from .controller import DualSteerController
from .evaluate import evaluate_dataset, compute_asr
from .utils import parse_layers, get_prompt_column, get_image_column

__all__ = [
    # Config
    'DualSteerConfig',
    'SteeringMode',
    'DATASETS',
    # Extractors
    'AttentionFeatureExtractor',
    'ActivationExtractor',
    # Steering components
    'AttentionPatternModeler',
    'AttentionSteering',
    'ActivationPatternAnalyzer',
    'ActivationSteering',
    # Controller
    'DualSteerController',
    # Evaluation
    'evaluate_dataset',
    'compute_asr',
    # Utils
    'parse_layers',
    'get_prompt_column',
    'get_image_column',
]

__version__ = '1.0.0'
