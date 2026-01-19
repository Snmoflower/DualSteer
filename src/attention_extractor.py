"""
Attention Feature Extractor Module

This module provides functionality to extract attention ratio features
from LVLM models for analysis and steering baseline construction.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from PIL import Image

# Import IMAGE_TOKEN_INDEX from model constants
try:
    from llava.constants import IMAGE_TOKEN_INDEX
except ImportError:
    IMAGE_TOKEN_INDEX = -200  # Default fallback


class AttentionFeatureExtractor:
    """
    Extracts attention ratio features from LVLM models.
    
    The attention ratio measures how much attention the last token
    allocates to different input regions (system, image, user).
    """
    
    def __init__(self, model, num_img_tokens: int = 576, target_layers: List[int] = None):
        """
        Initialize the attention feature extractor.
        
        Args:
            model: The LVLM model to extract features from
            num_img_tokens: Number of image tokens in the sequence
            target_layers: List of layer indices to extract features from
        """
        self.model = model
        self.num_img_tokens = num_img_tokens
        self.target_layers = target_layers or [-1]
        self.hooks = []
        self.attention_cache = {}
        self.current_boundaries = None
        
    @staticmethod
    def detect_boundaries(input_ids: torch.Tensor, num_img_tokens: int = 576) -> Tuple:
        """
        Detect region boundaries in the input sequence.
        
        Args:
            input_ids: Input token IDs tensor
            num_img_tokens: Number of image tokens
            
        Returns:
            Tuple of (bos_end, system_end, image_start, image_end) or (None, None, None, None)
        """
        bos_end = 1
        image_mask = (input_ids == IMAGE_TOKEN_INDEX)
        
        if not image_mask.any():
            return None, None, None, None
            
        image_start = image_mask.nonzero(as_tuple=True)[1][0].item()
        image_end = image_start + num_img_tokens
        system_end = image_start
        
        return bos_end, system_end, image_start, image_end

    @staticmethod
    def compute_attention_ratios(attn_weights: torch.Tensor, boundaries: Tuple) -> torch.Tensor:
        """
        Compute attention ratios for each region.
        
        Args:
            attn_weights: Attention weights tensor [B, H, Q, K] or [B, H, K]
            boundaries: Region boundaries tuple
            
        Returns:
            Attention ratios tensor [B, H, 3] for (system, image, user)
        """
        bos_end, system_end, image_start, image_end = boundaries
        
        # Extract last query's attention
        if attn_weights.dim() == 4:
            attn_sq = attn_weights[:, :, -1, :]
        else:
            attn_sq = attn_weights
        
        k_len = attn_sq.shape[-1]
        curr_img_end = min(image_end, k_len)
        curr_sys_end = min(system_end, k_len)
        
        # Compute attention to each region
        attn_sys = attn_sq[:, :, bos_end:curr_sys_end] if bos_end < curr_sys_end else attn_sq[:, :, 0:0]
        attn_img = attn_sq[:, :, image_start:curr_img_end] if image_start < curr_img_end else attn_sq[:, :, 0:0]
        attn_user = attn_sq[:, :, curr_img_end:] if curr_img_end < k_len else attn_sq[:, :, 0:0]
        
        # Sum attention for each region
        sys_sum = attn_sys.sum(dim=-1) if attn_sys.shape[-1] > 0 else torch.zeros_like(attn_sq[:, :, 0])
        img_sum = attn_img.sum(dim=-1) if attn_img.shape[-1] > 0 else torch.zeros_like(attn_sq[:, :, 0])
        user_sum = attn_user.sum(dim=-1) if attn_user.shape[-1] > 0 else torch.zeros_like(attn_sq[:, :, 0])
        
        # Normalize to get ratios
        total_sum = sys_sum + img_sum + user_sum + 1e-8
        ratios = torch.stack([sys_sum/total_sum, img_sum/total_sum, user_sum/total_sum], dim=-1)
        
        return ratios

    def _create_hook(self, layer_idx: int):
        """Create a forward hook for attention extraction."""
        def hook_fn(module, input, output):
            attn_weights = output[1]
            if attn_weights is None or self.current_boundaries is None:
                return
            ratios = self.compute_attention_ratios(attn_weights, self.current_boundaries)
            self.attention_cache[layer_idx] = ratios
        return hook_fn

    def extract(self, image: Image.Image, prompt: str) -> Optional[Dict[int, torch.Tensor]]:
        """
        Extract attention ratios for each layer.
        
        Args:
            image: Input PIL image
            prompt: Input text prompt
            
        Returns:
            Dictionary mapping layer indices to attention ratio tensors {layer_idx: [1, H, 3]}
        """
        self.attention_cache = {}
        self.hooks = []
        
        # Prepare inputs
        image_tensor, input_ids = self.model.prepare_inputs(image, prompt)
        boundaries = self.detect_boundaries(input_ids, self.num_img_tokens)
        
        if boundaries[2] is None:
            return None
        self.current_boundaries = boundaries
        
        # Get model layers
        try:
            layers = self.model.model.model.layers
        except AttributeError:
            layers = self.model.model.layers
        
        # Register hooks
        for i in self.target_layers:
            idx = i if i >= 0 else len(layers) + i
            hook = layers[idx].self_attn.register_forward_hook(self._create_hook(idx))
            self.hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = self.model.model(input_ids, images=image_tensor, 
                                return_dict=True, output_attentions=True)
        
        # Clean up hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        return self.attention_cache if self.attention_cache else None
