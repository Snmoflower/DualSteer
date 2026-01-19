"""
Activation Feature Extractor Module

This module provides functionality to extract hidden state activations
from LVLM models for steering vector computation.
"""

import torch
from typing import Dict, List
from PIL import Image


class ActivationExtractor:
    """
    Extracts hidden state activations from LVLM models.
    
    Captures the last-token hidden states at specified layers
    for contrastive analysis and steering vector computation.
    """
    
    def __init__(self, model, layer_indices: List[int] = None):
        """
        Initialize the activation extractor.
        
        Args:
            model: The LVLM model to extract activations from
            layer_indices: List of layer indices to extract activations from
        """
        self.model = model
        self.layer_indices = layer_indices or list(range(27, 32))
    
    def extract(self, image_path: str, prompt: str) -> Dict[int, torch.Tensor]:
        """
        Extract activations for specified layers.
        
        Args:
            image_path: Path to the input image
            prompt: Input text prompt
            
        Returns:
            Dictionary mapping layer indices to activation tensors {layer_idx: [d]}
        """
        try:
            image = Image.open(image_path).convert('RGB')
            with torch.no_grad():
                hidden_states = self.model.get_hidden_states(image, prompt)
            
            activations = {}
            for layer_idx in self.layer_indices:
                if layer_idx < hidden_states.shape[0]:
                    # Extract last token hidden state
                    activations[layer_idx] = hidden_states[layer_idx, 0, :].cpu()
            return activations
            
        except Exception as e:
            print(f"  [ActivationExtractor Error] {e}")
            return {}
