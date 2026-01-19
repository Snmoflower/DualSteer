"""
Attention Steering Module (Stage 1)

This module implements attention-level recalibration to counter
jailbreak distractors and restore safe attention allocation patterns.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional

from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv
)

from .config import DualSteerConfig, SteeringMode
from .attention_extractor import AttentionFeatureExtractor


class AttentionPatternModeler:
    """
    Models attention patterns from safe refusal samples.
    
    Maintains head-specific target ratios:
    - Prefill phase: per-layer head-specific baseline
    - Decode phase: last-layer baseline (shared across all decode layers)
    """
    
    def __init__(self):
        self.head_means_per_layer = {}  # {layer_idx: [H, 3]} - Prefill baseline
        self.last_layer_head_means = None  # [H, 3] - Decode baseline
        self.last_layer_idx = None
        self.fitted = False
        
    def fit(self, ratio_samples_per_layer: Dict[int, np.ndarray]):
        """
        Fit the pattern modeler with attention ratio samples.
        
        Args:
            ratio_samples_per_layer: {layer_idx: [N, H, 3]} attention ratio samples
        """
        print(f"\n[Attention Pattern] Learning from samples")
        print(f"  Prefill: per-layer head-specific baseline")
        print(f"  Decode:  last layer baseline")
        
        for layer_idx, ratio_samples in sorted(ratio_samples_per_layer.items()):
            head_means = ratio_samples.mean(axis=0)  # [H, 3]
            self.head_means_per_layer[layer_idx] = head_means
            
            global_mean = head_means.mean(axis=0)  # [3]
            print(f"  Layer {layer_idx:2d} ({ratio_samples.shape[0]} samples): "
                  f"Sys={global_mean[0]:.4f}, Img={global_mean[1]:.4f}, User={global_mean[2]:.4f}")
        
        # Decode uses last layer baseline
        self.last_layer_idx = max(ratio_samples_per_layer.keys())
        self.last_layer_head_means = self.head_means_per_layer[self.last_layer_idx]
        
        last_avg = self.last_layer_head_means.mean(axis=0)
        print(f"  Decode baseline (layer {self.last_layer_idx}): "
              f"Sys={last_avg[0]:.4f}, Img={last_avg[1]:.4f}, User={last_avg[2]:.4f}")
        
        self.fitted = True
        print(f"  Total layers fitted: {len(self.head_means_per_layer)}")
    
    def get_target_ratio_prefill(self, layer_idx: int) -> Optional[np.ndarray]:
        """Get prefill phase target ratio [H, 3] for a specific layer."""
        if not self.fitted:
            return None
        if layer_idx in self.head_means_per_layer:
            return self.head_means_per_layer[layer_idx]
        return self.last_layer_head_means  # fallback
    
    def get_target_ratio_decode(self) -> Optional[np.ndarray]:
        """Get decode phase target ratio [H, 3] using last layer baseline."""
        if not self.fitted:
            return None
        return self.last_layer_head_means
    
    def get_layers(self) -> List[int]:
        """Get list of fitted layer indices."""
        return list(self.head_means_per_layer.keys())


class AttentionSteering:
    """
    Stage 1: Attention Steering
    
    Steers attention distribution by adding bias to attention logits,
    guiding the model to maintain balanced attention across regions.
    """
    
    def __init__(self, model, pattern_modeler: AttentionPatternModeler, config: DualSteerConfig):
        """
        Initialize attention steering.
        
        Args:
            model: The LVLM model
            pattern_modeler: Fitted attention pattern modeler
            config: DualSteer configuration
        """
        self.model = model
        self.modeler = pattern_modeler
        self.config = config
        self.hooks = []
        self.boundaries = None
        
        # State tracking
        self.decode_token_count = 0
        self.prefill_steered = 0
        self.decode_steered = 0
        
    def reset(self):
        """Reset steering state."""
        self.decode_token_count = 0
        self.prefill_steered = 0
        self.decode_steered = 0
        
    def set_boundaries(self, boundaries):
        """Set input region boundaries."""
        self.boundaries = boundaries

    def _should_steer_decode(self) -> bool:
        """Determine if current decode token should be steered."""
        mode = self.config.attn_decode_mode
        if mode == SteeringMode.NONE:
            return False
        elif mode == SteeringMode.FIRST_ONLY:
            return self.decode_token_count == 0
        elif mode == SteeringMode.FIRST_N:
            return self.decode_token_count < self.config.attn_decode_n
        elif mode == SteeringMode.ALL:
            return True
        return False

    def _compute_bias_prefill(self, attn_logits: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Compute prefill phase attention bias using per-layer baseline."""
        if self.boundaries is None or not self.modeler.fitted:
            return torch.zeros_like(attn_logits)
        
        target_ratios_np = self.modeler.get_target_ratio_prefill(layer_idx)
        if target_ratios_np is None:
            return torch.zeros_like(attn_logits)
        
        return self._compute_bias_impl(attn_logits, target_ratios_np)
    
    def _compute_bias_decode(self, attn_logits: torch.Tensor) -> torch.Tensor:
        """Compute decode phase attention bias using last-layer baseline."""
        if self.boundaries is None or not self.modeler.fitted:
            return torch.zeros_like(attn_logits)
        
        target_ratios_np = self.modeler.get_target_ratio_decode()
        if target_ratios_np is None:
            return torch.zeros_like(attn_logits)
        
        return self._compute_bias_impl(attn_logits, target_ratios_np)
    
    def _compute_bias_impl(self, attn_logits: torch.Tensor, target_ratios_np: np.ndarray) -> torch.Tensor:
        """
        Compute attention bias: (target - current) * c * std_logits
        
        This adaptive bias correction allows regions receiving less attention
        than the safe pattern to obtain a positive bias, boosting their
        visibility in the softmax competition.
        """
        bos_end, system_end, image_start, image_end = self.boundaries
        batch, heads, q_len, k_len = attn_logits.shape
        
        if image_end > k_len:
            return torch.zeros_like(attn_logits)
        
        # Compute current attention ratios
        with torch.no_grad():
            probs = F.softmax(attn_logits, dim=-1)
            current_ratios = AttentionFeatureExtractor.compute_attention_ratios(
                probs, self.boundaries
            )  # [B, H, 3]
        
        # Compute standard deviation for normalization
        last_logits = attn_logits[:, :, -1, :]
        std_logits = last_logits.std(dim=-1, keepdim=True)  # [B, H, 1]
        
        # Target ratios
        target_ratios = torch.tensor(
            target_ratios_np,
            device=attn_logits.device,
            dtype=attn_logits.dtype
        ).unsqueeze(0)  # [1, H, 3]
        
        # Compute bias per region
        diff = target_ratios - current_ratios  # [B, H, 3]
        bias_per_region = diff * self.config.attn_c * std_logits  # [B, H, 3]
        
        # Distribute bias to token positions
        bias = torch.zeros_like(attn_logits)
        curr_img_end = min(image_end, k_len)
        curr_sys_end = min(system_end, k_len)
        
        if bos_end < curr_sys_end:
            bias[:, :, -1, bos_end:curr_sys_end] = bias_per_region[:, :, 0:1]
        if image_start < curr_img_end:
            bias[:, :, -1, image_start:curr_img_end] = bias_per_region[:, :, 1:2]
        if curr_img_end < k_len:
            bias[:, :, -1, curr_img_end:] = bias_per_region[:, :, 2:3]
        
        return bias

    def _create_hook(self, layer_idx: int, is_prefill_layer: bool, is_decode_layer: bool):
        """Create a forward hook for attention steering."""
        def hook_fn(module, args, kwargs, output):
            hidden_states = kwargs.get("hidden_states")
            attention_mask = kwargs.get("attention_mask")
            position_ids = kwargs.get("position_ids")
            past_key_value = kwargs.get("past_key_value")
            
            if hidden_states is None:
                return output
            
            bsz, q_len, _ = hidden_states.size()
            
            # Recompute attention with steering
            query_states = module.q_proj(hidden_states)
            key_states = module.k_proj(hidden_states)
            value_states = module.v_proj(hidden_states)
            
            query_states = query_states.view(bsz, q_len, module.num_heads, module.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, module.num_key_value_heads, module.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, module.num_key_value_heads, module.head_dim).transpose(1, 2)
            
            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, module.layer_idx)
            
            cos, sin = module.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
            
            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos}
                key_states, value_states = past_key_value.update(
                    key_states, value_states, module.layer_idx, cache_kwargs)
            
            key_states = repeat_kv(key_states, module.num_key_value_groups)
            value_states = repeat_kv(value_states, module.num_key_value_groups)
            
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(module.head_dim)
            
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            # Apply steering
            if self.boundaries is not None:
                _, _, _, image_end = self.boundaries
                k_len = attn_weights.shape[-1]
                
                if k_len >= image_end:
                    is_prefill = q_len > 1
                    is_decode = q_len == 1
                    
                    # Prefill phase steering
                    if is_prefill and is_prefill_layer:
                        bias = self._compute_bias_prefill(attn_weights, layer_idx)
                        attn_weights = attn_weights + bias
                        self.prefill_steered += 1
                    
                    # Decode phase steering
                    elif is_decode and is_decode_layer and self._should_steer_decode():
                        bias = self._compute_bias_decode(attn_weights)
                        attn_weights = attn_weights + bias
                        self.decode_steered += 1
                    
                    # Token counting (only at last decode layer)
                    if is_decode and is_decode_layer:
                        decode_layers = self.config.attn_decode_layers or []
                        if decode_layers and layer_idx == max(decode_layers):
                            self.decode_token_count += 1
            
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)
            
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, module.hidden_size)
            attn_output = module.o_proj(attn_output)
            
            return attn_output, None, past_key_value
        
        return hook_fn

    def apply(self):
        """Apply attention steering hooks to the model."""
        self.remove()
        self.reset()
        
        try:
            layers = self.model.model.model.layers
        except AttributeError:
            layers = self.model.model.layers
        
        prefill_layers = set(self.config.attn_prefill_layers or [])
        decode_layers = set(self.config.attn_decode_layers or [])
        all_layers = prefill_layers | decode_layers
        
        for layer_idx in all_layers:
            if layer_idx < len(layers):
                is_prefill = layer_idx in prefill_layers
                is_decode = layer_idx in decode_layers
                hook = layers[layer_idx].self_attn.register_forward_hook(
                    self._create_hook(layer_idx, is_prefill, is_decode), 
                    with_kwargs=True
                )
                self.hooks.append(hook)
        
        print(f"  [AttentionSteering]")
        print(f"    Prefill layers: {sorted(prefill_layers) if prefill_layers else 'None'}")
        print(f"    Decode layers: {sorted(decode_layers) if decode_layers else 'None'}")
        print(f"    Decode mode: {self.config.attn_decode_mode.value}, n={self.config.attn_decode_n}")

    def remove(self):
        """Remove all steering hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []
    
    def get_stats(self) -> dict:
        """Get steering statistics."""
        return {
            'prefill_steered': self.prefill_steered,
            'decode_steered': self.decode_steered,
            'decode_token_count': self.decode_token_count,
        }
