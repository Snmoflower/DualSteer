"""
Activation Steering Module (Stage 2)

This module implements hidden representation steering to guide the model
toward safe refusal responses by modifying intermediate hidden states.
"""

import numpy as np
import torch
from typing import Dict, List
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import DualSteerConfig, SteeringMode


class ActivationPatternAnalyzer:
    """
    Computes steering vectors from contrastive activation samples.
    
    Supports multiple methods for extracting discriminative directions:
    - mean_diff: Simple difference of class means
    - pca: PCA-based direction finding
    - svd: Contrastive SVD (recommended)
    - contrastive: Linear discriminant-style direction
    """
    
    def __init__(self, method: str = "svd", normalize: bool = True):
        """
        Initialize the pattern analyzer.
        
        Args:
            method: Method for computing steering vectors
            normalize: Whether to normalize the steering vectors
        """
        self.method = method
        self.normalize = normalize
        self.steering_vectors = {}
    
    def compute_steering_vectors(self, 
                                 success_acts: Dict[int, np.ndarray],
                                 fail_acts: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        Compute steering vectors from success/failure activation pairs.
        
        Args:
            success_acts: {layer_idx: [N, d]} activations from successful refusals
            fail_acts: {layer_idx: [N, d]} activations from jailbreak successes
            
        Returns:
            Dictionary mapping layer indices to steering vectors
        """
        print(f"\n[Activation Pattern] Computing vectors using {self.method.upper()}")
        
        for layer_idx in success_acts.keys():
            s_acts = success_acts[layer_idx].astype(np.float32)
            f_acts = fail_acts[layer_idx].astype(np.float32)
            
            if self.method == "mean_diff":
                vec = self._compute_mean_diff(s_acts, f_acts)
            elif self.method == "pca":
                vec = self._compute_pca_direction(s_acts, f_acts)
            elif self.method == "svd":
                vec = self._compute_svd_direction(s_acts, f_acts)
            elif self.method == "contrastive":
                vec = self._compute_contrastive_direction(s_acts, f_acts)
            else:
                vec = self._compute_mean_diff(s_acts, f_acts)
            
            if self.normalize and vec is not None:
                vec = vec / (np.linalg.norm(vec) + 1e-8)
            
            self.steering_vectors[layer_idx] = vec
            print(f"  Layer {layer_idx}: vector norm = {np.linalg.norm(vec):.4f}")
        
        return self.steering_vectors
    
    def _compute_mean_diff(self, s_acts: np.ndarray, f_acts: np.ndarray) -> np.ndarray:
        """Simple mean difference direction."""
        return np.mean(s_acts, axis=0) - np.mean(f_acts, axis=0)
    
    def _compute_pca_direction(self, s_acts: np.ndarray, f_acts: np.ndarray, 
                               n_components: int = 5) -> np.ndarray:
        """
        PCA-based direction finding.
        
        Finds the principal component that best separates the two classes.
        """
        all_acts = np.vstack([s_acts, f_acts])
        labels = np.array([1]*len(s_acts) + [0]*len(f_acts))
        
        scaler = StandardScaler()
        all_scaled = scaler.fit_transform(all_acts)
        
        pca = PCA(n_components=min(n_components, all_acts.shape[0]-1))
        pca.fit(all_scaled)
        
        projections = pca.transform(all_scaled)
        best_idx, best_sep = 0, 0
        
        # Find component with best class separation
        for i in range(projections.shape[1]):
            sep = abs(np.mean(projections[labels==1, i]) - np.mean(projections[labels==0, i]))
            if sep > best_sep:
                best_sep = sep
                best_idx = i
        
        return scaler.inverse_transform(pca.components_[best_idx].reshape(1,-1))[0]
    
    def _compute_svd_direction(self, s_acts: np.ndarray, f_acts: np.ndarray) -> np.ndarray:
        """
        Contrastive SVD direction (recommended method).
        
        Constructs a contrastive matrix from class displacements and extracts
        the principal direction via SVD. This filters out sample-specific noise
        that could trigger false refusals on benign inputs.
        """
        s_mean = np.mean(s_acts, axis=0)
        f_mean = np.mean(f_acts, axis=0)
        
        # Build contrastive matrix: displacements toward safe region
        # Each row: either (safe_sample - unsafe_mean) or (safe_mean - unsafe_sample)
        s_diff = s_acts - f_mean  # [m, d]
        f_diff = f_mean - f_acts  # [n, d] (negated: safe_mean would be close to s_mean)
        
        # Use symmetric construction for robustness
        combined = np.vstack([s_diff, s_mean - f_acts])  # [m+n, d]
        
        # SVD to find principal direction
        U, S, Vt = np.linalg.svd(combined, full_matrices=False)
        direction = Vt[0]
        
        # Orient toward safe manifold
        if np.dot(direction, s_mean - f_mean) < 0:
            direction = -direction
            
        return direction
    
    def _compute_contrastive_direction(self, s_acts: np.ndarray, f_acts: np.ndarray) -> np.ndarray:
        """
        Linear discriminant-style contrastive direction.
        
        Maximizes between-class variance relative to within-class variance.
        """
        s_mean = np.mean(s_acts, axis=0)
        f_mean = np.mean(f_acts, axis=0)
        
        # Within-class covariance
        s_cov = np.cov(s_acts.T)
        f_cov = np.cov(f_acts.T)
        within = s_cov + f_cov
        
        # Between-class covariance
        mean_diff = (s_mean - f_mean).reshape(-1, 1)
        between = mean_diff @ mean_diff.T
        
        try:
            # Solve generalized eigenvalue problem
            eigenvalues, eigenvectors = np.linalg.eigh(
                between, within + np.eye(within.shape[0]) * 1e-6)
            return eigenvectors[:, np.argmax(eigenvalues)]
        except:
            # Fallback to mean difference
            return s_mean - f_mean


class ActivationSteering:
    """
    Stage 2: Hidden Representation Steering
    
    Steers hidden states toward the refusal manifold by adding
    a computed steering vector to intermediate representations.
    
    Modes:
    - none: Only steer during prefill phase
    - first_only: Steer prefill + first decode token
    - first_n: Steer prefill + first N decode tokens
    - all: Steer all tokens
    """
    
    def __init__(self, model, steering_vectors: Dict[int, np.ndarray], config: DualSteerConfig):
        """
        Initialize activation steering.
        
        Args:
            model: The LVLM model
            steering_vectors: {layer_idx: vector} computed steering vectors
            config: DualSteer configuration
        """
        self.model = model
        self.steering_vectors = steering_vectors
        self.config = config
        self.hooks = []
        self.decode_token_count = 0
        self.prefill_steered = 0
        self.decode_steered = 0
    
    def reset(self):
        """Reset steering state."""
        self.decode_token_count = 0
        self.prefill_steered = 0
        self.decode_steered = 0
    
    def _should_steer_decode(self) -> bool:
        """Determine if current decode token should be steered."""
        mode = self.config.act_mode
        if mode == SteeringMode.NONE:
            return False
        elif mode == SteeringMode.FIRST_ONLY:
            return self.decode_token_count == 0
        elif mode == SteeringMode.FIRST_N:
            return self.decode_token_count < self.config.act_n
        elif mode == SteeringMode.ALL:
            return True
        return False
    
    def _steering_hook(self, layer_idx: int):
        """Create a forward hook for activation steering."""
        def hook_fn(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            
            bsz, seq_len, _ = hidden_states.size()
            is_prefill = seq_len > 1
            is_decode = seq_len == 1
            
            should_steer = False
            
            # Always steer during prefill
            if is_prefill:
                should_steer = True
                self.prefill_steered += 1
            # Decode phase: based on mode
            elif is_decode and self._should_steer_decode():
                should_steer = True
                self.decode_steered += 1
            
            if should_steer:
                vec = self.steering_vectors.get(layer_idx)
                if vec is not None:
                    vec_t = torch.tensor(vec, device=hidden_states.device, 
                                        dtype=hidden_states.dtype)
                    # Steer the last token's hidden state
                    hidden_states[:, -1:, :] += self.config.act_strength * vec_t
            
            # Token counting (only at last layer to avoid double counting)
            if is_decode and layer_idx == max(self.steering_vectors.keys()):
                self.decode_token_count += 1
            
            return (hidden_states,) + output[1:] if isinstance(output, tuple) else hidden_states
        return hook_fn
    
    def apply(self):
        """Apply activation steering hooks to the model."""
        self.remove()
        self.reset()
        
        try:
            layers = self.model.model.model.layers
        except AttributeError:
            layers = self.model.model.layers
        
        for layer_idx in self.steering_vectors.keys():
            if layer_idx < len(layers):
                hook = layers[layer_idx].register_forward_hook(self._steering_hook(layer_idx))
                self.hooks.append(hook)
        
        print(f"  [ActivationSteering]")
        print(f"    Layers: {sorted(self.steering_vectors.keys())}")
        print(f"    Strength: {self.config.act_strength}")
        print(f"    Mode: {self.config.act_mode.value}, n={self.config.act_n}")
    
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
