"""
DualSteer Controller Module

This module provides the main controller that coordinates both
attention steering (Stage 1) and activation steering (Stage 2)
for comprehensive jailbreak mitigation.
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from dataclasses import asdict

from .config import DualSteerConfig
from .attention_extractor import AttentionFeatureExtractor
from .attention_steering import AttentionPatternModeler, AttentionSteering
from .activation_extractor import ActivationExtractor
from .activation_steering import ActivationPatternAnalyzer, ActivationSteering


class DualSteerController:
    """
    DualSteer Defense Controller
    
    Coordinates dual-space steering for robust jailbreak mitigation:
    - Stage 1: Attention Steering - corrects where the model attends
    - Stage 2: Activation Steering - corrects what it extracts
    
    Both stages are training-free and lightweight, requiring only a few
    samples for offline calibration.
    """
    
    def __init__(self, model, config: DualSteerConfig):
        """
        Initialize the DualSteer controller.
        
        Args:
            model: The LVLM model to defend
            config: DualSteer configuration
        """
        self.model = model
        self.config = config
        
        # Stage 1 components
        self.attn_extractor = None
        self.attn_modeler = None
        self.attn_steering = None
        
        # Stage 2 components
        self.act_extractor = None
        self.act_analyzer = None
        self.act_steering = None
        
        # State flags
        self.stage1_ready = False
        self.stage2_ready = False
    
    def train_stage1(self, train_df: pd.DataFrame, image_dir: str, num_train: int = 50) -> bool:
        """
        Train Stage 1: Learn attention patterns from refusal samples.
        
        Args:
            train_df: DataFrame with columns 'image_path', 'jailbreak_query', 'jailbroken_claude'
            image_dir: Directory containing images
            num_train: Number of training samples
            
        Returns:
            True if training succeeded
        """
        print("\n" + "="*70)
        print("TRAINING STAGE 1: Attention Pattern Learning")
        print("="*70)
        
        # Select refusal samples (defense succeeded)
        safe_df = train_df[train_df["jailbroken_claude"] == False].head(num_train)
        print(f"Using {len(safe_df)} refusal samples")
        
        # Get target layers
        target_layers = self.config.get_all_attn_layers()
        if not target_layers:
            print("[Warning] No attention layers specified, skipping Stage 1")
            return False
        
        print(f"Target layers: {target_layers}")
        
        self.attn_extractor = AttentionFeatureExtractor(
            self.model, self.config.num_img_tokens, target_layers)
        
        # Collect attention ratio samples
        ratio_samples_per_layer = {layer_idx: [] for layer_idx in target_layers}
        
        for _, row in tqdm(safe_df.iterrows(), total=len(safe_df), desc="Extracting attention"):
            try:
                img_path = os.path.join(image_dir, row["image_path"])
                image = Image.open(img_path).convert('RGB')
                
                ratios_dict = self.attn_extractor.extract(image, row["jailbreak_query"])
                
                if ratios_dict is not None:
                    for layer_idx, ratios in ratios_dict.items():
                        ratio_samples_per_layer[layer_idx].append(
                            ratios.squeeze(0).cpu().numpy()
                        )
            except Exception as e:
                continue
        
        # Validate samples
        valid_layers = {}
        for layer_idx, samples in ratio_samples_per_layer.items():
            if len(samples) >= 5:
                valid_layers[layer_idx] = np.stack(samples)
                print(f"  Layer {layer_idx}: {len(samples)} samples")
            else:
                print(f"  Layer {layer_idx}: only {len(samples)} samples (skipped)")
        
        if len(valid_layers) == 0:
            print("[Error] No layer has enough samples")
            return False
        
        # Fit pattern modeler
        self.attn_modeler = AttentionPatternModeler()
        self.attn_modeler.fit(valid_layers)
        
        # Create steering component
        self.attn_steering = AttentionSteering(self.model, self.attn_modeler, self.config)
        self.stage1_ready = True
        return True
    
    def train_stage2(self, train_df: pd.DataFrame, image_dir: str, num_train: int = 50) -> bool:
        """
        Train Stage 2: Learn activation directions from success/failure contrast.
        
        Args:
            train_df: DataFrame with columns 'image_path', 'jailbreak_query', 'jailbroken_claude'
            image_dir: Directory containing images
            num_train: Number of training samples per class
            
        Returns:
            True if training succeeded
        """
        print("\n" + "="*70)
        print("TRAINING STAGE 2: Activation Pattern Learning")
        print("="*70)
        
        # Select contrasting samples
        success_df = train_df[train_df["jailbroken_claude"] == False].head(num_train)
        fail_df = train_df[train_df["jailbroken_claude"] == True].head(num_train)
        
        print(f"Success (defense worked): {len(success_df)}")
        print(f"Failed (jailbroken): {len(fail_df)}")
        
        act_layers = self.config.act_layers or list(range(27, 32))
        self.act_extractor = ActivationExtractor(self.model, act_layers)
        
        # Extract success activations
        success_acts = {idx: [] for idx in act_layers}
        for _, row in tqdm(success_df.iterrows(), total=len(success_df), desc="Success samples"):
            try:
                img_path = os.path.join(image_dir, row["image_path"])
                acts = self.act_extractor.extract(img_path, row["jailbreak_query"])
                for idx, act in acts.items():
                    success_acts[idx].append(act.numpy())
            except:
                continue
        
        # Extract failure activations
        fail_acts = {idx: [] for idx in act_layers}
        for _, row in tqdm(fail_df.iterrows(), total=len(fail_df), desc="Failed samples"):
            try:
                img_path = os.path.join(image_dir, row["image_path"])
                acts = self.act_extractor.extract(img_path, row["jailbreak_query"])
                for idx, act in acts.items():
                    fail_acts[idx].append(act.numpy())
            except:
                continue
        
        # Stack arrays
        for idx in act_layers:
            if success_acts[idx] and fail_acts[idx]:
                success_acts[idx] = np.vstack(success_acts[idx])
                fail_acts[idx] = np.vstack(fail_acts[idx])
                print(f"Layer {idx}: success={success_acts[idx].shape}, fail={fail_acts[idx].shape}")
        
        # Compute steering vectors
        self.act_analyzer = ActivationPatternAnalyzer(
            method=self.config.act_method, normalize=self.config.act_normalize)
        steering_vectors = self.act_analyzer.compute_steering_vectors(success_acts, fail_acts)
        
        # Create steering component
        self.act_steering = ActivationSteering(self.model, steering_vectors, self.config)
        self.stage2_ready = True
        return True
    
    def inference_original(self, image_path: str, prompt: str) -> str:
        """Run inference without any steering (baseline)."""
        self._remove_all()
        return self.model.inference(image_path, prompt)
    
    def inference_stage1(self, image_path: str, prompt: str) -> Tuple[str, dict]:
        """Run inference with Stage 1 (attention steering) only."""
        self._remove_all()
        
        if not self.stage1_ready:
            return self.model.inference(image_path, prompt), {}
        
        image = Image.open(image_path).convert('RGB')
        _, input_ids = self.model.prepare_inputs(image, prompt)
        boundaries = AttentionFeatureExtractor.detect_boundaries(
            input_ids, self.config.num_img_tokens)
        
        if boundaries[0] is None:
            return self.model.inference(image_path, prompt), {}
        
        self.attn_steering.set_boundaries(boundaries)
        self.attn_steering.apply()
        response = self.model.inference(image_path, prompt)
        
        info = self.attn_steering.get_stats()
        self.attn_steering.remove()
        return response, info
    
    def inference_stage2(self, image_path: str, prompt: str) -> Tuple[str, dict]:
        """Run inference with Stage 2 (activation steering) only."""
        self._remove_all()
        
        if not self.stage2_ready:
            return self.model.inference(image_path, prompt), {}
        
        self.act_steering.apply()
        response = self.model.inference(image_path, prompt)
        info = self.act_steering.get_stats()
        self.act_steering.remove()
        return response, info
    
    def inference_both(self, image_path: str, prompt: str) -> Tuple[str, dict]:
        """Run inference with both stages (full DualSteer)."""
        self._remove_all()
        
        info = {'attn': {}, 'act': {}}
        
        # Apply Stage 1
        if self.stage1_ready:
            image = Image.open(image_path).convert('RGB')
            _, input_ids = self.model.prepare_inputs(image, prompt)
            boundaries = AttentionFeatureExtractor.detect_boundaries(
                input_ids, self.config.num_img_tokens)
            
            if boundaries[0] is not None:
                self.attn_steering.set_boundaries(boundaries)
                self.attn_steering.apply()
        
        # Apply Stage 2
        if self.stage2_ready:
            self.act_steering.apply()
        
        # Generate response
        response = self.model.inference(image_path, prompt)
        
        # Collect statistics
        if self.stage1_ready:
            info['attn'] = self.attn_steering.get_stats()
        if self.stage2_ready:
            info['act'] = self.act_steering.get_stats()
        
        self._remove_all()
        return response, info
    
    def _remove_all(self):
        """Remove all steering hooks."""
        if self.attn_steering:
            self.attn_steering.remove()
        if self.act_steering:
            self.act_steering.remove()
    
    def save(self, path: str):
        """
        Save trained patterns to disk.
        
        Args:
            path: Output file path (pickle format)
        """
        data = {
            'version': 'dualsteer_v1.0',
            'config': asdict(self.config),
            'attn_head_means_per_layer': self.attn_modeler.head_means_per_layer if self.attn_modeler else None,
            'attn_last_layer_head_means': self.attn_modeler.last_layer_head_means if self.attn_modeler else None,
            'attn_last_layer_idx': self.attn_modeler.last_layer_idx if self.attn_modeler else None,
            'act_steering_vectors': self.act_analyzer.steering_vectors if self.act_analyzer else None
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved to {path}")
    
    def load(self, path: str):
        """
        Load trained patterns from disk.
        
        Args:
            path: Input file path (pickle format)
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        # Load Stage 1
        if data.get('attn_head_means_per_layer'):
            self.attn_modeler = AttentionPatternModeler()
            self.attn_modeler.head_means_per_layer = data['attn_head_means_per_layer']
            self.attn_modeler.last_layer_head_means = data.get('attn_last_layer_head_means')
            self.attn_modeler.last_layer_idx = data.get('attn_last_layer_idx')
            self.attn_modeler.fitted = True
            self.attn_steering = AttentionSteering(self.model, self.attn_modeler, self.config)
            self.stage1_ready = True
            print(f"Loaded Stage 1: {len(self.attn_modeler.head_means_per_layer)} layers")
        
        # Load Stage 2
        if data.get('act_steering_vectors'):
            self.act_steering = ActivationSteering(
                self.model, data['act_steering_vectors'], self.config)
            self.stage2_ready = True
            print(f"Loaded Stage 2: {len(data['act_steering_vectors'])} layers")
        
        print(f"Loaded from {path}")
