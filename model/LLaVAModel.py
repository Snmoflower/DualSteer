import sys
import os
import json
import math

import torch
from torch import nn
import numpy as np
from io import BytesIO
from typing import List, Tuple, Dict

from transformers import TextStreamer
from transformers.generation import BeamSearchDecoderOnlyOutput

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from model.base import LargeMultimodalModel

class LLaVA(LargeMultimodalModel):
    def __init__(self, model_path: str):
        """Initialize LLaVA model.
        
        Args:
            model_path: Path to pretrained LLaVA model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model configuration
        disable_torch_init()
        model_name = get_model_name_from_path(model_path)
        
        # Determine model base
        if "finetune-lora" in model_path:
            model_base = "liuhaotian/llava-v1.5-7b"
        elif "lora" in model_path:
            model_base = "lmsys/vicuna-7b-v1.5"
        else:
            model_base = None
            
        # Load model components
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, model_name, load_8bit=False, load_4bit=False
        )
        
        # Set conversation mode
        self.conv_mode = "llava_v1"
        self.conv = conv_templates[self.conv_mode].copy()
        
    def prepare_inputs(self, image, prompt: str):
        """Prepare model inputs from image and text prompt."""
        # Process image
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values']
        image_tensor = image_tensor.unsqueeze(0).half().to(self.device)
        
        # Prepare prompt with image tokens
        if self.model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
            
        # Update conversation
        self.conv = conv_templates[self.conv_mode].copy()
        self.conv.append_message(self.conv.roles[0], inp)
        self.conv.append_message(self.conv.roles[1], None)
        conv_prompt = self.conv.get_prompt()
        
        # Tokenize
        input_ids = tokenizer_image_token(
            conv_prompt, 
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors='pt'
        ).unsqueeze(0).to(self.device)

        #print(f"Input IDs shape: {input_ids.shape}")
        
        return image_tensor, input_ids
    
    def inference(self, image_path, prompt):
        from PIL import Image
        import torch

        # Load image
        image = Image.open(image_path)

        image_tensor, input_ids = self.prepare_inputs(image, prompt)

        # Inference with model
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                images=image_tensor,
                max_new_tokens=1024,
                use_cache=True
            )

        # Decode the output
        outputs = self.tokenizer.decode(
            output[0][len(input_ids)-1:], 
            skip_special_tokens=True
        )

        return outputs

          
    

    def get_hidden_states(self, image, prompt: str) -> torch.Tensor:
        """Get hidden states from each layer."""
        # Prepare inputs
        image_tensor, input_ids = self.prepare_inputs(image, prompt)
        
        # Use forward function directly with output_hidden_states=True
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                images=image_tensor,
                return_dict=True,
                output_hidden_states=True
            )
        
        # outputs.hidden_states contains hidden states from all layers, including embedding layer output
        # To maintain consistency with previous results, we take the last token position from each layer
        hidden_states_list = []
        
        # Start from layer 1 (skip embedding layer output)
        for layer_output in outputs.hidden_states[1:]:
            # Get hidden states at the last token position
            last_token_states = layer_output[:, -1, :]
            hidden_states_list.append(last_token_states)
        
        # Stack hidden states from all layers [num_layers, batch_size, hidden_size]
        hidden_states_tensor = torch.stack(hidden_states_list)
        
        return hidden_states_tensor
           

    def get_attention_weights(self, image, prompt: str) -> torch.Tensor:
        """
        Get attention weights from each layer's self-attention module.
        
        Returns:
            torch.Tensor: Shape [num_layers, num_heads, seq_len, seq_len]
                        Attention weights matrices for each layer
        """
        # Prepare inputs
        image_tensor, input_ids = self.prepare_inputs(image, prompt)
        
        attention_weights_list = []
        hooks = []
        
        def create_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple) and len(output) > 1:
                    attn_weights = output[1]  # [batch_size, num_heads, seq_len, seq_len]
                    if attn_weights is not None:
                        attention_weights_list.append(attn_weights.detach())
                
                return output
            
            return hook_fn
        
        for idx, layer in enumerate(self.model.model.layers):
            hook = layer.self_attn.register_forward_hook(create_hook(idx))
            hooks.append(hook)
        
        # Forward pass with output_attentions=True
        with torch.no_grad():
            _ = self.model(
                input_ids,
                images=image_tensor,
                output_attentions=True, 
                return_dict=True
            )
        
        # 清理 hooks
        for hook in hooks:
            hook.remove()
        
        # Stack attention weights from all layers
        # Shape: [num_layers, batch_size, num_heads, seq_len, seq_len]
        if attention_weights_list:
            attention_weights_tensor = torch.stack(attention_weights_list)
            # Squeeze batch dimension if batch_size=1
            if attention_weights_tensor.shape[1] == 1:
                attention_weights_tensor = attention_weights_tensor.squeeze(1)
                # Shape: [num_layers, num_heads, seq_len, seq_len]
            return attention_weights_tensor
        else:
            raise RuntimeError("No attention weights captured. Check if output_attentions=True is supported.")


    def get_attention_weights_last_token(self, image, prompt: str) -> torch.Tensor:
        """
        Get attention weights for the last token position only (more efficient).
        
        Returns:
            torch.Tensor: Shape [num_layers, num_heads, seq_len]
                        Last row of attention matrix for each layer
        """
        full_weights = self.get_attention_weights(image, prompt)
        # Extract last token's attention distribution
        # full_weights: [num_layers, num_heads, seq_len, seq_len]
        last_token_attn = full_weights[:, :, -1, :]  # [num_layers, num_heads, seq_len]
        return last_token_attn