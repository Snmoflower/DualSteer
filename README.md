# DualSteer: Dual-Space Steering for Robust Jailbreak Mitigation of LVLMs

This repository contains the implementation of **DualSteer**, a lightweight and training-free defense framework that protects Large Vision-Language Models (LVLMs) against multimodal jailbreak attacks through dual-space steering.

## Overview

DualSteer addresses the *attention distraction* phenomenon where jailbreak inputs hijack the model's attention resources, shifting focus from safety reasoning toward malicious content. Our approach provides coordinated defense through two stages:

1. **Stage 1: Attention Steering** - Recalibrates attention distributions to restore focus on safety-relevant system tokens
2. **Stage 2: Hidden Representation Steering** - Steers hidden states toward the refusal manifold using contrastive SVD

## Key Features

- **Training-free**: No model fine-tuning required
- **Lightweight**: Minimal computational overhead (~20ms per inference)
- **Robust**: Effective against diverse multimodal jailbreak attacks
- **Utility-preserving**: Maintains model performance on benign tasks

## Installation

### Environment Setup
Different models require different environments. We provide setup instructions for LLaVA as an example:

```bash
# Clone LLaVA repository
git clone <LLaVA official repository link>
cd LLaVA

# Create conda environment
conda create -n llava python=3.10 -y
conda activate llava

# Install dependencies following LLaVA's official instructions
pip install --upgrade pip
pip install -e .
```
For other models (MiniGPT-4, Qwen2.5-VL), please follow their respective official installation guides.

## Project Structure

```
dualsteer/
├── src/
│   ├── __init__.py              # Module exports
│   ├── config.py                # Configuration classes
│   ├── attention_extractor.py   # Attention feature extraction
│   ├── attention_steering.py    # Stage 1: Attention steering
│   ├── activation_extractor.py  # Activation feature extraction
│   ├── activation_steering.py   # Stage 2: Activation steering
│   ├── controller.py            # Main DualSteer controller
│   ├── evaluate.py              # Evaluation utilities
│   └── utils.py                 # Helper functions
├── main.py                      # Entry point
└── README.md                    # This file
```

## Quick Start

### 1. Prepare Calibration Data

Prepare a CSV file with the following columns:
- `image_path`: Relative path to the image
- `jailbreak_query`: The query/prompt text
- `jailbroken_claude`: Boolean indicating if the model was jailbroken (True) or refused (False)

### 2. Calibrate and Evaluate

```bash
python main.py \
    --model_path /path/to/llava-v1.5-7b \
    --train_samples_file data/train_samples.csv \
    --train_image_dir data/images \
    --attn_c 1.0 \
    --act_strength 1.0 \
    --attn_prefill_layers 7-31 \
    --act_layers 7-31 \
    --attn_decode_mode none \
    --act_mode all \
    --num_train 50 \
    --output_dir results \
    --datasets figstep qrel jood bap umk hades
```

### 3. Load from Checkpoint

```bash
python main.py \
    --model_path /path/to/model \
    --load_checkpoint checkpoints/dualsteer.pkl \
    --datasets figstep \
    --output_dir results
```

## Configuration

### Stage 1: Attention Steering

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--attn_c` | Attention steering coefficient | 1.0 |
| `--attn_prefill_layers` | Layers for prefill steering | "7-31" |
| `--attn_decode_layers` | Layers for decode steering | "none" |
| `--attn_decode_mode` | Decode mode: none/first_only/first_n/all | "none" |
| `--attn_decode_n` | Number of tokens for first_n mode | 10 |

### Stage 2: Activation Steering

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--act_method` | Steering vector method: mean_diff/pca/svd/contrastive | "svd" |
| `--act_strength` | Activation steering strength | 1.0 |
| `--act_layers` | Layers for activation steering | "7-31" |
| `--act_mode` | Decode mode: none/first_only/first_n/all | "all" |


## Datasets

The framework supports evaluation on various jailbreak benchmarks:
- **FigStep**: Typographic attacks
- **Q-Rel**: Query-relevant image attacks
- **BAP**: Bi-modal adversarial prompts
- **UMK**: Universal multimodal attacks
- **HADES**: Optimization-based attacks
- **JOOD**: Out-of-distribution attacks

Update the dataset paths in `src/config.py` or provide custom paths.

## Results

The ASR evaluated by Llama-Guard-3-8B.
| Model | Method | FigStep | Q-Rel | BAP | UMK | HADES | JOOD | Avg |
|-------|--------|---------|-------|-----|-----|-------|------|-----|
| LLaVA-1.5-7B | Vanilla | 0.383 | 0.567 | 0.400 | 0.983 | 0.750 | 0.900 | 0.664 |
| LLaVA-1.5-7B | **DualSteer** | **0.050** | **0.133** | **0.100** | **0.017** | **0.317** | **0.050** | **0.111** |


## Acknowledgments

We thank the developers of LLaVA, MiniGPT-4, and Qwen for their open-source contributions.
