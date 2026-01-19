"""
DualSteer: Dual-Space Steering for Robust Jailbreak Mitigation of LVLMs

Main entry point for training and evaluating the DualSteer defense framework.

Usage:
    python main.py --model_path /path/to/model --train_samples_file train.csv ...
    
See README.md for detailed usage instructions.
"""

import os
import sys
import argparse
import torch
import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DualSteerConfig, SteeringMode, DATASETS
from src.controller import DualSteerController
from src.evaluate import evaluate_dataset
from src.utils import parse_layers


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DualSteer: Dual-Space Steering for Robust Jailbreak Mitigation"
    )
    
    # Model and data paths
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the LVLM model")
    parser.add_argument("--train_samples_file", type=str, required=True,
                       help="Path to training samples CSV")
    parser.add_argument("--train_image_dir", type=str, required=True,
                       help="Directory containing training images")
    
    # Stage 1: Attention Steering
    parser.add_argument("--attn_c", type=float, default=0.5,
                       help="Attention steering coefficient (default: 0.5)")
    parser.add_argument("--attn_prefill_layers", type=str, default="7-31",
                       help="Layer range for prefill steering, e.g., '7-31' or 'none'")
    parser.add_argument("--attn_decode_layers", type=str, default="none",
                       help="Layer range for decode steering, e.g., '7-31' or 'none'")
    parser.add_argument("--attn_decode_mode", type=str, default="none",
                       choices=["none", "first_only", "first_n", "all"],
                       help="Decode phase steering mode (default: none)")
    parser.add_argument("--attn_decode_n", type=int, default=10,
                       help="Number of tokens for first_n mode (default: 10)")
    
    # Stage 2: Activation Steering
    parser.add_argument("--act_method", type=str, default="svd",
                       choices=["mean_diff", "pca", "svd", "contrastive"],
                       help="Method for computing steering vectors (default: svd)")
    parser.add_argument("--act_strength", type=float, default=1.0,
                       help="Activation steering strength (default: 1.0)")
    parser.add_argument("--act_layers", type=str, default="7-31",
                       help="Layer range for activation steering, e.g., '7-31'")
    parser.add_argument("--act_mode", type=str, default="all",
                       choices=["none", "first_only", "first_n", "all"],
                       help="Decode phase activation steering mode (default: all)")
    parser.add_argument("--act_n", type=int, default=10,
                       help="Number of tokens for first_n mode (default: 10)")
    
    # Training and evaluation
    parser.add_argument("--num_train", type=int, default=50,
                       help="Number of training samples (default: 50)")
    parser.add_argument("--num_test", type=int, default=None,
                       help="Number of test samples per dataset (default: all)")
    parser.add_argument("--datasets", type=str, nargs='+', default=None,
                       help="Datasets to evaluate (default: all)")
    
    # Checkpoints
    parser.add_argument("--load_checkpoint", type=str, default=None,
                       help="Load trained patterns from checkpoint")
    parser.add_argument("--save_checkpoint", type=str, default=None,
                       help="Save trained patterns to checkpoint")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./results",
                       help="Output directory for results")
    
    # Safety evaluation
    parser.add_argument("--use_llamaguard", action="store_true",
                       help="Use LlamaGuard for safety evaluation")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Parse layer specifications
    prefill_layers = parse_layers(args.attn_prefill_layers)
    decode_layers = parse_layers(args.attn_decode_layers)
    act_layers = parse_layers(args.act_layers)
    
    # Create configuration
    config = DualSteerConfig(
        attn_c=args.attn_c,
        attn_prefill_layers=prefill_layers,
        attn_decode_layers=decode_layers,
        attn_decode_mode=SteeringMode(args.attn_decode_mode),
        attn_decode_n=args.attn_decode_n,
        act_method=args.act_method,
        act_strength=args.act_strength,
        act_layers=act_layers,
        act_normalize=True,
        act_mode=SteeringMode(args.act_mode),
        act_n=args.act_n
    )
    
    # Print configuration
    print("\n" + "="*70)
    print("  DualSteer: Dual-Space Steering Framework")
    print("="*70)
    print(f"  Stage 1 (Attention Steering):")
    print(f"    Coefficient: {config.attn_c}")
    print(f"    Prefill layers: {prefill_layers}")
    print(f"    Decode layers: {decode_layers}")
    print(f"    Decode mode: {config.attn_decode_mode.value}")
    print(f"  Stage 2 (Activation Steering):")
    print(f"    Method: {config.act_method}")
    print(f"    Strength: {config.act_strength}")
    print(f"    Layers: {act_layers}")
    print(f"    Mode: {config.act_mode.value}")
    print("="*70)
    
    # Load model
    print("\nLoading model...")
    
    # Import model builder (adjust based on your model implementation)
    try:
        from model import build_model
        model = build_model(args)
    except ImportError:
        print("[ERROR] Could not import model builder. Please ensure the model module is available.")
        print("Expected: from model import build_model")
        sys.exit(1)
    
    # Initialize controller
    controller = DualSteerController(model, config)
    
    # Load or train
    if args.load_checkpoint and os.path.exists(args.load_checkpoint):
        print(f"\nLoading checkpoint from {args.load_checkpoint}")
        controller.load(args.load_checkpoint)
    else:
        print(f"\nLoading training data from {args.train_samples_file}")
        train_df = pd.read_csv(args.train_samples_file)
        
        # Check for required column
        if 'jailbroken_claude' not in train_df.columns:
            print("[ERROR] Training data must have 'jailbroken_claude' column")
            print(f"Available columns: {train_df.columns.tolist()}")
            sys.exit(1)
        
        safe_count = len(train_df[train_df['jailbroken_claude'] == False])
        unsafe_count = len(train_df[train_df['jailbroken_claude'] == True])
        print(f"  Total: {len(train_df)}, Safe: {safe_count}, Unsafe: {unsafe_count}")
        
        # Train both stages
        controller.train_stage1(train_df, args.train_image_dir, args.num_train)
        controller.train_stage2(train_df, args.train_image_dir, args.num_train)
        
        # Save checkpoint if requested
        if args.save_checkpoint:
            os.makedirs(os.path.dirname(args.save_checkpoint) or '.', exist_ok=True)
            controller.save(args.save_checkpoint)
    
    # Determine datasets to evaluate
    if args.datasets:
        datasets_to_test = {k: v for k, v in DATASETS.items() if k in args.datasets}
    else:
        datasets_to_test = DATASETS
    
    print(f"\n" + "="*70)
    print("EVALUATION PHASE")
    print("="*70)
    print(f"  Datasets: {list(datasets_to_test.keys())}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load LlamaGuard if requested
    llama_guard = None
    if args.use_llamaguard:
        print("Loading LlamaGuard...")
        try:
            from model.LlamaGuard import LlamaGuard
            device = "cuda" if torch.cuda.is_available() else "cpu"
            llama_guard = LlamaGuard(device)
        except ImportError:
            print("[WARNING] LlamaGuard not available, skipping safety evaluation")
    
    # Build config dict for logging
    config_dict = {
        "attn_c": args.attn_c,
        "attn_prefill_layers": args.attn_prefill_layers,
        "attn_decode_layers": args.attn_decode_layers,
        "attn_decode_mode": args.attn_decode_mode,
        "attn_decode_n": args.attn_decode_n,
        "act_method": args.act_method,
        "act_strength": args.act_strength,
        "act_layers": args.act_layers,
        "act_mode": args.act_mode,
        "act_n": args.act_n,
    }
    
    # Evaluate each dataset
    summary = {}
    for dataset_name, dataset_config in datasets_to_test.items():
        result = evaluate_dataset(
            controller=controller,
            dataset_name=dataset_name,
            csv_path=dataset_config["csv_path"],
            image_dir=dataset_config["image_dir"],
            output_dir=args.output_dir,
            config_dict=config_dict,
            num_test=args.num_test,
            llama_guard=llama_guard
        )
        if result:
            summary[dataset_name] = result["statistics"]
    
    # Print final summary
    print("\n" + "="*70)
    print("  EVALUATION COMPLETE")
    print("="*70)
    for dataset_name, stats in summary.items():
        print(f"  {dataset_name:10s}: {stats['successful']:4d} successful, {stats['errors']:4d} errors")
    print("="*70)


if __name__ == "__main__":
    main()
