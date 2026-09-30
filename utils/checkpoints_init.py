import torch
import argparse
import os
import sys
from safetensors.torch import save_file
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from diffsynth import load_state_dict

def linear_init_pisco(source_ckpt_path, save_path, new_in_dim=None, from_model="vace_", to_model="pisco_"):
    print(f"Loading source: {source_ckpt_path}")
    source_sd = load_state_dict(source_ckpt_path, device="cpu")
    new_sd = {}
    for k, v in source_sd.items():
        if from_model in k:
            new_k = k.replace(from_model, to_model) # Rename Key
            # Check if it is the input layer weight (usually patch_embedding.weight or input_proj.weight)
            if "patch_embedding.weight" in new_k and new_in_dim is not None:
                if v.shape[1] != new_in_dim:
                    print(f"Detected specific input layer mismatch: {k}")
                    print(f"  Old Shape: {v.shape} (Assuming 96 channels)")
                    print(f"  New Shape Target: {new_in_dim} channels (64)")
                    
                    # 1. Create a zero-filled new weight container
                    # Use Zero Init as base, so Depth (33-64) is automatically 0
                    new_shape = list(v.shape)
                    new_shape[1] = new_in_dim
                    new_weight = torch.zeros(new_shape, dtype=v.dtype)
                    
                    # 2. Apply surgical initialization strategy
                    # Ensure source dimension is 96
                    if v.shape[1] == 96 and new_in_dim == 132:
                        print("  Applying 'Surgical' Initialization Strategy...")
                        
                        # --- Part A: Video [0:16] ---
                        # Strategy: Average Fusion (Old FG + Old BG) / 2
                        
                        old_fg = v[:, 0:16]  # Old FG
                        old_bg = v[:, 16:32] # Old BG (indices 16-31)
                        
                        new_weight[:, 0:16] = (old_fg + old_bg) * 0.5
                        print("    -> [0:16] New Video initialized with avg(Old_FG, Old_BG)")

                        # --- Part B: Reference Foreground [17:32] ---
                        # Strategy: Direct Reuse (Copy Old FG)
                        new_weight[:, 16:32] = old_fg
                        print("    -> [16:32] New Ref_FG initialized with copy(Old_FG)")

                        # --- Part C: Depth & Depth FG [33:64] ---
                        # Strategy: Zero Initialization
                        print("    -> [32:64] New Depth features initialized with ZEROS")

                        # --- Part D: Mask [65:128] ---
                        # Strategy: Zero Initialization
                        new_weight[:, 64:128] = v[:, 32:96]
                        print("    -> [64:128] New Mask features initialized with copy(Old_Mask)")

                        # --- Part E: Temporal Mask [129:132] ---
                        # Strategy: Zero Initialization
                        print("    -> [128:132] New Temporal Mask features initialized with ZEROS")
                        
                    else:
                        # If dimension mismatch, fallback to Xavier initialization
                        print(f"  Warning: Dimensions match failed (expected 96->132, got {v.shape[1]}->{new_in_dim}). Fallback to Xavier.")
                        torch.nn.init.xavier_uniform_(new_weight)

                    new_sd[new_k] = new_weight
                    continue
            
            # Process other layers
            new_sd[new_k] = v
            
        else:
            # Keep non-vace_ base weights
            new_sd[k] = v 
            
    save_file(new_sd, save_path)
    print(f"Conversion Finished! Saved to: {save_path}")

# Default Configurations
MODEL_CONFIGS = {
    "1.3B": {
        "source": "models/Wan-AI/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors",
        "save_name": "PISCO-1.3B.safetensors",
        "save_dir": "models/PISCO/inits/PISCO-1.3B"
    },
    "14B-low": {
        "source": "models/PAI/Wan2.2-VACE-Fun-A14B/low_noise_model/diffusion_pytorch_model.safetensors",
        "save_name": "PISCO-14B-low-noise.safetensors",
        "save_dir": "models/PISCO/inits/PISCO-14B"
    },
    "14B-high": {
        "source": "models/PAI/Wan2.2-VACE-Fun-A14B/high_noise_model/diffusion_pytorch_model.safetensors",
        "save_name": "PISCO-14B-high-noise.safetensors",
        "save_dir": "models/PISCO/inits/PISCO-14B"
    }
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize PISCO model checkpoints.")
    
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        choices=list(MODEL_CONFIGS.keys()),
        help="Model configuration to initialize: " + ", ".join(MODEL_CONFIGS.keys())
    )
    
    parser.add_argument(
        "--source_ckpt_path", 
        type=str, 
        default=None, 
        help="Override source checkpoint path."
    )
    
    parser.add_argument(
        "--save_path", 
        type=str, 
        default=None, 
        help="Override save path (full path including filename)."
    )
    
    parser.add_argument(
        "--dim",
        type=int,
        default=132,
        help="New input dimension (default: 132)"
    )

    args = parser.parse_args()
    
    config = MODEL_CONFIGS[args.model]
    
    # Determine Source Path
    source_path = args.source_ckpt_path if args.source_ckpt_path else config["source"]
    
    # Determine Save Path
    if args.save_path:
        final_save_path = args.save_path
    else:
        # Construct default save path
        final_save_path = os.path.join(config["save_dir"], config["save_name"])
    
    print(f"Configuration: {args.model}")
    print(f"  Source: {source_path}")
    print(f"  Target: {final_save_path}")
    print(f"  Dim:    {args.dim}")

    # Ensure save directory exists
    save_dir = os.path.dirname(final_save_path)
    if not os.path.exists(save_dir):
        print(f"Creating directory: {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
    
    if not os.path.exists(source_path):
        print(f"Warning: Source file does not exist: {source_path}")
        # Not raising error to allow dry-run inspection of paths, 
        # but the load_state_dict will fail if file is missing.

    linear_init_pisco(
        source_ckpt_path=source_path, 
        save_path=final_save_path,
        new_in_dim=args.dim
    )