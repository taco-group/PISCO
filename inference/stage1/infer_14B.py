
import torch
from PIL import Image
import sys
import os
import argparse
import glob
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.core import load_state_dict

def get_checkpoint_path(stage_dir, step=None):
    if not os.path.exists(stage_dir):
        raise FileNotFoundError(f"Stage directory not found: {stage_dir}")

    if step is not None:
        ckpt_path = os.path.join(stage_dir, f"step-{step}.safetensors")
        if not os.path.exists(ckpt_path):
            # Fallback: Check if user provided different steps for high/low via separate args?
            # For now, strict checking.
            raise FileNotFoundError(f"Checkpoint for step {step} not found at {ckpt_path}")
        return ckpt_path, step

    # Find max step
    files = glob.glob(os.path.join(stage_dir, "step-*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No checkponits found in {stage_dir}")

    max_step = -1
    for f in files:
        match = re.search(r'step-(\d+)\.safetensors', os.path.basename(f))
        if match:
            s = int(match.group(1))
            if s > max_step:
                max_step = s
    
    if max_step == -1:
        raise ValueError(f"Could not parse step numbers from files in {stage_dir}")
        
    return os.path.join(stage_dir, f"step-{max_step}.safetensors"), max_step

def get_clean_state_dict(ckpt_path):
    print(f"Loading checkpoint: {ckpt_path}")
    state_dict = load_state_dict(ckpt_path)
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove prefixes introduced by training wrapper if present
        # Assuming training script wraps model components with 'pipe.pisco.' or 'pipe.dit.'
        # Or if saved directly from accelerate, it might be clean or have 'module.'
        # Let's clean standard prefixes.
        new_k = k.replace("pipe.pisco.", "").replace("pipe.dit.", "").replace("module.", "")
        new_state_dict[new_k] = v
    return new_state_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=None, help="Specific step to load. If not provided, loads the latest.")
    args = parser.parse_args()

    # Paths
    high_noise_dir = "models/train/PISCO-14B-high-noise_stage1"
    low_noise_dir = "models/train/PISCO-14B-low-noise_stage1"
    
    # Get checkpoints
    ckpt_high, step_high = get_checkpoint_path(high_noise_dir, args.step)
    ckpt_low, step_low = get_checkpoint_path(low_noise_dir, args.step)

    print(f"Detected High Noise Checkpoint: {ckpt_high} (Step {step_high})")
    print(f"Detected Low Noise Checkpoint: {ckpt_low} (Step {step_low})")

    device = "cuda"
    vram_config = {
        "offload_dtype": torch.bfloat16,
        "offload_device": "cpu",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cpu",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": device,
        "computation_dtype": torch.bfloat16,
        "computation_device": device,
    }

    print("Initializing Pipeline...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id="PAI/Wan2.2-VACE-Fun-A14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **vram_config),
            ModelConfig(model_id="PAI/Wan2.2-VACE-Fun-A14B", origin_file_pattern="Wan2.1_VAE.pth", **vram_config),
            ModelConfig(path="models/PISCO/inits/PISCO-14B/PISCO-14B-high-noise.safetensors", **vram_config),
            ModelConfig(path="models/PISCO/inits/PISCO-14B/PISCO-14B-low-noise.safetensors", **vram_config),
        ],
        tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
        vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
    )

    # Load State Dicts
    sd_high = get_clean_state_dict(ckpt_high)
    sd_low = get_clean_state_dict(ckpt_low)

    print("Loading state dicts into models...")
    pipe.pisco.pisco_patch_embedding.load_state_dict(sd_high, strict=True)
    pipe.pisco2.pisco_patch_embedding.load_state_dict(sd_low, strict=True)

    os.makedirs(f"logs/PISCO/PISCO-14B_stage1", exist_ok=True)

    for i in range(5):
        print(f"Processing example {i+1}...")
        pisco_video = VideoData(f"eval/example{i+1}/clean.mp4", height=480, width=832)
        pisco_video.set_length(49)
        pisco_video_mask = VideoData(f"eval/example{i+1}/mask.mp4", height=480, width=832)
        pisco_video_mask.set_length(49)
        pisco_reference_video = VideoData(f"eval/example{i+1}/video_masked.mp4", height=480, width=832)
        pisco_reference_video.set_length(49)
        pisco_depth = VideoData(f"eval/example{i+1}/clean_depth.mp4", height=480, width=832)
        pisco_depth.set_length(49)
        pisco_reference_depth = VideoData(f"eval/example{i+1}/video_depth_masked.mp4", height=480, width=832)
        pisco_reference_depth.set_length(49)

        mask = torch.zeros(49, dtype=torch.bool).to(pipe.device)
        mask[[10, 20, 46]] = True

        video = pipe(
            prompt="",
            negative_prompt="",
            pisco_video=pisco_video,
            pisco_video_mask=(pisco_video_mask, mask),
            pisco_reference_video=(pisco_reference_video, mask),
            pisco_depth=(pisco_depth, mask),
            pisco_reference_depth=(pisco_reference_depth, mask),
            num_frames=49,
            seed=1, tiled=False
        )
        
        output_path = f"logs/PISCO/PISCO-14B_stage1/step-high_{step_high}_step-low_{step_low}_example{i+1}_three_frames_10_20_46_serial.mp4"
        save_video(video, output_path, fps=15, quality=5)
        print(f"Saved to {output_path}")

    print("Inference completed.")

if __name__ == "__main__":
    main()
