
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


parser = argparse.ArgumentParser()
parser.add_argument("--step", type=int, default=None, help="Specific step to load. If not provided, loads the latest.")
args = parser.parse_args()

stage1_dir = "models/train/PISCO-1.3B_stage2"
checkpoint_path, step = get_checkpoint_path(stage1_dir, args.step)
print(f"Loading checkpoint from: {checkpoint_path}")

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        # ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="Wan2.1_VAE.pth"),
        ModelConfig(path="models/PISCO/inits/PISCO-1.3B/PISCO-1.3B.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
)

state_dict = load_state_dict(checkpoint_path)
pipe.pisco.load_state_dict(state_dict)


for i in range(5):
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

    # Keep only 3 frames for reference
    mask = torch.zeros(49, dtype=torch.bool).to(pipe.device)
    mask[[10,20,46]] = True

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
    os.makedirs(f"logs/PISCO/PISCO-1.3B_stage2", exist_ok=True)
    save_video(video, f"logs/PISCO/PISCO-1.3B_stage2/step-{step}_example{i+1}_three_frames_10_20_46.mp4", fps=15, quality=5)
