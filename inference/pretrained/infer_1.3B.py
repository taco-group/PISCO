
import torch
from PIL import Image
import sys
import os
import argparse
import glob
import re
import numpy as np


# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.core import load_state_dict
from inference.utils import load_image_sequence


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="xiangbog/PISCO-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="xiangbog/PISCO-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.safetensors"),
        ModelConfig(model_id="xiangbog/PISCO-1.3B", origin_file_pattern="Wan2.1_VAE.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="xiangbog/PISCO-1.3B", origin_file_pattern="google/umt5-xxl/"),
)

for i in range(5):
    example_id = i + 1
    base_path = f"eval/example{example_id}"
    
    # 1. Load clean videos
    pisco_video = VideoData(video_file=os.path.join(base_path, "clean.mp4"), height=480, width=832)
    pisco_video.set_length(49)
    pisco_depth = VideoData(video_file=os.path.join(base_path, "clean_depth.mp4"), height=480, width=832)
    pisco_depth.set_length(49)

    # 2. Load pre-masked image sequences (sparse)
    pisco_video_mask = load_image_sequence(os.path.join(base_path, "mask"), height=480, width=832)
    pisco_video_mask.set_length(49)
    
    pisco_reference_video = load_image_sequence(os.path.join(base_path, "video_masked"), height=480, width=832)
    pisco_reference_video.set_length(49)
    
    pisco_reference_depth = load_image_sequence(os.path.join(base_path, "video_depth_masked"), height=480, width=832)
    pisco_reference_depth.set_length(49)

    # 3. Create Mask
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
    
    os.makedirs(f"logs/PISCO_1.3B_img", exist_ok=True)
    save_video(video, f"logs/PISCO_1.3B_img/example{i+1}_three_frames.mp4", fps=24, quality=3)
