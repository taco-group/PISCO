
import torch
from PIL import Image
import sys
import os
import argparse
import glob
import re
import numpy as np

# Add parent directory to path

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.core import load_state_dict
from inference.utils import load_image_sequence, vram_config


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="xiangbog/PISCO-14B", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id="xiangbog/PISCO-14B", origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id="xiangbog/PISCO-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.safetensors", **vram_config),
        ModelConfig(model_id="xiangbog/PISCO-14B", origin_file_pattern="Wan2.1_VAE.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="xiangbog/PISCO-14B", origin_file_pattern="google/umt5-xxl/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
)


for i in range(5):
    example_id = i + 1
    base_path = f"eval/example{example_id}"
    
    pisco_video = VideoData(video_file=os.path.join(base_path, "clean.mp4"), height=720, width=1280)
    pisco_video.set_length(49)
    pisco_depth = VideoData(video_file=os.path.join(base_path, "clean_depth.mp4"), height=720, width=1280)
    pisco_depth.set_length(49)

    pisco_video_mask = load_image_sequence(os.path.join(base_path, "mask"), height=720, width=1280)
    pisco_video_mask.set_length(49)
    
    pisco_reference_video = load_image_sequence(os.path.join(base_path, "video_masked"), height=720, width=1280)
    pisco_reference_video.set_length(49)
    
    pisco_reference_depth = load_image_sequence(os.path.join(base_path, "video_depth_masked"), height=720, width=1280)
    pisco_reference_depth.set_length(49)

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
        height=720,
        width=1280,
        switch_DiT_boundary=0.642,
        num_frames=49,
        seed=1, tiled=False
    )
    os.makedirs(f"logs/PISCO_14B_720p_img", exist_ok=True)
    save_video(video, f"logs/PISCO_14B_720p_img/example{i+1}_three_frames.mp4", fps=24, quality=3)
