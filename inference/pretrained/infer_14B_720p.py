
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


vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

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
)


for i in range(5):
    pisco_video = VideoData(f"eval/example{i+1}/clean.mp4", height=720, width=1280)
    pisco_video.set_length(49)
    pisco_video_mask = VideoData(f"eval/example{i+1}/mask.mp4", height=720, width=1280)
    pisco_video_mask.set_length(49)
    pisco_reference_video = VideoData(f"eval/example{i+1}/video_masked.mp4", height=720, width=1280)
    pisco_reference_video.set_length(49)
    pisco_depth = VideoData(f"eval/example{i+1}/clean_depth.mp4", height=720, width=1280)
    pisco_depth.set_length(49)
    pisco_reference_depth = VideoData(f"eval/example{i+1}/video_depth_masked.mp4", height=720, width=1280)
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
        height=720,
        width=1280,
        switch_DiT_boundary=0.642,
        num_inference_steps=4,
        num_frames=49,
        seed=1, tiled=False
    )
    os.makedirs(f"logs/PISCO_14B_720p", exist_ok=True)
    save_video(video, f"logs/PISCO_14B_720p/example{i+1}_three_frames.mp4", fps=24, quality=3)
