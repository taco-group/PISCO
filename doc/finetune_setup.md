# PISCO Fine-tuning Setup for Instance Insertion

## Overview

This document describes the setup for fine-tuning the PISCO model on a video instance insertion dataset, where the model learns to insert object instances into clean background videos given a single reference frame.

## Changes Made

### New Files

| File | Description |
|------|-------------|
| `utils/preprocess_data.py` | Preprocesses raw dataset into training-ready video files |
| `utils/data_operators.py` | Custom data loading operators for deterministic temporal masking |
| `utils/generate_dataset_json.py` | Generates metadata JSON for the training pipeline |
| `training/finetune/PISCO-1.3B.sh` | Training launch script for 1.3B model fine-tuning |
| `training/finetune/PISCO-14B.sh` | Training launch script for 14B model fine-tuning (DeepSpeed ZeRO-2) |
| `training/accelerate_config_14B_7gpu.yaml` | Accelerate/DeepSpeed config for 14B training on 7 GPUs |

### Modified Files

| File | Change |
|------|--------|
| `training/train.py` | Added `--pisco_first_frame_only` flag and import of custom operators. When enabled, uses `LoadVideoKeepFirstFrame` for `pisco_reference_video` and `pisco_reference_depth` (temporal mask = first frame only), and `LoadVideoAllFrames` for `pisco_video_mask` and `pisco_depth` (all frames valid). Also added `--tokenizer_path` support for `model_id:pattern` format. No changes to diffsynth library code. |
| `diffsynth/configs/model_configs.py` | Added hash entries for 14B PISCO model (`0eb85c72daf854598632637381da680d`) which includes `vace_blocks` and `vace_patch_embedding` keys alongside the PISCO keys. |

### No Changes to diffsynth/

The core diffsynth library remains unmodified. Custom data operators are defined in `utils/data_operators.py` and injected via `train.py`'s `special_operator_map`.

## Data Pipeline

### Raw Data Structure

```
Dataset/data/
├── clean/           # Clean background videos (1280x720, 120 frames)
│   └── {ID}_removal.mp4
├── edited/          # Videos with inserted instances (ground truth)
│   └── {ID}.mp4
├── depth/           # Depth map videos
│   └── {ID}_removal_depth.mp4
└── instance/        # Per-sample instance segmentation
    └── {ID}png_{hash}_{date}_{time}_{hash2}/
        ├── combined_cutout.png   # RGBA cutout of instances
        └── combined_mask.png     # Binary mask (grayscale)
```

### Preprocessing

```bash
python utils/preprocess_data.py --dataset_dir Dataset/data --num_frames 21
```

This generates:

```
Dataset/data/processed/
├── reference_video/{ID}.mp4   # Frame 0 = RGB cutout (black bg), rest = black
├── mask_video/{ID}.mp4        # All frames = spatial mask (white on black)
└── reference_depth/{ID}.mp4   # Frame 0 = depth masked by instance mask, rest = black
```

Processing details:
- **reference_video**: `combined_cutout.png` (RGBA) → RGB with black background (alpha compositing), resized to video resolution (1280x720), placed as frame 0, remaining frames are black
- **mask_video**: `combined_mask.png` (grayscale) → RGB (white=instance, black=background), same mask repeated for all frames
- **reference_depth**: First frame of depth video masked by `combined_mask.png`, remaining frames are black

### Dataset JSON

```bash
python utils/generate_dataset_json.py --dataset_dir Dataset/data --repeat 10
```

Generates `Dataset/data/pisco_finetune.json` with entries like:

```json
{
  "video": "edited/{ID}.mp4",
  "pisco_video": "clean/{ID}_removal.mp4",
  "pisco_reference_video": "processed/reference_video/{ID}.mp4",
  "pisco_video_mask": "processed/mask_video/{ID}.mp4",
  "pisco_depth": "depth/{ID}_removal_depth.mp4",
  "pisco_reference_depth": "processed/reference_depth/{ID}.mp4",
  "prompt": ""
}
```

### Data Flow During Training

| Input Key | Source | Role | Temporal Mask |
|-----------|--------|------|---------------|
| `video` | edited/*.mp4 | Ground truth (target) | N/A (standard video) |
| `pisco_video` | clean/*.mp4 | Clean background (condition) | N/A (standard video) |
| `pisco_reference_video` | processed/reference_video/*.mp4 | Instance reference | **First frame only** |
| `pisco_video_mask` | processed/mask_video/*.mp4 | Where to insert | All frames |
| `pisco_depth` | depth/*.mp4 | Scene depth | All frames |
| `pisco_reference_depth` | processed/reference_depth/*.mp4 | Instance depth reference | **First frame only** |

## Model Loading

Fine-tuning loads the pretrained PISCO model from HuggingFace:

```
xiangbog/PISCO-1.3B (contains):
├── diffusion_pytorch_model.safetensors   # DIT + PISCO weights (auto-detected by hash)
├── models_t5_umt5-xxl-enc-bf16.safetensors  # Text encoder
├── Wan2.1_VAE.safetensors               # VAE
└── google/umt5-xxl/                     # Tokenizer
```

The single `diffusion_pytorch_model.safetensors` file is loaded as **both** `wan_video_dit` and `wan_video_pisco` via hash-based auto-detection (same hash maps to multiple model configs in `model_configs.py`).

## Training

### Quick Start

```bash
# 1. Install environment
conda create -n pisco python=3.12 && conda activate pisco
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# 2. Preprocess data
python utils/preprocess_data.py --dataset_dir Dataset/data --num_frames 21

# 3. Generate metadata JSON
python utils/generate_dataset_json.py --dataset_dir Dataset/data --repeat 10

# 4. Launch training
export WANDB_API_KEY="your_key"
bash training/finetune/PISCO-1.3B.sh
```

### Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Resolution | 480×832 | Reduced from 720p for debugging |
| Frames | 21 | Reduced from 121 max for debugging |
| Learning rate | 1e-4 | |
| Gradient accumulation | 8 | |
| Trainable | `pisco` (full module) | Patch embedding + 15 attention blocks |
| Checkpoint prefix | `pipe.pisco.` | Saves only PISCO weights |
| Mask strategy | First frame only | `--pisco_first_frame_only` |
| Model source | `xiangbog/PISCO-1.3B` | HuggingFace download |

### Custom Data Operators

`utils/data_operators.py` provides two operators:

- **`LoadVideoKeepFirstFrame`**: Returns `(frames, mask)` where `mask = [True, False, False, ...]`. Used for reference inputs where only frame 0 has meaningful data. The pipeline's `interpolate_fill` copies frame 0 to all positions, then `block_mask` zeros out latents for frames 1+.

- **`LoadVideoAllFrames`**: Returns `(frames, mask)` where `mask = [True, True, ...]`. Used for spatial mask and depth whose temporal masks are discarded by the pipeline but still need to return a `(frames, mask)` tuple.

### Scaling Up

To train at full resolution, modify the training script:

```bash
--height 720
--width 1280
--num_frames 121  # or 49 for moderate
```

For 14B model, use the provided shell script with DeepSpeed ZeRO-2:
```bash
# Adjust CUDA_VISIBLE_DEVICES as needed (default uses all 7 GPUs 0-6)
# If GPU 2 is occupied: CUDA_VISIBLE_DEVICES=0,1,3,4,5,6 and update num_processes in yaml
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash training/finetune/PISCO-14B.sh
```

The 14B training uses:
- `training/accelerate_config_14B_7gpu.yaml`: DeepSpeed ZeRO-2 config (no CPU offload needed with H100 80GB GPUs)
- `--initialize_model_on_cpu`: Loads model on CPU first to avoid per-GPU OOM during initialization
- ~5.8s/step on 6× H100 80GB GPUs
