#!/bin/bash
# Fine-tune PISCO-1.3B on instance insertion dataset
# Uses 480p, 21 frames for fast debugging
#
# Prerequisites:
#   1. Run: python utils/preprocess_data.py --dataset_dir Dataset/data --num_frames 21
#   2. Run: python utils/generate_dataset_json.py --dataset_dir Dataset/data --repeat 10

# Logging goes to Weights & Biases: set WANDB_API_KEY in your environment, or WANDB_MODE=offline to log locally.
export DIFFSYNTH_DOWNLOAD_SOURCE="huggingface"

accelerate launch ./training/train.py \
    --dataset_base_path ./Dataset/data \
    --dataset_metadata_path ./Dataset/data/pisco_finetune.json \
    --data_file_keys "video,pisco_video,pisco_video_mask,pisco_reference_video,pisco_depth,pisco_reference_depth" \
    --dataset_num_workers 2 \
    --height 480 \
    --width 832 \
    --num_frames 21 \
    --dataset_repeat 1 \
    --gradient_accumulation_steps 8 \
    --auto_load_checkpoints \
    --model_id_with_origin_paths "xiangbog/PISCO-1.3B:diffusion_pytorch_model.safetensors,xiangbog/PISCO-1.3B:models_t5_umt5-xxl-enc-bf16.safetensors,xiangbog/PISCO-1.3B:Wan2.1_VAE.safetensors" \
    --learning_rate 1e-4 \
    --num_epochs 100 \
    --remove_prefix_in_ckpt "pipe.pisco." \
    --output_path "./models/train/PISCO-1.3B_finetune" \
    --trainable_models "pisco" \
    --extra_inputs "pisco_video,pisco_video_mask,pisco_reference_video,pisco_depth,pisco_reference_depth" \
    --save_steps 100 \
    --warmup_steps 100 \
    --tokenizer_path "xiangbog/PISCO-1.3B:google/umt5-xxl/" \
    --pisco_first_frame_only
