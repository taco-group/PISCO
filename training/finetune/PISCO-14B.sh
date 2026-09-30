#!/bin/bash
# Fine-tune PISCO-14B on instance insertion dataset
# Uses DeepSpeed ZeRO-2 with 7 GPUs, 480p, 21 frames for fast debugging
#
# Prerequisites:
#   1. Run: python utils/preprocess_data.py --dataset_dir Dataset/data --num_frames 21
#   2. Run: python utils/generate_dataset_json.py --dataset_dir Dataset/data --repeat 10
#   3. pip install deepspeed

# Logging goes to Weights & Biases: set WANDB_API_KEY in your environment, or WANDB_MODE=offline to log locally.
export DIFFSYNTH_DOWNLOAD_SOURCE="huggingface"

accelerate launch --config_file training/accelerate_config_14B_7gpu.yaml ./training/train.py \
    --dataset_base_path ./Dataset/data \
    --dataset_metadata_path ./Dataset/data/pisco_finetune.json \
    --data_file_keys "video,pisco_video,pisco_video_mask,pisco_reference_video,pisco_depth,pisco_reference_depth" \
    --dataset_num_workers 2 \
    --height 480 \
    --width 832 \
    --num_frames 21 \
    --dataset_repeat 1 \
    --auto_load_checkpoints \
    --initialize_model_on_cpu \
    --model_id_with_origin_paths "xiangbog/PISCO-14B:low_noise_model/diffusion_pytorch_model.safetensors,xiangbog/PISCO-14B:models_t5_umt5-xxl-enc-bf16.safetensors,xiangbog/PISCO-14B:Wan2.1_VAE.safetensors" \
    --learning_rate 1e-4 \
    --num_epochs 100 \
    --remove_prefix_in_ckpt "pipe.pisco." \
    --output_path "./models/train/PISCO-14B_finetune" \
    --trainable_models "pisco" \
    --extra_inputs "pisco_video,pisco_video_mask,pisco_reference_video,pisco_depth,pisco_reference_depth" \
    --save_steps 100 \
    --warmup_steps 100 \
    --tokenizer_path "xiangbog/PISCO-14B:google/umt5-xxl/" \
    --pisco_first_frame_only
