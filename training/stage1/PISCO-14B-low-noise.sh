# Logging goes to Weights & Biases: set WANDB_API_KEY in your environment, or WANDB_MODE=offline to log locally.

accelerate launch --config_file training/accelerate_config_14B.yaml ./training/train.py \
    --dataset_base_path ./dataset/PISCO \
    --dataset_metadata_path ./dataset/PISCO/PISCO.json \
    --data_file_keys "video,pisco_video,pisco_video_mask,pisco_reference_video,pisco_depth,pisco_reference_depth" \
    --dataset_num_workers 2 \
    --height 480 \
    --width 832 \
    --num_frames 49 \
    --dataset_repeat 1 \
    --auto_load_checkpoints \
    --model_paths "models/PISCO/inits/PISCO-14B/PISCO-14B-low-noise.safetensors" \
    --model_id_with_origin_paths "PAI/Wan2.2-VACE-Fun-A14B:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.2-VACE-Fun-A14B:Wan2.1_VAE.pth" \
    --learning_rate 1e-4 \
    --num_epochs 100 \
    --remove_prefix_in_ckpt "pipe.pisco.pisco_patch_embedding." \
    --output_path "./models/train/PISCO-14B-low-noise_stage1" \
    --loading_model "pisco.pisco_patch_embedding" \
    --trainable_models "pisco.pisco_patch_embedding" \
    --extra_inputs "pisco_video,pisco_video_mask,pisco_reference_video,pisco_depth,pisco_reference_depth" \
    --min_timestep_boundary 0.358 \
    --max_timestep_boundary 1 \
    --save_steps 100 \
    --warmup_steps 100
