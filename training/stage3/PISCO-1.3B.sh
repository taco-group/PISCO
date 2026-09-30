# Logging goes to Weights & Biases: set WANDB_API_KEY in your environment, or WANDB_MODE=offline to log locally.

accelerate launch ./training/train.py \
    --dataset_base_path ./dataset/PISCO \
    --dataset_metadata_path ./dataset/PISCO/PISCO.json \
    --data_file_keys "video,pisco_video,pisco_video_mask,pisco_reference_video,pisco_depth,pisco_reference_depth" \
    --dataset_num_workers 2 \
    --height 480 \
    --width 832 \
    --num_frames 49 \
    --dataset_repeat 1 \
    --gradient_accumulation_steps 4 \
    --auto_load_checkpoints \
    --model_paths "models/PISCO/inits/PISCO-1.3B/PISCO-1.3B.safetensors" \
    --model_id_with_origin_paths "Wan-AI/Wan2.1-VACE-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-1.3B:Wan2.1_VAE.pth" \
    --learning_rate 1e-5 \
    --num_epochs 100 \
    --remove_prefix_in_ckpt "pipe.pisco.,pipe.dit." \
    --output_path "./models/train/PISCO-1.3B_stage3" \
    --loading_model "pisco" \
    --trainable_models "pisco,dit" \
    --extra_inputs "pisco_video,pisco_video_mask,pisco_reference_video,pisco_depth,pisco_reference_depth" \
    --save_steps 100 \
    --warmup_steps 100
    
