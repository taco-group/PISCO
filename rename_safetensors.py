import os
from safetensors.torch import load_file, save_file
import argparse

def rename_keys(file_path):
    print(f"Loading {file_path}...")
    try:
        tensors = load_file(file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    new_tensors = {}
    renamed_count = 0

    for key, tensor in tensors.items():
        new_key = key
        
        # Replace "pipe.xgv2." with empty string (assuming user meant pipe, not pip based on example)
        if "pipe.xgv2." in new_key:
            new_key = new_key.replace("pipe.xgv2.", "")

        # Replace "pipe.dit." with empty string
        if "pipe.dit." in new_key:
            new_key = new_key.replace("pipe.dit.", "")

        # Replace "xgv2_" with "pisco_"
        if "xgv2_" in new_key:
            new_key = new_key.replace("xgv2_", "pisco_")


            
        if new_key != key:
            renamed_count += 1
            # print(f"Renamed: {key} -> {new_key}") 
        
        new_tensors[new_key] = tensor

    if renamed_count > 0:
        new_path = file_path.replace(".safetensors", "_renamed.safetensors")
        print(f"Renamed {renamed_count} keys.")
        print(f"Saving to {new_path}...")
        save_file(new_tensors, new_path)
        print("Done.")
    else:
        print("No keys matched the criteria. No file saved.")

# if __name__ == "__main__":
#     target_path = "/mnt/beegfs/xiangbo/Folder/Research/SE/PISCO/models/PISCO/PISCO-14B/high_noise_model/diffusion_pytorch_model.safetensors"
#     if os.path.exists(target_path):
#         rename_keys(target_path)
#     else:
#         print(f"File not found: {target_path}")

#     target_path = "/mnt/beegfs/xiangbo/Folder/Research/SE/PISCO/models/PISCO/PISCO-14B/low_noise_model/diffusion_pytorch_model.safetensors"
#     if os.path.exists(target_path):
#         rename_keys(target_path)
#     else:
#         print(f"File not found: {target_path}")

if __name__ == "__main__":
    target_path = "/mnt/beegfs/xiangbo/Folder/Research/SE/DiffSynth-Studio/models/train/XG/xgv2_14B_softinit_full_high_noise_weighted_dataset_720/step-14050.safetensors"
    if os.path.exists(target_path):
        rename_keys(target_path)
    else:
        print(f"File not found: {target_path}")

    target_path = "/mnt/beegfs/xiangbo/Folder/Research/SE/DiffSynth-Studio/models/train/XG/xgv2_14B_softinit_full_low_noise_weighted_dataset_720/step-10650.safetensors"
    if os.path.exists(target_path):
        rename_keys(target_path)
    else:
        print(f"File not found: {target_path}")
