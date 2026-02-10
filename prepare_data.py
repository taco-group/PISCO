
import os
import shutil
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

def process_video(input_path, output_dir, keep_indices):
    """
    Reads a video, keeps specified frames, masks others with black, and saves as PNGs.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(input_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        if i in keep_indices:
            # Keep frame
            cv2.imwrite(os.path.join(output_dir, f"{i:05d}.png"), frame)
        else:
            # Mask frame (black) - Do not save to disk
            pass
    
    cap.release()

def main():
    base_dir = "eval"
    output_base_dir = "eval_img"
    keep_indices = {10, 20, 46}

    # Ensure output directory exists
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    # Find all example directories
    example_dirs = sorted(glob(os.path.join(base_dir, "example*")))
    
    for example_dir in tqdm(example_dirs, desc="Processing examples"):
        example_name = os.path.basename(example_dir)
        output_example_dir = os.path.join(output_base_dir, example_name)
        
        if not os.path.exists(output_example_dir):
            os.makedirs(output_example_dir)

        # 1. Copy clean.mp4 and clean_depth.mp4
        for filename in ["clean.mp4", "clean_depth.mp4"]:
            src = os.path.join(example_dir, filename)
            dst = os.path.join(output_example_dir, filename)
            if os.path.exists(src):
                shutil.copy2(src, dst)
            else:
                print(f"Warning: {src} not found.")

        # 2. Process masked videos
        # pisco_video_mask -> mask.mp4
        # pisco_reference_video -> video_masked.mp4
        # pisco_reference_depth -> video_depth_masked.mp4
        
        videos_to_process = {
            "mask.mp4": "mask", # Save to eval_img/exampleX/mask/
            "video_masked.mp4": "video_masked",
            "video_depth_masked.mp4": "video_depth_masked"
        }

        for filename, subfolder_name in videos_to_process.items():
            src = os.path.join(example_dir, filename)
            if os.path.exists(src):
                 # Create subfolder for images
                output_subfolder = os.path.join(output_example_dir, subfolder_name)
                process_video(src, output_subfolder, keep_indices)
            else:
                print(f"Warning: {src} not found.")

if __name__ == "__main__":
    main()
