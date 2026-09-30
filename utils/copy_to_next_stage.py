
import argparse
import os
import shutil
import glob
import re
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Copy checkpoint to next stage directory.")
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["1.3B", "14B"], 
        required=True, 
        help="Model size (1.3B or 14B)"
    )
    parser.add_argument(
        "--noise", 
        type=str, 
        choices=["high", "low"], 
        help="Noise level (high or low). Required for 14B model."
    )
    parser.add_argument(
        "--stage", 
        type=int, 
        required=True, 
        help="Current stage number (e.g., 1 for moving from stage1 to stage2)"
    )
    parser.add_argument(
        "--step", 
        type=int, 
        default=None, 
        help="Specific checkpoint step to copy. If not provided, defaults to the highest step."
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Print what would happen without actually copying."
    )
    return parser.parse_args()

def get_max_step(files):
    max_step = -1
    for f in files:
        match = re.search(r'step-(\d+)\.safetensors', os.path.basename(f))
        if match:
            step = int(match.group(1))
            if step > max_step:
                max_step = step
    return max_step

def get_source_path(models_train_dir, model_name, stage):
    # Try different naming conventions
    # Convention 1: PISCO-1.3B_stage1
    path1 = os.path.join(models_train_dir, f"{model_name}_stage{stage}")
    if os.path.isdir(path1):
        return path1
    
    # Convention 2: PISCO-1.3B-stage1 (hyphen instead of underscore - just in case)
    path2 = os.path.join(models_train_dir, f"{model_name}-stage{stage}")
    if os.path.isdir(path2):
        return path2

    return path1 # Default to convention 1

def main():
    args = parse_args()

    # validate args for 14B
    if args.model == "14B" and not args.noise:
        print("Error: --noise argument is required for 14B model.")
        sys.exit(1)

    # construct model base name
    if args.model == "1.3B":
        base_name = "PISCO-1.3B"
    else:  # 14B
        base_name = f"PISCO-14B-{args.noise}-noise"

    # Assume we are running from project root or find the models directory
    # If running from utils/, we need to go up one level.
    # If running from root, models/ is right there.
    cwd = os.getcwd()
    if os.path.basename(cwd) == "utils":
        project_root = os.path.dirname(cwd)
    else:
        project_root = cwd

    models_train_dir = os.path.join(project_root, "models", "train")
    
    if not os.path.exists(models_train_dir):
        # Last ditch effort: try absolute path if we know it? 
        # But let's just error out if we can't find it relative to CWD.
        print(f"Error: Could not find 'models/train' directory in {models_train_dir}")
        print("Please run this script from the project root.")
        sys.exit(1)

    source_dir = get_source_path(models_train_dir, base_name, args.stage)
    dest_dir = os.path.join(models_train_dir, f"{base_name}_stage{args.stage + 1}")

    if not os.path.exists(source_dir):
        print(f"Error: Source directory does not exist: {source_dir}")
        sys.exit(1)

    # Find checkpoints
    files = glob.glob(os.path.join(source_dir, "step-*.safetensors"))

    if not files:
        print(f"Error: No 'step-*.safetensors' files found in {source_dir}")
        sys.exit(1)

    target_file = None
    target_step = args.step

    if target_step is not None:
        potential_file = os.path.join(source_dir, f"step-{target_step}.safetensors")
        if os.path.exists(potential_file):
            target_file = potential_file
        else:
             print(f"Error: Checkpoint for step {target_step} not found.")
             sys.exit(1)
    else:
        max_step = get_max_step(files)
        if max_step == -1:
            print(f"Error: Could not parse step numbers from files in {source_dir}")
            sys.exit(1)
        target_file = os.path.join(source_dir, f"step-{max_step}.safetensors")

    print("-" * 40)
    print(f"Source Directory:      {source_dir}")
    print(f"Destination Directory: {dest_dir}")
    print(f"Checkpoint File:       {os.path.basename(target_file)}")
    print("-" * 40)

    if args.dry_run:
        print("[DRY RUN] Would create destination directory and copy file.")
    else:
        print("Creating destination directory...")
        os.makedirs(dest_dir, exist_ok=True)
        
        dest_file_path = os.path.join(dest_dir, os.path.basename(target_file))
        
        if os.path.exists(dest_file_path):
             print(f"Warning: Destination file already exists: {dest_file_path}")
             val = input("Overwrite? (y/n): ")
             if val.lower() != 'y':
                 print("Aborting copy.")
                 sys.exit(0)

        print(f"Copying {os.path.basename(target_file)} -> {dest_dir}...")
        shutil.copy2(target_file, dest_file_path)
        print("Done.")

if __name__ == "__main__":
    main()
