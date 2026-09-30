"""
Generate dataset metadata JSON for PISCO fine-tuning.

Maps each sample to its source and preprocessed video files.
Output JSON is consumed by diffsynth's UnifiedDataset.

Usage:
    python utils/generate_dataset_json.py \
        --dataset_dir Dataset/data \
        --output Dataset/data/pisco_finetune.json \
        --repeat 10
"""

import os
import json
import argparse


def discover_samples(dataset_dir):
    """Discover all preprocessed samples."""
    clean_dir = os.path.join(dataset_dir, "clean")
    edited_dir = os.path.join(dataset_dir, "edited")
    depth_dir = os.path.join(dataset_dir, "depth")
    proc_dir = os.path.join(dataset_dir, "processed")

    ref_video_dir = os.path.join(proc_dir, "reference_video")
    mask_video_dir = os.path.join(proc_dir, "mask_video")
    ref_depth_dir = os.path.join(proc_dir, "reference_depth")

    samples = []
    for fname in sorted(os.listdir(clean_dir)):
        if not fname.endswith(".mp4"):
            continue
        sample_id = fname.replace("_removal.mp4", "")

        paths = {
            # video = edited (target/ground truth for training)
            "video": os.path.join("edited", f"{sample_id}.mp4"),
            # pisco_video = clean background (input condition)
            "pisco_video": os.path.join("clean", fname),
            # Preprocessed files
            "pisco_reference_video": os.path.join("processed", "reference_video", f"{sample_id}.mp4"),
            "pisco_video_mask": os.path.join("processed", "mask_video", f"{sample_id}.mp4"),
            "pisco_depth": os.path.join("depth", f"{sample_id}_removal_depth.mp4"),
            "pisco_reference_depth": os.path.join("processed", "reference_depth", f"{sample_id}.mp4"),
            "prompt": "",
        }

        # Verify all files exist
        missing = []
        for key, relpath in paths.items():
            if key == "prompt":
                continue
            fullpath = os.path.join(dataset_dir, relpath)
            if not os.path.exists(fullpath):
                missing.append(f"{key}: {relpath}")

        if missing:
            print(f"[SKIP] {sample_id} - missing: {', '.join(missing)}")
            continue

        samples.append(paths)

    return samples


def main():
    parser = argparse.ArgumentParser(description="Generate PISCO dataset metadata JSON")
    parser.add_argument("--dataset_dir", type=str, default="Dataset/data",
                        help="Path to dataset directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: {dataset_dir}/pisco_finetune.json)")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Repeat dataset entries N times (for small datasets)")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.dataset_dir, "pisco_finetune.json")

    print(f"Scanning {args.dataset_dir}...")
    samples = discover_samples(args.dataset_dir)
    print(f"Found {len(samples)} valid samples")

    if args.repeat > 1:
        samples = samples * args.repeat
        print(f"After {args.repeat}x repeat: {len(samples)} entries")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"Wrote {len(samples)} entries to {args.output}")


if __name__ == "__main__":
    main()
