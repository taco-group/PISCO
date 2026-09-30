"""
Preprocess raw PISCO dataset into training-ready video files.

Input structure (Dataset/data/):
    clean/          - Clean background videos (1280x720, 120 frames)
    edited/         - Edited videos with inserted instances
    depth/          - Depth map videos
    instance/       - Per-sample directories with:
                      combined_cutout.png (RGBA), combined_mask.png (grayscale)

Output structure (Dataset/data/processed/):
    reference_video/  - Frame 0 = RGB cutout (black bg), rest = black
    mask_video/       - All frames = mask (white on black, RGB)
    reference_depth/  - Frame 0 = depth masked by instance mask, rest = black

Usage:
    python utils/preprocess_data.py --dataset_dir Dataset/data --num_frames 21
"""

import os
import re
import argparse
import glob
import numpy as np
from PIL import Image
import imageio


def find_instance_dir(instance_base, sample_id):
    """Find instance directory matching a sample ID (e.g., '0000043_00000')."""
    for dirname in os.listdir(instance_base):
        if dirname.startswith(sample_id + "png_"):
            return os.path.join(instance_base, dirname)
    return None


def discover_samples(dataset_dir):
    """Discover all samples by matching files across clean/, edited/, depth/, instance/."""
    clean_dir = os.path.join(dataset_dir, "clean")
    edited_dir = os.path.join(dataset_dir, "edited")
    depth_dir = os.path.join(dataset_dir, "depth")
    instance_dir = os.path.join(dataset_dir, "instance")

    samples = []
    for fname in sorted(os.listdir(clean_dir)):
        if not fname.endswith(".mp4"):
            continue
        # Extract sample ID: 0000043_00000_removal.mp4 → 0000043_00000
        sample_id = fname.replace("_removal.mp4", "")

        clean_path = os.path.join(clean_dir, fname)
        edited_path = os.path.join(edited_dir, f"{sample_id}.mp4")
        depth_path = os.path.join(depth_dir, f"{sample_id}_removal_depth.mp4")
        inst_dir = find_instance_dir(instance_dir, sample_id)

        if not os.path.exists(edited_path):
            print(f"[SKIP] Missing edited: {edited_path}")
            continue
        if not os.path.exists(depth_path):
            print(f"[SKIP] Missing depth: {depth_path}")
            continue
        if inst_dir is None:
            print(f"[SKIP] Missing instance dir for {sample_id}")
            continue

        cutout_path = os.path.join(inst_dir, "combined_cutout.png")
        mask_path = os.path.join(inst_dir, "combined_mask.png")
        if not os.path.exists(cutout_path) or not os.path.exists(mask_path):
            print(f"[SKIP] Missing cutout/mask in {inst_dir}")
            continue

        samples.append({
            "sample_id": sample_id,
            "clean": clean_path,
            "edited": edited_path,
            "depth": depth_path,
            "cutout": cutout_path,
            "mask": mask_path,
        })

    return samples


def get_video_resolution(video_path):
    """Get (width, height) of a video."""
    reader = imageio.get_reader(video_path)
    frame = reader.get_data(0)
    reader.close()
    return frame.shape[1], frame.shape[0]  # width, height


def rgba_to_rgb_black_bg(rgba_image):
    """Convert RGBA image to RGB with black background for transparent pixels."""
    rgba = np.array(rgba_image)
    rgb = rgba[:, :, :3].copy()
    alpha = rgba[:, :, 3:4] / 255.0
    rgb = (rgb * alpha).astype(np.uint8)
    return Image.fromarray(rgb)


def save_video(frames_np, output_path, fps=24):
    """Save numpy frames as mp4 video."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264",
                                output_params=["-crf", "18", "-pix_fmt", "yuv420p"])
    for frame in frames_np:
        writer.append_data(frame)
    writer.close()


def process_reference_video(cutout_path, target_w, target_h, num_frames):
    """Create reference video: frame 0 = RGB cutout, rest = black."""
    cutout = Image.open(cutout_path).convert("RGBA")
    cutout_rgb = rgba_to_rgb_black_bg(cutout)
    cutout_rgb = cutout_rgb.resize((target_w, target_h), Image.LANCZOS)

    frames = []
    frames.append(np.array(cutout_rgb))
    black_frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    for _ in range(num_frames - 1):
        frames.append(black_frame)
    return frames


def process_mask_video(mask_path, target_w, target_h, num_frames):
    """Create mask video: all frames have the same mask (white on black, RGB)."""
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((target_w, target_h), Image.NEAREST)
    mask_np = np.array(mask)
    # Convert to RGB: white where mask > 0, black otherwise
    mask_rgb = np.stack([mask_np, mask_np, mask_np], axis=-1)

    frames = []
    for _ in range(num_frames):
        frames.append(mask_rgb)
    return frames


def process_reference_depth(depth_path, mask_path, target_w, target_h, num_frames):
    """Create reference depth: frame 0 = depth masked by instance mask, rest = black."""
    # Load first frame of depth video
    reader = imageio.get_reader(depth_path)
    depth_frame = reader.get_data(0)
    reader.close()

    # Resize depth to target
    depth_img = Image.fromarray(depth_frame).resize((target_w, target_h), Image.LANCZOS)
    depth_np = np.array(depth_img)

    # Load and resize mask
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((target_w, target_h), Image.NEAREST)
    mask_np = np.array(mask)[:, :, np.newaxis] / 255.0

    # Apply mask
    masked_depth = (depth_np * mask_np).astype(np.uint8)

    frames = []
    frames.append(masked_depth)
    black_frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    for _ in range(num_frames - 1):
        frames.append(black_frame)
    return frames


def process_sample(sample, output_dir, target_w, target_h, num_frames, fps=24):
    """Process a single sample: create reference_video, mask_video, reference_depth."""
    sid = sample["sample_id"]

    ref_video_path = os.path.join(output_dir, "reference_video", f"{sid}.mp4")
    mask_video_path = os.path.join(output_dir, "mask_video", f"{sid}.mp4")
    ref_depth_path = os.path.join(output_dir, "reference_depth", f"{sid}.mp4")

    # Skip if all outputs already exist
    if (os.path.exists(ref_video_path) and os.path.exists(mask_video_path)
            and os.path.exists(ref_depth_path)):
        print(f"  [SKIP] {sid} - already processed")
        return

    print(f"  Processing {sid}...")

    # Reference video
    ref_frames = process_reference_video(
        sample["cutout"], target_w, target_h, num_frames
    )
    save_video(ref_frames, ref_video_path, fps)

    # Mask video
    mask_frames = process_mask_video(
        sample["mask"], target_w, target_h, num_frames
    )
    save_video(mask_frames, mask_video_path, fps)

    # Reference depth
    ref_depth_frames = process_reference_depth(
        sample["depth"], sample["mask"], target_w, target_h, num_frames
    )
    save_video(ref_depth_frames, ref_depth_path, fps)


def main():
    parser = argparse.ArgumentParser(description="Preprocess PISCO dataset for training")
    parser.add_argument("--dataset_dir", type=str, default="Dataset/data",
                        help="Path to raw dataset directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: {dataset_dir}/processed)")
    parser.add_argument("--num_frames", type=int, default=21,
                        help="Number of frames per video (must satisfy T%%4==1)")
    parser.add_argument("--fps", type=int, default=24,
                        help="Output video FPS")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.dataset_dir, "processed")

    assert args.num_frames % 4 == 1, \
        f"num_frames must satisfy T%%4==1 for VAE compression. Got {args.num_frames}"

    print(f"Discovering samples in {args.dataset_dir}...")
    samples = discover_samples(args.dataset_dir)
    print(f"Found {len(samples)} valid samples")

    if len(samples) == 0:
        print("No samples found. Check dataset structure.")
        return

    # Get target resolution from first clean video
    target_w, target_h = get_video_resolution(samples[0]["clean"])
    print(f"Target resolution: {target_w}x{target_h}")
    print(f"Number of frames: {args.num_frames}")
    print(f"Output directory: {args.output_dir}")

    for sample in samples:
        process_sample(sample, args.output_dir, target_w, target_h,
                       args.num_frames, args.fps)

    print(f"\nPreprocessing complete. {len(samples)} samples processed.")
    print(f"Output in: {args.output_dir}")


if __name__ == "__main__":
    main()
