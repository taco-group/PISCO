"""Build the PISCO training index from a dataset folder.

Layout: <dataset_dir>/<subset>/<subset>_{Unedited,Edited,Masked,Entity}/<name>.mp4
        <dataset_dir>/<subset>/<subset>_{Edited_Depth,Depth_Entity}/<name>_viz.mp4
Every folder <subset> containing <subset>_Unedited is indexed; samples with a missing file are skipped.
"""
import argparse
import json
from pathlib import Path

# training key -> (modality folder, filename suffix)
KEYS = {
    "video": ("Unedited", ""),                          # target video with the instance
    "pisco_video": ("Edited", ""),                      # background video, instance removed
    "pisco_video_mask": ("Masked", ""),                 # spatial mask of the instance
    "pisco_reference_video": ("Entity", ""),            # segmented instance
    "pisco_depth": ("Edited_Depth", "_viz"),            # depth of the background video
    "pisco_reference_depth": ("Depth_Entity", "_viz"),  # depth of the segmented instance
}


def index_subset(root, subset):
    entries, skipped = [], 0
    for video in sorted((root / subset / f"{subset}_Unedited").glob("*.mp4")):
        entry = {key: f"{subset}/{subset}_{folder}/{video.stem}{suffix}.mp4" for key, (folder, suffix) in KEYS.items()}
        if all((root / path).exists() for path in entry.values()):
            entries.append({**entry, "prompt": ""})
        else:
            skipped += 1
    return entries, skipped


def main():
    parser = argparse.ArgumentParser(description="Generate the PISCO training index.")
    parser.add_argument("--dataset_dir", default="dataset/PISCO")
    parser.add_argument("--output", default=None, help="Default: <dataset_dir>/PISCO.json")
    parser.add_argument("--subsets", nargs="*", default=None, help="Default: every subset found in dataset_dir.")
    args = parser.parse_args()

    root = Path(args.dataset_dir)
    subsets = args.subsets or sorted(d.name for d in root.iterdir() if (d / f"{d.name}_Unedited").is_dir())
    index = []
    for subset in subsets:
        entries, skipped = index_subset(root, subset)
        index += entries
        print(f"{subset}: {len(entries)} samples" + (f" ({skipped} skipped: missing files)" if skipped else ""))

    output = Path(args.output) if args.output else root / "PISCO.json"
    output.write_text(json.dumps(index, indent=4))
    print(f"{len(index)} samples -> {output}")


if __name__ == "__main__":
    main()
