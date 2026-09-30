import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import argparse

def get_paths_map(base_path, env, video_path, enabled_mappings=None):
    """Generates the dictionary of expected file paths for PISCO."""
    if enabled_mappings is None:
        enabled_mappings = []
        
    stem = video_path.stem
    
    # Mapping for PISCO schema
    mappings = {
        "video":                (f"{env}_Unedited", video_path.name),
        "pisco_video_mask":      (f"{env}_Masked",   video_path.name),
        "pisco_video":           (f"{env}_Edited",   video_path.name),
        "pisco_reference_video": (f"{env}_Entity",   video_path.name),
        "pisco_depth":           (f"{env}_Edited_Depth", f"{stem}_viz.mp4"),
        "pisco_reference_depth": (f"{env}_Depth_Entity", f"{stem}_viz.mp4"),
    }
    
    mappings_2 = {}
    if "relight" in enabled_mappings:
        mappings_2 = {
            "video":                (f"{env}_Unedited", video_path.name),
            "pisco_video_mask":      (f"{env}_Masked",   video_path.name),
            "pisco_video":           (f"{env}_Edited",   video_path.name),
            "pisco_reference_video": (f"{env}_Entity_relighted",   video_path.name),
            "pisco_depth":           (f"{env}_Edited_Depth", f"{stem}_viz.mp4"),
            "pisco_reference_depth": (f"{env}_Depth_Entity", f"{stem}_viz.mp4"),
        }

    mappings_3 = {}
    if "complete" in enabled_mappings:
        mappings_3 = {
            "video":                (f"{env}_Unedited", video_path.name),
            "pisco_video_mask":      (f"{env}_Complete_Masked",   video_path.name),
            "pisco_video":           (f"{env}_Edited",   video_path.name),
            "pisco_reference_video": (f"{env}_Complete_Videos",   video_path.name),
            "pisco_depth":           (f"{env}_Edited_Depth", f"{stem}_viz.mp4"),
            "pisco_reference_depth": (f"{env}_Depth_Complete", f"{stem}_viz.mp4"),
        }

    mappings_4 = {}
    if "complete_relight" in enabled_mappings:
        mappings_4 = {
            "video":                (f"{env}_Unedited", video_path.name),
            "pisco_video_mask":      (f"{env}_Complete_Masked",   video_path.name),
            "pisco_video":           (f"{env}_Edited",   video_path.name),
            "pisco_reference_video": (f"{env}_Complete_Videos_relighted",   video_path.name),
            "pisco_depth":           (f"{env}_Edited_Depth", f"{stem}_viz.mp4"),
            "pisco_reference_depth": (f"{env}_Depth_Complete", f"{stem}_viz.mp4"),
        }
    
    return ({
        key: base_path / env / subfolder / filename
        for key, (subfolder, filename) in mappings.items()
    },
    {
        key: base_path / env / subfolder / filename
        for key, (subfolder, filename) in mappings_2.items()
    },
    {
        key: base_path / env / subfolder / filename
        for key, (subfolder, filename) in mappings_3.items()
    },
    {
        key: base_path / env / subfolder / filename
        for key, (subfolder, filename) in mappings_4.items()
    })

def process_single_env(args):
    """Worker function to process one Environment folder."""
    env, base_path, limit, repeat, enabled_mappings = args
    env_entries = []
    local_total = 0
    local_valid = 0

    unedited_dir = base_path / env / f"{env}_Unedited"
    if not unedited_dir.exists():
        return [], 0, 0

    for video_file in unedited_dir.glob("*.mp4"):
        if limit is not None and len(env_entries) >= limit:
            break

        local_total += 1
        path_map, path_map_2, path_map_3, path_map_4 = get_paths_map(base_path, env, video_file, enabled_mappings)
        
        if path_map:
            # Identify missing files
            missing = [p for p in path_map.values() if not p.exists()]

            if not missing:
                local_valid += 1
                entry = {
                    key: f"{path.relative_to(base_path).as_posix()}"
                    for key, path in path_map.items()
                }
                entry["prompt"] = ""
                env_entries.append(entry)
            else:
                # Uncomment to debug missing files
                print(f"[{env}] Skipping {video_file.name}. Missing: {missing}")
                pass
        
        if limit is not None and len(env_entries) >= limit: break

        if path_map_2:
            missing_2 = [p for p in path_map_2.values() if not p.exists()]
            if not missing_2:
                local_valid += 1
                entry = {
                    key: f"{path.relative_to(base_path).as_posix()}"
                    for key, path in path_map_2.items()
                }
                entry["prompt"] = ""
                env_entries.append(entry)
            else:
                # Uncomment to debug missing files
                print(f"[{env}] Skipping {video_file.name}. Missing: {missing_2}")
                pass

        if limit is not None and len(env_entries) >= limit: break

        if path_map_3:
            missing_3 = [p for p in path_map_3.values() if not p.exists()]
            if not missing_3:
                local_valid += 1
                entry = {
                    key: f"{path.relative_to(base_path).as_posix()}"
                    for key, path in path_map_3.items()
                }
                entry["prompt"] = ""
                env_entries.append(entry)
            else:
                # Uncomment to debug missing files
                print(f"[{env}] Skipping {video_file.name}. Missing: {missing_3}")
                pass

        if limit is not None and len(env_entries) >= limit: break

        if path_map_4:
            missing_4 = [p for p in path_map_4.values() if not p.exists()]
            if not missing_4:
                local_valid += 1
                entry = {
                    key: f"{path.relative_to(base_path).as_posix()}"
                    for key, path in path_map_4.items()
                }
                entry["prompt"] = ""
                env_entries.append(entry)
            else:
                # Uncomment to debug missing files
                print(f"[{env}] Skipping {video_file.name}. Missing: {missing_4}")
                pass

    if limit is not None:
        env_entries = env_entries[:limit]
        
    # Repeat the data
    env_entries = env_entries * repeat
    
    return env_entries, local_total, local_valid

def discover_subsets(base_path):
    """A subset is any folder <name> that contains <name>/<name>_Unedited (e.g. Env1, data_v2, DAVIS)."""
    return sorted(d.name for d in base_path.iterdir() if (d / f"{d.name}_Unedited").is_dir())


def generate_index_parallel(dataset_root, dataset_name, max_workers=8, enabled_mappings=None, subsets=None, limits=None, repeats=None):
    base_path = dataset_root / dataset_name
    subsets = subsets or discover_subsets(base_path)
    limits = limits or {}
    repeats = repeats or {}

    tasks = [(name, base_path, limits.get(name), repeats.get(name, 1), enabled_mappings) for name in subsets]

    final_index = []
    global_total = 0
    global_valid = 0

    print(f"Subsets: {', '.join(subsets)}")
    print(f"Starting parallel processing on {max_workers} cores...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_env, task): task for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing Envs"):
            entries, total, valid = future.result()
            final_index.extend(entries)
            global_total += total
            global_valid += valid

    ratio = (global_valid / global_total) if global_total > 0 else 0
    print(f"Valid Ratio: {ratio:.2%} ({global_valid}/{global_total})")
    
    return final_index


def parse_subset_values(items, cast):
    """Parse ["Env*=60", "data_v2=3000"] into a {subset: value} dict; "*" is a trailing wildcard."""
    patterns = []
    for item in items:
        name, value = item.split("=", 1)
        patterns.append((name, cast(value)))
    return patterns


def expand_patterns(patterns, subsets):
    values = {}
    for name, value in patterns:
        for subset in subsets:
            if subset == name or (name.endswith("*") and subset.startswith(name[:-1])):
                values[subset] = value
    return values

def parse_args():
    parser = argparse.ArgumentParser(description="Generate JSON dataset index for PISCO.")
    
    parser.add_argument("--dataset_folder", type=str, default="Dataset", help="Path to the dataset folder.")
    parser.add_argument("--dataset_name", type=str, default="VISAR-Dataset-V2", help="Name of the dataset.")
    parser.add_argument("--output_filename", type=str, default="video_dataset_PISCO.json", help="Output JSON filename.")
    parser.add_argument("--max_workers", type=int, default=8, help="Number of parallel workers.")
    
    parser.add_argument("--extra_mappings", type=str, nargs="*", default=[], 
                        choices=["relight", "complete", "complete_relight"], 
                        help="Enable extra mappings.")
    parser.add_argument("--subsets", type=str, nargs="*", default=None,
                        help="Subsets to index (default: every <name>/<name>_Unedited folder).")
    parser.add_argument("--limit", type=str, nargs="*", default=[],
                        help='Cap samples per subset, e.g. --limit "Env*=60" data_v2=3000.')
    parser.add_argument("--repeat", type=str, nargs="*", default=[],
                        help='Repeat samples per subset, e.g. --repeat DAVIS=5.')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # --- Configuration ---
    DATASET_FOLDER = Path(args.dataset_folder)
    DATASET_NAME = args.dataset_name
    OUTPUT_FILENAME = args.output_filename
    
    # --- Execution ---
    subsets = args.subsets or discover_subsets(DATASET_FOLDER / DATASET_NAME)
    index_data = generate_index_parallel(
        DATASET_FOLDER, 
        DATASET_NAME, 
        max_workers=args.max_workers,
        enabled_mappings=args.extra_mappings,
        subsets=subsets,
        limits=expand_patterns(parse_subset_values(args.limit, int), subsets),
        repeats=expand_patterns(parse_subset_values(args.repeat, int), subsets),
    )

    output_file = DATASET_FOLDER / DATASET_NAME / OUTPUT_FILENAME
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=4)

    print(f"JSON generated at: {output_file}")
    print(f"Total pairs: {len(index_data)}")