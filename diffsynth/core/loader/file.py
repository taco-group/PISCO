from safetensors import safe_open
import torch, hashlib
import os
import re

def find_recent_checkpoint(output_path):
    # Check if directory exists to avoid errors
    if not os.path.exists(output_path):
        return None

    max_step = -1
    latest_checkpoint_step = None
    
    max_epoch = -1
    latest_checkpoint_epoch = None
    
    # Regex to match 'step-[number].safetensors' or 'epoch-[number].safetensors'
    pattern_step = re.compile(r"step-(\d+)\.safetensors")
    pattern_epoch = re.compile(r"epoch-(\d+)\.safetensors")

    for filename in os.listdir(output_path):
        # Check for step
        match_step = pattern_step.search(filename)
        if match_step:
            step = int(match_step.group(1))
            if step > max_step:
                max_step = step
                latest_checkpoint_step = os.path.join(output_path, filename)
        
        # Check for epoch
        match_epoch = pattern_epoch.search(filename)
        if match_epoch:
            epoch = int(match_epoch.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                latest_checkpoint_epoch = os.path.join(output_path, filename)
                
    if latest_checkpoint_step is not None:
        return latest_checkpoint_step
    else:
        return latest_checkpoint_epoch

def load_state_dict(file_path, torch_dtype=None, device="cpu"):
    if isinstance(file_path, list):
        state_dict = {}
        for file_path_ in file_path:
            state_dict.update(load_state_dict(file_path_, torch_dtype, device))
        return state_dict
    if file_path.endswith(".safetensors"):
        return load_state_dict_from_safetensors(file_path, torch_dtype=torch_dtype, device=device)
    else:
        return load_state_dict_from_bin(file_path, torch_dtype=torch_dtype, device=device)


def load_state_dict_from_safetensors(file_path, torch_dtype=None, device="cpu"):
    state_dict = {}
    with safe_open(file_path, framework="pt", device=str(device)) as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
            if torch_dtype is not None:
                state_dict[k] = state_dict[k].to(torch_dtype)
    return state_dict


def load_state_dict_from_bin(file_path, torch_dtype=None, device="cpu"):
    state_dict = torch.load(file_path, map_location=device, weights_only=True)
    if len(state_dict) == 1:
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "module" in state_dict:
            state_dict = state_dict["module"]
        elif "model_state" in state_dict:
            state_dict = state_dict["model_state"]
    if torch_dtype is not None:
        for i in state_dict:
            if isinstance(state_dict[i], torch.Tensor):
                state_dict[i] = state_dict[i].to(torch_dtype)
    return state_dict


def convert_state_dict_keys_to_single_str(state_dict, with_shape=True):
    keys = []
    for key, value in state_dict.items():
        if isinstance(key, str):
            if isinstance(value, torch.Tensor):
                if with_shape:
                    shape = "_".join(map(str, list(value.shape)))
                    keys.append(key + ":" + shape)
                keys.append(key)
            elif isinstance(value, dict):
                keys.append(key + "|" + convert_state_dict_keys_to_single_str(value, with_shape=with_shape))
    keys.sort()
    keys_str = ",".join(keys)
    return keys_str


def hash_state_dict_keys(state_dict, with_shape=True):
    keys_str = convert_state_dict_keys_to_single_str(state_dict, with_shape=with_shape)
    keys_str = keys_str.encode(encoding="UTF-8")
    return hashlib.md5(keys_str).hexdigest()


def load_keys_dict(file_path):
    if isinstance(file_path, list):
        state_dict = {}
        for file_path_ in file_path:
            state_dict.update(load_keys_dict(file_path_))
        return state_dict
    if file_path.endswith(".safetensors"):
        return load_keys_dict_from_safetensors(file_path)
    else:
        return load_keys_dict_from_bin(file_path)


def load_keys_dict_from_safetensors(file_path):
    keys_dict = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            keys_dict[k] = f.get_slice(k).get_shape()
    return keys_dict


def convert_state_dict_to_keys_dict(state_dict):
    keys_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            keys_dict[k] = list(v.shape)
        else:
            keys_dict[k] = convert_state_dict_to_keys_dict(v)
    return keys_dict


def load_keys_dict_from_bin(file_path):
    state_dict = load_state_dict_from_bin(file_path)
    keys_dict = convert_state_dict_to_keys_dict(state_dict)
    return keys_dict


def convert_keys_dict_to_single_str(state_dict, with_shape=True):
    keys = []
    for key, value in state_dict.items():
        if isinstance(key, str):
            if isinstance(value, dict):
                keys.append(key + "|" + convert_keys_dict_to_single_str(value, with_shape=with_shape))
            else:
                if with_shape:
                    shape = "_".join(map(str, list(value)))
                    keys.append(key + ":" + shape)
                keys.append(key)
    keys.sort()
    keys_str = ",".join(keys)
    return keys_str


def hash_model_file(path, with_shape=True):
    keys_dict = load_keys_dict(path)
    keys_str = convert_keys_dict_to_single_str(keys_dict, with_shape=with_shape)
    keys_str = keys_str.encode(encoding="UTF-8")
    return hashlib.md5(keys_str).hexdigest()


if __name__ == "__main__":
    path = "/mnt/beegfs/xiangbo/Folder/Research/SE/DiffSynth-Studio/models/XG/xgtemp.safetensors"
    print("xgtemp:", hash_model_file(path))
    path = "/mnt/beegfs/xiangbo/Folder/Research/SE/DiffSynth-Studio/models/XG/pisco.safetensors"
    print("pisco:", hash_model_file(path))