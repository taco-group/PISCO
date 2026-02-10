
import os
import glob
from PIL import Image
import torch

class SparseVideoData:
    def __init__(self, folder, height=480, width=832, length=49):
        self.folder = folder
        self.height = height
        self.width = width
        self.length = length
        self.data_type = "images" 
        
        self.image_map = {}
        if os.path.exists(folder):
            files = sorted(glob.glob(os.path.join(folder, "*.png")))
            for f in files:
                try:
                    # Expect filenames like "00010.png"
                    idx = int(os.path.splitext(os.path.basename(f))[0])
                    self.image_map[idx] = f
                except ValueError:
                    continue

    def set_length(self, length):
        self.length = length
    
    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        if item in self.image_map:
            img = Image.open(self.image_map[item]).convert("RGB")
            # Resize if needed
            if img.size != (self.width, self.height):
                img = img.resize((self.width, self.height))
            return img
        else:
            # Return black image
            return Image.new("RGB", (self.width, self.height), (0, 0, 0))

def load_image_sequence(folder, height=480, width=832, length=49):
    if not os.path.isdir(folder):
        raise ValueError(f"Directory not found: {folder}")
    return SparseVideoData(folder, height=height, width=width, length=length)

vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
