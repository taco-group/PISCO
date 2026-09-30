"""
Custom data loading operators for PISCO fine-tuning.

These operators extend diffsynth's LoadVideo to provide deterministic
temporal masking strategies needed for instance insertion training,
where reference information is only available at frame 0.
"""

import torch
from diffsynth.core.data.operators import LoadVideo, ImageCropAndResize


class LoadVideoKeepFirstFrame(LoadVideo):
    """
    Loads video and returns (frames, temporal_mask) where only frame 0 is marked valid.
    Used for pisco_reference_video and pisco_reference_depth where only the first
    frame contains meaningful data.
    """
    def __init__(self, num_frames=81, height=None, width=None, max_pixels=None,
                 height_division_factor=16, width_division_factor=16):
        super().__init__(num_frames, 1, 1, lambda x: x)
        self.image_crop_and_resize = ImageCropAndResize(
            height, width, max_pixels, height_division_factor, width_division_factor
        )

    def __call__(self, data: str):
        import imageio
        from PIL import Image

        reader = imageio.get_reader(data)
        target_num_frames = self.num_frames
        frames = []
        for frame_id in range(target_num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.image_crop_and_resize(frame)
            frames.append(frame)
        reader.close()

        mask = torch.zeros(len(frames), dtype=torch.bool)
        if len(frames) > 0:
            mask[0] = True
        return frames, mask

    def get_num_frames(self, reader):
        return self.num_frames


class LoadVideoAllFrames(LoadVideo):
    """
    Loads video and returns (frames, temporal_mask) where all frames are marked valid.
    Used for pisco_video_mask and pisco_depth whose temporal masks are discarded
    by the pipeline but still need to return a (frames, mask) tuple.
    """
    def __init__(self, num_frames=81, height=None, width=None, max_pixels=None,
                 height_division_factor=16, width_division_factor=16):
        super().__init__(num_frames, 1, 1, lambda x: x)
        self.image_crop_and_resize = ImageCropAndResize(
            height, width, max_pixels, height_division_factor, width_division_factor
        )

    def __call__(self, data: str):
        import imageio
        from PIL import Image

        reader = imageio.get_reader(data)
        target_num_frames = self.num_frames
        frames = []
        for frame_id in range(target_num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.image_crop_and_resize(frame)
            frames.append(frame)
        reader.close()

        mask = torch.ones(len(frames), dtype=torch.bool)
        return frames, mask

    def get_num_frames(self, reader):
        return self.num_frames
