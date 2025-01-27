import cv2
import numpy as np

def stack_frames(frames, new_frame):
    if frames is None:
        return np.stack([new_frame] * 4, axis=-1)  # Stack 4 frames
    else:
        return np.concatenate([frames[..., 1:], new_frame[..., np.newaxis]], axis=-1)