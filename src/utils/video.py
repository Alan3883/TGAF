import cv2
import numpy as np
from typing import List


def sample_video_frames(path: str, max_frames: int = 16, fps: int = 2) -> List[np.ndarray]:
    """
    Sample frames from video file with temporal sampling.
    
    Args:
        path: Path to video file
        max_frames: Maximum number of frames to extract
        fps: Target frames per second for sampling
        
    Returns:
        List of frame arrays in RGB format [H, W, 3]
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {path}")
    
    frames = []
    rate = cap.get(cv2.CAP_PROP_FPS) or 30 # set default 30 fps.
    step = max(int(rate // fps), 1)
    
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(frame[:, :, ::-1])
        i += 1
    
    cap.release()
    
    # Uniform temporal subsampling if we have more frames than needed
    if len(frames) > max_frames:
        # np.linspace(start, end, num)
        idxs = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = [frames[j] for j in idxs]
    
    return frames