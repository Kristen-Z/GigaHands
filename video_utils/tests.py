import numpy as np
from video_utils.reader_cpu import Reader

def test_list_views(reader: Reader):
    print("Available views for the ith capture:")
    for view_name in reader.streams.keys():
        print(f" - {view_name}")

def test_load_all_frames_for_view(reader: Reader, view_name: str, undistort=True, intr = None, proj = None, dist = None):
    """
    Test loading all frames for a specific view only.
    """
    print(f"\n--- Loading all frames for view '{view_name}' ---")
    frames_np, ts_list = reader.buffer_view_frames(view_name, undistort=undistort, intr = intr, proj = proj, dist = dist)
    print(f"Loaded {frames_np.shape[0]} frames with shape: {frames_np.shape}")
    return frames_np, ts_list

def test_multiview_frame_batch(reader: Reader, frame_ids: [], undistort=True, intrs = None, projs = None, dists = None):
    import cv2
    view_names = list(reader.streams.keys())
    print("Available views:", view_names)

    for frame_id in frame_ids:
        frames = reader._get_next_frame(frame_id)
        if undistort:
            frame_list = [cv2.undistort(frames[view][0], intrs[view_id], dists[view_id], None, intrs[view_id]) for view_id, view in enumerate(view_names)]
        else:
            frame_list = [frames[view][0] for view in view_names]
        ts_list = [frames[view][1] for view in view_names]
        frame_batch = np.stack(frame_list)  # Shape: (V, H, W, 3)
        print(f"Frame ID {frame_id}: shape {frame_batch.shape}")
        yield view_names, frame_batch, ts_list