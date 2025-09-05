import cv2
import os
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import json

def frame_preprocess(path, undistort=False, intr=None, dist_intr=None, dist=None):
    stream = cv2.VideoCapture(path)
    assert stream.isOpened(), 'Cannot capture source'

    datalen = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_imgs = []
    im_names = []
    
    frame_num = 0
    for k in range(datalen):
        (grabbed, frame) = stream.read()
        # if the `grabbed` boolean is `False`, then we have
        # reached the end of the video file
        if not grabbed:
            stream.release()
            break

        # orig_imgs.append(frame[:, :, ::-1])
        if undistort:
            frame = cv2.undistort(frame, intr, dist, None, dist_intr)
        orig_imgs.append(frame)
        im_names.append(f'{frame_num:08d}' + '.jpg')
        frame_num += 1
    H, W, _ = frame.shape
    stream.release()

    # print(f'Total number of frames: {frame_num} in {path}')
    return im_names, orig_imgs, H, W

def create_video_writer(filename, frame_size, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)

def convert_video_ffmpeg(input_path):
    temp_path = input_path + '.temp'
    output_path = input_path
    
    # Rename the original file
    os.rename(input_path, temp_path)
    
    # Construct the ffmpeg command with the -y option
    ffmpeg_command = f'ffmpeg -i {temp_path} -vcodec libx264 -y {output_path}'
    
    # Execute the command
    os.system(ffmpeg_command)
    
    # Remove the temporary file
    os.remove(temp_path)

# Sample function: test 3 with multiview + pressure in 7x7 grid
def render_multiview(view_batches, view_names_list, output_dir, grid_size=(7, 7), frame_size=(360, 640)):
    os.makedirs(output_dir, exist_ok=True)
    frame_count = len(view_batches)
    grid_h, grid_w = grid_size
    cell_h, cell_w = frame_size

    # Calculate grid and pressure map dimensions
    grid_img_h = grid_h * cell_h
    grid_img_w = grid_w * cell_w

    canvas_h = grid_img_h
    canvas_w = grid_img_w

    for idx in range(frame_count):
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
        frames = view_batches[idx]
        names = view_names_list[idx]
        # Draw grid of views on the left
        for i, (name, frame) in enumerate(zip(names, frames)):
            row, col = divmod(i, grid_w)
            x, y = col * cell_w, row * cell_h
            frame_resized = cv2.resize(frame, (cell_w, cell_h))
            canvas[y:y+cell_h, x:x+cell_w] = frame_resized

        out_path = os.path.join(output_dir, f"frame_{idx:04d}.jpg")
        cv2.imwrite(out_path, canvas)

        print(f'Plot Frame {idx} at {out_path}')
    return True