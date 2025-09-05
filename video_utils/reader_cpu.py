import cv2
import os
import ffmpeg
import numpy as np
from glob import glob
from natsort import natsorted
from typing import Dict, Iterable, Generator, Tuple
import json
import ipdb
import datetime
from tqdm import tqdm

def parse_timestamp(filename):
    # Strip the extension and parse the remaining part as a datetime
    timestamp_str = filename.split('_')[-1].split('.')[0]
    return datetime.datetime.fromtimestamp(int(timestamp_str) / 1e6)

def find_closest_video(folder, anchor_timestamp):
    min_diff = datetime.timedelta(seconds=0.05)  # Max allowed difference
    closest_video = None

    for filename in os.listdir(folder):
        if filename.endswith('.mp4'):
            video_timestamp = parse_timestamp(filename)
            time_diff = abs(video_timestamp - anchor_timestamp)
            
            if time_diff < min_diff:
                min_diff = time_diff
                closest_video = filename
    
    return closest_video, min_diff

class Reader():
    iterator = []

    def __init__(
        self,
        video_path: str,
        undistort: bool = False,
        cams_to_remove=None,
        ith: int = 0,
        anchor_camera=None,
    ):
        """ith: the ith video in each folder will be processed."""
        self.ith = ith
        self.video_path = video_path
        self.undistort = undistort
        self.cams_to_remove = set(cams_to_remove or [])
        self.to_delete = []
        self.streams = {}
        self.vids = []
        self.frame_count = int(1e9)
        self.cur_frame = 0
        self.timestamps = {}

        assert anchor_camera is not None, "Anchor camera must be specified."

        # Step 1: Get ith anchor capture
        anchor_video, anchor_timestamp = self.find_ith_anchor_capture(anchor_camera)
        self.vids.append(anchor_video)
        self.anchor_timestamp = anchor_timestamp

        # Step 2: Find aligned videos from other views
        self._align_other_views(anchor_camera)

        # Step 3: Initialize video capture objects
        self.init_videos()

        assert self.frame_count < int(1e9), "No valid frames found."
        if self.frame_count <= 0:
            print("Warning: No usable frames detected across views.")


    def find_ith_anchor_capture(self, anchor_camera: str):
        """Find the ith video from the anchor camera folder and parse its timestamp."""
        anchor_cam_path = os.path.join(self.video_path, anchor_camera)
        anchor_mp4s = natsorted(glob(f"{anchor_cam_path}/*.mp4"))
        if len(anchor_mp4s) <= self.ith:
            raise IndexError(f"Not enough videos in {anchor_cam_path} to get index {self.ith}")
        anchor_video = anchor_mp4s[self.ith]
        anchor_timestamp = parse_timestamp(os.path.basename(anchor_video))
        return anchor_video, anchor_timestamp

    def _align_other_views(self, anchor_camera: str):
        """Find videos in other camera folders that align with anchor by timestamp."""
        for cam_folder in os.listdir(self.video_path):
            full_path = os.path.join(self.video_path, cam_folder)
            if (
                not os.path.isdir(full_path)
                or cam_folder in self.cams_to_remove
                or cam_folder == anchor_camera
                or "imu" in cam_folder
                or "mic" in cam_folder
            ):
                continue

            closest_file, _ = find_closest_video(full_path, self.anchor_timestamp)
            if closest_file:
                self.vids.append(os.path.join(full_path, closest_file))
            else:
                self.to_delete.append(cam_folder)

    def load_frame_timestamps_for_video(self, video_path: str) -> np.ndarray:
        txt_path = video_path.replace(".mp4", ".txt")
        if not os.path.isfile(txt_path):
            raise FileNotFoundError(f"Timestamp file not found: {txt_path}")

        timestamps = []
        with open(txt_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        ts = int(line.split("_")[1])
                        timestamps.append(ts)
                    except Exception as e:
                        print(f"Warning: Skipping malformed line in {txt_path}: {line}")
        return np.array(timestamps, dtype=np.int64)

    def init_videos(self):
        """Create video capture objects and compute minimum shared frame count."""
        self.vids = sorted(self.vids)
        for vid_path in self.vids:
            cap = cv2.VideoCapture(vid_path)
            try:
                frame_info = ffmpeg.probe(vid_path, cmd="ffprobe")["streams"][0]
                num_frames = int(frame_info.get("nb_frames", 0))
            except Exception as e:
                raise RuntimeError(f"ffprobe failed on {vid_path}: {e}")

            self.frame_count = min(self.frame_count, num_frames)
            cam_name = "_".join(vid_path.split("/")[-1].split("_")[:-1])
            self.streams[cam_name] = cap
            
            ts = self.load_frame_timestamps_for_video(vid_path)
            self.timestamps[cam_name] = ts

        self.frame_count = max(0, self.frame_count - 5)

    def _get_next_frame(self, frame_idx) -> Dict[str, np.ndarray]:
        """ Get next frame (stride 1) from each camera"""
        self.cur_frame = frame_idx
        
        if self.cur_frame == self.frame_count:
            return {}

        frames = {}
        for cam_name, cam_cap in self.streams.items():
            cam_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            suc, frame = cam_cap.read()
            if not suc:
                raise RuntimeError(f"Couldn't retrieve frame from {cam_name}")
            timestamp = self.timestamps.get(cam_name, [0] * self.frame_count)[frame_idx]
            frames[cam_name] = (frame, timestamp)

        return frames

    def buffer_view_frames(self, view_name: str, undistort=True, intr = None, proj = None, dist = None) -> np.ndarray:
        """ Buffer all frames from a specific view."""
        if view_name not in self.streams:
            raise ValueError(f"View '{view_name}' not found in loaded streams: {list(self.streams.keys())}")
        
        cap = self.streams[view_name]
        timestamps = self.timestamps.get(view_name, [])
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for _ in tqdm(range(min(self.frame_count, len(timestamps))), desc=f"Buffering {view_name}", leave=False):
            success, frame = cap.read()
            if not success:
                break
            if undistort:
                frame = cv2.undistort(frame, intr, dist, None, intr)
            frames.append(frame)

        return np.stack(frames), timestamps[:len(frames)]
            
    def reinit(self):
        """ Reinitialize the reader """
        self.release_videos()
        self.init_videos()

        self.cur_frame = 0

    def release_videos(self):
        for cap in self.streams.values():
            cap.release()
    
    def __call__(self, frames: Iterable[int]=[]):
        # Sort the frames so that we access them in order
        frames = sorted(frames)
        self.iterator = frames
        
        for frame_idx in frames:
            frame = self._get_next_frame(frame_idx)
            if not frame:
                break
                
            yield frame, self.cur_frame

        # Reinitialize the videos
        self.reinit()

if __name__ == "__main__":
    reader = Reader("video", "/hdd_data/common/BRICS/hands/peisen/actions/abduction_adduction/", 5, 16, 3)
    for i in range(len(reader)):
        frames, frame_num = reader.get_frames()
        print(frame_num)
