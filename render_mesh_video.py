import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import trimesh
from scipy.spatial.transform import Slerp, Rotation
from scipy.interpolate import interp1d
import imageio

import sys
sys.path.append(".")
sys.path.append("./third-party/EasyMocap")
os.environ["PYOPENGL_PLATFORM"] = "egl"

from easymocap.smplmodel import select_nf
from easymocap.visualize.renderer import Renderer
from utils.easymocap_utils import load_model
from utils import camparam_utils as param_utils

# ==== Configuration ====
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Render mesh/hand pose sequences from tracked data and save as video."
    )
    parser.add_argument('--dataset_root', type=Path, required=True,
                        help='Path to the dataset root folder')
    parser.add_argument('--scene_name', type=str, required=True,
                        help='Scene name (e.g. 17_instruments)')
    parser.add_argument('--session_name', type=str, required=True,
                        help='Session name (e.g. p003-instrument)')
    parser.add_argument('--seq_id', type=int, required=True,
                        help='Sequence ID (e.g. 33)')
    parser.add_argument('--object_name', type=str, required=True,
                        help='Object name (e.g. ukelele_scan)')
    parser.add_argument('--mesh_name', type=str, required=True,
                        help='Mesh filename (e.g. ukelele-simplified1_1.obj)')
    parser.add_argument('--render_camera', type=str, default='brics-odroid-011_cam0',
                        help='Camera name for rendering')
    parser.add_argument('--save_root', type=Path, default=Path('visualizations'),
                        help='Output folder for visualizations')
    return parser.parse_args()

# ==== Utility Functions ====

def read_params(params_path):
    """Reads camera intrinsics and extrinsics from a formatted txt file."""
    params = np.loadtxt(
        params_path,
        dtype=[
            ("cam_id", int),
            ("width", int),
            ("height", int),
            ("fx", float),
            ("fy", float),
            ("cx", float),
            ("cy", float),
            ("k1", float),
            ("k2", float),
            ("p1", float),
            ("p2", float),
            ("cam_name", "<U22"),
            ("qvecw", float),
            ("qvecx", float),
            ("qvecy", float),
            ("qvecz", float),
            ("tvecx", float),
            ("tvecy", float),
            ("tvecz", float),
        ]
    )
    params = np.sort(params, order="cam_name")
    return params

def get_projections(params, cam_names, n_Frames=1):
    """Returns camera intrinsics, extrinsics, projections, and distortion parameters for the named camera."""
    projs, intrs, dists, rot, trans = [], [], [], [], []
    for param in params:
        if param["cam_name"] == cam_names:
            extr = param_utils.get_extr(param)
            intr, dist = param_utils.get_intr(param)
            r, t = param_utils.get_rot_trans(param)
            rot.append(r)
            trans.append(t)
            intrs.append(intr.copy())
            projs.append(intr @ extr)
            dists.append(dist)
    cameras = {
        'K': np.repeat(np.asarray(intrs), n_Frames, axis=0),
        'R': np.repeat(np.asarray(rot), n_Frames, axis=0),
        'T': np.repeat(np.asarray(trans), n_Frames, axis=0),
        'dist': np.repeat(np.asarray(dists), n_Frames, axis=0),
        'P': np.repeat(np.asarray(projs), n_Frames, axis=0)
    }
    return cameras

def object_pose_loader(object_pose_path, use_filter=True, use_smoother=True):
    """Load and optionally filter/smooth/interpolate object pose sequence from JSON."""
    with open(object_pose_path, "r") as f:
        object_poses = json.load(f)
    tracked_object_frames = sorted([int(frame_id) for frame_id in object_poses.keys()])
    chosen_object_poses, chosen_object_frames = interpolate_object_poses(object_poses, tracked_object_frames, use_filter=use_filter, use_smoother=use_smoother)
    return chosen_object_poses, chosen_object_frames

def hand_pose_loader(keypoints3d_path):
    """Find frame indices for which both left and right hand pose data are present."""
    chosen_path_left = Path(keypoints3d_path) / "chosen_frames_left.json"
    chosen_path_right = Path(keypoints3d_path) / "chosen_frames_right.json"
    with open(chosen_path_right, "r") as f:
        chosen_frames_right = set(json.load(f))
    with open(chosen_path_left, "r") as f:
        chosen_frames_left = set(json.load(f))
    chosen_hand_union_frames = list(chosen_frames_right | chosen_frames_left)
    chosen_hand_intersect_frames = list(chosen_frames_right & chosen_frames_left)
    return chosen_hand_union_frames, chosen_hand_intersect_frames

def interpolate_object_poses(object_poses, tracked_frames, use_filter=True, use_smoother=True):
    """Interpolate and smooth translation and rotation in the dictionary."""
    trans, rots, idxs = [], [], []
    for cid in tracked_frames:
        frame = str(cid).zfill(6)
        if frame in object_poses:
            trans.append(np.asarray(object_poses[frame]["mesh_translation"]).squeeze())
            rots.append(np.asarray(object_poses[frame]["mesh_rotation"]))
            idxs.append(cid)
    trans, rots, idxs = np.array(trans), np.array(rots), np.array(idxs)
    # Interpolate translations and rotations
    full_idx = np.arange(idxs[0], idxs[-1] + 1)
    interp_t = interp1d(idxs, trans, axis=0, kind='linear', fill_value="extrapolate")(full_idx)
    if rots.shape[1] == 3:
        r = Rotation.from_rotvec(rots)
        out_fmt = "rotvec"
    else:
        r = Rotation.from_quat(rots)
        out_fmt = "quat"
    interp_r = Slerp(idxs, r)(full_idx)
    interp_r = interp_r.as_rotvec() if out_fmt == "rotvec" else interp_r.as_quat()
    # Smoothing using moving average
    if use_smoother:
        interp_t = moving_average_filter(interp_t, window_size=9)
        interp_r = moving_average_filter(interp_r, window_size=9)
    inter_poses = {
        str(fid).zfill(6): {
            "mesh_translation": interp_t[i].tolist(),
            "mesh_rotation": interp_r[i].tolist()
        }
        for i, fid in enumerate(full_idx)
    }
    frame_idx = [fid for i, fid in enumerate(full_idx)]
    return inter_poses, frame_idx

def moving_average_filter(signal, window_size=5):
    """Apply moving average filter with edge-padding to smooth the signal."""
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    pad_len = window_size // 2
    padded = np.pad(signal, ((pad_len, pad_len), (0, 0)), mode='edge')
    kernel = np.ones(window_size) / window_size
    smoothed_signal = np.array([
        np.convolve(padded[:, i], kernel, mode='valid')
        for i in range(signal.shape[1])
    ]).T
    return smoothed_signal.squeeze()

def load_body_model(model_path="body_models"):
    """Load left and right MANO hand models."""
    body_model_right = load_model(
        gender='neutral', model_type='manor', model_path=model_path,
        num_pca_comps=6, use_pose_blending=True, use_shape_blending=True,
        use_pca=False, use_flat_mean=False)
    body_model_left = load_model(
        gender='neutral', model_type='manol', model_path=model_path,
        num_pca_comps=6, use_pose_blending=True, use_shape_blending=True,
        use_pca=False, use_flat_mean=False)
    return body_model_right, body_model_left

def hand_mano_loader(path, idx_in_hand_iou_indices):
    """Load and select MANO hand model parameters for frames of interest."""
    with open(path, 'r') as f:
        manos_params = json.load(f)
    params_left_list = manos_params['left']
    params_right_list = manos_params['right']
    params_left = {k: np.asarray(v) for k, v in params_left_list.items()}
    params_right = {k: np.asarray(v) for k, v in params_right_list.items()}
    choosen_frame = np.asarray(list(range(len(params_left_list['poses']))))
    param_right_all, param_left_all = [], []
    for nf in choosen_frame[idx_in_hand_iou_indices]:
        param_right = select_nf(params_right, nf)
        param_left = select_nf(params_left, nf)
        param_right_all.append(param_right)
        param_left_all.append(param_left)
    return param_right_all, param_left_all

def pytorch3d_quat_to_rotmat(quat_wxyz):
    """Convert a quaternion from PyTorch3D ([w, x, y, z]) to a rotation matrix using scipy, handling coordinate conventions."""
    # Convert to [x, y, z, w]
    quat_xyzw = np.asarray([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    # .T is required for our dataset's handedness (PyTorch3D to scipy)
    return Rotation.from_quat(quat_xyzw).as_matrix().T

def object_transform_loader(valid_frames, chosen_object_poses):
    """
    For each valid frame, construct a 4x4 transformation matrix from quaternion/rotvec and translation.
    The quaternion case expects [w, x, y, z] as from PyTorch3D and converts to scipy format.
    """
    all_transforms = []
    for cid in valid_frames:
        pose = chosen_object_poses[str(cid).zfill(6)]
        mesh_translation = np.asarray(pose["mesh_translation"]).reshape(3,)
        mesh_rotation = np.asarray(pose["mesh_rotation"])
        if mesh_rotation.shape[0] == 3:
            # Rotation vector (axis-angle)
            R = Rotation.from_rotvec(mesh_rotation).as_matrix()
        else:
            # Quaternion [w, x, y, z] from PyTorch3D
            R = pytorch3d_quat_to_rotmat(mesh_rotation)
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = mesh_translation
        all_transforms.append(transform)
    return all_transforms

# ==== Main Script ====

def main():
    args = parse_args()

    object_pose_path = args.dataset_root / "object_poses" / args.scene_name / args.object_name / f"{args.session_name}_{str(args.seq_id).zfill(4)}" / "pose" / "optimized_pose.json"
    hand_pose_path = args.dataset_root / "hand_poses" / args.session_name
    mesh_path = args.dataset_root / "object_meta" / args.scene_name / args.object_name / args.mesh_name
    save_path = args.save_root / args.scene_name / f"{args.session_name}_{str(args.seq_id).zfill(4)}"
    save_path.mkdir(parents=True, exist_ok=True)
    keypoints3d_path = hand_pose_path / "keypoints_3d" / f"{int(args.seq_id):03d}"
    mano_main_path = hand_pose_path / "params" / f"{int(args.seq_id):03d}.json"
    camera_params_path = hand_pose_path / "optim_params.txt"

    # Load camera parameters
    params = read_params(camera_params_path)
    cameras = get_projections(params, args.render_camera, n_Frames=1)

    # Load object and hand poses
    chosen_object_poses, chosen_object_frames = object_pose_loader(object_pose_path, use_filter=True, use_smoother=True)
    chosen_hand_union_frames, chosen_hand_intersect_frames = hand_pose_loader(keypoints3d_path)
    # Find joint-valid frame numbers
    chosen_video_frames, _, _ = np.intersect1d(chosen_object_frames, chosen_hand_intersect_frames, return_indices=True)
    valid_length = len(chosen_video_frames)
    if valid_length == 0:
        print("No frames with both object and hand pose detected.")
        return

    # Load MANO parameters of joint-valid frames
    video_indices_in_hand_iou_indices = np.asarray([chosen_hand_union_frames.index(f) for f in chosen_video_frames])
    mano_params_right, mano_params_left = hand_mano_loader(mano_main_path, video_indices_in_hand_iou_indices)
    # Load object transforms for valid frames
    object_transforms = object_transform_loader(chosen_video_frames, chosen_object_poses)

    # Load object mesh (keep base copy)
    object_mesh_base = trimesh.load(mesh_path, process=False)
    object_vertices = np.hstack([object_mesh_base.vertices, np.ones((object_mesh_base.vertices.shape[0], 1))])
    body_model_right, body_model_left = load_body_model(model_path="body_models")

    render = Renderer(height=720, width=1280, faces=None, extra_mesh=[])

    frames = []
    for abs_idx in tqdm(range(valid_length), desc="Rendering frames"):
        param_right = mano_params_right[abs_idx]
        param_left = mano_params_left[abs_idx]
        transform = object_transforms[abs_idx]
        vertices_right = body_model_right(return_verts=True, return_tensor=False, **param_right)[0]
        vertices_left = body_model_left(return_verts=True, return_tensor=False, **param_left)[0]
        faces = body_model_left.faces
        # Transform object mesh for this frame (make a copy)
        frame_mesh = object_mesh_base.copy()
        vertices_transformed = (transform @ object_vertices.T).T[:, :3]
        frame_mesh.vertices = vertices_transformed

        image = np.zeros((720, 1280, 3))
        render_data = {
            0: {'vertices': vertices_right, 'faces': faces, 'vid': 1, 'name': f'human_{abs_idx}_0'},
            1: {'vertices': vertices_left, 'faces': faces, 'vid': 4, 'name': f'human_{abs_idx}_1'}
        }
        render.extra_mesh = [frame_mesh]
        render_results = render.render(render_data, cameras, [image], add_back=False)
        image_vis = render_results[0][:, :, [2, 1, 0, 3]]  # BGR -> RGB, preserve alpha
        frames.append(image_vis.astype(np.uint8))

    # Save video
    output_video = save_path / 'output.mp4'
    imageio.mimsave(str(output_video), frames, fps=30)
    print(f"Saved output video to {output_video}")

if __name__ == "__main__":
    main()
