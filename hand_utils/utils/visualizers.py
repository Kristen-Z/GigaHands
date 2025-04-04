import numpy as np
import argparse

import os
import cv2
import sys
from natsort import natsorted
import datetime
from glob import glob

import numpy as np
from tqdm import tqdm

sys.path.append(".")
from hand_utils.utils.easymocap_utils import vis_smpl, projectN3, vis_repro, load_model
from hand_utils.utils.video_handler import create_video_writer, convert_video_ffmpeg, add_text_to_frame
import hand_utils.utils.camera_utils as param_utils
sys.path.append("./third-party/EasyMocap")
from easymocap.dataset import CONFIG
from easymocap.pipeline import smpl_from_keypoints3d
from easymocap.smplmodel import select_nf
os.environ["PYOPENGL_PLATFORM"] = "egl"

def read_params(params_path):
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

def get_projections(params, cam_names, n_Frames):
    # Gets the projection matrices and distortion parameters
    projs = []
    intrs = []
    dist_intrs = []
    dists = []
    rot = []
    trans = []

    for param in params:
        if (param["cam_name"] == cam_names):
            extr = param_utils.get_extr(param)
            intr, dist = param_utils.get_intr(param)
            r, t = param_utils.get_rot_trans(param)

            rot.append(r)
            trans.append(t)

            intrs.append(intr.copy())
            
            dist_intrs.append(intr.copy())

            projs.append(intr @ extr)
            dists.append(dist)

    cameras = { 
        'K': np.repeat(np.asarray(intrs), n_Frames, axis=0),
        'R': np.repeat(np.asarray(rot), n_Frames, axis=0), 
        'T': np.repeat(np.asarray(trans), n_Frames, axis=0),
        'dist': np.repeat(np.asarray(dists), n_Frames, axis=0),
        'P': np.repeat(np.asarray(projs), n_Frames, axis=0)
        }

    return intrs, np.asarray(projs), dist_intrs, dists, cameras

def parse_timestamp(filename):
    # Strip the extension and parse the remaining part as a datetime
    timestamp_str = filename.split('_')[-1].split('.')[0]
    return datetime.datetime.fromtimestamp(int(timestamp_str) / 1e6)

def find_closest_video(folder, anchor_timestamp):
    min_diff = datetime.timedelta(seconds=1)  # Max allowed difference
    closest_video = None

    for filename in os.listdir(folder):
        if filename.endswith('.mp4'):
            video_timestamp = parse_timestamp(filename)
            time_diff = abs(video_timestamp - anchor_timestamp)
            
            if time_diff < min_diff:
                min_diff = time_diff
                closest_video = filename
    
    return closest_video, min_diff

# -------------------- Visualization Functions -------------------- #
def plot_3d_hand_motion(motion, gt_motion=None, data_path='./dataset/hand_pose', out_path=None, save_file=None, to_vis_mano=True, add_back=False, scene_path='./p001-folder', camera_view='brics-odroid-021_cam1', save_mesh=False, vis_3d_repro=False):

    parser = argparse.ArgumentParser("Mano Fitting Argument Parser")
    parser.add_argument('--name', type=str)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--text_prompt', type=str)
    parser.add_argument('--input_text', type=str)
    parser.add_argument('--num_repetitions', type=int)
    parser.add_argument('--model', type=str, default='manor')
    parser.add_argument('--body', type=str, default='handr')
    parser.add_argument('--gender', type=str, default='neutral')
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--robust3d', type=bool, default=False)
    parser.add_argument('--save_origin', type=bool, default=False)
    parser.add_argument('--save_frame', type=bool, default=False)
    parser.add_argument('--proj_share_weight', action="store_true", help='Training iterations')
    parser.add_argument('--sample', action="store_true")
    
    args = parser.parse_args()
    
    # Total Frames
    n_Frames = motion.shape[0]
    # Loads the camera parameters
    params_path = os.path.join(data_path, scene_path, "optim_params.txt")
    params = read_params(params_path)
    
    intrs, projs, dist_intrs, dists, cameras = get_projections(params, camera_view, n_Frames)

    # adjust the camera
    # cameras['T'] += [0, 0.25, -0.5] # move up Y+, move left X+

    body_model_right = load_model(gender='neutral', model_type='manor', model_path="body_models", num_pca_comps=6, use_pose_blending=True, use_shape_blending=True, use_pca=False, use_flat_mean=False)
    body_model_left = load_model(gender='neutral', model_type='manol', model_path="body_models", num_pca_comps=6, use_pose_blending=True, use_shape_blending=True, use_pca=False, use_flat_mean=False)

    # fits the mano model
    dataset_config = CONFIG['handr']
    weight_pose = {
        'k3d': 1e2, 'k2d': 2e-3,
        'reg_poses': 1e-3, 'smooth_body': 1e2, 'smooth_poses': 1e2,
    }

    cof_motion_right = np.pad(motion[:,21:,:], ((0, 0), (0, 0), (0, 1)), 'constant', constant_values=1)
    cof_motion_left = np.pad(motion[:,:21,:], ((0, 0), (0, 0), (0, 1)), 'constant', constant_values=1)
    params_right = smpl_from_keypoints3d(body_model_right, cof_motion_right, 
        config=dataset_config, args=args, weight_shape={'s3d': 1e5, 'reg_shapes': 5e3}, weight_pose=weight_pose)
    params_left = smpl_from_keypoints3d(body_model_left, cof_motion_left, 
        config=dataset_config, args=args, weight_shape={'s3d': 1e5, 'reg_shapes': 5e3}, weight_pose=weight_pose)
    
    if gt_motion is not None:
        gt_n_Frames = gt_motion.shape[0]
        gt_motion = np.pad(gt_motion, ((0, 0), (0, 0), (0, 1)), 'constant', constant_values=1)
        gt_motion_right = gt_motion[:,21:,:]
        gt_motion_left = gt_motion[:,:21,:]
        gt_params_right = smpl_from_keypoints3d(body_model_right, gt_motion_right, 
            config=dataset_config, args=args, weight_shape={'s3d': 1e5, 'reg_shapes': 5e3}, weight_pose=weight_pose)
        gt_params_left = smpl_from_keypoints3d(body_model_left, gt_motion_left, 
            config=dataset_config, args=args, weight_shape={'s3d': 1e5, 'reg_shapes': 5e3}, weight_pose=weight_pose)        
            
    # visualize model
    if to_vis_mano or save_mesh or vis_3d_repro:
        import trimesh
        white_background = np.zeros((720, 1280, 3), dtype=np.uint8)+255 # use white background when raw video is missing
        images = [white_background]
        text = save_file.replace('.mp4', '').replace('_', ' ')
        if to_vis_mano:
            os.makedirs(f'{out_path}/mano_vid', exist_ok=True)
            outhand_mano_path = f'{out_path}/mano_vid/{save_file}'

        if vis_3d_repro:
            os.makedirs(f'{out_path}/repro_3d_vid', exist_ok=True)
            outhand_3d_path = f'{out_path}/repro_3d_vid/{save_file}'
        
        nf = 0
        
        for abs_idx in tqdm(range(n_Frames), total=n_Frames):
            # hand color
            ocean_breeze_scheme = np.array([[20,150,255], [0, 60, 255]])
            # save the mano render
            if to_vis_mano:
                vertices = []
                faces = []
                colors = [ocean_breeze_scheme[1]/255]
                
                param_right = select_nf(params_right, nf)
                param_left = select_nf(params_left, nf)
                vertices_right = body_model_right(return_verts=True, return_tensor=False, **param_right)
                vertices_left = body_model_left(return_verts=True, return_tensor=False, **param_left)   
                vertice = np.concatenate((vertices_left[0], vertices_right[0]), axis=0)
                face = np.concatenate((body_model_left.faces, body_model_right.faces+vertices_left[0].shape[0]), axis=0)
                vertices.append(vertice)
                faces.append(face)
                image_vis = vis_smpl(args, vertices=vertices, faces=faces, images=images, nf=nf, cameras=cameras, colors=colors, add_back=add_back, out_dir=outhand_mano_path)

                if gt_motion is not None and nf < gt_n_Frames:
                    vertices = []
                    faces = []
                    colors = [ocean_breeze_scheme[0]/255]
                    gt_param_right = select_nf(gt_params_right, nf)
                    gt_param_left = select_nf(gt_params_left, nf)
                    gt_vertices_right = body_model_right(return_verts=True, return_tensor=False, **gt_param_right)
                    gt_vertices_left = body_model_left(return_verts=True, return_tensor=False, **gt_param_left)
                    gt_vertice = np.concatenate((gt_vertices_left[0], gt_vertices_right[0]), axis=0)
                    gt_face = np.concatenate((body_model_left.faces, body_model_right.faces+gt_vertices_left[0].shape[0]), axis=0)
                    vertices.append(gt_vertice)
                    faces.append(gt_face)                         
                    image_vis_gt = vis_smpl(args, vertices=vertices, faces=faces, images=images, nf=nf, cameras=cameras, colors=colors, add_back=add_back, out_dir=outhand_mano_path)
                    image_vis = np.hstack((image_vis[:, :, :3], image_vis_gt[:, :, :3]))
                
                image_vis = add_text_to_frame(image_vis, text)
                
                if nf == 0:
                    outhand_mano = create_video_writer(outhand_mano_path, (image_vis.shape[1], image_vis.shape[0]), fps=30)

                outhand_mano.write(image_vis)
                        
            vis_config = CONFIG['handlr']
            # save the 3d reprojected hands
            if vis_3d_repro:
                kpts_repros = []
                keypoints = np.concatenate((cof_motion_left[abs_idx], cof_motion_right[abs_idx]), axis=0)
                kpts_repro = projectN3(keypoints, projs)
                kpts_repro[:, :, 2] = 0.5
                kpts_repros.append(kpts_repro)
                if gt_motion is not None and nf < gt_n_Frames:
                    gt_kpts_repro = projectN3(gt_motion[abs_idx], projs)
                    kpts_repros.append(gt_kpts_repro)

                image_vis = vis_repro(args, images, kpts_repros, config=vis_config, nf=nf, mode='repro_smpl', outdir=outhand_3d_path)
                image_vis = add_text_to_frame(image_vis, text)
                if abs_idx == 0:
                    outhand_3d = create_video_writer(outhand_3d_path, (image_vis.shape[1], image_vis.shape[0]), fps=30)
                
                outhand_3d.write(image_vis)

            # save the mesh
            if save_mesh:
                vertices = np.concatenate((vertices_left[0], vertices_right[0]), axis=0)
                faces = np.concatenate((body_model_left.faces, body_model_right.faces+vertices_left[0].shape[0]), axis=0)
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                outdir = os.path.join(out_path, f'meshes/{save_file[:-4]}')
                os.makedirs(outdir, exist_ok=True)
                outname = os.path.join(outdir, '{:08d}.obj'.format(nf))
                mesh.export(outname)
                
            nf += 1

        if to_vis_mano:
            outhand_mano.release()
            print(outhand_mano_path)
            convert_video_ffmpeg(outhand_mano_path)
            print('Mano Video Handler Released')
        if vis_3d_repro:
            outhand_3d.release()
            convert_video_ffmpeg(outhand_3d_path)
            print('Repro Video Handler Released')


