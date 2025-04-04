import os
from os.path import join
import numpy as np
import cv2
import sys
sys.path.append(".")
from hand_utils.utils.renderer import Renderer

sys.path.append("./third-party/EasyMocap")
from easymocap.mytools.file_utils import get_bbox_from_pose
from easymocap.mytools.vis_base import plot_bbox, plot_keypoints, merge

def interpolate_vertices_and_colors(verts1, verts2, color_scheme1, color_scheme2, interpolation_factor):
    """
    Interpolate between two vertex sets and assign colors based on interpolation factor.
    
    Parameters:
        verts1 (np.ndarray): First set of vertices of shape (n, 3).
        verts2 (np.ndarray): Second set of vertices of shape (n, 3).
        color_scheme1 (np.ndarray): Color range for the first set of vertices, shape (2, 3) or (2, 4) for RGB/RGBA.
        color_scheme2 (np.ndarray): Color range for the second set of vertices, shape (2, 3) or (2, 4) for RGB/RGBA.
        interpolation_factor (float): A value in the range [0, 1] to interpolate between verts1 and verts2.
    
    Returns:
        np.ndarray: Concatenated set of colors with shape (2n, 3) or (2n, 4).
    """
    
    
    # Interpolate colors for verts1
    colors1_start = color_scheme1[0]
    colors1_end = color_scheme1[1]
    interpolated_colors1 = colors1_start * (1 - interpolation_factor) + colors1_end * interpolation_factor
    
    # Interpolate colors for verts2
    colors2_start = color_scheme2[0]
    colors2_end = color_scheme2[1]
    interpolated_colors2 = colors2_start * (1 - interpolation_factor) + colors2_end * interpolation_factor
    
    # Stack the interpolated colors to match the vertices
    colors1 = np.tile(interpolated_colors1, (verts1.shape[0], 1))
    colors2 = np.tile(interpolated_colors2, (verts2.shape[0], 1))
    
    # Concatenate the color sets
    concatenated_colors = np.vstack((colors1, colors2))
    
    return concatenated_colors

# Modified from EasyMocap/easymocap/smplmodel/body_model.py
def load_model(gender='neutral', use_cuda=True, model_type='smpl', skel_type='body25', device=None, model_path='data/smplx', **kwargs):
    # prepare SMPL model
    # print('[Load model {}/{}]'.format(model_type, gender))
    import torch
    if device is None:
        if use_cuda and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    from easymocap.smplmodel.body_model import SMPLlayer
    if model_type == 'smpl':
        if skel_type == 'body25':
            reg_path = join(model_path, 'J_regressor_body25.npy')
        elif skel_type == 'h36m':
            reg_path = join(model_path, 'J_regressor_h36m.npy')
        else:
            raise NotImplementedError
        body_model = SMPLlayer(join(model_path, 'smpl'), gender=gender, device=device,
            regressor_path=reg_path, **kwargs)
    elif model_type == 'smplh':
        body_model = SMPLlayer(join(model_path, 'smplh/SMPLH_MALE.pkl'), model_type='smplh', gender=gender, device=device,
            regressor_path=join(model_path, 'J_regressor_body25_smplh.txt'), **kwargs)
    elif model_type == 'smplx':
        body_model = SMPLlayer(join(model_path, 'smplx/SMPLX_{}.pkl'.format(gender.upper())), model_type='smplx', gender=gender, device=device,
            regressor_path=join(model_path, 'J_regressor_body25_smplx.txt'), **kwargs)
    elif model_type == 'manol' or model_type == 'manor':
        lr = {'manol': 'LEFT', 'manor': 'RIGHT'}
        body_model = SMPLlayer(join(model_path, 'smplh/MANO_{}.pkl'.format(lr[model_type])), model_type='mano', gender=gender, device=device,
            regressor_path=join(model_path, 'J_regressor_mano_{}.txt'.format(lr[model_type])), **kwargs)
    else:
        body_model = None
    body_model.to(device)
    return body_model

def vis_smpl(args, vertices, faces, images, nf, cameras, colors=None, mode='smpl', extra_data=[], add_back=True, out_dir='mano'):
    render_data = {}
    pid = 0
    for vertice, face in zip(vertices, faces):
        assert vertice.shape[1] == 3 and len(vertice.shape) == 2, 'shape {} != (N, 3)'.format(vertice.shape)
        render_data[pid] = {'vertices': vertice, 'faces': face, 
            'vid': pid, 'name': 'human_{}_{}'.format(nf, pid)}
        if colors is not None:
            render_data[pid]['colors'] = colors
        pid += 1
    render = Renderer(height=1024, width=1024, faces=None)
    render_results = render.render(render_data, cameras, images, add_back=add_back)
    image_vis = merge(render_results, resize=not args.save_origin)
    if args.save_frame:
        outname = os.path.join(out_dir, '{:08d}.jpg'.format(nf))
        cv2.imwrite(outname, image_vis)
    # else:
    #     out_dir.write(image_vis)
    
    return image_vis


def vis_smpl_stack(args, vertices, faces, images, nf, cameras, mode='smpl', extra_data=[], out_dir='mano', inter_factor=1):
    sunset_scheme = np.array([[255, 69, 0], [106, 90, 205]])  # Deep Orange to Soft Purple
    ocean_breeze_scheme = np.array([[0, 255, 255], [0, 0, 139]])  # Aqua to Deep Blue
    
    
    render_data = {}
    pid = 0
    for vertice, face in zip(vertices, faces):
        assert vertice.shape[1] == 3 and len(vertice.shape) == 2, 'shape {} != (N, 3)'.format(vertice.shape)
        sunset_colors = interpolate_vertices_and_colors(vertice[:vertice.shape[0]//2, :], vertice[vertice.shape[0]//2:, :], sunset_scheme, ocean_breeze_scheme, inter_factor)
        render_data[pid] = {'vertices': vertice, 'faces': face, 'colors':sunset_colors/255, 
            'vid': pid, 'name': 'human_{}_{}'.format(nf, pid)}
        pid += 1
    render = Renderer(height=1024, width=1024, faces=None)
    render_results = render.render(render_data, cameras, images, add_back=False)
    image_vis = merge(render_results, resize=not args.save_origin)
    
    return render_results


# project 3d keypoints from easymocap/mytools/reconstruction.py
def projectN3(kpts3d, cameras):
    # kpts3d: (N, 3)
    nViews = len(cameras)
    kp3d = np.hstack((kpts3d[:, :3], np.ones((kpts3d.shape[0], 1))))
    kp2ds = []
    for nv in range(nViews):
        kp2d = cameras[nv] @ kp3d.T
        kp2d[:2, :] /= kp2d[2:, :]
        kp2ds.append(kp2d.T[None, :, :])
    kp2ds = np.vstack(kp2ds)
    if kpts3d.shape[-1] == 4:
        kp2ds[..., -1] = kp2ds[..., -1] * (kpts3d[None, :, -1] > 0.)
    return kp2ds

# visualize reprojection from easymocap/dataset/mv1pmf.py
def vis_repro(args, images, kpts_repros, nf, config, to_img=True, mode='repro', outdir='mano_keypoints', vis_id=True):
    
    images_vis = []
    for nv, image in enumerate(images):
        img = image.copy()
        for pid, kpts_repro in enumerate(kpts_repros):
            keypoints = kpts_repro[nv]
            bbox = get_bbox_from_pose(kpts_repro[nv], images[nv])
            plot_bbox(img, bbox, pid=pid, vis_id=vis_id)
            plot_keypoints(img, keypoints, pid=pid, config=config, use_limb_color=True if pid==0 else False, lw=4)
        images_vis.append(img)
    if len(images_vis) > 1:
        images_vis = merge(images_vis, resize=not args.save_origin)
    else:
        images_vis = images_vis[0]
    if args.save_frame:
        outname = os.path.join(outdir, '{:06d}.jpg'.format(nf))
        cv2.imwrite(outname, images_vis)
    # else:
    #     outdir.write(images_vis)
    return images_vis
# ----------------------------------------------------------------- #
