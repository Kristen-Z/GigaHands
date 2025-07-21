import os
from os.path import join
import numpy as np
import cv2
from easymocap.visualize.renderer import Renderer
from easymocap.mytools.file_utils import get_bbox_from_pose
from easymocap.mytools.vis_base import plot_bbox, plot_keypoints, merge

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

def vis_smpl(args, vertices, faces, images, nf, cameras, mode='smpl', extra_data=[], add_back=True, out_dir='mano'):
    render_data = {}
    assert vertices.shape[1] == 3 and len(vertices.shape) == 2, 'shape {} != (N, 3)'.format(vertices.shape)
    pid = 0
    render_data[pid] = {'vertices': vertices, 'faces': faces, 
        'vid': pid, 'name': 'human_{}_{}'.format(nf, pid)}
    render = Renderer(height=1024, width=1024, faces=None)
    render_results = render.render(render_data, cameras, images, add_back=add_back)
    image_vis = merge(render_results, resize=not args.save_origin)
    if args.save_frame:
        outname = os.path.join(out_dir, '{:08d}.jpg'.format(nf))
        cv2.imwrite(outname, image_vis)
    # else:
    #     out_dir.write(image_vis)
    return image_vis


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
def vis_repro(args, images, kpts_repro, nf, config, to_img=True, mode='repro', outdir='mano_keypoints', vis_id=True):
    lDetections = []
    for nv in range(len(images)):
        det = {
            'id': -1,
            'keypoints2d': kpts_repro[nv],
            'bbox': get_bbox_from_pose(kpts_repro[nv], images[nv])
        }
        lDetections.append([det])
    
    images_vis = []
    for nv, image in enumerate(images):
        img = image.copy()
        for det in lDetections[nv]:
            pid = det['id']
            if 'keypoints2d' in det.keys():
                keypoints = det['keypoints2d']
            else:
                keypoints = det['keypoints']
            if 'bbox' not in det.keys():
                bbox = get_bbox_from_pose(keypoints, img)
            else:
                bbox = det['bbox']
            plot_bbox(img, bbox, pid=pid, vis_id=vis_id)
            plot_keypoints(img, keypoints, pid=pid, config=config, use_limb_color=True, lw=4)
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