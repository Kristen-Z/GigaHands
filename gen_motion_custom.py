import os 
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import json
import clip
import codecs as cs

import options.option_transformer as option_trans
import models.vqvae as vqvae
import utils.utils_model as utils_model
from dataset.dataset_hands import GigaHands
from hand_utils.utils.visualizers import plot_3d_hand_motion
import hand_utils.eval_trans_hand as eval_trans_hand
import models.t2m_trans as trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')

##### ---- Exp dirs ---- #####
args = option_trans.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)

##### ---- Dataset ---- #####
mode = 'giga'
dataset_opt_path = f'./dataset/opt.txt'
wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda:0'))

# load motion mean, std
mean_path = 'hand_utils/giga_mean_kp.npy'
std_path = 'hand_utils/giga_std_kp.npy'
mean = np.load(mean_path)
std = np.load(std_path)

##### ---- Network ---- #####

## load clip model
clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)  # Must set jit=False for training
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

net = vqvae.HandVQVAE(args, ## use args to define different parameters in different quantizers
                    args.nb_code,
                    args.code_dim,
                    args.output_emb_width,
                    args.down_t,
                    args.stride_t,
                    args.width,
                    args.depth,
                    args.dilation_growth_rate)

trans_encoder = trans.Text2Motion_Transformer(num_vq=args.nb_code, 
                                embed_dim=args.embed_dim_gpt, 
                                clip_dim=args.clip_dim, 
                                block_size=args.block_size, 
                                num_layers=args.num_layers, 
                                n_head=args.n_head_gpt, 
                                drop_out_rate=args.drop_out_rate, 
                                fc_rate=args.ff_rate)

print ('loading VQVAE checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda()

if args.resume_trans is not None:
    print ('loading transformer checkpoint from {}'.format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    trans_encoder.load_state_dict(ckpt['trans'], strict=True)
trans_encoder.train()
trans_encoder.cuda()

with cs.open(args.input_text) as f:
    for line in f.readlines():
        clip_text = line.strip()
        text = clip.tokenize(clip_text, truncate=True).cuda()

        feat_clip_text = clip_model.encode_text(text).float()
        index_motion = trans_encoder.sample(feat_clip_text[0:1], False)
        pred_pose = net.forward_decoder(index_motion).detach().cpu().numpy()
        pred_denorm = pred_pose*std + mean
        pred_xyz = pred_denorm.reshape(pred_denorm.shape[1],-1,3)

        save_dir = os.path.join(args.out_dir, 'motions')
        os.makedirs(save_dir, exist_ok=True)
        save_file = clip_text.replace('\n', '').replace(' ', '_') + '.mp4'
        plot_3d_hand_motion(pred_xyz, gt_motion=None, out_path=save_dir, save_file=save_file, add_back=True, data_path=wrapper_opt.dataset_root)
        print('Saved motion:', clip_text)
