import os 
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
from torch.distributions import Categorical
import json
import clip
from tqdm import tqdm

import options.option_transformer as option_trans
import models.vqvae as vqvae
import utils.utils_model as utils_model
import hand_utils.eval_trans_hand as eval_trans_hand
from dataset import dataset_hands
from dataset.dataset_hands import GigaHands
import models.t2m_trans as trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')


##### ---- Exp dirs ---- #####
args = option_trans.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
args.vq_dir= os.path.join(f"./tokens", f'{args.vq_name}')
os.makedirs(args.out_dir, exist_ok = True)
os.makedirs(args.vq_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

##### ---- Dataloader ---- #####
from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vocab')


mode = 'giga'
dataset_opt_path = f'./dataset/opt.txt'
wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda:0'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

train_dataset = GigaHands(opt=wrapper_opt, mode=f'{mode}_motion', split='all', w_vectorizer=w_vectorizer, unit_length=2**args.down_t)
train_loader_token = torch.utils.data.DataLoader(train_dataset, 1, shuffle=False, num_workers=4, drop_last = False)

val_dataset = GigaHands(opt=wrapper_opt, mode=f'{mode}_motiontext_eval', split='val',w_vectorizer=w_vectorizer, augment_text = False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, drop_last=False, num_workers=4, shuffle=True,)


##### ---- Network ---- #####
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

print ('loading checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda()

trans_encoder = trans.Text2Motion_Transformer(num_vq=args.nb_code, 
                                embed_dim=args.embed_dim_gpt, 
                                clip_dim=args.clip_dim, 
                                block_size=args.block_size, 
                                num_layers=args.num_layers, 
                                n_head=args.n_head_gpt, 
                                drop_out_rate=args.drop_out_rate, 
                                fc_rate=args.ff_rate)

if args.resume_trans is not None:
    print ('loading transformer checkpoint from {}'.format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    trans_encoder.load_state_dict(ckpt['trans'], strict=True)
trans_encoder.train()
trans_encoder.cuda()

##### ---- Optimizer & Scheduler ---- #####
optimizer = utils_model.initial_optim(args.decay_option, args.lr, args.weight_decay, trans_encoder, args.optimizer)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

##### ---- Optimization goals ---- #####
loss_ce = torch.nn.CrossEntropyLoss()

nb_iter, avg_loss_cls, avg_acc = 0, 0., 0.
right_num = 0
nb_sample_train = 0

##### ---- get code ---- #####
# mtoken_data = []
for batch in tqdm(train_loader_token):
    pose, path = batch # bs, nb_joints, joints_dim, seq_len
    target = net.encode(pose.cuda().float())
    target = target.cpu().numpy()

    # For giga
    separa_path = os.path.normpath(path[0]).split(os.sep)
    os.makedirs(os.path.join(args.vq_dir, separa_path[-3]), exist_ok=True)
    dir_name = separa_path[-3]
    seq_name = f'{separa_path[-1][:-5]}.npy'

    np.save(pjoin(args.vq_dir, dir_name, seq_name), target)

train_dataset = GigaHands(opt=wrapper_opt, mode=f'{mode}_motiontext_train', split='train', codebook_size=args.nb_code, unit_length=2**args.down_t, augment_text = False, token_dir=args.vq_dir)
train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4, drop_last = False)
train_loader_iter = dataset_hands.cycle(train_loader)

        
##### ---- Training ---- #####
best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans_hand.evaluation_transformer(args.out_dir, val_loader, net, trans_encoder, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, clip_model=clip_model, eval_wrapper=eval_wrapper, mode=mode, draw=True)
while nb_iter <= args.total_iter:
    batch = next(train_loader_iter)
    clip_text, m_tokens, m_tokens_len = batch
    m_tokens, m_tokens_len = m_tokens.cuda(), m_tokens_len.cuda()
    bs = m_tokens.shape[0]
    target = m_tokens    # (bs, 26)
    target = target.cuda()
    
    text = clip.tokenize(clip_text, truncate=True).cuda()
    
    feat_clip_text = clip_model.encode_text(text).float()

    input_index = target[:,:-1]

    if args.pkeep == -1:
        proba = np.random.rand(1)[0]
        mask = torch.bernoulli(proba * torch.ones(input_index.shape,
                                                         device=input_index.device))
    else:
        mask = torch.bernoulli(args.pkeep * torch.ones(input_index.shape,
                                                         device=input_index.device))
    mask = mask.round().to(dtype=torch.int64)
    r_indices = torch.randint_like(input_index, args.nb_code)
    a_indices = mask*input_index+(1-mask)*r_indices

    cls_pred = trans_encoder(a_indices, feat_clip_text)
    cls_pred = cls_pred.contiguous()

    loss_cls = 0.0
    for i in range(bs):
        # loss function     (26), (26, 513)
        loss_cls += loss_ce(cls_pred[i][:m_tokens_len[i] + 1], target[i][:m_tokens_len[i] + 1]) / bs

        # Accuracy
        probs = torch.softmax(cls_pred[i][:m_tokens_len[i] + 1], dim=-1)

        if args.if_maxtest:
            _, cls_pred_index = torch.max(probs, dim=-1)

        else:
            dist = Categorical(probs)
            cls_pred_index = dist.sample()
        right_num += (cls_pred_index.flatten(0) == target[i][:m_tokens_len[i] + 1].flatten(0)).sum().item()

    ## global loss
    optimizer.zero_grad()
    loss_cls.backward()
    optimizer.step()
    scheduler.step()

    avg_loss_cls = avg_loss_cls + loss_cls.item()
    nb_sample_train = nb_sample_train + (m_tokens_len + 1).sum().item()

    nb_iter += 1
    if nb_iter % args.print_iter ==  0 :
        avg_loss_cls = avg_loss_cls / args.print_iter
        avg_acc = right_num * 100 / nb_sample_train
        writer.add_scalar('./Loss/train', avg_loss_cls, nb_iter)
        writer.add_scalar('./ACC/train', avg_acc, nb_iter)
        msg = f"Train. Iter {nb_iter} : Loss. {avg_loss_cls:.5f}, ACC. {avg_acc:.4f}"
        logger.info(msg)
        avg_loss_cls = 0.
        right_num = 0
        nb_sample_train = 0

    if nb_iter % args.eval_iter ==  0:
        best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans_hand.evaluation_transformer(args.out_dir, val_loader, net, trans_encoder, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, clip_model=clip_model, eval_wrapper=eval_wrapper, mode=mode, draw=True)

    if nb_iter == args.total_iter: 
        msg_final = f"Train. Iter {best_iter} : FID. {best_fid:.5f}, Diversity. {best_div:.4f}, TOP1. {best_top1:.4f}, TOP2. {best_top2:.4f}, TOP3. {best_top3:.4f}"
        logger.info(msg_final)
        break            
