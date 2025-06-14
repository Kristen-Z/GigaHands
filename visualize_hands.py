import os
import torch
import argparse
import warnings
from dataset.dataset_hands import GigaHands
from hand_utils.utils.visualizers import plot_3d_hand_motion
from options.get_eval_option import get_opt

warnings.filterwarnings('ignore')

if __name__ == '__main__':

    # Load configuration
    dataset_opt_path = './dataset/opt.txt'
    save_dir = './visualizations'
    wrapper_opt = get_opt(dataset_opt_path)
    from utils.word_vectorizer import WordVectorizer
    w_vectorizer = WordVectorizer('./glove', 'our_vocab')

    # Load dataset
    dataset = GigaHands(opt=wrapper_opt, mode='giga_motiontext_eval', w_vectorizer=w_vectorizer)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4, shuffle=True)

    # Visualize motions
    os.makedirs(save_dir, exist_ok=True)
    for batch in loader:
        word_embeddings, pos_one_hots, text, sent_len, motion, m_length, _, name = batch
        pose_gt = dataset.giga_dataset.inv_transform(motion[0]).reshape(motion.shape[1], -1, 3).cpu().numpy()
        save_file = text[0].replace('\n', '').replace(' ', '_') + '.mp4'
        plot_3d_hand_motion(pose_gt, out_path=save_dir, save_file=save_file, add_back=True, data_path=wrapper_opt.dataset_root, vis_3d_repro=True)
        print(f"Saved: {os.path.join(save_dir, save_file)}")

