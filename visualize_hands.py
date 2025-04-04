import os
import torch
import argparse
import warnings
from dataset.dataset_hands import GigaHands
from hand_utils.utils.visualizers import plot_3d_hand_motion
from options.get_eval_option import get_opt

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='./dataset/hand_pose', type=str,
                        help='Root path for datasets')
    parser.add_argument('--save_dir', default='./visualizations', type=str,
                        help='Save path for visualizations')
    args = parser.parse_args()

    # Load configuration
    dataset_opt_path = './dataset/opt.txt'
    wrapper_opt = get_opt(dataset_opt_path)
    wrapper_opt.dataset_root = args.dataset_root 

    # Load dataset
    dataset = GigaHands(opt=wrapper_opt, mode='motion_text')
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4, shuffle=True)

    # Visualize motions
    os.makedirs(args.save_dir, exist_ok=True)
    for motion, text in loader:
        pose_gt = dataset.giga_dataset.inv_transform(motion[0]).reshape(motion.shape[1], -1, 3).cpu().numpy()
        save_file = text[0].replace('\n', '').replace(' ', '_') + '.mp4'
        plot_3d_hand_motion(pose_gt, out_path=args.save_dir, save_file=save_file, add_back=True, data_path=args.dataset_root, vis_3d_repro=True)
        print(f"Saved: {os.path.join(args.save_dir, save_file)}")

