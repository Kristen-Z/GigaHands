import os, random
from os.path import join as pjoin
import numpy as np
from torch.utils import data
import json

class GigaMotion(data.Dataset):
    def __init__(self, opt, mean, std, split='train', rep='keypoints'):
        self.opt = opt
        self.mean = mean
        self.std = std
        self.rep = rep

        data_list = []
        text_file = os.path.join(opt.dataset_root, 'annotations_v0.jsonl')
        motion_root = opt.dataset_root
        self.max_motion_length = opt.max_motion_length
        params_path = 'keypoints_3d_mano' if self.rep == 'keypoints' else 'params'

        with open(text_file, 'r', encoding='utf-8') as t_file:
            for t_line in t_file:
                script_info = json.loads(t_line)
                seq = script_info['sequence'][0] if isinstance(script_info['sequence'], list) else script_info['sequence']
                script_text = script_info['description']
                scene_name = script_info['scene']
                sf, ef = script_info['start_frame_id'], script_info['end_frame_id']

                v_path = os.path.join(motion_root, scene_name, params_path, seq+'.json')
                if os.path.exists(v_path):
                    data = {'path': v_path,'chosen_frames': (sf, ef),'text':script_text}
                    data_list.append(data)
        
        self.data_list = data_list 
        self.split = split
        print(f'{len(self.data_list)} motions in {split} split')
    
    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, item):

        data_path = self.data_list[int(item)]['path']
        with open(data_path, "r") as f:
            mano_kp = json.load(f) # [F, 42*3]

        start, end = self.data_list[int(item)]['chosen_frames']
        text = self.data_list[int(item)]['text']
        if end == -1:
            motion = np.array(mano_kp)[start:]
        else:
            motion = np.array(mano_kp)[start:end+1] # [valid_F, 42*3]

        motion = (motion - self.mean) / self.std
        # m_length = motion.shape[0]
        # if m_length > self.max_motion_length:
        #     motion = motion[:self.max_motion_length]
        #     m_length = self.max_motion_length
        
        # if m_length < self.max_motion_length:
        #     motion = np.concatenate([motion,
        #                              np.zeros((self.max_motion_length - m_length, motion.shape[1]))
        #                              ], axis=0)
        return motion, text

# A wrapper class for GigaHands for different applications
class GigaHands(data.Dataset):
    def __init__(self, opt, mode='motion_text', split='train', rep='keypoints', **kwargs):
        self.split = split
        self.rep = rep
        self.mode = mode
        
        self.opt = opt
        print(f'Loading dataset {opt.dataset_name} in mode {self.mode} ...')

        if rep == 'keypoints':
            mean_path = 'hand_utils/giga_mean_kp.npy'
            std_path = 'hand_utils/giga_std_kp.npy'

        mean = np.load(mean_path)
        std = np.load(std_path)         

        if self.mode == 'motion_text':
            self.giga_dataset = GigaMotion(self.opt, mean, std, split, rep)
            self.giga_len = len(self.giga_dataset)

    def __getitem__(self, item):
        return self.giga_dataset.__getitem__(item)

    def __len__(self):
       return self.giga_len
    

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
