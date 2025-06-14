import os, random
from os.path import join as pjoin
from sklearn.model_selection import train_test_split
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
        text_file = opt.text_file
        motion_root = opt.dataset_root
        self.max_motion_length = opt.max_motion_length
        params_path = 'keypoints_3d_mano' if self.rep == 'keypoints' else 'params'

        with open(text_file, 'r', encoding='utf-8') as file:
            for t_line in file:
                script_info = json.loads(t_line)
                seq = script_info['sequence'][0] if isinstance(script_info['sequence'], list) else script_info['sequence']
                sf, ef = script_info['start_frame_id'], script_info['end_frame_id']
                script_text = script_info['clarify_annotation']
                scene_name = script_info['scene']
            
                v_path = os.path.join(motion_root, scene_name, params_path, seq+'.json')
                if split == 'all':
                    if not os.path.exists(v_path):
                        print('path in json not exists', v_path)
                        continue
                    data_list.append({'path': v_path,'chosen_frames': (sf, ef)})
                elif os.path.exists(v_path) and script_text!='None' and script_text!='Buggy':
                    data = {'path': v_path,'chosen_frames': (sf, ef),}
                    data_list.append(data)
        
        if split!='all':
            # train:test:val = 0.8:0.15:0.05
            train_data, remaining_data = train_test_split(data_list, test_size=0.2, random_state=42)
            test_data, val_data = train_test_split(remaining_data, test_size=0.25, random_state=42)
            if split == 'val':
                self.data_list = val_data
            else:
                self.data_list = train_data if split=='train' else test_data 
        else:
            self.data_list = data_list 

        self.split = split
        print('GigaHand', split, len(self.data_list))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, item):

        data_path = self.data_list[int(item)]['path']
            
        with open(data_path, "r") as f:
            mano_kp = json.load(f) # [F, 42*3]

        start, end = self.data_list[int(item)]['chosen_frames']
        if end == -1:
            motion = np.array(mano_kp)[start:]
        else:
            motion = np.array(mano_kp)[start:end+1] # [valid_F, 2*J,3]

        motion = (motion - self.mean) / self.std
        m_length = motion.shape[0]
        if m_length > self.max_motion_length:
            # print('DownSampling')
            # random sampling
            # selected_indices = random.sample(range(m_length), self.max_motion_length)
            # selected_indices.sort()
            # motion = motion[selected_indices]
            
            motion = motion[:self.max_motion_length]
            m_length = self.max_motion_length
        
        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        if self.split == 'all':
            return motion, data_path
        else:
            return motion

class GigaTextMotionTrain(data.Dataset):
    def __init__(self, opt, w_vectorizer, codebook_size=1024, split='train', augment_text=False, token_dir='./VQVAE'):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.token_file = token_dir
        data_list = []

        text_file = opt.text_file
        self.augment_text = augment_text

        self.mot_end_idx = codebook_size
        self.mot_pad_idx = codebook_size + 1
        self.max_motion_length = 51

        with open(text_file, 'r', encoding='utf-8') as t_file:
            for t_line in t_file:
                script_info = json.loads(t_line)
                seq = script_info['sequence'][0] if isinstance(script_info['sequence'], list) else script_info['sequence']
                script_text = script_info['clarify_annotation']
                scene_name = script_info['scene']
                all_scripts = [script_text]
                if self.augment_text:
                    all_scripts += [caption.rstrip('.') for caption in script_info['rewritten_annotation']] 
                    if len(all_scripts) != 6:
                        continue
                
                if script_text!='None' and script_text!='Buggy':
                    token_path = os.path.join(self.token_file, scene_name, seq+'.npy')
                    
                    if not os.path.exists(token_path):
                        continue
                    m_tokens = np.load(token_path)
                    data = {'m_token_list': m_tokens.tolist(), 'captions': all_scripts}
                    data_list.append(data)
        
        if split!='all':
            # train:test:val = 0.8:0.15:0.05
            train_data, remaining_data = train_test_split(data_list, test_size=0.2, random_state=42)
            test_data, val_data = train_test_split(remaining_data, test_size=0.25, random_state=42)
            if split == 'val':
                self.data_list = val_data
            else:
                self.data_list = train_data if split=='train' else test_data 
        else:
            self.data_list = data_list 

        print('GigaHand', split, len(self.data_list))
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        data = self.data_list[int(item)]
        m_tokens = random.choice(data['m_token_list'])

        caption = random.choice(data['captions'])
        coin = np.random.choice([False, False, True])
        if coin:
            # drop one token at the head or tail
            coin2 = np.random.choice([True, False])
            if coin2:
                m_tokens = m_tokens[:-1]
            else:
                m_tokens = m_tokens[1:]
        m_tokens_len = len(m_tokens)

        if m_tokens_len+1 < self.max_motion_length:
            m_tokens = np.concatenate([m_tokens, np.ones((1), dtype=int) * self.mot_end_idx, np.ones((self.max_motion_length-1-m_tokens_len), dtype=int) * self.mot_pad_idx], axis=0)
        else:
            m_tokens = np.concatenate([m_tokens, np.ones((1), dtype=int) * self.mot_end_idx], axis=0)

        return caption, m_tokens.reshape(-1), m_tokens_len


class GigaTextMotionEval(data.Dataset):
    def __init__(self, opt, mean, std, w_vectorizer, split='train', augment_text=True, rep='keypoints'):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.mean = mean
        self.std = std
        self.max_motion_frame = opt.max_motion_frame
        self.augment_text = augment_text
        self.rep = rep
        params_path = 'keypoints_3d_mano' if self.rep == 'keypoints' else 'params'

        data_list = []
        text_file = opt.text_file
        motion_root = opt.dataset_root

        with open(text_file, 'r', encoding='utf-8') as t_file:
            for t_line in t_file:
                script_info = json.loads(t_line)
                seq = script_info['sequence'][0] if isinstance(script_info['sequence'], list) else script_info['sequence']
                sf, ef = script_info['start_frame_id'], script_info['end_frame_id']
                script_text = script_info['clarify_annotation'].replace('\n', '').replace('.', '')
                scene_name = script_info['scene']
                all_scripts = [script_text]
                if self.augment_text:
                        all_scripts += [caption.rstrip('.') for caption in script_info['rewritten_annotation']] 
                        if len(all_scripts) != 6:
                            continue
                all_tokens = script_info['all_tokens']
                v_path = os.path.join(motion_root, scene_name, params_path, seq+'.json')
                if os.path.exists(v_path) and script_text!='None' and script_text!='Buggy' and (ef==-1 or (ef-sf+1)>self.opt.unit_length):
                        text_data = []
                        for script, tokens in zip(all_scripts, all_tokens):
                            text_dict = {}
                            text_dict['caption'] = script
                            text_dict['tokens'] = tokens.split(' ')
                            text_data.append(text_dict)
            
                        data = {'path': v_path, 'text': text_data, 'chosen_frames': (sf, ef)}
                        data_list.append(data)

        if split!='all':
            # train:test:val = 0.8:0.15:0.05
            train_data, remaining_data = train_test_split(data_list, test_size=0.2, random_state=42)
            test_data, val_data = train_test_split(remaining_data, test_size=0.25, random_state=42)
            if split == 'val':
                self.data_list = val_data
            else:
                self.data_list = train_data if split=='train' else test_data 
        else:
            self.data_list = data_list 
        print('GigaHand', split, len(self.data_list))

    def __len__(self):
        return len(self.data_list)
    
    def inv_transform(self, data):
        return data * self.std + self.mean
        
    def __getitem__(self, item):
        data = self.data_list[int(item)]
        data_path, text_list = data['path'], data['text']
        separa_path = os.path.normpath(data_path).split(os.sep)
        name = f'{separa_path[-3]}-{separa_path[-1][:-5]}'

        text_data = random.choice(text_list)
        caption, t_tokens = text_data['caption'], text_data['tokens']

        if len(t_tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + t_tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = t_tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Load motions
        with open(data_path, "r") as f:
            mano_kp = json.load(f) # [F, 42*3]

        start, end = self.data_list[int(item)]['chosen_frames']
        if end == -1:
            motion = np.array(mano_kp)[start:]
        else:
            motion = np.array(mano_kp)[start:end+1] # [valid_F, 2*J,3]

        m_length = motion.shape[0]
        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]
        motion = (motion - self.mean) / self.std
        
        if m_length < self.max_motion_frame:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_frame - m_length, motion.shape[1]))
                                     ], axis=0)
        else:
            motion = motion[:self.max_motion_frame]
            m_length = self.max_motion_frame
            
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), name

    
# A wrapper class for GigaHands for different applications
class GigaHands(data.Dataset):
    def __init__(self, opt, mode='single_motion', split="train", rep = 'keypoints', augment_text=False, codebook_size=1024, w_vectorizer=None, token_dir='./VQVAE', **kwargs):
        self.split = split
        self.rep = rep
        self.mode = mode

        # Load annotation files
        opt.text_file = os.path.join(opt.dataset_root, 'annotations_v2.jsonl')
            
        self.opt = opt
        print(f'Loading dataset {opt.dataset_name} in mode {self.mode} ...')

        if rep == 'keypoints':
            mean_path = 'hand_utils/giga_mean_kp.npy'
            std_path = 'hand_utils/giga_std_kp.npy'

        mean = np.load(mean_path)
        std = np.load(std_path)         

        if self.mode == 'giga_motion':
            self.giga_dataset = GigaMotion(self.opt, mean, std, split, rep=rep)
            self.giga_len = len(self.giga_dataset)
        
        elif self.mode == 'giga_motiontext_train':
            self.giga_dataset = GigaTextMotionTrain(self.opt, w_vectorizer=w_vectorizer, codebook_size=codebook_size, augment_text = augment_text, split=split, token_dir=token_dir)
            self.giga_len = len(self.giga_dataset)

        elif self.mode == 'giga_motiontext_eval':
            self.giga_dataset = GigaTextMotionEval(self.opt, mean, std, w_vectorizer=w_vectorizer, augment_text = augment_text, split=split, rep=rep)
            self.giga_len = len(self.giga_dataset)

    def __getitem__(self, item):
        return self.giga_dataset.__getitem__(item)

    def __len__(self):
       return self.giga_len
    

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
