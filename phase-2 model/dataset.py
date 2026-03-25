import numpy as np

from navsim.common.dataloader import SceneLoader

import torch
class Dataset(torch.utils.data.Dataset):
    def __init__(self, scene_loader: SceneLoader, max_len=None, random_sample=False):
        self.scene_loader = scene_loader
        if random_sample:
            self.tokens = np.random.choice(scene_loader.tokens, size=max_len, replace=False)
        else:
            self.max_len = min(max_len, len(scene_loader)) if max_len is not None else len(scene_loader)
            self.tokens = scene_loader.tokens[:self.max_len]

    def __len__(self):
        # Return the total number of samples in the dataset
        return self.max_len

    def __getitem__(self, idx):
        # Retrieve a sample from the dataset at the given index
        token = self.tokens[idx]
        scene = self.scene_loader.get_scene_from_token(token)
        num_his = scene.scene_metadata.num_history_frames
        cur_frame = scene.frames[num_his]
        inputs = {}
        trajectory = scene.get_future_trajectory()
        inputs['cam'] = cur_frame.cameras
        
        return inputs, trajectory