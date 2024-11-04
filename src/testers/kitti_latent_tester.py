from .base_tester import BaseTester
import torch
import numpy as np
from typing import Dict, Any
from src.utils.kitti_utils import read_pose_from_text
from src.utils.kitti_latent_eval import KITTI_tester_latent

from dataclasses import dataclass
import os

class KITTILatentTester(BaseTester):
    def __init__(self, val_seqs, data_dir, seq_len, folder, img_w, img_h, wrapper_weights_path, device, v_f_len, i_f_len, use_history_in_eval=False):
        super().__init__()
        self.val_seq = val_seqs
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.folder = folder
        self.img_w = img_w
        self.img_h = img_h
        self.wrapper_weights_path = wrapper_weights_path
        self.device = device
        self.v_f_len = v_f_len
        self.i_f_len = i_f_len

        @dataclass
        class Args:
            val_seq: list
            data_dir: str
            seq_len: int
            folder: str
            img_w: int
            img_h: int
            device: str
            v_f_len: int
            i_f_len: int
            imu_dropout: float

        self.args = Args(self.val_seq, self.data_dir, self.seq_len, self.folder, self.img_w, self.img_h, self.device, self.v_f_len, self.i_f_len, 0.1)

        self.kitti_latent_tester = KITTI_tester_latent(self.args, self.wrapper_weights_path, use_history_in_eval=use_history_in_eval)
    
    def test(self, model: torch.nn.Module) -> Dict[str, Any]:
        results = {}
        for i, seq in enumerate(self.val_seq):
            print(f"Testing sequence {i+1} of {len(self.val_seq)}")
            pose_est = self.kitti_latent_tester.test_one_path(model, self.kitti_latent_tester.dataloader[i])
            pose_gt = self.kitti_latent_tester.dataloader[i].poses_rel
            
            results[seq] = {
                'estimated_poses': pose_est,
                'gt_poses': pose_gt
            }

        return results

    def save_results(self, results: Dict[str, Any], save_dir: str):
        for seq_name, seq_data in results.items():
            np.save(os.path.join(save_dir, f'{seq_name}_estimated_poses.npy'), seq_data['estimated_poses'])
            np.save(os.path.join(save_dir, f'{seq_name}_gt_poses.npy'), seq_data['gt_poses'])
