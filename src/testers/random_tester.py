import torch
import numpy as np
from typing import Dict, Any
from .base_tester import BaseTester

class RandomTester(BaseTester):
    def __init__(self, seq_len, num_sequences, sequence_lengths):
        self.seq_len = seq_len
        self.num_sequences = num_sequences
        self.sequence_lengths = sequence_lengths

    def test(self, model: torch.nn.Module) -> Dict[str, Any]:
        results = {}
        for i, seq_length in enumerate(self.sequence_lengths):
            estimated_poses = []
            gt_poses = []
            
            for j in range(0, seq_length - self.seq_len + 1):
                # Generate random input
                output = torch.randn(1, self.seq_len, 6)  # Assuming 6 DoF poses
                estimated_poses.append(output[-1].cpu().numpy())
                gt_poses.append(np.random.randn(6))  # Random ground truth pose
            
            results[f'sequence_{i}'] = {
                'estimated_poses': np.array(estimated_poses),
                'gt_poses': np.array(gt_poses)
            }
        
        return results

    def save_results(self, results: Dict[str, Any], save_dir: str):
        for seq_name, seq_data in results.items():
            np.save(f"{save_dir}/{seq_name}_estimated.npy", seq_data['estimated_poses'])
            np.save(f"{save_dir}/{seq_name}_gt.npy", seq_data['gt_poses'])