from .base_metrics_calculator import BaseMetricsCalculator
from typing import Dict, Any
from src.utils.kitti_eval import kitti_err_cal
from src.utils.kitti_utils import path_accu
import numpy as np

class KITTIMetricsCalculator(BaseMetricsCalculator):
    def calculate_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        metrics = {}
        for seq_name, seq_data in results.items():
            pose_est = seq_data['estimated_poses']
            pose_gt = seq_data['gt_poses']

            # Convert to global poses
            pose_est_global = path_accu(pose_est)
            pose_gt_global = path_accu(pose_gt)

            # Calculate errors
            err, t_rel, r_rel, speed = kitti_err_cal(pose_est_global, pose_gt_global)
            t_rmse, r_rmse = self.calculate_rmse(pose_est, pose_gt)

            metrics[f'{seq_name}_t_rel'] = t_rel * 100
            metrics[f'{seq_name}_r_rel'] = r_rel / np.pi * 180 * 100
            metrics[f'{seq_name}_t_rmse'] = t_rmse
            metrics[f'{seq_name}_r_rmse'] = r_rmse / np.pi * 180

        return metrics

    def calculate_rmse(self, pose_est, pose_gt):
        t_rmse = np.sqrt(np.mean(np.sum((pose_est[:, 3:] - pose_gt[:, 3:])**2, -1)))
        r_rmse = np.sqrt(np.mean(np.sum((pose_est[:, :3] - pose_gt[:, :3])**2, -1)))
        return t_rmse, r_rmse