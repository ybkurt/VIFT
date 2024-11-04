import torch
import torch.nn as nn
from utils.kitti_utils import eulerAnglesToRotationMatrixTorch as etr
from utils import rpmg

class WeightedMSEPoseLoss(nn.Module):
    def __init__(self, angle_weight=100):
        super(WeightedMSEPoseLoss, self).__init__()
        self.angle_weight = angle_weight

    def forward(self, poses, gts):
        angle_loss = torch.nn.functional.mse_loss(poses[:,:,:3], gts[:, :, :3])
        translation_loss = torch.nn.functional.mse_loss(poses[:,:,3:], gts[:, :, 3:])
        
        pose_loss = self.angle_weight * angle_loss + translation_loss
        return pose_loss
    
class WeightedMAEPoseLoss(nn.Module):
    def __init__(self, angle_weight=10):
        super(WeightedMAEPoseLoss, self).__init__()
        self.angle_weight = angle_weight

    def forward(self, poses, gts):
        angle_loss = torch.nn.functional.l1_loss(poses[:,:,:3], gts[:, :, :3])
        translation_loss = torch.nn.functional.l1_loss(poses[:,:,3:], gts[:, :, 3:])
        
        pose_loss = self.angle_weight * angle_loss + translation_loss
        return pose_loss

class LieTorchPoseLoss(nn.Module):
    def __init__(self, angle_weight=100):
        super().__init__()
        self.angle_weight= angle_weight
    def forward(self, poses, gts): # poses : translation + SE3 matrix, gts : translation + Euler Angles

        pass

class RPMGPoseLoss(nn.Module):
    def __init__(self, base_loss_fn, angle_weight=100):
        super().__init__()
        self.angle_weight = angle_weight
    def forward(self, poses, gts, weights, use_weighted_loss=True):
        angle_loss = torch.nn.functional.l1_loss(
            rpmg.simple_RPMG.apply(
                                   etr(poses[:,:,:3]).view(poses.shape[0]*poses.shape[1],9),
                                   1/4,
                                   0.01
                                   ).view(-1,9),
            etr(gts[:,:,:3]).view(poses.shape[0]*poses.shape[1],9)
        )
        
        translation_loss = torch.nn.functional.l1_loss(poses[:,:,3:], gts[:, :, 3:])
        
        pose_loss = self.angle_weight * angle_loss + translation_loss
        return pose_loss

class DataWeightedRPMGPoseLoss(nn.Module):
    def __init__(self, base_loss_fn, angle_weight=100):
        super().__init__()
        self.angle_weight = angle_weight
        self.base_loss_fn = base_loss_fn # make sure reduction is None
    def forward(self, poses, gts, weights=None, use_weighted_loss=True):
        angle_loss = self.base_loss_fn(
            rpmg.simple_RPMG.apply(
                                   etr(poses[:,:,:3]).view(poses.shape[0]*poses.shape[1],9),
                                   1/4,
                                   0.01
                                   ).view(-1,9),
            etr(gts[:,:,:3]).view(poses.shape[0]*poses.shape[1],9)
        )
        
        translation_loss = self.base_loss_fn(poses[:,:,3:], gts[:, :, 3:])
        
        total_loss = self.angle_weight * torch.sum(angle_loss,-1).view(poses.shape[0],-1) + torch.sum(translation_loss,-1).view(poses.shape[0],-1)
        if use_weighted_loss and (weights is not None):
            # Normalize weights
            weights = weights / weights.sum()
            weights = weights.view(-1, 1)  # reshape to (batch_size, 1, 1)
            
            # Apply weights
            total_loss = weights * total_loss.sum(dim=(1)).view(-1, 1)
        return total_loss.mean()

class DataWeightedPoseLoss(nn.Module):
    def __init__(self, base_loss_fn, angle_weight=100):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.angle_weight = angle_weight

    def forward(self, poses, gts, weights=None, use_weighted_loss=True):
        batch_size, seq_len, _ = poses.shape

        # Calculate losses for rotation and translation separately
        angle_loss = self.base_loss_fn(poses[:,:,:3], gts[:,:,:3])
        translation_loss = self.base_loss_fn(poses[:,:,3:], gts[:,:,3:])

        # Combine losses
        total_loss = self.angle_weight * angle_loss + translation_loss

        if use_weighted_loss and (weights is not None):
            # Normalize weights
            weights = weights / weights.sum()
            weights = weights.view(-1, 1, 1)  # reshape to (batch_size, 1, 1)
            
            # Apply weights
            total_loss = weights * total_loss.sum(dim=(1, 2)).view(-1, 1, 1)
        
        return total_loss.mean()

class CustomWeightedPoseLoss(nn.Module):
    def __init__(self, base_loss_fn, angle_weight=100):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.angle_weight = angle_weight

    def forward(self, poses, gts, weights=None, use_weighted_loss=True):
        batch_size, seq_len, _ = poses.shape
        loss1 = self.base_loss_fn(poses[:,:,0], gts[:,:,0])
        loss2 = self.base_loss_fn(poses[:,:,1], gts[:,:,1])
        loss3 = self.base_loss_fn(poses[:,:,2], gts[:,:,2])
        loss4 = self.base_loss_fn(poses[:,:,3], gts[:,:,3])
        loss5 = self.base_loss_fn(poses[:,:,4], gts[:,:,4])
        loss6 = self.base_loss_fn(poses[:,:,5], gts[:,:,5])

        total_loss = loss1 * (2/3) + loss2 * (1/5) + loss3 + loss4 * (0.1) + loss5 * (0.1) + loss6 * (0.03)
 
        if use_weighted_loss and (weights is not None):
            # Normalize weights
            weights = weights / weights.sum()
            weights = weights.view(-1, 1, 1)  # reshape to (batch_size, 1, 1)
            
            # Apply weights
            total_loss = weights * total_loss.sum(dim=(1)).view(-1, 1, 1)
        
        return total_loss.mean()


class AngleWeightedPoseLoss(nn.Module):
    def __init__(self, base_loss_fn, angle_weight=100):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.angle_weight = angle_weight

    def forward(self, poses, gts, weights=None, use_weighted_loss=True):
        batch_size, seq_len, _ = poses.shape

        # Calculate losses for rotation and translation separately
        angle_loss = self.base_loss_fn(poses[:,:,:3], gts[:,:,:3])
        translation_loss = self.base_loss_fn(poses[:,:,3:], gts[:,:,3:])

        # Combine losses
        total_loss = self.angle_weight * angle_loss + translation_loss
        
        return total_loss.mean()

class TokenizedPoseLoss(nn.Module):
    def __init__(self, angle_weight=100):
        super().__init__()

    def forward(self, poses, gts):
        
        out, loss = poses
        return loss


class WeightedTokenizedPoseLoss(nn.Module):
    def __init__(self, base_loss_fn, angle_weight=100):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.angle_weight = angle_weight

    def forward(self, poses, gts, weights=None, use_weighted_loss=True):
        out, ce_loss = poses
        return ce_loss
