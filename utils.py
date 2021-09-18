"""
Modified from:
https://github.com/yewzijian/RPMNet/blob/master/src/models/rpmnet.py
"""
import torch
from scipy.spatial.transform import Rotation as R
import numpy as np


def compute_rigid_transform(x, y, match, match_score):
    """
    Compute the rigid body transform from x to y based on
    match and match_scores.
    
    Args:
        x (torch.Tensor): size = (B, M, 3)
        y (torch.Tensor): size = (B, N, 3)
        match (torch.Tensor): size = (B, M), value -1 indicates no match.
        match_score (torch.Tensor): size = (B, M)
        
    Return:
        Transform T of size (B, 3, 4) from x to y.
    """
    B, M, _ = x.shape
    mask = match==-1
    match_score[mask] = 0
    normalized_match_score = torch.unsqueeze(match_score / torch.sum(match_score, dim=-1, keepdim=True), dim=-1)
    x_match = torch.zeros_like(x)
    
    for i in range(0, B):
        x_match[i, :, :] = y[i, match[i,:], :]
    
    # Centralize x and x_match
    x_centroid = torch.sum(x * normalized_match_score, dim=1)
    x_match_centroid = torch.sum(x_match * normalized_match_score, dim=1)
    x_centered = x - x_centroid[:, None, :]
    x_match_centered = x_match - x_match_centroid[:, None, :]
    
    # Compute the rotation matrix
    cov = x_centered.transpose(-2, -1) @ (x_match_centered * normalized_match_score)
    
    # Solve the unstable torch.svd error as described here:
    # https://github.com/pytorch/pytorch/issues/28293
    
    try:
        u, s, v = torch.svd(cov, some=False, compute_uv=True)
    except:                     # torch.svd may have convergence issues for GPU and CPU.
        print("Failed to compute cov")
        u, s, v = torch.svd(cov + 1e-3*cov.mean()*(torch.rand(3, 3).cuda()), some=False, compute_uv=True)
    
    rot_mat_pos = v @ u.transpose(-1, -2)
    v_neg = v.clone()
    v_neg[:, :, 2] *= -1
    rot_mat_neg = v_neg @ u.transpose(-1, -2)
    rot_mat = torch.where(torch.det(rot_mat_pos)[:,None,None] > 0, rot_mat_pos, rot_mat_neg)
    assert torch.all(torch.det(rot_mat) > 0)
    
    # Compute the translation matrix
    translation = -rot_mat @ x_centroid[:, :, None] + x_match_centroid[:, :, None]
    
    T = torch.cat((rot_mat, translation), dim=-1)
    
    return T


def compute_error(T, gt_rotation, gt_translation):
    """
    Compute Error.
    
    Args:
        T (torch.Tensor): size = (B, 3, 4)
        gt_rotation (torch.Tensor): size = (B, 3, 3)
        gt_translation (torch.Tensor): size = (B, 3)
        
    Return:
        anisotropic/isotropic rotation/translation error
    """
    gt_rotation = gt_rotation.float()
    gt_translation = gt_translation.float()
    
    B = T.shape[0]
    
    pred_rotation = T[:, :, :3]
    pred_translation = T[:, :, 3]
    euler_pred = R.from_matrix(pred_rotation.detach().cpu().numpy())
    euler_gt = R.from_matrix(gt_rotation.detach().cpu().numpy())
    
    ani_rot_error = np.mean(np.absolute(euler_pred.as_euler('zyx', degrees=True) - euler_gt.as_euler('zyx', degrees=True)))
    ani_trans_error = np.mean(np.absolute(pred_translation.detach().cpu().numpy() - gt_translation.detach().cpu().numpy()))
    
    iso_rot_error = 0
    iso_trans_error = 0
    
    for i in range(B):
        gt_rot = gt_rotation[i, :, :]
        pred_rot = pred_rotation[i, :, :]
        
        tmp = torch.acos((torch.trace(torch.inverse(gt_rot) @ pred_rot) - 1) / 2.0)
        if tmp == tmp:
            # In case trace is 3.0000
            iso_rot_error += tmp
        iso_trans_error += torch.mean((gt_translation[i, :] - pred_translation[i, :]) ** 2)
        
    iso_rot_error /= B
    iso_trans_error /= B
    
    return ani_rot_error, ani_trans_error, iso_rot_error.detach().cpu().numpy(), iso_trans_error.detach().cpu().numpy()
    

def loss_fn(T, gt_rotation, gt_translation, matches, matching_scores, point_set):
    """
    Compute Loss.
    
    Args:
        T (torch.Tensor): size = (B, 3, 4)
        gt_rotation (torch.Tensor): size = (B, 3, 3)
        gt_translation (torch.Tensor): size = (B, 3)
        matches (torch.Tensor): size = (B, n_points)
        matching_scores (torch.Tensor): size = (B, n_points)
        point_set (torch.Tensor): size = (B, 3, n_points)
        
    Return:
        Reprojection Error, cnt
    """
    
    gt_rotation = gt_rotation.float()
    gt_translation = gt_translation.float()
    
    B = T.shape[0]
    n_points = matches.shape[1]
    pred_rotation = T[:, :, :3]
    pred_translation = T[:, :, 3]
    
    # Compute re-projection loss
    loss_reproj = (torch.bmm(gt_rotation, point_set) + torch.unsqueeze(gt_translation, 2)) - (torch.bmm(pred_rotation, point_set) + torch.unsqueeze(pred_translation, 2))
    loss_reproj = torch.sum(torch.abs(loss_reproj)) / (B * n_points)
    
    # Compute inlier loss
    loss_inlier = 0
    cnt = 0
    
    for i in range(B):
        for j in range(n_points):
            if matches[i, j] == j:
                loss_inlier -= torch.log(matching_scores[i, j])
                cnt += 1
                    
    return loss_reproj + 0.3 * loss_inlier / cnt, cnt / B, loss_reproj, loss_inlier / cnt