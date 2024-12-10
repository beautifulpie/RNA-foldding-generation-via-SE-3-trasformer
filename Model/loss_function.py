import math
import numpy as np
import torch as nn
from . import rigid_utils as ru
import random
from datasets import all_atom

def cal_rmsd_numpy(coord_1, coord_2):
    rmsd = np.sqrt(((coord_1 - coord_2) ** 2).mean())    ## this would be the formula
    return rmsd

def denoising_score_matching_loss(R_t, R_0):
    # Denoising Score Matching Loss
    # λtR=1/𝔼[||∇logp_t|0(R(t)|R(0))||_SO⁢(3) ^ 2].
    loss_R = 0
    loss_X = 0
    return loss_R + loss_X

def torsion_angle_loss(torsion, torsion_gt):
    loss_torsion = 0
    for i in range(len(torsion)):
        alpha_i = torsion[i]
        alpha_i_gt = torsion_gt[i]
        alpha_i_altgt = torsion_gt[i] + math.pi
        loss_torsion += math.min((math.hypot(alpha_i-alpha_i_gt))**2, (math.hypot(alpha_i-alpha_i_altgt))**2)
    
    return loss_torsion/len(torsion)

def auxiliary_loss(predict_trajectory, true_trajectory):
    omega = ["N", "C", "C_alpha", "C1", "O", "CA", "Ca"]
    predict_frames = []
    true_frames = []
    loss_omega = 0
    
    for predict_atom in predict_trajectory:
        if predict_atom in omega :
            predict_frames.append(predict_atom)

    for true_atom in true_trajectory:
        if true_atom in omega :
            true_frames.append(true_atom)

    loss_omega = cal_rmsd_numpy(true_frames, predict_frames) # 오류 나면 이 부분 수정할 것
    
    loss_2D = 0

    for i in range(len(predict_trajectory)):
        for j in range(len(predict_trajectory)):
            d_ab = math.hypot(predict_frames[i]-true_frames[j])
            if d_ab[i][j] < 0.6:
                loss_2D += d_ab[i][j]**2

    return loss_omega+loss_2D

def get_torsion_data(trajectory):
    rotation = trajectory[0]
    return rotation

def total_loss(predict_trajectory, true_trajectory, true_folding_structure, predict_folding_structure):
    w1 = 0.25
    w2 = 1.00

    R_t = 0
    R_0 = 0

    torsion_angle = get_torsion_data(predict_trajectory)
    true_torsion_angle = get_torsion_data(true_trajectory)

    RMSD = cal_rmsd_numpy(true_folding_structure, predict_folding_structure)
    print(RMSD)

    return denoising_score_matching_loss(R_t, R_0) + w1 * auxiliary_loss(true_trajectory, predict_trajectory) + w2 * torsion_angle_loss(torsion_angle, true_torsion_angle) #+ RMSD



def loss_fn(self, batch):
    """Computes loss and auxiliary data.

    Args:
        batch: Batched data.
        model_out: Output of model ran on batch.

    Returns:
        loss: Final training loss scalar.
        aux_data: Additional logging data.
    """
    if self._model_conf.embed.embed_self_conditioning and random.random() > 0.5:
        with nn.no_grad():
            batch = self._self_conditioning(batch)
    model_out = self.model(batch)
    bb_mask = batch['res_mask']
    diffuse_mask = 1 - batch['fixed_mask']
    loss_mask = bb_mask * diffuse_mask
    batch_size, num_res = bb_mask.shape

    gt_rot_score = batch['rot_score']
    gt_trans_score = batch['trans_score']
    rot_score_scaling = batch['rot_score_scaling']
    trans_score_scaling = batch['trans_score_scaling']
    batch_loss_mask = nn.any(bb_mask, dim=-1)

    pred_rot_score = model_out['rot_score'] * diffuse_mask[..., None]
    pred_trans_score = model_out['trans_score'] * diffuse_mask[..., None]

    # Translation score loss
    trans_score_mse = (gt_trans_score - pred_trans_score)**2 * loss_mask[..., None]
    trans_score_loss = nn.sum(
        trans_score_mse / trans_score_scaling[:, None, None]**2,
        dim=(-1, -2)
    ) / (loss_mask.sum(dim=-1) + 1e-10)

    # Translation x0 loss
    gt_trans_x0 = batch['rigids_0'][..., 4:] * self._exp_conf.coordinate_scaling
    pred_trans_x0 = model_out['rigids'][..., 4:] * self._exp_conf.coordinate_scaling
    trans_x0_loss = nn.sum(
        (gt_trans_x0 - pred_trans_x0)**2 * loss_mask[..., None],
        dim=(-1, -2)
    ) / (loss_mask.sum(dim=-1) + 1e-10)

    trans_loss = (
        trans_score_loss * (batch['t'] > self._exp_conf.trans_x0_threshold)
        + trans_x0_loss * (batch['t'] <= self._exp_conf.trans_x0_threshold)
    )
    trans_loss *= self._exp_conf.trans_loss_weight
    trans_loss *= int(self._diff_conf.diffuse_trans)

    # Rotation loss
    if self._exp_conf.separate_rot_loss:
        gt_rot_angle = nn.norm(gt_rot_score, dim=-1, keepdim=True)
        gt_rot_axis = gt_rot_score / (gt_rot_angle + 1e-6)

        pred_rot_angle = nn.norm(pred_rot_score, dim=-1, keepdim=True)
        pred_rot_axis = pred_rot_score / (pred_rot_angle + 1e-6)

        # Separate loss on the axis
        axis_loss = (gt_rot_axis - pred_rot_axis)**2 * loss_mask[..., None]
        axis_loss = nn.sum(
            axis_loss, dim=(-1, -2)
        ) / (loss_mask.sum(dim=-1) + 1e-10)

        # Separate loss on the angle
        angle_loss = (gt_rot_angle - pred_rot_angle)**2 * loss_mask[..., None]
        angle_loss = nn.sum(
            angle_loss / rot_score_scaling[:, None, None]**2,
            dim=(-1, -2)
        ) / (loss_mask.sum(dim=-1) + 1e-10)
        angle_loss *= self._exp_conf.rot_loss_weight
        angle_loss *= batch['t'] > self._exp_conf.rot_loss_t_threshold
        rot_loss = angle_loss + axis_loss
    else:
        rot_mse = (gt_rot_score - pred_rot_score)**2 * loss_mask[..., None]
        rot_loss = nn.sum(
            rot_mse / rot_score_scaling[:, None, None]**2,
            dim=(-1, -2)
        ) / (loss_mask.sum(dim=-1) + 1e-10)
        rot_loss *= self._exp_conf.rot_loss_weight
        rot_loss *= batch['t'] > self._exp_conf.rot_loss_t_threshold
    rot_loss *= int(self._diff_conf.diffuse_rot)

    # Backbone atom loss
    pred_atom37 = model_out['atom37'][:, :, :5]
    gt_rigids = ru.Rigid.from_tensor_7(batch['rigids_0'].type(nn.float32))
    gt_psi = batch['torsion_angles_sin_cos'][..., 2, :]
    gt_atom37, atom37_mask, _, _ = all_atom.compute_backbone(
        gt_rigids, gt_psi)
    gt_atom37 = gt_atom37[:, :, :5]
    atom37_mask = atom37_mask[:, :, :5]

    gt_atom37 = gt_atom37.to(pred_atom37.device)
    atom37_mask = atom37_mask.to(pred_atom37.device)
    bb_atom_loss_mask = atom37_mask * loss_mask[..., None]
    bb_atom_loss = nn.sum(
        (pred_atom37 - gt_atom37)**2 * bb_atom_loss_mask[..., None],
        dim=(-1, -2, -3)
    ) / (bb_atom_loss_mask.sum(dim=(-1, -2)) + 1e-10)
    bb_atom_loss *= self._exp_conf.bb_atom_loss_weight
    bb_atom_loss *= batch['t'] < self._exp_conf.bb_atom_loss_t_filter
    bb_atom_loss *= self._exp_conf.aux_loss_weight

    # Pairwise distance loss
    gt_flat_atoms = gt_atom37.reshape([batch_size, num_res*5, 3])
    gt_pair_dists = nn.linalg.norm(
        gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1)
    pred_flat_atoms = pred_atom37.reshape([batch_size, num_res*5, 3])
    pred_pair_dists = nn.linalg.norm(
        pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1)

    flat_losjys_mask = nn.tile(loss_mask[:, :, None], (1, 1, 5))
    flat_loss_mask = flat_loss_mask.reshape([batch_size, num_res*5])
    flat_res_mask = nn.tile(bb_mask[:, :, None], (1, 1, 5))
    flat_res_mask = flat_res_mask.reshape([batch_size, num_res*5])

    gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
    pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
    pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

    # No loss on anything >6A
    proximity_mask = gt_pair_dists < 6
    pair_dist_mask  = pair_dist_mask * proximity_mask

    dist_mat_loss = nn.sum(
        (gt_pair_dists - pred_pair_dists)**2 * pair_dist_mask,
        dim=(1, 2))
    dist_mat_loss /= (nn.sum(pair_dist_mask, dim=(1, 2)) - num_res)
    dist_mat_loss *= self._exp_conf.dist_mat_loss_weight
    dist_mat_loss *= batch['t'] < self._exp_conf.dist_mat_loss_t_filter
    dist_mat_loss *= self._exp_conf.aux_loss_weight

    final_loss = (
        rot_loss
        + trans_loss
        + bb_atom_loss
        + dist_mat_loss
    )

    return final_loss