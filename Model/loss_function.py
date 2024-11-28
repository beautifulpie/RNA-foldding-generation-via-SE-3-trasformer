import math
import numpy as np

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

