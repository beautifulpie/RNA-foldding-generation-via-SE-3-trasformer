import torch.nn as nn

import sys
sys.path.append("/workspace/4D-Diff-RNA_test_1/")

import torch
from torch.utils.data import dataloader, Dataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from datasets import edge_embedder, node_embedder
from invariant_point_attention import invariant_point_attention
from modules import MotionAlignment, SpatialModule, EdgeUpdate, BackboneUpdate, Process_trajectory, SinusoidalTimeEmbedding

import torch
import torch.nn as nn
import datasets.ipa_pytorch as ipa_pytorch
import torsion_net
from datasets import utils as du  

class FlowModel(nn.Module):

    def __init__(self, model_conf):
        super(FlowModel, self).__init__()
        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * du.ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * du.NM_TO_ANG_SCALE) 
        self.num_iterations = 1

        self.node_embedder = node_embedder.NodeEmbedder(model_conf.node_features)
        self.edge_embedder = edge_embedder.EdgeEmbedder(model_conf.edge_features)

        self.spatial_module = SpatialModule(input_dim=128, output_dim=128, num_heads=4)
        self.motion_alignment = MotionAlignment(input_dim=128, output_dim=128, num_heads=4)
        self.edge_update = EdgeUpdate(D_v= 128, D_z= 64)
        self.backbone_update = BackboneUpdate(D_v= 128)

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s  
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks-1:
                edge_in = self._model_conf.edge_embed_size
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._model_conf.edge_embed_size,
                )

        self.angle_pred_net = torsion_net.TorsionAngleHead(c_in=self._ipa_conf.c_s, c_hidden=128, no_blocks=2, no_angles=8, epsilon=1e-12)

    def forward(self, seq, coord_4d):
        S = coord_4d.shape[0]
        node_mask = seq['res_mask']
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        continuous_t = seq['t']
        trans_t = seq['trans_t']
        rotmats_t = seq['rotmats_t']

        init_node_embed = self.node_embedder(continuous_t, node_mask)
        trans_sc = seq.get('trans_sc', torch.zeros_like(trans_t))
        init_edge_embed = self.edge_embedder(init_node_embed, trans_t, trans_sc, edge_mask)

        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)

        node_embed = init_node_embed * node_mask[..., None]
        edge_embed = init_edge_embed * edge_mask[..., None]
        backbone_trajectory = []

        for i in range(S):
            T = coord_4d[i]

            for _ in range(self.num_iterations):
                V_1 = self.trunk[f'ipa_{i}'](node_embed, edge_embed, curr_rigids, node_mask)
                V_1 *= node_mask[..., None]
                node_embed = self.trunk[f'ipa_ln_{i}'](node_embed + V_1)
                seq_tfmr_out = self.trunk[f'seq_tfmr_{i}'](node_embed, src_key_padding_mask=(1 - node_mask).bool())
                node_embed = node_embed + self.trunk[f'post_tfmr_{i}'](seq_tfmr_out)
                node_embed = self.trunk[f'node_transition_{i}'](node_embed)
                node_embed = node_embed * node_mask[..., None]
                rigid_update = self.trunk[f'bb_update_{i}'](node_embed * node_mask[..., None])
                curr_rigids = curr_rigids.compose_q_update_vec(rigid_update, node_mask[..., None])

                if i < self._ipa_conf.num_blocks - 1:
                    edge_embed = self.trunk[f'edge_transition_{i}'](node_embed, edge_embed)
                    edge_embed *= edge_mask[..., None]

            curr_rigids = self.rigids_nm_to_ang(curr_rigids)
            pred_trans = curr_rigids.get_trans()
            pred_rotmats = curr_rigids.get_rots().get_rot_mats()

            backbone_trajectory.append((pred_trans, pred_rotmats))

        _, pred_torsions = self.angle_pred_net(node_embed, init_node_embed)

        return {
            'pred_torsions': pred_torsions,
            'backbone_trajectory': backbone_trajectory,
        }

# 모델 구성 설정 (모델 구성 객체는 실제 사용에 맞게 정의해야 함)

# NodeEmbedderConfig 클래스 정의
class NodeEmbedderConfig:
    def __init__(self):
        self.single_bias_transition_n = 2
        self.c_s = 256  # 최종 임베딩 차원
        self.c_pos_emb = 128  # 위치 임베딩 차원
        self.c_timestep_emb = 128  # 시간 임베딩 차원
        self.embed_diffuse_mask = False
        self.max_num_res = 2000
        self.timestep_int = 1000

class EdgeEmbedderConfig:
    def __init__(self):
        self.c_s = 256  # 노드 임베딩 차원
        self.c_p = 128   # 엣지 임베딩 차원
        self.relpos_k = 64
        self.use_rbf = True
        self.num_rbf = 32
        self.feat_dim = 64
        self.num_bins = 22
        self.self_condition = True

class ipaConfig :
    def __init__(self):
        self.num_blocks = 6
        self.c_s = 256  
        self.c_z = 128
        self.c_hidden = 128
        self.no_heads = 8
        self.no_qk_points = 8
        self.no_v_points = 12
        self.seq_tfmr_num_heads = 4
        self.seq_tfmr_num_layers = 2

class ModelConfig:
    def __init__(self):
        
        self.edge_embed_size = 128
        self.node_embed_size = 256
        self.symmetric = False

        # ipa, Node&Edge Embedder setting
        self.ipa = ipaConfig()
        self.node_features =  NodeEmbedderConfig()
        self.edge_features = EdgeEmbedderConfig()

# 입력 값 생성 함수
def create_input_tensors(batch_size, num_residues, seq_len):
    seq = {
        'res_mask': torch.ones(seq_len, batch_size),  # Mask indicating valid residues
        't': torch.rand(seq_len, batch_size),  # Continuous time steps
        'trans_t': torch.rand(seq_len, batch_size, 3),  # Translations (3D coordinates)
        'rotmats_t': torch.rand(seq_len, batch_size, 3, 3),  # Rotation matrices
        'trans_sc': torch.rand(seq_len, batch_size, 3)  # Side chain translations (optional)
    }

    timesteps = torch.rand(batch_size * num_residues * 3)
    print("2D Timesteps Shape:", len(timesteps.shape))
    # timesteps = timesteps.view(batch_size , num_residues, 2)

    coord_4d = torch.randn(batch_size, seq_length, 4)

    return seq, timesteps, coord_4d

model_conf = ModelConfig()
batch_size = 2
num_residues = 5
seq_length = 10

# 입력 텐서 생성
seq, timesteps, coord_4d = create_input_tensors(batch_size, num_residues, seq_length)

# 모델 확인
print("Sequence Data:")
print("res_mask:", seq['res_mask'])
print("t:", seq['t'])
print("trans_t:", seq['trans_t'])
print("rotmats_t:", seq['rotmats_t'])
print("timesteps:", timesteps)  # 1D 텐서 확인
# print("\n4D Coordinates:")
# print(coord_4d)

# 모델 초기화
model_conf = ModelConfig()
model = FlowModel(model_conf)

print(model)
output = model(seq, coord_4d)
print(output)

# # 학습 루프
# num_epochs = 10

# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
    
#     for inputs, labels in train_loader:
#         inputs, labels = inputs.cuda(), labels.cuda()

#         # 옵티마이저 초기화
#         optimizer.zero_grad()

#         # 모델 출력
#         outputs = model(inputs)

#         # 손실 계산
#         loss = criterion(outputs['backbone_trajectory'], labels)
        
#         # 역전파 및 옵티마이저 스텝
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#     print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

# print('Training Finished.')