import torch.nn as nn
import numpy as np

import sys
sys.path.append("/workspace/4D-Diff-RNA_test_1/")

import torch
from torch.utils.data import dataloader, Dataset, DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from datasets import edge_embedder, node_embedder
from invariant_point_attention import invariant_point_attention
from Model.modules import MotionAlignment, SpatialModule, EdgeUpdate, BackboneUpdate, Process_trajectory, SinusoidalTimeEmbedding

import torch
import torch.nn as nn
import datasets.ipa_pytorch as ipa_pytorch
from Model import torsion_net
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

    def forward(self, input_feat):
        """
        input_feat:
            Shape coord_4d : torch.Size([B, N, 4])
            Shape trans_sc : torch.Size([B, N, 3])
            Shape res_mask : torch.Size([B, N])
            Shape trans_t : torch.Size([B, N, 3])
            Shape rotmats_t : torch.Size([B, N, 3, 3])
            Shape t : torch.Size([B, 1])
        """
        # for key in input_feat.keys():
        #     print(f"Shape {key} : {input_feat[key].shape}")
        
        coord_4d = input_feat['coord_4d']

        S = coord_4d.shape[0]
        node_mask = input_feat['res_mask']
        edge_mask = node_mask[:, None] * node_mask[:, :, None]  ## node_mask[:, :, None] * node_mask[:, :, :, None] 
        continuous_t = input_feat['t']   # 
        trans_t = input_feat['trans_t']
        rotmats_t = input_feat['rotmats_t']

        init_node_embed = self.node_embedder(continuous_t, node_mask)
        trans_sc = input_feat.get('trans_sc', torch.zeros_like(trans_t))
        init_edge_embed = self.edge_embedder(init_node_embed, trans_t, trans_sc, edge_mask)

        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)

        node_embed = init_node_embed * node_mask[..., None]   # 여기가 고비
        edge_embed = init_edge_embed * edge_mask[..., None]   # 여기가 고비 2
        backbone_trajectory = []

        for i in range(S):   # 이거 대신 reshape 해가지고 한번에 넣어 버리기~
            T = coord_4d[i]

            for b in range(self._ipa_conf.num_blocks):
                V_1 = self.trunk[f'ipa_{b}'](node_embed, edge_embed, curr_rigids, node_mask)
                V_1 *= node_mask[..., None]
                node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + V_1)
                seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](node_embed, src_key_padding_mask=(1 - node_mask).bool())
                node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
                node_embed = self.trunk[f'node_transition_{b}'](node_embed)
                node_embed = node_embed * node_mask[..., None]
                rigid_update = self.trunk[f'bb_update_{b}'](node_embed * node_mask[..., None])
                curr_rigids = curr_rigids.compose_q_update_vec(rigid_update, node_mask[..., None])

                if b < self._ipa_conf.num_blocks - 1:
                    edge_embed = self.trunk[f'edge_transition_{b}'](node_embed, edge_embed)
                    edge_embed *= edge_mask[..., None]

            curr_rigids = self.rigids_nm_to_ang(curr_rigids)
            pred_trans = curr_rigids.get_trans()
            pred_rotmats = curr_rigids.get_rots().get_rot_mats()

            backbone_trajectory.append((pred_trans, pred_rotmats))

        _, pred_torsions = self.angle_pred_net(node_embed, init_node_embed)

        return {
            'pred_torsions': pred_torsions,
            'pred_trans' : pred_trans ,
            'pred_rotmats' : pred_rotmats,
            'backbone_trajectory': backbone_trajectory,
        }

# Example dataset class
class ExampleDataset(Dataset):
    def __init__(self, num_samples, seq_len, embedding_dim):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_feat = {
            'res_mask': torch.ones(self.seq_len),       # [N]
            't': torch.rand(self.seq_len),              # [N]
            'trans_t': torch.rand(self.seq_len, 3),     # [N, 3]
            'rotmats_t': torch.rand(self.seq_len, 3, 3),# [N, 3, 3]
            'trans_sc': torch.rand(self.seq_len, 3),    # [N, 3]
            'gt_torsions': torch.rand(self.seq_len, 8), # [N, 8]
            'gt_backbone_trajectory': [(
                    torch.rand(3),       # 랜덤 translation vector [3]
                    torch.rand(3, 3)    # 랜덤 rotation matrix [3, 3]
                ) 
                for _ in range(self.seq_len)
            ],
            'coord_4d' : torch.rand(self.seq_len, 4)  # [N, 4]
        }
        return input_feat

class NodeEmbedderConfig:
    def __init__(self):
        self.single_bias_transition_n = 2
        self.c_s = 256
        self.c_pos_emb = 128
        self.c_timestep_emb = 128
        self.embed_diffuse_mask = False
        self.max_num_res = 2000
        self.timestep_int = 1000

class EdgeEmbedderConfig:
    def __init__(self):
        self.c_s = 256
        self.c_p = 128
        self.relpos_k = 64
        self.use_rbf = True
        self.num_rbf = 32
        self.feat_dim = 64
        self.num_bins = 22
        self.self_condition = True

class ipaConfig:
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
        self.ipa = ipaConfig()
        self.node_features = NodeEmbedderConfig()
        self.edge_features = EdgeEmbedderConfig()

def custom_collate_fn(batch):
    collated_batch = {}
    for key in batch[0].keys():
        if key == 'gt_backbone_trajectory':
            # `gt_backbone_trajectory`는 리스트로 유지
            collated_batch[key] = [item[key] for item in batch]
        elif isinstance(batch[0][key], torch.Tensor):
            collated_batch[key] = torch.stack([item[key] for item in batch])
        else:
            collated_batch[key] = [item[key] for item in batch]
    return collated_batch

if __name__ == '__main__':
    # Hyperparameters
    num_epochs = 14
    batch_size = 1
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model configuration and model
    model_conf = ModelConfig()
    model = FlowModel(model_conf).to(device)

    # Example dataset and data loader
    dataset = ExampleDataset(num_samples=100, seq_len=50, embedding_dim=model_conf.ipa.c_s)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, input_feat in enumerate(dataloader):
            # Move tensors to device
            for key, val in input_feat.items():
                if key == 'gt_backbone_trajectory':
                    seq_trajectories = []
                    for trajectory in val:  # Process each batch
                        new_traj = []
                        for item in trajectory:
                            if isinstance(item, tuple) and len(item) == 2:
                                new_traj.append((item[0].to(device), item[1].to(device)))
                            else:
                                print(f"Invalid trajectory format: {item}")
                        seq_trajectories.append(new_traj)
                    input_feat[key] = seq_trajectories
                elif isinstance(val, torch.Tensor):
                    input_feat[key] = val.to(device)

            # Forward pass
            outputs = model(input_feat)

            # Calculate loss
            pred_torsions = outputs['pred_torsions']
            backbone_trajectory = outputs['backbone_trajectory']

            gt_torsions = input_feat['gt_torsions']
            gt_backbone_trajectory = input_feat['gt_backbone_trajectory']
            gt_torsions_sin_cos = torch.stack([torch.sin(gt_torsions), torch.cos(gt_torsions)], dim=-1)

            torsion_loss = criterion(pred_torsions, gt_torsions_sin_cos)

            backbone_loss = 0.0
            for (pred_trans, pred_rot), (gt_trans, gt_rot) in zip(backbone_trajectory, gt_backbone_trajectory[0]):
                backbone_loss += criterion(pred_trans, gt_trans) + criterion(pred_rot, gt_rot)

            loss = torsion_loss + backbone_loss

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 1 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {running_loss:.4f}')
                running_loss = 0.0

    print('Finished Training')
