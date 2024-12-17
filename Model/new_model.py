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
        seq = {
            'res_mask': torch.ones(self.seq_len),       # [L]
            't': torch.rand(self.seq_len),              # [L]
            'trans_t': torch.rand(self.seq_len, 3),     # [L, 3]
            'rotmats_t': torch.rand(self.seq_len, 3, 3),# [L, 3, 3]
            'trans_sc': torch.rand(self.seq_len, 3),    # [L, 3]
            'gt_torsions': torch.rand(self.seq_len, 8), # [L, 8]
            'gt_backbone_trajectory': [
                (torch.rand(3), torch.rand(3, 3)) for _ in range(self.seq_len)
            ]  # List of length L, each entry a (trans, rotmat)
        }
        coord_4d = torch.rand(self.seq_len, 4)  # [L, 4]
        return seq, coord_4d

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

# Suppose we have a FlowModel defined elsewhere
# from your_project import FlowModel

def custom_collate_fn(batch):
    # batch: list of (seq, coord_4d) pairs
    seq_list, coord_4d_list = zip(*batch)

    # Keys should be the same for all seq dictionaries
    seq_keys = seq_list[0].keys()
    collated_seq = {}
    for key in seq_keys:
        vals = [d[key] for d in seq_list]
        if torch.is_tensor(vals[0]):
            # Stack tensors along batch dimension
            collated_seq[key] = torch.stack(vals, dim=0)  # [B, L, ...]
        else:
            # For gt_backbone_trajectory, we have a list of lists of tuples.
            # We'll keep it as is, resulting in a list of length batch_size.
            # Each element: list of length L with (trans, rotmat) tuples.
            collated_seq[key] = vals

    coord_4d = torch.stack(coord_4d_list, dim=0)  # [B, L, 4]
    return collated_seq, coord_4d

if __name__ == '__main__':
    # Hyperparameters
    num_epochs = 50
    batch_size = 1  # Start with 1 for simplicity; handling lists for multiple batches is trickier.
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model configuration and model
    model_conf = ModelConfig()
    model = FlowModel(model_conf).to(device)

    # Example dataset and data loader
    dataset = ExampleDataset(num_samples=10, seq_len=10, embedding_dim=model_conf.ipa.c_s)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    # Loss function and optimizer
    criterion = nn.MSELoss()  # Just as an example
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (seq, coord_4d) in enumerate(dataloader):
            # Move tensors to device
            for key, val in seq.items():
                if torch.is_tensor(val):
                    seq[key] = val.to(device)
                else:
                    # val should be gt_backbone_trajectory: List of length B (here B=1)
                    # Each element: list of L tuples (trans, rotmat)
                    # Move each element to device
                    seq_trajectories = []
                    for trajectory in val:  # val is a list of length B
                        new_traj = [(x.to(device), y.to(device)) for (x, y) in trajectory]
                        seq_trajectories.append(new_traj)
                    seq[key] = seq_trajectories

            coord_4d = coord_4d.to(device)

            # Forward pass
            outputs = model(seq, coord_4d)

            # Calculate loss
            pred_torsions = outputs['pred_torsions']  # [B, L, ...]
            backbone_trajectory = outputs['backbone_trajectory']  # As returned by model

            gt_torsions = seq['gt_torsions']           # [B, L, 8]
            gt_backbone_trajectory = seq['gt_backbone_trajectory'] # list for each batch element
            gt_torsions_sin_cos = torch.stack([torch.sin(gt_torsions), torch.cos(gt_torsions)], dim=-1)


            # If batch_size > 1, you'll need a loop. For now, B=1.
            torsion_loss = criterion(pred_torsions, gt_torsions_sin_cos)

            # backbone_trajectory and gt_backbone_trajectory might need careful handling.
            # The model likely returns a list of length L with (pred_trans, pred_rotmat) for each residue.
            # The gt is similarly structured.
            # We'll sum losses over L:
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