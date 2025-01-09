import torch.nn as nn

import sys
sys.path.append("/workspace/4D-Diff-RNA_test_1/")

import torch
import torch.nn as nn
import torch.nn.functional as F
from data import edge_embedder, node_embedder
from invariant_point_attention import invariant_point_attention
from modules import MotionAlignment, SpatialModule, EdgeUpdate, BackboneUpdate, Process_trajectory, SinusoidalTimeEmbedding


class IPATransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_tokens=None,
        predict_points=False,
        detach_rotations=True,
        **kwargs
    ):
        super().__init__()

        # using quaternion functions from pytorch3d
        try:
            from pytorch3d.transforms import quaternion_multiply, quaternion_to_matrix
            self.quaternion_to_matrix = quaternion_to_matrix
            self.quaternion_multiply = quaternion_multiply
        except (ImportError, ModuleNotFoundError) as err:
            print('unable to import pytorch3d - please install with `conda install pytorch3d -c pytorch3d`')
            raise err

        # embedding
        self.token_emb = nn.Embedding(num_tokens, dim) if invariant_point_attention.exists(num_tokens) else None

        # layers
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                invariant_point_attention.IPABlock(dim=dim, **kwargs),
                nn.Linear(dim, 6)
            ]))

        # whether to detach rotations or not, for stability during training
        self.detach_rotations = detach_rotations

        # output
        self.predict_points = predict_points
        if predict_points:
            self.to_points = nn.Linear(dim, 3)

    def forward(
        self,
        single_repr,
        translations=None,
        quaternions=None,
        pairwise_repr=None,
        mask=None
    ):
        x, device = single_repr, single_repr.device
        b, n, *_ = x.shape

        if invariant_point_attention.exists(self.token_emb):
            x = self.token_emb(x)

        # if no initial quaternions passed in, start from identity
        if not invariant_point_attention.exists(quaternions):
            quaternions = torch.tensor([1., 0., 0., 0.], device=device)  # initial rotations
            quaternions = quaternions.unsqueeze(0).unsqueeze(0).repeat(b, n, 1)

        # if not translations passed in, start from identity
        if not invariant_point_attention.exists(translations):
            translations = torch.zeros((b, n, 3), device=device)

        # go through the layers and apply invariant point attention and feedforward
        for block, to_update in self.layers:
            rotations = self.quaternion_to_matrix(quaternions)

            if self.detach_rotations:
                rotations = rotations.detach()

            x = block(
                x,
                pairwise_repr=pairwise_repr,
                rotations=rotations,
                translations=translations
            )

            # update quaternion and translation
            quaternion_update, translation_update = to_update(x).chunk(2, dim=-1)
            quaternion_update = F.pad(quaternion_update, (1, 0), value=1.)
            quaternion_update = quaternion_update / torch.linalg.norm(quaternion_update, dim=-1, keepdim=True)
            quaternions = self.quaternion_multiply(quaternions, quaternion_update)
            translations = translations + torch.einsum('b n c, b n c r -> b n r', translation_update, rotations)

        if not self.predict_points:
            return x, translations, quaternions

        points_local = self.to_points(x)
        rotations = self.quaternion_to_matrix(quaternions)
        points_global = torch.einsum('b n c, b n c d -> b n d', points_local, rotations) + translations
        return points_global


class DenoisingModel(nn.Module):
    def __init__(self, model_conf):
        super(DenoisingModel, self).__init__()
        self._model_conf = model_conf
        self.num_iterations = 4  #config.num_iterations
        self.ipa = IPATransformer()
        self.spatial_module = SpatialModule()
        self.motion_alignment = MotionAlignment()
        self.edge_update = EdgeUpdate()
        self.backbone_update = BackboneUpdate()

        self.node_embedder = node_embedder.NodeEmbedder(model_conf.node_features)
        self.edge_embedder = edge_embedder.EdgeEmbedder(model_conf.edge_features)

    def forward(self, seq, coord_4d):
        S = len(coord_4d)   # Length of Frame
        node_emb = self.node_embedder(seq)
        edge_emb = self.edge_embedder(seq)
        # Time_emb = SinusoidalTimeEmbedding()

        V_0 = node_emb.repeat(S, 1, 1)   # (Frame, residue, residue, Node_dim) 
        Z_0 = edge_emb.repeat(S, 1, 1, 1)    # (Frame, residue, Edge_dim)
        
        T_0 = Process_trajectory(coord_4d)
        V = V_0
        Z = Z_0
        T = coord_4d  # [Frame, Residue, coord(x, y, z)]
        backbone_trajectory = []

        for i in range(1, coord_4d[-1]):
            T = coord_4d[i]
            
            for _ in range(self.num_iterations):
                V_1 = self.ipa(V, Z, T)
                V_2 = V_1 + V
                V_3 = torch.concat((V_2, V_0), dim = 0)
                V_new_spatial = self.spatial_module(V_3)
                V_4 = V_new_spatial + V_2
                V_5 = torch.concat((V_4, V_0), dim = 0)
                V_new_motion = self.motion_alignment(V_5)
                V_new = V_4 + V_new_motion

                V = V_new
                Z = self.edge_update(Z, V_new)
                T = self.backbone_update(T, V_new)
            
            backbone_trajectory.append(T)
        
        result_trajectory = backbone_trajectory
        
        return result_trajectory
    
    def _apply_mask(self, aatype_diff, aatype_0, diff_mask):
        return diff_mask * aatype_diff + (1 - diff_mask) * aatype_0
    
    # Define model configuration and create instance
model_conf = {
    'node_features': 128,  # Example value
    'edge_features': 64,  # Example value
    # Add other necessary configuration parameters
}

# Create model instance
model = DenoisingModel(model_conf)