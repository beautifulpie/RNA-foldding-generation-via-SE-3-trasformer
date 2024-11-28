import numpy as np
import torch.nn as nn
import torch

class MotionAlignment(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads):
        super().__init__()
        self.position_embedding = nn.Embedding(100, input_dim)  # 예를 들어, 100개의 위치 임베딩
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, motion_nodes, reference_nodes, noisy_nodes, time_step):
        # 위치 임베딩 가져오기
        position_embeds = self.position_embedding(time_step)

        # 노드 결합
        combined_nodes = torch.cat([motion_nodes, reference_nodes, noisy_nodes], dim=0)

        # Attention 적용
        attn_output, _ = self.attention(combined_nodes + position_embeds, combined_nodes + position_embeds, combined_nodes + position_embeds)

        # 선형 변환
        output = self.linear(attn_output)

        return output

class SpatialModule(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, node_features, reference_nodes, noisy_nodes):
        # 입력 노드와 기준 노드 결합
        combined_nodes = torch.cat([node_features, reference_nodes, noisy_nodes], dim=0)
        
        # Attention 적용
        attn_output, _ = self.attention(combined_nodes, combined_nodes, combined_nodes)
        
        # 선형 변환
        output = self.linear(attn_output)
        
        return output

class EdgeUpdate(nn.Module):
    def __init__(self, D_v, D_z):
        super(EdgeUpdate, self).__init__()
        self.linear = nn.Linear(D_v, D_v // 2)
        self.mlp = nn.Sequential(
            nn.LayerNorm(D_v // 2 + D_v // 2 + D_z),
            nn.Linear(D_v // 2 + D_v // 2 + D_z, D_v // 2 + D_v // 2 + D_z)
        )

    def forward(self, V, Z):
        V_down = self.linear(V)  # Linear(V^{l+1})
        Z_in = torch.cat((V_down.unsqueeze(1).expand(-1, V.size(1), -1), 
                          V_down.unsqueeze(0).expand(V.size(1), -1, -1), 
                          Z), dim=-1)  # concat(V_down, V_down, Z^{l+1})

        Z_out = self.mlp(Z_in)  # MLP(Z_in)
        return Z_out
    
def quat_to_rot_matrix(quat):
    """
    Convert a quaternion to a rotation matrix.
    """
    a, b, c, d = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    ab, ac, ad, bc, bd, cd = a*b, a*c, a*d, b*c, b*d, c*d
    
    rot_matrix = torch.stack([
        1 - 2*(cc + dd), 2*(bc - ad), 2*(bd + ac),
        2*(bc + ad), 1 - 2*(bb + dd), 2*(cd - ab),
        2*(bd - ac), 2*(cd + ab), 1 - 2*(bb + cc)
    ], dim=-1).reshape(-1, 3, 3)
    
    return rot_matrix

class BackboneUpdate(nn.Module):
    def __init__(self, D_v):
        super(BackboneUpdate, self).__init__()
        self.linear = nn.Linear(D_v, 6)  # Output: (b_i, c_i, d_i, X_update_i)
    
    def forward(self, V):
        updates = self.linear(V)  # Linear(V_i^l)
        b, c, d, X_update = updates[:, :1], updates[:, 1:2], updates[:, 2:3], updates[:, 3:]
        
        # Normalize quaternion
        a = torch.ones_like(b)
        norm_factor = torch.sqrt(1 + b**2 + c**2 + d**2)
        a, b, c, d = a / norm_factor, b / norm_factor, c / norm_factor, d / norm_factor
        
        # Update rotation matrix
        R_update = quat_to_rot_matrix(torch.cat([a, b, c, d], dim=-1))
        
        return R_update, X_update


def Process_trajectory():
    return
