import numpy as np
import torch.nn as nn
import torch

class MotionAlignment(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, max_seq_len=100):
        super().__init__()
        self.position_embedding = nn.Embedding(max_seq_len, input_dim)  # 최대 시퀀스 길이만큼 위치 임베딩
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, V_s, V_mot, V_ref):
        """
        V_s: Noised Node (B, S, D)  # 배치, 시퀀스 길이, 특성 차원
        V_mot: Motion Node (B, 1, D)
        V_ref: Reference Node (B, 1, D)
        """
        batch_size, seq_len, dim = V_s.shape

        # 노드 결합 (V_mot, V_ref는 1개 시퀀스지만, V_s는 S개 시퀀스를 가짐)
        combined_nodes = torch.cat([V_mot, V_ref, V_s], dim=1)  # (B, 2+S, D)

        # 위치 임베딩 추가 (시퀀스 길이에 맞게 생성)
        position_ids = torch.arange(combined_nodes.shape[1], device=combined_nodes.device).unsqueeze(0)
        position_embeds = self.position_embedding(position_ids)  # (1, 2+S, D)

        # Attention 적용 (Q=K=V로 사용)
        attn_output, _ = self.attention(combined_nodes + position_embeds, 
                                        combined_nodes + position_embeds, 
                                        combined_nodes + position_embeds)

        # 선형 변환
        output = self.linear(attn_output)  # (B, 2+S, output_dim)

        # Residual Connection 추가 (V_s 부분만 사용)
        output[:, 2:, :] += V_s  # 원래의 V_s에 output 값 더하기 (Residual)

        return output[:, 2:, :]  # (B, S, output_dim) -> V_s와 같은 차원으로 반환

class SpatialModule(nn.Module):
    def __init__(self, input_dim, num_heads):
        """
        Reference Network using Self-Attention.

        Args:
            input_dim (int): Feature dimension (D).
            num_heads (int): Number of heads for multi-head self-attention.
        """
        super().__init__()
        
        # Self-Attention Layer
        self.self_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        
        # Projection Matrix Wr (2D × D)
        self.projection = nn.Linear(2 * input_dim, input_dim)

        # Linear Projection Wr (D × D)
        self.linear = nn.Linear(input_dim, input_dim)


    def forward(self, V_ref, V_s):
        """
        Forward pass of Reference Network.

        Args:
            V_ref (torch.Tensor): Reference node features (S × N × D).
            V_s (torch.Tensor): Noisy node features (S × N × D).

        Returns:
            torch.Tensor: Updated noisy node features (V̂_s^l).
        """
        # Step 1: Concatenate reference and noisy node features along the last dimension
        combined_features = torch.cat([V_ref, V_s], dim=-1)  # Shape: (S, N, 2D)

        # Step 2: Project to D-dimensional space for Self-Attention
        projected_features = self.projection(combined_features)  # Shape: (S, N, D)

        # Step 3: Apply Self-Attention
        attn_output, _ = self.self_attention(projected_features, projected_features, projected_features)

        # Step 4: Linear transformation
        A_s = self.linear(attn_output)  # Shape: (S, N, D)

        # Step 5: Residual connection with original noisy node features
        V_s_hat = A_s + V_s  # Shape: (S, N, D)

        return V_s_hat
    
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


import torch
import math

class SinusoidalTimeEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim):
        super(SinusoidalTimeEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
    
    def forward(self, time_tensor):
        """
        Args:
            time_tensor: Tensor of shape [seq_len, batch_size] or [batch_size, seq_len] containing time steps or time values
        """
        # Ensure time_tensor is [seq_len, batch_size]
        if time_tensor.dim() == 2 and time_tensor.size(0) != self.embedding_dim:
            time_tensor = time_tensor.transpose(0, 1)
        
        seq_len, batch_size = time_tensor.size()
        
        # Create a tensor to hold the time embeddings
        time_emb = torch.zeros(seq_len, batch_size, self.embedding_dim, device=time_tensor.device)
        
        # Compute the scaling factors for the sine and cosine functions
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2, device=time_tensor.device).float() * (-math.log(10000.0) / self.embedding_dim))
        
        # Apply sine to even indices
        time_emb[:, :, 0::2] = torch.sin(time_tensor.unsqueeze(-1) * div_term)
        
        # Apply cosine to odd indices
        time_emb[:, :, 1::2] = torch.cos(time_tensor.unsqueeze(-1) * div_term)
        
        return time_emb
