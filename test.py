import torch
from einops import repeat
from invariant_point_attention import InvariantPointAttention

attn = InvariantPointAttention(
    dim = 64,                  # single (and pairwise) representation dimension
    heads = 8,                 # number of attention heads
    scalar_key_dim = 16,       # scalar query-key dimension
    scalar_value_dim = 16,     # scalar value dimension
    point_key_dim = 4,         # point query-key dimension
    point_value_dim = 4        # point value dimension
)

single_repr   = torch.randn(1, 10, 256, 64)      # (batch x seq x dim)
pairwise_repr = torch.randn(1, 10, 256, 256, 64) # (batch x seq x seq x dim)
mask          = torch.ones(1, 10, 256).bool()    # (batch x seq)

rotations     = repeat(torch.eye(3), '... -> b n ...', b = 1, n = 256)  # (batch x seq x rot1 x rot2) - example is identity
translations  = torch.zeros(1, 256, 3) # translation, also identity for example

attn_out = attn(
    single_repr,
    pairwise_repr,
    rotations = rotations,
    translations = translations,
    mask = mask
)

print(attn_out.shape) # (1, 256, 64)