import torch
from einops import repeat
from invariant_point_attention import InvariantPointAttention
from datasets import edge_embedder, node_embedder
attn = InvariantPointAttention(
    dim = 64,                  # single (and pairwise) representation dimension
    heads = 8,                 # number of attention heads
    scalar_key_dim = 16,       # scalar query-key dimension
    scalar_value_dim = 16,     # scalar value dimension
    point_key_dim = 4,         # point query-key dimension
    point_value_dim = 4        # point value dimension
)

frame = 10
seq = 256

single_repr   = torch.randn(frame, seq, 64)      # (frame x seq x dim)
pairwise_repr = torch.randn(frame, seq, seq, 64) # (frame x seq x seq x dim)
mask          = torch.ones(frame, seq).bool()    # (frame x seq)

rotations     = repeat(torch.eye(3), '... -> b n ...', b = frame, n = seq)  # (frame x seq x rot1 x rot2) - example is identity
translations  = torch.zeros(frame, seq, 3) # translation, also identity for example

attn_out = attn(
    single_repr,
    pairwise_repr,
    rotations = rotations,
    translations = translations,
    mask = mask
)

print(attn_out.shape)


import torch
import torch.nn as nn
import pickle

# Load the parameter
with open("./node_embedder_param.pickle", "rb") as f:
    model_params = pickle.load(f)

print("Model parameter:", model_params[0].shape, model_params[1].shape)

# Model config
class Config:
    def __init__(self):
        self.c_s = 128
        self.c_pos_emb = 128
        self.c_timestep_emb = 128
        self.embed_diffuse_mask = False
        self.max_num_res = 2000
        self.timestep_int = 1000

module_cfg = Config()
model = node_embedder.NodeEmbedder(module_cfg)

# Paramter 할당
model.linear.weight.data = model_params[0]
model.linear.bias.data = model_params[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.eval()  

batch_size = 2
num_res = 256
timesteps = torch.rand(batch_size, 1).to(device)  # [b, 1] 형태의 timesteps
mask = torch.ones(batch_size, num_res).to(device)  # [b, n_res] 형태의 mask

with torch.no_grad():
    node_feats = model(timesteps, mask)

print(f"Node Feature : {node_feats.shape}")
print(node_feats)
