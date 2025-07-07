import torch
import os
import numpy as np
from dataclasses import dataclass

from pkm.util.latent_obj_util import LatentObjects
from pkm.models.rl.net.ev.collision_decoder import DSLRCollisionDecoder



@dataclass
class Config:
    oricorn_path: str = "/input/DGN/meta-v8/oricorn"
    reduce_k = 128
    reduce_mode = "closest" # ["closest", "closest_farthest", "all"]
    embed_mode = "concat" # ["concat", "decoder"]
    net = DSLRCollisionDecoder.Config()
cfg = Config()

device = "cuda"
with open(os.path.join(cfg.oricorn_path, '..', 'rot_configs.npy'), 'rb') as f:
    rot_configs = np.load(f, allow_pickle=True).item()

rot_configs = data = {
    "type": rot_configs["type"],
    "Js": [
        None if j is None else torch.tensor(j, dtype=torch.float32, device=device) for j in rot_configs["Js"]
    ],
    "Y_basis": [
        None if y is None else torch.tensor(y, dtype=torch.float32, device=device) for y in rot_configs["Y_basis"]
    ],
    "D_basis": [
        None if d is None else torch.tensor(d, dtype=torch.float32, device=device) for d in rot_configs["D_basis"]
    ],
    "dim_list": rot_configs["dim_list"],
    "Y_linear_coef": [
        None if y is None else torch.tensor(y, dtype=torch.float32, device=device) for y in rot_configs["Y_linear_coef"]
    ],
    "constant_scale": rot_configs["constant_scale"],
}
decoder = DSLRCollisionDecoder(cfg.net, rot_configs)
ckpt = torch.load(cfg.net.ckpt, map_location=device)
decoder.load_state_dict(ckpt, strict=True)

decoder.to(device)
decoder.eval()


pos = torch.tensor([0.1, 0.2, 0.3], device=device, dtype=torch.float32)
z = torch.tensor([[
    [0.0, 0.1],
    [0.2, 0.3],
    [0.4, 0.5],
    [0.6, 0.7],
    [0.8, 0.9],
    [1.0, 1.1],
    [1.2, 1.3],
    [1.4, 1.5],
], [
    [1.6, 1.7],
    [1.8, 1.9],
    [2.0, 2.1],
    [2.2, 2.3],
    [2.4, 2.5],
    [2.6, 2.7],
    [2.8, 2.9],
    [3.0, 3.1],
]], device=device, dtype=torch.float32)  # shape: (2, 8, 2)
rel_fps = torch.tensor([[-0.1, 0.2, 0.3], [0.4, -0.5, 0.6]], device=device, dtype=torch.float32)  # shape: (2, 3)

objects_a = LatentObjects(
    pos = pos,
    rel_fps = rel_fps,
    z = z,
)
objects_b = LatentObjects(
    pos = pos + 0.1,
    rel_fps = rel_fps + 0.1,
    z = z + 0.1,
)
idx_a = torch.tensor([0, 1], device=device, dtype=torch.int64)
idx_b = torch.tensor([0, 1], device=device, dtype=torch.int64)

# with torch.no_grad():
#     collision_cost = decoder(objects_a[None], objects_b[None], idx_a[None], idx_b[None])
#     print("Collision Cost:", collision_cost)
#     print(collision_cost.shape)

print(
    torch.arange(0, 2).expand(32, 2)
)