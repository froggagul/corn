#!/usr/bin/env python3

import torch as th
import torch.nn as nn
import torch.nn.functional as F

import einops
from pathlib import Path

from dataclasses import dataclass, fields

from typing import Tuple, Dict, Optional

from pkm.models.rl.net.base import FeatureBase
from pkm.models.cloud.point_mae import (
    PointMAEEncoder,
    get_group_module,
    get_patch_module,
    get_pos_enc_module,
    get_group_module_v2
)
from pkm.models.common import (transfer, MultiHeadLinear, MLP)
from pkm.train.ckpt import load_ckpt, last_ckpt
from pkm.util.config import recursive_replace_map, ConfigBase
from pkm.util.latent_obj_util import LatentObjects
import pkm.util.transform_util as tutil
import pkm.util.ev_util.rotm_util as rmutil
from icecream import ic

def _broadcast_to(x: th.Tensor, shape: th.Size) -> th.Tensor:
    """Torch does not yet have `broadcast_to` in stable, so emulate it."""
    if list(x.shape) == list(shape):
        return x
    repeats = []
    for s_orig, s_new in zip(x.shape, shape):
        repeats.append(1 if s_orig == s_new else s_new)
    return x.expand(*shape)

class DSLRCollisionDecoder(nn.Module):
    @dataclass
    class Config(FeatureBase.Config):
        dim_in: Tuple[int, ...] = (2, 19)  # pair of p and z
        dim_out: int = 1 # collision logit
        depth: int = 3
        ckpt: str = "/input/collision_dec.pt"

    def __init__(self, cfg: Config, rot_configs: Optional[Dict[str, th.Tensor]] = None):
        super().__init__()
        self.cfg = cfg
        self.depth = cfg.depth
        self.rot_configs = rot_configs

        self.line_segment_mlps = nn.ModuleList()
        self.line_segment_mlps.append(nn.Linear(
            in_features=6,
            out_features=16,
        ))
        self.line_segment_mlps.append(nn.Linear(
            in_features=16,
            out_features=16,
        ))

        self.z_b_mlps = nn.ModuleList()
        self.z_b_mlps.append(
            nn.Linear(
                in_features=32,
                out_features=16,
            ))
        self.z_b_mlps.append(nn.Linear(
                in_features=16,
                out_features=16,
            ))

        self.pos_mlps = nn.ModuleList()
        self.pos_mlps.append(nn.Linear(
            in_features=1,
            out_features=16,
        ))
        self.pos_mlps.append(nn.Linear(
            in_features=16,
            out_features=16,
        ))
        self.pairwise_mlps = nn.ModuleList()
        self.pairwise_mlps.append(nn.Linear(
            in_features=48,
            out_features=64,
        ))
        self.pairwise_mlps.append(nn.Linear(
            in_features=64,
            out_features=64,
        ))
        self.pairwise_mlps.append(nn.Linear(
            in_features=64,
            out_features=64,
        ))
        self.out_linear = nn.Linear(
            in_features=64,
            out_features=1,
        )

    def forward(self, latent_obj_a: LatentObjects, latent_obj_b: LatentObjects, a_idx: th.tensor, b_idx: th.tensor) -> th.Tensor:
        z_flat_a = th.gather(
            latent_obj_a.z_flat,
            dim=1,
            index=a_idx.unsqueeze(-1).expand(-1, -1, *latent_obj_a.z_flat.shape[2:])
        ) # N, k, 16
        z_flat_b = th.gather(
            latent_obj_b.z_flat,
            dim=1,
            index=b_idx.unsqueeze(-1).expand(-1, -1, *latent_obj_b.z_flat.shape[2:])
        )
        z_a = th.gather(
            latent_obj_a.z,
            dim=1,
            index=a_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, *latent_obj_a.z.shape[2:])
        ) # N, k, 8, 2
        z_b = th.gather(
            latent_obj_b.z,
            dim=1,
            index=b_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, *latent_obj_b.z.shape[2:])
        ) # N, k, 8, 2
        latent_obj_a_fps_tf = th.gather(
            latent_obj_a.fps_tf,
            dim=1,
            index=a_idx.unsqueeze(-1).expand(-1, -1, *latent_obj_a.fps_tf.shape[2:])
        ) # N, K, 80, 3
        latent_obj_b_fps_tf = th.gather(
            latent_obj_b.fps_tf,
            dim=1,
            index=b_idx.unsqueeze(-1).expand(-1, -1, *latent_obj_b.fps_tf.shape[2:])
        ) # N, K, 80, 3

        pairwise_pos_dif = latent_obj_a_fps_tf[:, :, None, :] - latent_obj_b_fps_tf[:, None, :, :] # N, K, K, 3

        z_norm_a: th.Tensor = th.norm(z_flat_a, dim=-1) # N, K
        z_norm_b: th.Tensor = th.norm(z_flat_b, dim=-1) # N, K

        pairwise_outer_shape = pairwise_pos_dif.shape[:-1] # N, K, K
        if list(z_norm_b.shape) != list(pairwise_outer_shape):
            z_norm_b = z_norm_b[...,None,:] # N, 1, K
            z_b = z_b[...,None,:,:,:] # N, 1, K, 8, 2
        else:
            z_b = z_b

        z_norm = th.maximum(z_norm_a.unsqueeze(-1), z_norm_b) # N, K, K

        pairwise_dist = th.norm(pairwise_pos_dif, dim=-1) # N, K, K
        scale = th.where(z_norm > 2 * pairwise_dist, z_norm, 2 * pairwise_dist) # N, K, K
        scale = scale.detach()  # stop‑grad equivalent

        pairwise_z_shape = pairwise_outer_shape + latent_obj_a.z.shape[-2:] # N, K, K, 8, 2

        pairwise_z_a = _broadcast_to(z_a[...,None,:,:], pairwise_z_shape)
        pairwise_z_b = _broadcast_to(z_b, pairwise_z_shape)
        pairwise_z = th.cat([pairwise_z_a, pairwise_z_b], dim=-1)    # (..., J, K, j_a/b, k_a/b*2)

        swap_mask = z_norm_a[...,None] < z_norm_b
        pairwise_pos_dif = th.where(swap_mask[...,None], -pairwise_pos_dif, pairwise_pos_dif)
        pairwise_z_swapped = th.cat([pairwise_z_b, pairwise_z_a], dim=-1)
        pairwise_z = th.where(swap_mask[...,None,None], pairwise_z_swapped, pairwise_z)

        line_align_Rm = tutil.line2Rm(
            pairwise_pos_dif,
            th.tensor([1, 0, 0], dtype=pairwise_pos_dif.dtype, device=pairwise_pos_dif.device).view(1, 1, 1, 3)
        )                 # custom util (torch)
        line_align_Rm_inv = tutil.Rm_inv(line_align_Rm)
        pairwise_z = rmutil.apply_rot(pairwise_z,                       # custom util (torch)
                                       line_align_Rm_inv,
                                       self.rot_configs,
                                       feature_axis=-2)

        pairwise_pos_dif = th.einsum('...ij,...j', line_align_Rm_inv, pairwise_pos_dif)

        pairwise_z = pairwise_z / scale[...,None,None]
        pairwise_pos_dif = pairwise_pos_dif / scale[...,None]

        z_feat_dim = z_flat_a.shape[-1]
        original_z_feat_dim = z_feat_dim
        
        line_seg_feat = th.zeros((6, ), dtype=pairwise_z.dtype, device=pairwise_z.device)
        for layer in self.line_segment_mlps:
            line_seg_feat = layer(line_seg_feat)
            line_seg_feat = F.gelu(line_seg_feat)
        
        pairwise_z_a, pairwise_z_b = pairwise_z[...,:z_a.shape[-1]], pairwise_z[...,z_a.shape[-1]:]
        pairwise_z_b_feat = th.zeros((original_z_feat_dim+z_feat_dim,), dtype=pairwise_z.dtype, device=pairwise_z.device)

        for layer in self.z_b_mlps:
            pairwise_z_b_feat = layer(pairwise_z_b_feat)
            pairwise_z_b_feat = F.gelu(pairwise_z_b_feat)

        pairwise_z = th.cat([pairwise_z_a, pairwise_z_b], dim=-1)
        pairwise_pos_feat = pairwise_pos_dif[..., 2:]  # drop first two components, because we already have them in z

        for layer in self.pos_mlps:
            pairwise_pos_feat = F.gelu(layer(pairwise_pos_feat))

        pairwise_z_flat = einops.rearrange(pairwise_z, '... j k -> ... (j k)')
        pairwise_feat = th.cat([pairwise_z_flat, pairwise_pos_feat], dim=-1)

        # skip = None
        x = pairwise_feat
        for idx, layer in enumerate(self.pairwise_mlps):
            x = F.gelu(layer(x))
            if idx == 0:
                skip = x
        embed = x + skip

        # out = self.out_linear(x)  # (..., 1)
        # # normalize output with scale
        # out = (out - 30.0) / 50.0

        K = latent_obj_a_fps_tf.shape[1]  # K is the number of points in each object
        pairwise_pos_cat = th.cat([
            latent_obj_a_fps_tf[:, :, None, :].expand(-1, -1, K, -1), # N, K, K, 3
            latent_obj_b_fps_tf[:, None, :, :].expand(-1, K, -1, -1), # N, K, K, 3
        ], dim=-1) # N, K, K, 6
        z_flat_cat = th.cat([
            z_flat_a[:, :, None, :].expand(-1, -1, K, -1),  # N, K, K, 16
            z_flat_b[:, None, :, :].expand(-1, K, -1, -1),  # N, K, K, 16
        ], dim=-1)
        out = th.cat([
            pairwise_pos_cat,
            z_flat_cat,
            embed,
        ], dim=-1)  # N, K, K, 6 + 32 + 64
        
        return out
