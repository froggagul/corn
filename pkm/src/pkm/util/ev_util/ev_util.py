"""
Modified from https://github.com/FlyingGiraffe/vnn-pc
"""

import numpy as np
import torch
import einops

# from ev_utils.rotm_util import Y_func_V2

EPS = 1e-6

def knn(x:torch.Tensor, k):
    '''
    (... P F)
    '''

    # pairwise_distance = jnp.sum((x[...,None,:] - x[...,None,:,:])**2, axis=-1)

    inner = -2*torch.matmul(x, x.swapaxes(-1, -2)) # (... P P)
    xx:torch.Tensor = torch.sum(x**2, dim=-1, keepdim=True) # (... P 1)
    pairwise_distance:torch.Tensor = xx + inner + xx.swapaxes(-1, -2)
    
    return torch.argsort(pairwise_distance, dim=-1)[...,:k]


def get_graph_feature(x, k=20, cross=False, idx_out=False):
    nf = x.shape[-2]
    x = einops.rearrange(x, '... f d -> ... (f d)')
    idx = knn(x, k=k)   # (batch_size, num_points, k)
    feature = torch.take_along_dim(x[...,None,:], idx[...,None], dim=-3) # (B, P, K, FD)
    x = einops.repeat(x, '... j -> ... r j', r=k)
    if cross:
        cross = torch.cross(feature, x)
        feature = torch.stack((feature-x, x, cross), dim=-1)
    else:
        feature = torch.stack((feature-x, x), dim=-1)
    feature = einops.rearrange(feature, '... (f d) k -> ... f (d k)', f=nf)
    if idx_out:
        return feature, idx
    else:
        return feature


if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    # jkey = jax.random.PRNGKey(0)
    # x = jax.random.normal(jkey, shape=(10,3,1))
    x = torch.randn(10, 3, 1)
    
    feat, idx = get_graph_feature(x, k=4, cross=True, idx_out=True)
    import pkm.util.transform_util as trutil
    randquat = trutil.qrand((1,), 1)
    randR = trutil.q2R(randquat)
    
    featrot2 = torch.einsum('...ij,...kjd->...kid', randR, feat)
    rotx = torch.einsum('...ij,...jd->...id', randR, x)
    featrot, idx2 = get_graph_feature(rotx, k=4, cross=True, idx_out=True)

    print(1)