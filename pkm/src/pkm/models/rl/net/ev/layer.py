# import math
# import typing
# import einops
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from pkm.util.ev_util.ev_util import *
# from pkm.util.ev_util.rotm_util import Y_func_V2
# import pkm.util.transform_util as tutil
# EPS = 1e-8

# class _NormMLP(nn.Sequential):
#     """
#     3-layer MLP used when `mlp_norm=True`.
#     Builds lazily so it can match the channel size encountered at run-time.
#     """
#     def __init__(self, channels: int):
#         super().__init__(
#             nn.Linear(channels, 16),
#             nn.SELU(inplace=True),
#             nn.Linear(16, 16),
#             nn.SELU(inplace=True),
#             nn.Linear(16, channels),
#         )

# class MakeHDFeature(nn.Module):
#     """
#     PyTorch re-implementation of the original Flax `MakeHDFeature`.

#     Args
#     ----
#     rot_configs : dict or Namespace with key ``'dim_list'``
#     mlp_norm    : bool – whether to use the tiny MLP that rescales ‖x‖.
#     """
#     def __init__(self, rot_configs: typing.Mapping, channels: int, mlp_norm: bool = False):
#         super().__init__()
#         self.rot_configs = rot_configs
#         self.mlp_norm = mlp_norm

#         # We create one tiny MLP per entry in dim_list *once* here,
#         # because Flax created fresh Dense layers inside the loop.
#         if mlp_norm:
#             self.norm_mlps = nn.ModuleList(
#                 [_NormMLP(channels) for _ in rot_configs["dim_list"]]
#             )

#     def forward(self, x: torch.Tensor, feat_axis: int = -2) -> torch.Tensor:
#         x_norm = tutil.safe_norm(x, axis=feat_axis, keepdims=True)
#         zero_norm = x_norm < EPS                 # mask for zero vectors
#         x_hat = x / x_norm
#         ft_list = []
#         for i, dl in enumerate(self.rot_configs["dim_list"]):
#             x_ = Y_func_V2(
#                 dl,
#                 x_hat.transpose(-1, feat_axis),
#                 self.rot_configs,
#                 normalized_input=True,
#             ).transpose(-1, feat_axis)

#             x_ = torch.where(zero_norm, torch.zeros_like(x_), x_)

#             if self.mlp_norm:
#                 x_norm_scaled = self.norm_mlps[i](x_norm)
#             else:
#                 x_norm_scaled = x_norm

#             x_ = torch.where(zero_norm, x_, x_ * x_norm_scaled)
#             ft_list.append(x_)
#         feat = torch.cat(ft_list, dim=-2)
#         return feat

# class EVNNonLinearity(nn.Module):
#     def __init__(self, channels: int, feat_axis: int = -2):
#         super().__init__()
#         self.feat_axis = feat_axis

#         # Dense(max(channels, 1)) ➜ Dense(channels) – `channels` is ≥1 anyway
#         self.fc1 = nn.Linear(channels, channels, bias=True)
#         self.fc2 = nn.Linear(channels, channels, bias=True)

#         # --- match Flax zeros_init ------------------------------------------------
#         nn.init.zeros_(self.fc1.weight)
#         nn.init.zeros_(self.fc1.bias)
#         nn.init.zeros_(self.fc2.weight)
#         nn.init.zeros_(self.fc2.bias)
#         # ------------------------------------------------------------------------- #

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         norm = safe_norm(x, axis=self.feat_axis, keepdims=True)  # (..., 1, C)
#         zero_norm = norm <= EPS

#         norm_bn = self.fc1(norm)
#         norm_bn = F.selu(norm_bn)
#         norm_bn = self.fc2(norm_bn) + norm       # residual connection

#         x_scaled = torch.where(
#             zero_norm,
#             x,                          # leave true zero vectors untouched
#             x / norm * norm_bn          # scale non-zeros
#         )
#         return x_scaled

        
# class EVLinearLeakyReLU(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, feat_axis: int = -2):
#         super().__init__()
#         self.linear = nn.Linear(in_channels, out_channels, bias=False)
#         self.evn    = EVNNonLinearity(channels=out_channels, feat_axis=feat_axis)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.linear(x)
#         x = self.evn(x)
#         return x

# class EVNStdFeature(nn.Module):
#     """
#     Parameters
#     ----------
#     in_feat_dim  : int   – original feature size  (== `nf` in Flax code)
#     out_feat_dim : int   – size after final dense layer
#     feat_axis    : int   – axis of 3-vectors (default -3)
#     """
#     def __init__(self,
#                  in_feat_dim : int,
#                  out_feat_dim: int,
#                  feat_axis   : int = -3):
#         super().__init__()
#         self.in_feat_dim  = in_feat_dim
#         self.out_feat_dim = out_feat_dim
#         self.feat_axis    = feat_axis

#         hid1 = max(in_feat_dim // 2, 1)
#         hid2 = max(in_feat_dim // 4, 1)

#         self.ll1   = EVLinearLeakyReLU(in_feat_dim, hid1, feat_axis=feat_axis)
#         self.ll2   = EVLinearLeakyReLU(hid1,       hid2, feat_axis=feat_axis)
#         self.dense = nn.Linear(hid2, out_feat_dim, bias=False)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # shape check
#         assert x.shape[-1] == self.in_feat_dim, \
#             f"EVNStdFeature expected last dim {self.in_feat_dim}, got {x.shape[-1]}"

#         z = self.ll1(x)
#         z = self.ll2(z)
#         z = self.dense(z)                                   #  ... j i
#         y = torch.einsum('...ji,...jk->...ik', z, x)        #  ... i k
#         return y



# class EVSTNkd(nn.Module):
#     def __init__(
#             self,
#             base_dim : int,
#             in_dim   : int,
#             feat_axis: int = -2):
#         super().__init__()
#         self.base_dim  = base_dim
#         self.in_dim    = in_dim
#         self.feat_axis = feat_axis

#         # First three stages
#         self.l1 = EVLinearLeakyReLU(in_dim,       base_dim,   feat_axis, args)
#         self.l2 = EVLinearLeakyReLU(base_dim,     base_dim*2, feat_axis, args)
#         self.l3 = EVLinearLeakyReLU(base_dim*2,   base_dim*4, feat_axis, args)

#         # After mean-pool over the 3-vector axis (-3)
#         self.l4 = EVLinearLeakyReLU(base_dim*4,   base_dim*2, feat_axis, args)
#         self.l5 = EVLinearLeakyReLU(base_dim*2,   base_dim,   feat_axis, args)
#         self.l6 = EVLinearLeakyReLU(base_dim,     in_dim,     feat_axis, args)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         assert x.shape[-1] == self.in_dim, \
#             f"EVSTNkd expected last dim {self.in_dim}, got {x.shape[-1]}"

#         x = self.l1(x)
#         x = self.l2(x)
#         x = self.l3(x)

#         # mean over the 3-vector axis (default = −3)
#         x = torch.mean(x, dim=-3)

#         x = self.l4(x)
#         x = self.l5(x)
#         x = self.l6(x)
#         return x


# def safe_norm(x, axis, keepdims=False, eps=0.0):
#     is_zero = torch.all(torch.isclose(x,torch.zeros_like(x)), dim=axis, keepdim=True)
#     # temporarily swap x with ones if is_zero, then swap back
#     x = torch.where(is_zero, torch.ones_like(x), x)
#     n = torch.norm(x, dim=axis, keepdim=keepdims)
#     is_zero = is_zero if keepdims else torch.squeeze(is_zero, -1)
#     n = torch.where(is_zero, torch.zeros_like(n), n)
#     return n.clip(eps)


# class AttnDropout(nn.Module):
#     def __init__(self, dropout: float):
#         super().__init__()
#         if not (0.0 <= dropout < 1.0):
#             raise ValueError("dropout probability has to be in [0, 1).")
#         self.dropout = dropout
#         self.keep_prob = 1.0 - dropout

#     def forward(
#         self,
#         attn: torch.Tensor,
#         det : bool = True,
#     ):
#         if det or self.p == 0.0:
#             return attn

#         keep_mask = torch.rand_like(attn) < self.keep_prob
#         multiplier = keep_mask.type_as(attn) / self.keep_prob
#         return attn * multiplier


# class EVLayerNorm(nn.Module):
#     def __init__(self, vector_axis: int = -2, eps: float = EPS):
#         super().__init__()
#         self.vector_axis = vector_axis
#         self.eps = eps

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x_norm = torch.norm(x, dim=self.vector_axis, keepdim=True)
#         # symmetric concat → variance identical to that of ±norm
#         sigma = torch.std(
#             torch.cat([ x_norm, -x_norm ], dim=-1),
#             dim=-1,
#             keepdim=True
#         )
#         return x / (sigma + self.eps)

# class CrossAttention(nn.Module):
#     def __init__(self,
#                  in_dim       : int,
#                  qk_dim       : int,
#                  v_dim        : int,
#                  n_heads      : int,
#                  attn_type    : str  = "dot",
#                  dropout      : float = 0.0,
#                  normalize_qk : bool  = False,
#                  feat_axis    : int   = -2):
#         super().__init__()
#         if qk_dim % n_heads != 0 or v_dim % n_heads != 0:
#             raise ValueError("qk_dim and v_dim must be divisible by n_heads")

#         self.qk_dim      = qk_dim
#         self.v_dim       = v_dim
#         self.n_heads     = n_heads
#         self.attn_type   = attn_type.lower()
#         self.dropout_p   = dropout
#         self.normalize_qk= normalize_qk
#         self.feat_axis   = feat_axis

#         # 1) initial EV-linear projections
#         self.q_proj = EVLinearLeakyReLU(in_dim, qk_dim, feat_axis)
#         self.k_proj = EVLinearLeakyReLU(in_dim, qk_dim, feat_axis)
#         self.v_proj = EVLinearLeakyReLU(in_dim, v_dim,  feat_axis)

#         # 2) optional normalisation of q & k
#         if normalize_qk:
#             self.q_norm = EVLayerNorm(vector_axis=feat_axis)
#             self.k_norm = EVLayerNorm(vector_axis=feat_axis)

#         # 3) final EV-linear after the attention read-out
#         self.o_proj = EVLinearLeakyReLU(v_dim, v_dim, feat_axis)

#         # 4) dropout helper
#         self.attn_drop = AttnDropout(dropout)

#         self.head_dim_qk = qk_dim // n_heads
#         self.head_dim_v  = v_dim // n_heads

#     def _split_heads(self, x: torch.Tensor, head_dim: int) -> torch.Tensor:
#         """
#         (..., D) → (H, ..., head_dim)   where D = H·head_dim
#         """
#         x = x.view(*x.shape[:-1], self.n_heads, head_dim)

#     def forward(self,
#                 qfts : torch.Tensor,
#                 kvfts: torch.Tensor,
#                 *,
#                 det : bool = True,
#                 generator: torch.Generator | None = None) -> torch.Tensor:
#         """
#         Shapes follow the original comment:

#             qfts  : (B, C, 3, Nq)
#             kvfts : (B, C, 3, Nk)

#         Only the **last** axis (feature dim) is fixed here; everything left of
#         it is treated as “batch + token” dimensions and remains untouched.
#         """
#         if qfts.shape[-1] != kvfts.shape[-1]:
#             raise ValueError("qfts and kvfts must share the feature dimension")

#         # 1) linear + split heads
#         q = self._split_heads(self.q_proj(qfts), self.head_dim_qk)  # (H, ..., qdim)
#         k = self._split_heads(self.k_proj(kvfts), self.head_dim_qk)
#         v = self._split_heads(self.v_proj(kvfts), self.head_dim_v)

#         # 2) optional normalisation along the vector axis
#         if self.normalize_qk:
#             q = self.q_norm(q)
#             k = self.k_norm(k)

#         # ---------------------------------------------------------------
#         # 3) build attention scores
#         # ---------------------------------------------------------------
#         if self.attn_type == "dot":
#             scale = math.sqrt(k.shape[-1])
#             attn  = torch.einsum('h...d,h...d->h...', q, k) / scale

#         elif self.attn_type == "sub":
#             scale = math.sqrt(k.shape[-1])
#             diff  = q.unsqueeze(-3) - k.unsqueeze(-4)      # broadcast
#             mu    = diff.mean(dim=-1, keepdim=True)
#             attn  = torch.einsum('h...d,h...d->h...', diff, mu.squeeze(-1))
#             attn  = attn / scale

#         elif self.attn_type == "slot":
#             # same as dot, but softmax over *head* axis then L2-norm
#             attn = torch.einsum('h...d,h...d->h...', q, k)
#             attn = F.softmax(attn, dim=-2)
#             attn = safe_norm(attn, axis=-2, keepdims=True)
#         else:
#             raise ValueError(f"Unsupported attn_type '{self.attn_type}'")

#         attn = F.softmax(attn, dim=-1)              # softmax over keys
#         attn = self.attn_drop(attn, det=det, generator=generator)

#         # 4) read-out and merge heads
#         resi = torch.einsum('h...k,h...kd->h...d', attn, v)
#         resi = einops.rearrange(resi, 'h ... d -> ... (h d)')
#         resi = self.o_proj(resi)
#         return resi
#         return x.movedim(-2, 0)         # head axis to the very front

        
# class EVNResnetBlockFC(nn.Module):
#     """
#     Simple 2-layer ResNet‐style block with EVN non-linearities.
#     """
#     def __init__(self,
#                  in_dim  : int,
#                  hidden  : int,
#                  out_dim : int,
#                  *,
#                  feat_axis: int = -2):
#         super().__init__()
#         self.in_dim  = in_dim
#         self.out_dim = out_dim

#         self.act1 = EVNNonLinearity(in_dim,  feat_axis)
#         self.fc1  = nn.Linear(in_dim, hidden, bias=False)
#         self.act2 = EVNNonLinearity(hidden,  feat_axis)
#         self.fc2  = nn.Linear(hidden, out_dim, bias=False)

#         if in_dim != out_dim:
#             self.skip = nn.Linear(in_dim, out_dim, bias=False)
#         else:
#             self.skip = nn.Identity()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         assert x.shape[-1] == self.in_dim
#         dx  = self.act1(x)
#         dx  = self.fc1(dx)
#         dx  = self.act2(dx)
#         dx  = self.fc2(dx)
#         return self.skip(x) + dx


# class InvCrossAttention(nn.Module):
#     args:typing.NamedTuple
#     qk_dim:int
#     v_dim:int
#     query_size:int
#     multi_head:int
    
#     @nn.compact
#     def __call__(self, x, jkey=None, det=True):
#         '''
#         x: (... F D)
#         '''
#         learnable_query = self.param('learnable_query', nn.initializers.lecun_normal(),
#                                           (self.query_size, self.qk_dim))
#         kvfts = EVLinearLeakyReLU(self.args, self.qk_dim)(x)
#         kvfts_inv = jnp.einsum('...fd,...f', kvfts, jnp.mean(kvfts, axis=-1))

#         q = nn.DenseGeneral(axis=-1, features=(self.multi_head, self.qk_dim//self.multi_head), use_bias=False)(learnable_query)
#         k = nn.DenseGeneral(axis=-1, features=(self.multi_head, self.qk_dim//self.multi_head), use_bias=False)(kvfts_inv)
#         v = nn.DenseGeneral(axis=-1, features=(self.multi_head, self.v_dim//self.multi_head), use_bias=False)(kvfts)
#         q,k,v = map(lambda x: jnp.moveaxis(x, -2, 0), (q,k,v))
        
#         if self.args.normalize_qk:
#             q = nn.LayerNorm()(q)
#             k = nn.LayerNorm()(k)

#         scale = np.sqrt(k.shape[-1])
#         attn = jnp.einsum('...qi,...bki->...bqk', q, k) # (b n n)
#         attn = nn.softmax(attn/scale, axis=-1)
        
#         if self.args.dropout > 0.0 and not det:
#             attn = AttnDropout(self.args)(attn, jkey, det=det)

#         resi = jnp.einsum('...qk,...kfd->...qfd', attn, v)
#         resi = einops.rearrange(resi, 'h ... f d -> ... f (h d)')
#         resi = EVLinearLeakyReLU(self.args, self.v_dim)(resi)

#         return resi



# class QueryElements(nn.Module):
#     args:typing.NamedTuple
#     rot_configs:typing.Sequence=None
    
#     @nn.compact
#     def __call__(self, qpnts, kvfts):
#         '''
#         qpnts: (... q f)
#         kvfts: (... c f d)
#         '''
#         if qpnts.shape[-1] == 3:
#             assert self.rot_configs is not None
#             qpnts = MakeHDFeature(self.args, self.rot_configs)(qpnts[...,None]).squeeze(-1)
#         scale = np.sqrt(kvfts.shape[-1])
#         attn = jnp.einsum('...qf,...cfd->...qc', qpnts, kvfts) # (b n n)
#         attn = nn.softmax(attn/scale, axis=-1)
#         resi = jnp.einsum('...qc,...cfd->...qfd', attn, kvfts)
#         resi = resi[...,None,:,:]
#         return resi



# if __name__ == '__main__':
#     class Args:
#         pass
    
#     args = Args()
#     args.feat_dim=8
#     args.psi_scale_type=0
#     args.negative_slope=0.0
#     args.skip_connection=1
#     args.normalize_qk = 1
#     args.dropout = 0.0

#     import pkm.util.ev_util.rotm_util as rmutil
#     import pkm.util.transform_util as trutil

#     rot_configs = rmutil.init_rot_config(0, [1, 2], rot_type='wigner')

#     jkey = jax.random.PRNGKey(0)
#     x = jax.random.normal(jkey, shape=(10, 3, 8, 2))
#     randR = rmutil.rand_matrix(10)

#     model = InvCrossAttention(args, 16, 16, 4, 2)
#     outputs, params = model.init_with_output(jkey, x)


#     # model = EVLinearLeakyReLU(args, 4)
#     # outputs, params = model.init_with_output(jkey, x)

#     res = model.apply(params, x)
#     resrot = rmutil.apply_rot(res, randR[...,None,:,:], rot_configs, -2)

#     rotx = rmutil.apply_rot(x, randR[...,None,:,:], rot_configs, -2)
#     res2 = model.apply(params, rotx)

#     make_feature = MakeHDFeature(args, rot_configs)

#     jkey = jax.random.PRNGKey(0)

#     outputs, params = make_feature.init_with_output(jkey, x)

#     randR = rmutil.rand_matrix(10)
#     res = make_feature.apply(params, x)
#     resrot = rmutil.apply_rot(res, randR[...,None,:,:], rot_configs, -2)
#     rotx = jnp.einsum('...ij,...jk->...ik',randR,x)
#     resrot2 = make_feature.apply(params, rotx)

#     resid = jnp.abs(resrot - resrot2)

#     print(1)