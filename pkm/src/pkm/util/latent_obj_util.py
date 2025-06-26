import torch
import einops
import numpy as np
import numpy.typing as npt
from typing import Tuple
from dataclasses import replace
from flax.struct import dataclass as flax_dataclass
import jax

import pkm.util.transform_util as tutil
import pkm.util.ev_util.rotm_util as rmutil

POS_COEF = 1.0
FPS_COEF = 1.0
@flax_dataclass
class PointObjects:
    pos: torch.Tensor=None  # (... 3)
    color: torch.Tensor=None         # (... 3)
    rel_fps:torch.Tensor=None        # (... NP 3)
    rel_pcd: torch.Tensor=None       # (... NP 3)
    nmls: torch.Tensor=None          # (... NP 3)

    def __getitem__(self, idx: torch.Tensor)->"PointObjects":
        """Convenient indexing for dataclass"""
        return jax.tree_util.tree_map(lambda x: x[idx], self) 
        
    def init_pcd(self, pos, rel_fps, rel_pcd, nmls=None) -> "PointObjects":
        self = replace(self, rel_pcd=rel_pcd)
        self = replace(self, pos=pos)
        self = replace(self, rel_fps=rel_fps)
        self = replace(self, nmls=nmls)
        return self

    def random_color(self, jkey) -> "PointObjects":
        color = torch.randn(self.outer_shape + (3,))
        color[...,-1] = 1.0  # Set alpha channel to 1
        return replace(self, color=color)
    
    def apply_pq_pcd(self, pos, quat) -> "PointObjects":
        pos_ = tutil.pq_action(pos, quat, self.pos)
        self = replace(self, pos=pos_)
        self = self.rotate_rel_pcd(quat)
        return self
    
    def translate(self, pos) -> "PointObjects":
        if self.pos is None:
            self = self.init_pos_zero()
        return replace(self, pos=self.pos+pos)

    def apply_scale(self, scale, center=None) -> "PointObjects":        
        for _ in range(self.rel_pcd.ndim-scale.ndim):
            scale = scale[...,None,:]
        if self.rel_fps is not None:
            self = replace(self, rel_fps=scale*self.rel_fps)
        if center is not None:
            self = replace(self, pos=scale.squeeze(-1)*(self.pos-center)+center)
        return replace(self, rel_pcd=scale*self.rel_pcd)

    def rotate_rel_pcd(self, quat) -> "PointObjects":
        self = replace(self, rel_pcd=tutil.qaction(quat[...,None,:], self.rel_pcd))
        if self.rel_fps is not None:
            self = replace(self, rel_fps=tutil.qaction(quat[...,None,:], self.rel_fps))
        if self.nmls is not None:
            self = replace(self, nmls=tutil.qaction(quat[...,None,:], self.nmls))
        return self
    
    def padding_or_none(self, objB, padding) -> "PointObjects":
        def padding_or_none(a:torch.Tensor, b:torch.Tensor):
            if a is None and b is None:
                return None, None
            if a is not None and b is not None:
                return a, b
            if not padding:
                return None, None
            if a is None and b is not None:
                a = torch.zeros(self.outer_shape + b.shape[len(objB.outer_shape):])
            if a is not None and b is None:
                b = torch.zeros(objB.outer_shape + a.shape[len(self.outer_shape):])
            return a, b
        return jax.tree_util.tree_map(lambda a, b: padding_or_none(a,b), self, objB)
    
    def concat(self, objectsB, axis, padding=False) -> "PointObjects":
        '''
        axis : axis within outer shape
        '''
        if axis < 0:
            axis = len(self.outer_shape) + axis
        def padding_or_none(a:torch.Tensor, b:torch.Tensor):
            if a is None and b is None:
                return None
            if a is not None and b is not None:
                return torch.cat((a, b), dim=axis)
            if not padding:
                return None
            if a is None and b is not None:
                a = torch.zeros(self.outer_shape + b.shape[len(objectsB.outer_shape):])
            if a is not None and b is None:
                b = torch.zeros(objectsB.outer_shape + a.shape[len(self.outer_shape):])
            return torch.cat((a, b), dim=axis)
        return jax.tree_util.tree_map(lambda *x: padding_or_none(*x), self, objectsB)


    def stack(self, objectsB, axis, padding=False) -> "PointObjects":
        '''
        axis : axis within outer shape
        '''
        if axis < 0:
            axis = len(self.outer_shape) + 1 + axis
        def padding_or_none(a:torch.Tensor, b:torch.Tensor):
            if a is None and b is None:
                return None
            if a is not None and b is not None:
                return torch.stack((a, b), dim=axis)
            if not padding:
                return None
            if a is None and b is not None:
                a = torch.zeros(self.outer_shape + b.shape[len(objectsB.outer_shape):])
            if a is not None and b is None:
                b = torch.zeros(objectsB.outer_shape + a.shape[len(self.outer_shape):])
            return torch.stack((a, b), dim=axis)
        return jax.tree_util.tree_map(lambda *x: padding_or_none(*x), self, objectsB)

    def extend_outer_shape(self, axis) -> "PointObjects":
        if axis < 0:
            axis = len(self.outer_shape) + 1 + axis
        return jax.tree_util.tree_map(lambda x: torch.expand_dims(x, axis), self)
    
    def squeeze_outer_shape(self, axis) -> "PointObjects":
        if axis < 0:
            axis = len(self.outer_shape) + 1 + axis
        return jax.tree_util.tree_map(lambda x: torch.squeeze(x, axis), self)

    def extend_and_repeat_outer_shape(self, r, axis) -> "PointObjects":
        if axis < 0:
            axis = len(self.outer_shape) + 1 + axis
        self = self.extend_outer_shape(axis)
        return self.repeat_outer_shape(r, axis)

    def repeat_outer_shape(self, r, axis) -> "PointObjects":
        if axis < 0:
            axis = len(self.outer_shape) + 1 + axis
        return jax.tree_util.tree_map(lambda x: torch.broadcast_to(x, x.shape[:axis] +(r,) + x.shape[axis+1:]), self)

    def pair_split(self, axis):
        objA = jax.tree_util.tree_map(lambda x : torch.split(x, 2, axis=axis)[0], self)
        objB = jax.tree_util.tree_map(lambda x : torch.split(x, 2, axis=axis)[1], self)
        return objA, objB
    
    def take_along_outer_axis(self, indices, axis) -> "PointObjects":
        def align_indices_ndim(x, indices_):
            for _ in range(x.ndim - indices_.ndim):
                indices_ = indices_[...,None]
            return indices_
        return jax.tree_util.tree_map(lambda x: torch.take_along_axis(x, align_indices_ndim(x, indices), axis), self)

    def reshape_outer_shape(self, new_shape) -> "PointObjects":
        outer_ndim = len(self.outer_shape)
        return jax.tree_util.tree_map(lambda x: x.reshape(new_shape + x.shape[outer_ndim:]), self)

    def init_pos_zero(self) -> "PointObjects":
        self = replace(self, pos=torch.zeros(self.outer_shape + (3,)))
        return self
    
    @property
    def outer_shape(self):
        return self.rel_pcd.shape[:-2]
    
    @property
    def shape(self):
        return self.outer_shape
    
    @property
    def ndim(self):
        return len(self.outer_shape)

    @property
    def nfps(self):
        return self.rel_fps.shape[-2]

    @property
    def pcd_tf(self):
        return self.pos[...,None,:] + self.rel_pcd
    
    @property
    def fps_tf(self):
        return self.pos[...,None,:].detach() + self.rel_fps
    
    @property
    def len(self):
        rel_pcd_norm = torch.norm(self.rel_pcd, dim=-1)
        return torch.mean(torch.sort(rel_pcd_norm)[...,-rel_pcd_norm.shape[-1]//16:], dim=-1) # (nob, )
    


@flax_dataclass
class LatentObjects(PointObjects):
    z:torch.Tensor=None
    conf:torch.Tensor=None

    def __getitem__(self, idx: torch.Tensor)->"LatentObjects":
        """Convenient indexing for dataclass"""
        return jax.tree_util.tree_map(lambda x: x[idx], self) 

    def set_z_with_idx(self, indices, obj_fps_list, obj_z_list) -> "LatentObjects":
        self = replace(self, rel_fps=obj_fps_list[indices])
        self = replace(self, z=obj_z_list[indices])
        return self

    def set_z_list(self, obj_fps_list, obj_z_list) -> "LatentObjects":
        self = replace(self, rel_fps=obj_fps_list)
        self = replace(self, z=obj_z_list)
        return self

    def init_obj_info(self, obj_info, latent_obj_list:"LatentObjects", rot_configs) -> "LatentObjects":
        if isinstance(obj_info, dict):
            pq = obj_info['obj_posquats'].astype(torch.float32)
            scale = obj_info["scale"].astype(torch.float32)
            if 'oriCORN_z' in obj_info:
                latent_obj = self.replace(z=obj_info['oriCORN_z'].astype(torch.float32), rel_fps=obj_info['oriCORN_fps'].astype(torch.float32))
                latent_obj = latent_obj.init_pos_zero()
                # latent_obj.apply_scale(1/models.scale_to_origin[obj_info["idx"]]).init_pos_zero().translate(-models.translation_to_origin[obj_info["idx"]])
            else:
                latent_obj = latent_obj_list[obj_info["idx"]]
            # valid_obj_mask = obj_info["idx"] >= 0
            valid_obj_mask = scale>1e-6
            latent_obj = latent_obj.apply_scale(scale).apply_pq_z(pq[...,:3], pq[...,3:], rot_configs)
            def fill_with_rank(x, base):
                for _ in range(base.ndim-x.ndim):
                    x = x[...,None]
                return x
            latent_obj = jax.tree_util.tree_map(lambda x: torch.where(fill_with_rank(valid_obj_mask, x), x, torch.zeros_like(x)), latent_obj)
            return latent_obj
        elif isinstance(obj_info, tuple) or isinstance(obj_info, list):
            raise ValueError
    
    def valid_obj_padding(self, obj_no, jkey, z_scale:float=None, obj_valid_mask=None) -> "LatentObjects":
        '''
        self outer shape: (NB, NO)
        '''
        cur_obj_no = self.nobj
        if obj_valid_mask is None:
            obj_valid_mask = self.obj_valid_mask
        prob = obj_valid_mask.astype(torch.float32)
        prob = prob/torch.sum(prob, axis=-1, keepdims=True)
        if prob.ndim == 1:
            # valid_idx = jax.random.choice(jkey, torch.arange(self.outer_shape[-1], dtype=torch.int32), shape=(obj_no,), p=prob)
            valid_idx = torch.multinomial(prob, obj_no, replacement=True).to(torch.int32)
        else:
            # valid_idx = jax.vmap(lambda jk, a, p: jax.random.choice(jk, a, shape=(obj_no,), p=p))(jax.random.split(jkey, prob.shape[0]), 
            #                                                             einops.repeat(torch.arange(self.outer_shape[-1], dtype=torch.int32), 'i -> r i', r=prob.shape[0]), 
            #                                                             prob)
            valid_idx = torch.multinomial(prob, obj_no, replacement=True).to(torch.int32)  # (batch, obj_no)

        valid_h = torch.take_along_axis(self.h, valid_idx[...,None], axis=-2)
        if z_scale is not None:
            valid_x = self.set_h(valid_h)
            valid_h = replace(valid_x, z=z_scale*valid_x.z).h
        x0_pred_origin = self.set_h(torch.where(obj_valid_mask[...,None], self.h, valid_h[...,:cur_obj_no,:]))
        x0_origin_pad = x0_pred_origin.drop_gt_info().concat(LatentObjects().init_h(valid_h[...,cur_obj_no:,:], x0_pred_origin.latent_shape), axis=-1)

        return x0_origin_pad

    def init_pos_zero(self) -> "LatentObjects":
        self = replace(self, pos=torch.zeros(self.outer_shape + (3,)).astype(torch.float32))
        return self
    
    def apply_scale(self, scale, center=None) -> "LatentObjects":
        if isinstance(scale, float) or isinstance(scale, int):
            scale = torch.tensor([scale])
        # assert scale.shape[-1] != 3
        for _ in range(self.rel_fps.ndim-scale.ndim):
            scale = scale[...,None]
        self = replace(self, rel_fps=scale*self.rel_fps)
        self = replace(self, z=scale[...,None]*self.z)
        if center is not None:
            self = replace(self, pos=scale.squeeze(-1)*(self.pos-center)+center)
        return self

    def drop_gt_info(self, color=False) -> "LatentObjects":
        self = replace(self, rel_pcd=None)
        if color:
            self = replace(self, color=None)
        return self
    
    def set_z(self, z) -> "LatentObjects":
        self = replace(self, z=z)
        return self

    def set_fps_tf(self, fps_tf) -> "LatentObjects":
        self = replace(self, rel_fps=fps_tf-self.pos.unsqueeze(-2).detach()) # jax.lax.stop_gradient(self.pos[...,None,:])
        return self

    def init_h(self, h, latent_shape) -> "LatentObjects":
        nfps, nf, nz = latent_shape
        z = einops.rearrange(h[...,:nfps*nf*nz], '... (i j k) -> ... i j k', i=nfps, k=nz)
        pos = h[...,-3:]*POS_COEF
        fps_tf = h[...,nfps*nf*nz:-3]*FPS_COEF
        # rel_fps = einops.rearrange(fps_tf, '... (i j) -> ... i j', j=3) - pos[...,None,:]
        # rel_fps = h[...,nfps*nf*nz:-3]*FPS_COEF
        # rel_fps = einops.rearrange(rel_fps, '... (i j) -> ... i j', j=3)
        return self.replace(z=z, pos=pos).set_fps_tf(einops.rearrange(fps_tf, '... (i j) -> ... i j', j=3))
    
    def set_conf(self, conf) -> "LatentObjects":
        if conf.ndim != self.ndim:
            conf = conf.squeeze(-1)
        return replace(self, conf=conf)
    
    def init_conf_zero(self) -> "LatentObjects":
        return replace(self, conf=torch.zeros(self.shape))

    def set_h(self, h) -> "LatentObjects":
        latent_shape = self.nfps, self.nf, self.nz
        return self.init_h(h, latent_shape)

    def apply_pq_z(self, pos, quat, rot_configs=None) -> "LatentObjects":
        '''
        pos quat should have same outer shape
        '''
        if pos.shape[-1] == 7:
            assert rot_configs is None
            rot_configs = quat
            pos, quat = pos[...,:3], pos[...,3:]
        else:
            assert rot_configs is not None
        quat = quat/torch.norm(quat, dim=-1, keepdim=True)
        
        self = replace(self, pos=tutil.pq_action(pos, quat, self.pos)).set_fps_tf(tutil.pq_action(pos[...,None,:], quat[...,None,:], self.fps_tf))
        self = self.rotate_z(quat, rot_configs, rot_rel_fps=False)
        return self

    def rotate_z(self, quat, rot_configs, rot_rel_fps=True) -> "LatentObjects":
        '''
        rotate z while keeping center point
        '''
        if self.z is not None:
            self = replace(self, z=rmutil.apply_rot(self.z, tutil.q2R(quat), rot_configs, feature_axis=-2, expand_R_no=2))
        if rot_rel_fps:
            self = replace(self, rel_fps=tutil.qaction(quat[...,None,:], self.rel_fps))
        return self

    def translate(self, pos) -> "LatentObjects":
        if self.pos is None:
            self = self.init_pos_zero()
        return replace(self, pos=self.pos+pos, rel_fps=self.rel_fps+pos[...,None,:]-pos.unsqueeze(-2).detach()) # jax.lax.stop_gradient(pos[...,None,:])

    def AABB_IoU(self, objB:"LatentObjects"):
        A_min, A_max = self.AABB_fps
        B_min, B_max = objB.AABB_fps
        intersection = torch.minimum(A_max, B_max) - torch.maximum(A_min, B_min)
        return torch.prod(intersection, dim=-1) / (torch.prod(A_max-A_min, dim=-1)+torch.prod(B_max-B_min, dim=-1))

    def reorder_fps(self, idx)->"LatentObjects":
        '''
        idx : (outer_shape, nfps)
        '''
        fps_reorder = torch.take_along_dim(self.rel_fps, idx[...,None], dim=-2)
        z_reorder = torch.take_along_dim(self.z, idx[...,None,None], dim=-3)
        self = replace(self, rel_fps=fps_reorder, z=z_reorder)
        return self

    def merge(self, keepdims=False)->"LatentObjects":
        if self.ndim == 0:
            return self
        rel_fps = einops.rearrange(self.fps_tf, '... i j k -> ... (i j) k')
        z = einops.rearrange(self.z, '... i j k p -> ... (i j) k p')
        # volume weight sum
        weight = self.len**3
        weight = weight.clip(1e-6)
        weight = weight / torch.sum(weight, dim=-1, keepdim=True)
        center = torch.sum(self.pos * weight[...,None], dim=-2)
        res_obj:"LatentObjects" = self.replace(rel_fps=rel_fps-center[...,None,:], z=z, pos=center)
        if keepdims:
            return res_obj.extend_outer_shape(-1)
        else:
            return res_obj

    def reduce_fps(self, reduced_nfps, jkey)->"LatentObjects":
        if reduced_nfps == self.nfps:
            return self
        # sample random idx
        random_idx = torch.arange(self.nfps)
        random_idx = torch.broadcast_to(random_idx, self.outer_shape + (self.nfps,))
        # random_idx = jax.random.permutation(jkey, random_idx, axis=-1)[...,:reduced_nfps, None]
        perm = torch.randperm(random_idx.size(-1), device=random_idx.device)
        random_idx = random_idx.index_select(-1, perm)[..., :reduced_nfps].unsqueeze(-1)

        # random_idx = jax.random.choice(jkey, self.nfps, shape=self.outer_shape + (reduced_nfps, 1), replace=False)
        reduced_rel_fps = torch.take_along_dim(self.rel_fps, random_idx, dim=-2)
        reduced_z = torch.take_along_dim(self.z, random_idx[...,None], dim=-3)
        return self.replace(rel_fps=reduced_rel_fps, z=reduced_z)

    def remove_idx_by_axis(self, idx, axis)->"LatentObjects":
        # jax.delete(x, idx, dim=axis)
        return jax.tree_util.tree_map(lambda x: torch.cat((x.narrow(axis, 0, idx), x.narrow(axis, idx + 1, x.size(axis) - idx - 1)), dim=axis), self)

    def broadcast_outershape(self, outer_shape)->"LatentObjects":
        return self.replace(pos=torch.broadcast_to(self.pos, outer_shape + self.pos.shape[-1:]),
                            rel_fps=torch.broadcast_to(self.rel_fps, outer_shape + self.rel_fps.shape[-2:]),
                            z=torch.broadcast_to(self.z, outer_shape + self.z.shape[-3:]),
                            conf=None if self.conf is None else torch.broadcast_to(self.conf, outer_shape + self.conf.shape[-1:])
                            )
        # return jax.tree_util.tree_map(lambda x: torch.broadcast_to(x, outer_shape + x.shape[len(self.outer_shape):]), self)

    def canonicalize(self):
        pqc = torch.cat([self.pos, tutil.qExp(torch.zeros_like(self.pos))], dim=-1)
        return self.init_pos_zero(), pqc

    def get_valid_oriCORNs(self)->"LatentObjects":
        # valid_obj_mask = torch.all(torch.abs(self.pos) < bound, axis=-1)
        return self[self.obj_valid_mask]

    def deprecate_obj(self, valid_obj_mask)->"LatentObjects":
        if valid_obj_mask.ndim == self.ndim:
            valid_obj_mask = valid_obj_mask[...,None]
        assert valid_obj_mask.shape[-1] == 1
        return replace(self, 
                       pos=torch.where(valid_obj_mask, self.pos, torch.tensor([0,0,10.0], dtype=self.pos.dtype, device=self.pos.device)),
                       rel_fps=torch.where(valid_obj_mask[...,None], self.rel_fps, torch.zeros_like(self.rel_fps)), 
                       z=torch.where(valid_obj_mask[...,None,None], self.z, torch.zeros_like(self.z)),
                       conf=None if self.conf is None else torch.where(valid_obj_mask.squeeze(-1), self.conf, -1e5)
                       )
    
    def get_fps_o3d(self, color=None):
        import open3d as o3d
        if color is None:
            color = np.random.rand(3)
        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.fps_tf.reshape(-1,3))).paint_uniform_color(color)

    def get_fps_sphere_o3d(self, color=None):
        import open3d as o3d
        if color is None:
            color = np.random.rand(3)
        radius = self.mean_fps_dist[...,None].repeat_interleave(self.nfps, dim=-1) # repeat(self.nfps, axis=-1)
        # radius = self.fps_dist
        fps_centers = self.fps_tf.reshape(-1,3)
        radius = radius.reshape(-1)
        sphere_list = []
        for i in range(fps_centers.shape[0]):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius[i])
            sphere.translate(fps_centers[i])
            sphere.compute_vertex_normals()
            sphere.paint_uniform_color(color)
            sphere_list.append(sphere)
        return sphere_list
        

    def sort_by_conf(self)->"LatentObjects":
        assert self.conf is not None
        idx = torch.argsort(self.conf, dim=-1)
        return self.take_along_outer_axis(idx, axis=self.ndim-1)

    @property
    def h(self)->torch.Tensor:
        return torch.cat([einops.rearrange(self.z_flat, '... i j -> ... (i j)'), einops.rearrange(self.fps_tf, '... i j -> ... (i j)')/FPS_COEF, self.pos/POS_COEF], -1)

    @property
    def z_flat(self)->torch.Tensor:
        return einops.rearrange(self.z, '... f d -> ... (f d)')
    
    @property
    def nz(self):
        return self.z.shape[-1]
    
    @property
    def nobj(self):
        return self.z.shape[-4]
    
    @property
    def nf(self):
        return self.z.shape[-2]

    @property
    def outer_shape(self):
        return self.z.shape[:-3]
    
    @property
    def shape(self):
        return self.outer_shape
    
    @property
    def ndim(self):
        return len(self.outer_shape)
    
    @property
    def latent_shape(self):
        return (self.nfps, self.nf, self.nz)
    
    @property
    def obj_valid_mask(self):
        return torch.logical_or(torch.any(torch.abs(self.z) > 1e-5, dim=(-1,-2,-3)), torch.any(torch.abs(self.rel_fps) > 1e-5, dim=(-1,-2)))
        # return torch.any(torch.abs(self.rel_fps) > 1e-5, axis=(-1,-2))
    
    @property
    def AABB_fps(self):
        fps_tf = self.fps_tf
        return torch.min(fps_tf, dim=-2), torch.max(fps_tf, dim=-2)
    
    @property
    def len(self):
        rel_fps_norm = torch.norm(self.rel_fps, dim=-1)
        return torch.mean(torch.sort(rel_fps_norm)[...,-4:], dim=-1) # (nob, )
    
    @property
    def nh(self)->torch.Tensor:
        return self.latent_shape[0] * self.latent_shape[1] * self.latent_shape[2] + 3 + 3 * self.latent_shape[0]

    @property
    def fps_dist(self)->torch.Tensor:
        if self.nfps > 4:
            pairwise_dist = torch.norm(self.rel_fps[...,None,:] - self.rel_fps[...,None,:,:], dim=-1)
            return torch.mean(torch.sort(pairwise_dist, dim=-1)[...,:np.maximum(self.nfps//16, 1)], dim=(-1,))
        else:
            return torch.norm(self.z_flat, dim=-1)/200


    @property
    def mean_fps_dist(self)->torch.Tensor:
        # pairwise_dist = torch.norm(self.rel_fps[...,None,:] - self.rel_fps[...,None,:,:], axis=-1)
        # return torch.mean(torch.sort(pairwise_dist, axis=-1)[...,:4], axis=(-1,-2))
        return torch.mean(self.fps_dist, dim=-1)