'''
author: Dongwon Son
2023-10-17
'''
import numpy as np
import einops
import torch

def rand_sphere(outer_shape):
    ext = np.random.normal(size=outer_shape + (5,))
    return (ext / np.linalg.norm(ext, axis=-1, keepdims=True))[...,-3:]


def safe_norm(x, axis, keepdims=False, eps=0.0):
    is_zero = torch.all(torch.isclose(x, torch.zeros_like(x)), dim=axis, keepdim=True)
    # temporarily swap x with ones if is_zero, then swap back
    x = torch.where(is_zero, torch.ones_like(x), x)
    n = torch.norm(x, dim=axis, keepdim=keepdims)
    n = torch.where(is_zero if keepdims else torch.squeeze(is_zero, -1), torch.zeros_like(n), n)
    return n.clip(eps)

# quaternion operations
def normalize(vec, eps=1e-8):
    # return vec/(safe_norm(vec, axis=-1, keepdims=True, eps=eps) + 1e-8)
    return vec/safe_norm(vec, axis=-1, keepdims=True, eps=eps)

def quw2wu(quw):
    return torch.cat([quw[...,-1:], quw[...,:3]], dim=-1)

def qrand(outer_shape, jkey=None):
    if jkey is None:
        return qrand_np(outer_shape)
    else:
        return normalize(torch.randn(outer_shape + (4,)))

def qrand_np(outer_shape):
    q = np.random.normal(size=outer_shape+(4,))
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return q

def line2q(zaxis, yaxis=np.array([1,0,0])):
    Rm = line2Rm(zaxis, yaxis)
    return Rm2q(Rm)

def qmulti(q1, q2):
    b,c,d,a = torch.split(q1, (1, 1, 1, 1), dim=-1)
    f,g,h,e = torch.split(q2, (1, 1, 1, 1), dim=-1)
    w,x,y,z = a*e-b*f-c*g-d*h, a*f+b*e+c*h-d*g, a*g-b*h+c*e+d*f, a*h+b*g-c*f+d*e
    return torch.cat([x,y,z,w], dim=-1)

def qmulti_np(q1, q2):
    b,c,d,a = np.split(q1, 4, axis=-1)
    f,g,h,e = np.split(q2, 4, axis=-1)
    w,x,y,z = a*e-b*f-c*g-d*h, a*f+b*e+c*h-d*g, a*g-b*h+c*e+d*f, a*h+b*g-c*f+d*e
    return np.concatenate([x,y,z,w], axis=-1)

def qinv(q):
    x,y,z,w = torch.split(q, (1,1,1,1), dim=-1)
    return torch.cat([-x,-y,-z,w], dim=-1)

def qinv_np(q):
    x,y,z,w = np.split(q, 4, axis=-1)
    return np.concatenate([-x,-y,-z,w], axis=-1)

def q2aa(q):
    return 2*qlog(q)[...,:3]

def qlog(q):
    # Clamp to avoid domain errors in arccos due to floating-point inaccuracies
    q_w = torch.clip(q[..., 3:], -1 + 1e-7, 1 - 1e-7)
    
    # Compute alpha with clamped w-component
    alpha = torch.arccos(q_w)
    sinalpha = torch.sin(alpha)
    
    # Ensure stable division by using a safe minimum threshold for sinalpha
    safe_sinalpha = torch.where(torch.abs(sinalpha) < 1e-6, 1e-6, sinalpha)
    n = q[..., :3] / (safe_sinalpha * torch.sign(sinalpha))
    
    # Use a threshold to check for small values of alpha
    res = torch.where(torch.abs(q_w) < 1 - 1e-6, n * alpha, torch.zeros_like(n))
    
    # Concatenate result with an additional zero for the w-component
    return torch.cat([res, torch.zeros_like(res[..., :1])], dim=-1)

def qLog(q):
    return qvee(qlog(q))

def qvee(phi):
    return 2*phi[...,:-1]

def qhat(w):
    return torch.cat([w*0.5, torch.zeros_like(w[...,0:1])], dim=-1)

def aa2q(aa):
    return qexp(aa*0.5)

def q2R(q):
    i,j,k,r = torch.split(q, (1,1,1,1), dim=-1)
    R1 = torch.cat([1-2*(j**2+k**2), 2*(i*j-k*r), 2*(i*k+j*r)], dim=-1)
    R2 = torch.cat([2*(i*j+k*r), 1-2*(i**2+k**2), 2*(j*k-i*r)], dim=-1)
    R3 = torch.cat([2*(i*k-j*r), 2*(j*k+i*r), 1-2*(i**2+j**2)], dim=-1)
    return torch.stack([R1,R2,R3], dim=-2)

def qexp(logq):
    if isinstance(logq, np.ndarray):
        alpha = np.linalg.norm(logq[...,:3], axis=-1, keepdims=True)
        alpha = np.maximum(alpha, 1e-6)
        return np.concatenate([logq[...,:3]/alpha*np.sin(alpha), np.cos(alpha)], axis=-1)
    else:
        alpha = safe_norm(logq[...,:3], axis=-1, keepdims=True)
        alpha = torch.clip(alpha, min=1e-6)
        return torch.cat([logq[...,:3]/alpha*torch.sin(alpha), torch.cos(alpha)], dim=-1)

def pq_quatnormalize(pqc):
    return torch.cat([pqc[...,:3], normalize(pqc[...,3:])], dim=-1)

def qExp(w):
    return qexp(qhat(w))

def qaction(quat, pos):
    return qmulti(qmulti(quat, torch.cat([pos, torch.zeros_like(pos[...,:1])], dim=-1)), qinv(quat))[...,:3]

def qaction_np(quat, pos):
    return qmulti_np(qmulti_np(quat, np.concatenate([pos, np.zeros_like(pos[...,:1])], axis=-1)), qinv_np(quat))[...,:3]

def qnoise(quat, scale=np.pi*10/180):
    lq = np.random.normal(scale=scale, size=quat[...,:3].shape)
    return qmulti(quat, qexp(lq))

def qzero(outer_shape):
    return torch.cat([torch.zeros(outer_shape + (3,)), torch.ones(outer_shape + (1,))], dim=-1)

# posquat operations
def pq_inv(pos, quat=None):
    is_pqc = False
    if pos.shape[-1] == 7:
        is_pqc = True
        assert quat is None
        quat = pos[...,3:]
        pos = pos[...,:3]
    quat_inv = qinv(quat)
    if is_pqc:
        return torch.cat([-qaction(quat_inv, pos), quat_inv], dim=-1)
    else:
        return -qaction(quat_inv, pos), quat_inv

def pq_action(translate, rotate, pnt=None):
    if translate.shape[-1] == 7:
        assert pnt is None
        assert rotate.shape[-1] == 3
        pnt = rotate
        pos = translate[...,:3]
        quat = translate[...,3:]
        return qaction(quat, pnt) + pos
    return qaction(rotate, pnt) + translate

def pq_multi(pos1, quat1, pos2=None, quat2=None):
    if pos1.shape[-1] == 7:
        assert quat1.shape[-1] == 7
        assert pos2 is None
        assert quat2 is None
        pos2 = quat1[...,:3]
        quat2 = quat1[...,3:]
        quat1 = pos1[...,3:]
        pos1 = pos1[...,:3]
        return torch.cat([qaction(quat1, pos2)+pos1, qmulti(quat1, quat2)], dim=-1)
    else:
        assert pos2 is not None
        assert quat2 is not None
        return qaction(quat1, pos2)+pos1, qmulti(quat1, quat2)

def pqc_Exp(twist):
    return torch.cat([twist[...,:3], qExp(twist[...,3:])], dim=-1)

def pqc_Log(pqc):
    return torch.cat([pqc[...,:3], qLog(pqc[...,3:])], dim=-1)

def pqc_minus(pqc1, pqc2):
    '''
    pqc1 - pqc2
    '''
    if pqc1.shape[-1] != 7:
        # only position
        return pqc1 - pqc2
    pqc_exp = pq_multi(pq_inv(pqc2), pqc1)
    pqc_exp = pq_quatnormalize(pqc_exp)
    return pqc_Log(pqc_exp)


def pq2H(pos, quat=None):
    if pos.shape[-1] == 7:
        assert quat is None
        quat = pos[...,-4:]
        pos = pos[...,:3]
    else:
        assert quat is not None

    R = q2R(quat)
    return H_from_Rpos(R, pos)

# homogineous transforms
def H_from_Rpos(R, pos):
    H = torch.zeros(pos.shape[:-1] + (4,4))
    H = H.at[...,-1,-1].set(1)
    H = H.at[...,:3,:3].set(R)
    H = H.at[...,:3,3].set(pos)
    return H

def H_inv(H):
    R = H[...,:3,:3]
    p = H[...,:3, 3:]
    return H_from_Rpos(T(R), (-T(R)@p)[...,0])

def H2pq(H, concat=False):
    R = H[...,:3,:3]
    p = H[...,:3, 3]
    if concat:
        return torch.cat([p, Rm2q(R)], dim=-1)
    else:
        return p, Rm2q(R)

# Rm util
def Rm_inv(Rm):
    return T(Rm)

def line2Rm(zaxis, yaxis=np.array([1,0,0])):
    zaxis = normalize(zaxis + torch.tensor([0,1e-6,0], dtype=zaxis.dtype, device=zaxis.device))
    xaxis = torch.linalg.cross(yaxis, zaxis)
    xaxis = normalize(xaxis)
    yaxis = torch.linalg.cross(zaxis, xaxis)
    Rm = torch.stack([xaxis, yaxis, zaxis], dim=-1)
    return Rm

def line2Rm_np(zaxis, yaxis=np.array([1,0,0])):
    zaxis = (zaxis + torch.tensor([0,1e-6,0], dtype=zaxis.dtype, device=zaxis.device))
    zaxis = zaxis/np.linalg.norm(zaxis, axis=-1, keepdims=True)
    xaxis = np.cross(yaxis, zaxis)
    xaxis = xaxis/np.linalg.norm(xaxis, axis=-1, keepdims=True)
    yaxis = np.cross(zaxis, xaxis)
    Rm = np.stack([xaxis, yaxis, zaxis], axis=-1)
    return Rm

def Rm2q(Rm):
    Rm = einops.rearrange(Rm, '... i j -> ... j i')
    con1 = (Rm[...,2,2] < 0) & (Rm[...,0,0] > Rm[...,1,1])
    con2 = (Rm[...,2,2] < 0) & (Rm[...,0,0] <= Rm[...,1,1])
    con3 = (Rm[...,2,2] >= 0) & (Rm[...,0,0] < -Rm[...,1,1])
    con4 = (Rm[...,2,2] >= 0) & (Rm[...,0,0] >= -Rm[...,1,1]) 

    t1 = 1 + Rm[...,0,0] - Rm[...,1,1] - Rm[...,2,2]
    t2 = 1 - Rm[...,0,0] + Rm[...,1,1] - Rm[...,2,2]
    t3 = 1 - Rm[...,0,0] - Rm[...,1,1] + Rm[...,2,2]
    t4 = 1 + Rm[...,0,0] + Rm[...,1,1] + Rm[...,2,2]

    q1 = torch.stack([t1, Rm[...,0,1]+Rm[...,1,0], Rm[...,2,0]+Rm[...,0,2], Rm[...,1,2]-Rm[...,2,1]], dim=-1) / torch.sqrt(t1.clip(1e-7))[...,None]
    q2 = torch.stack([Rm[...,0,1]+Rm[...,1,0], t2, Rm[...,1,2]+Rm[...,2,1], Rm[...,2,0]-Rm[...,0,2]], dim=-1) / torch.sqrt(t2.clip(1e-7))[...,None]
    q3 = torch.stack([Rm[...,2,0]+Rm[...,0,2], Rm[...,1,2]+Rm[...,2,1], t3, Rm[...,0,1]-Rm[...,1,0]], dim=-1) / torch.sqrt(t3.clip(1e-7))[...,None]
    q4 = torch.stack([Rm[...,1,2]-Rm[...,2,1], Rm[...,2,0]-Rm[...,0,2], Rm[...,0,1]-Rm[...,1,0], t4], dim=-1) / torch.sqrt(t4.clip(1e-7))[...,None]
 
    q = torch.zeros(Rm.shape[:-2]+(4,))
    q = torch.where(con1[...,None], q1, q)
    q = torch.where(con2[...,None], q2, q)
    q = torch.where(con3[...,None], q3, q)
    q = torch.where(con4[...,None], q4, q)
    q *= 0.5

    return q

def pRm_inv(pos, Rm):
    return (-T(Rm)@pos[...,None,:])[...,0], T(Rm)

def pRm_action(pos, Rm, x):
    return (Rm @ x[...,None,:])[...,0] + pos

def se3_rot(se3, quat):
    if se3.shape[-1] == 3:
        # only position
        return se3
    return torch.cat([qaction(quat, se3[...,:3]), qaction(quat, se3[...,3:])], dim=-1)


# 6d utils
def R6d2Rm(x, gram_schmidt=False):
    xv, yv = x[...,:3], x[...,3:]
    xv = normalize(xv)
    if gram_schmidt:
        yv = normalize(yv - torch.einsum('...i,...i',yv,xv)[...,None]*xv)
        zv = torch.cross(xv, yv)
    else:
        zv = torch.cross(xv, yv)
        zv = normalize(zv)
        yv = torch.cross(zv, xv)
    return torch.stack([xv,yv,zv], -1)

# 9d utils
def R9d2Rm(x):
    xm = einops.rearrange(x, '... (t i) -> ... t i', t=3)
    u, s, vt = torch.svd(xm)
    # vt = einops.rearrange(v, '... i j -> ... j i')
    det = torch.det(torch.matmul(u,vt))
    vtn = torch.cat([vt[...,:2,:], vt[...,2:,:]*det[...,None,None]], dim=-2)
    return torch.matmul(u,vtn)


# general
def T(mat):
    return einops.rearrange(mat, '... i j -> ... j i')

def pq2SE2h(pos, quat=None):
    if pos.shape[-1] == 7:
        assert quat is None
        quat = pos[...,-4:]
        pos = pos[...,:3]
    z_angle = q2aa(quat)[...,2]
    SE2 = torch.cat([pos[...,:2], z_angle[...,None]], dim=-1)
    height = pos[...,2]
    return SE2, height

def SE2h2pq(SE2, height, concat=False):
    height = torch.tensor(height)
    pos = torch.cat([SE2[...,:2], height[...,None]], dim=-1)
    quat = aa2q(torch.cat([torch.zeros_like(SE2[...,:2]), SE2[...,2:]], dim=-1))
    if concat:
        return torch.cat([pos, quat], dim=-1)
    else:
        return pos, quat

# euler angle
def Rm2ZYZeuler(Rm):
    sy = torch.sqrt(Rm[...,0,2]**2+Rm[...,1,2]**2)
    v1 = torch.arctan2(Rm[...,1,2], Rm[...,0,2])
    v2 = torch.arctan2(sy, Rm[...,2,2])
    v3 = torch.arctan2(Rm[...,2,1], -Rm[...,2,0])

    v1n = torch.arctan2(-Rm[...,0,1], Rm[...,1,1])
    v1 = torch.where(sy < 1e-6, v1n, v1)
    v3 = torch.where(sy < 1e-6, torch.zeros_like(v1), v3)

    return torch.stack([v1,v2,v3],-1)

def Rm2YXYeuler(Rm):
    sy = torch.sqrt(torch.sqrt(Rm[...,0,1]**2+Rm[...,2,1]**2))
    v1 = torch.arctan2(Rm[...,0,1], Rm[...,2,1])
    v2 = torch.arctan2(sy, Rm[...,1,1])
    v3 = torch.arctan2(Rm[...,1,0], -Rm[...,1,2])

    v1n = torch.arctan2(-Rm[...,2,0], Rm[...,0,0])
    v1 = torch.where(sy < 1e-6, v1n, v1)
    v3 = torch.where(sy < 1e-6, torch.zeros_like(v1), v3)

    return torch.stack([v1,v2,v3],-1)

def YXYeuler2Rm(YXYeuler):
    c1,c2,c3 = torch.split(torch.cos(YXYeuler), (3,3,3), -1)
    s1,s2,s3 = torch.split(torch.sin(YXYeuler), (3,3,3), -1)
    return torch.stack([torch.cat([c1*c3-c2*s1*s3, s1*s2, c1*s3+c2*c3*s1],-1),
            torch.cat([s2*s3, c2, -c3*s2],-1),
            torch.cat([-c3*s1-c1*c2*s3, c1*s2, c1*c2*c3-s1*s3],-1)], -2)

def wigner_D_order1_from_Rm(Rm):
    r1,r2,r3 = torch.split(Rm,(3, 3, 3),-2)
    r11,r12,r13 = torch.split(r1,(3, 3, 3),-1)
    r21,r22,r23 = torch.split(r2,(3, 3, 3),-1)
    r31,r32,r33 = torch.split(r3,(3, 3, 3),-1)

    return torch.cat([torch.c_[r22, r23, r21],
                torch.c_[r32, r33, r31],
                torch.c_[r12, r13, r11]], axis=-2)

def q2ZYZeuler(q):
    return Rm2ZYZeuler(q2R(q))

def q2XYZeuler(q):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x, y, z, w = torch.split(q, (1,1,1,1), -1)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.arctan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = torch.arcsin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.arctan2(t3, t4)
    
    return torch.cat([roll_x, pitch_y, yaw_z], -1) # in radians

def XYZeuler2q(euler):
    """
    Convert euler angles (roll, pitch, yaw) to quaternion
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    roll_x, pitch_y, yaw_z = torch.split(euler, (1, 1, 1), -1)
    cy = torch.cos(yaw_z * 0.5)
    sy = torch.sin(yaw_z * 0.5)
    cp = torch.cos(pitch_y * 0.5)
    sp = torch.sin(pitch_y * 0.5)
    cr = torch.cos(roll_x * 0.5)
    sr = torch.sin(roll_x * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return torch.cat([x, y, z, w], -1)
