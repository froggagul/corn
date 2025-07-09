# import torch
# import os
# import numpy as np
# from dataclasses import dataclass

# from pkm.util.latent_obj_util import LatentObjects
# from pkm.models.rl.net.ev.collision_decoder import DSLRCollisionDecoder



# @dataclass
# class Config:
#     oricorn_path: str = "/input/DGN/meta-v8/oricorn"
#     reduce_k = 128
#     reduce_mode = "closest" # ["closest", "closest_farthest", "all"]
#     embed_mode = "concat" # ["concat", "decoder"]
#     net = DSLRCollisionDecoder.Config()
# cfg = Config()

# device = "cuda"
# with open(os.path.join(cfg.oricorn_path, '..', 'rot_configs.npy'), 'rb') as f:
#     rot_configs = np.load(f, allow_pickle=True).item()

# rot_configs = data = {
#     "type": rot_configs["type"],
#     "Js": [
#         None if j is None else torch.tensor(j, dtype=torch.float32, device=device) for j in rot_configs["Js"]
#     ],
#     "Y_basis": [
#         None if y is None else torch.tensor(y, dtype=torch.float32, device=device) for y in rot_configs["Y_basis"]
#     ],
#     "D_basis": [
#         None if d is None else torch.tensor(d, dtype=torch.float32, device=device) for d in rot_configs["D_basis"]
#     ],
#     "dim_list": rot_configs["dim_list"],
#     "Y_linear_coef": [
#         None if y is None else torch.tensor(y, dtype=torch.float32, device=device) for y in rot_configs["Y_linear_coef"]
#     ],
#     "constant_scale": rot_configs["constant_scale"],
# }
# decoder = DSLRCollisionDecoder(cfg.net, rot_configs)
# ckpt = torch.load(cfg.net.ckpt, map_location=device)
# decoder.load_state_dict(ckpt, strict=True)

# decoder.to(device)
# decoder.eval()


# pos = torch.tensor([0.1, 0.2, 0.3], device=device, dtype=torch.float32)
# z = torch.tensor([[
#     [0.0, 0.1],
#     [0.2, 0.3],
#     [0.4, 0.5],
#     [0.6, 0.7],
#     [0.8, 0.9],
#     [1.0, 1.1],
#     [1.2, 1.3],
#     [1.4, 1.5],
# ], [
#     [1.6, 1.7],
#     [1.8, 1.9],
#     [2.0, 2.1],
#     [2.2, 2.3],
#     [2.4, 2.5],
#     [2.6, 2.7],
#     [2.8, 2.9],
#     [3.0, 3.1],
# ]], device=device, dtype=torch.float32)  # shape: (2, 8, 2)
# rel_fps = torch.tensor([[-0.1, 0.2, 0.3], [0.4, -0.5, 0.6]], device=device, dtype=torch.float32)  # shape: (2, 3)

# objects_a = LatentObjects(
#     pos = pos,
#     rel_fps = rel_fps,
#     z = z,
# )
# objects_b = LatentObjects(
#     pos = pos + 0.1,
#     rel_fps = rel_fps + 0.1,
#     z = z + 0.1,
# )
# idx_a = torch.tensor([0, 1], device=device, dtype=torch.int64)
# idx_b = torch.tensor([0, 1], device=device, dtype=torch.int64)

# # with torch.no_grad():
# #     collision_cost = decoder(objects_a[None], objects_b[None], idx_a[None], idx_b[None])
# #     print("Collision Cost:", collision_cost)
# #     print(collision_cost.shape)

# print(
#     torch.arange(0, 2).expand(32, 2)
# )


import torch

def balanced_kmeans(x: torch.Tensor,
                    k: int = 16,
                    cap: int = 5,
                    n_iter: int = 30):
    """
    Balanced K-means for small data sets.

    Parameters
    ----------
    x      : (N, D) points              (N must equal k * cap)
    k      : number of clusters
    cap    : maximum (== exact) size of every cluster
    n_iter : Lloyd iterations

    Returns
    -------
    labels  : (N,) cluster index 0‥k-1   (every index appears exactly `cap` times)
    centers : (k, D) final centroids
    """
    N, D = x.shape
    assert N == k * cap, "N must equal k × cap"

    # --- 1.  initialise centroids (random sample without replacement)
    perm    = torch.randperm(N, device=x.device)
    centers = x[perm[:k]].clone()                         # (k, D)

    for _ in range(n_iter):
        # --- 2.  compute pairwise distances (N, k)
        dists = torch.cdist(x, centers)                   # Euclidean

        # --- 3.  greedy balanced assignment
        #
        # We flatten the cost matrix, sort all (point, cluster) pairs
        # by distance, and pick the cheapest still-feasible pair until
        #   • every point is assigned once
        #   • each cluster reaches `cap` members.
        #
        labels          = torch.full((N,), -1, device=x.device, dtype=torch.long)
        cluster_counts  = torch.zeros(k,  device=x.device, dtype=torch.long)

        flat_idx        = torch.argsort(dists.reshape(-1))     # ascending
        for idx in flat_idx:
            p  = idx // k     # point index
            c  = idx %  k     # cluster index
            if labels[p] == -1 and cluster_counts[c] < cap:
                labels[p]        = c
                cluster_counts[c] += 1
                # Early exit when all points placed
                if (cluster_counts == cap).all():
                    break

        # --- 4.  recompute centroids
        for j in range(k):
            members = x[labels == j]
            centers[j] = members.mean(dim=0)

    return labels, centers


# -------------------------- demo --------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    N, D, K, CAP = 80, 3, 16, 5
    pts = torch.randn(N, D) * 0.3                        # dummy data
    pts[:20]  += torch.tensor([ 2,  2, 0])
    pts[20:40] += torch.tensor([-2,  2, 0])
    pts[40:60] += torch.tensor([-2, -2, 0])
    pts[60:]   += torch.tensor([ 2, -2, 0])

    labels, centers = balanced_kmeans(pts, k=K, cap=CAP, n_iter=40)
    print(labels)

    # quick sanity check
    unique, counts = torch.unique(labels, return_counts=True)
    print(dict(zip(unique.tolist(), counts.tolist())))
    # -> every cluster id appears exactly 5 times


