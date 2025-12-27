import torch

from src.leo_pg.data.io import load_bundle

path = "data/starlink_like.pt"   # 改成你实际生成的文件

obj = load_bundle(path)

steps = obj["steps"] if isinstance(obj, dict) and "steps" in obj else obj

print("num_steps:", len(steps))

s0 = steps[0]

for k in ["node_x","edge_index","edge_z","y","meta"]:

    if k in s0:

        v = s0[k]

        if hasattr(v, "shape"): print(k, tuple(v.shape))

        else: print(k, v)

# 数值检查

import math

def check_tensor(x, name):

    if not torch.is_tensor(x): return

    nan = torch.isnan(x).any().item()

    inf = torch.isinf(x).any().item()

    print(name, "nan", nan, "inf", inf, "min", float(x.min()), "max", float(x.max()))

check_tensor(s0["node_x"], "node_x")

check_tensor(s0["edge_z"], "edge_z")

check_tensor(s0["y"], "y")

# 边数分布（前 20 步）

edges = [int(st["edge_index"].shape[1]) for st in steps[:20]]

print("edge_count first20:", edges)

print("edge_count mean:", sum(edges)/len(edges))

