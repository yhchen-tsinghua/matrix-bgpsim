#!/usr/bin/env python
from pathlib import Path
import numpy as np

script_dir = Path(__file__).resolve().parent

max_asn_num = 10000 # at most so many ASes are kept in the sampled topology
original = script_dir/"20250101.as-rel2.txt" # original CAIDA AS relationship
C2P, P2P, P2C = 1, 0, -1 # CAIDA norms

# construct original topology
ngbrs = {}
edges = []
for line in open(original, "r"):
    if line.startswith("#"): continue
    a, b, rel = line.strip().split("|")[:3]

    if a not in ngbrs: ngbrs[a] = {C2P: [], P2P: [], P2C: []}
    ngbrs[a][int(rel)].append(b)

    if b not in ngbrs: ngbrs[b] = {C2P: [], P2P: [], P2C: []}
    ngbrs[b][-int(rel)].append(a)

    edges.append((a, b, rel))

# sample a core network starting with Tier-1 ASes
queue = ["174", "209", "286", "701", "1239", "1299", "2828", "2914", "3257", "3320", "3356", "3491", "5511", "6453", "6461", "6762", "6830", "7018", "12956"]

sample_asn = set()
while queue and len(sample_asn) < max_asn_num:
    a = queue.pop(0)
    if a in sample_asn: continue
    sample_asn.add(a)
    for b in ngbrs[a][P2C]:
        if b not in sample_asn:
            queue.append(b)
    for b in ngbrs[a][P2P]:
        if b not in sample_asn:
            queue.append(b)

sample_edges = []
for a, b, rel in edges:
    if a in sample_asn and b in sample_asn:
        sample_edges.append(f"{a}|{b}|{rel}\n")

save_path = script_dir/f"core-{max_asn_num}.{original.name}"
save_path.write_text("".join(sample_edges))

print(f"{save_path.name}: {len(sample_asn)} ASes, {len(sample_edges)} Rels")

# sample a random network (might not interconnected)
np.random.seed(42)
sample_asn = set(np.random.choice(list(ngbrs.keys()), size=max_asn_num, replace=False).tolist())
sample_edges = []
for a, b, rel in edges:
    if a in sample_asn and b in sample_asn:
        sample_edges.append(f"{a}|{b}|{rel}\n")

n_area = 0
while sample_asn:
    area = [sample_asn.pop()]
    n_area += 1
    while area:
        asn = area.pop(0)
        for neighbors in ngbrs[asn].values():
            for neighbor in neighbors:
                if neighbor in sample_asn:
                    sample_asn.remove(neighbor)
                    area.append(neighbor)

save_path = script_dir/f"random-{max_asn_num}.{original.name}"
save_path.write_text("".join(sample_edges))

print(f"{save_path.name}: {max_asn_num} ASes, {len(sample_edges)} Rels, {n_area} isolated areas")
