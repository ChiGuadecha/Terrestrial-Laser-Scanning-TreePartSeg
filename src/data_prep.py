# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# 梁艺萌
# 树干 / 树枝 / 树叶 三维点云分离识别 —— 项目源代码
# 未经本人允许不得商用。third_party/ 内组件许可归其原作者，见 NOTICE.txt。
# ------------------------------------------------------------------
"""Tree data prep: parse .laz -> centered coord + normals + labels.

Classes:  0=leaf, 1=branch, 2=trunk   (label3)
          0=leaf, 1=wood               (label_woodleaf, matches provided annotation)
Trunk/branch split of annotated wood is a geometric pseudo-label:
horizontal slicing + DBSCAN clusters + trunk-axis tracking.
"""
import os, sys, glob, json
import numpy as np
import laspy
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

ROOT = r"E:\Desktop\场景"
OUT  = os.path.join(ROOT, "treeseg", "data")
os.makedirs(OUT, exist_ok=True)

# ---------------- trunk/branch pseudo labeling ----------------

def slice_clusters(pts, z_lo, z_hi, cell=0.05, min_pts=15):
    """Rasterize slice xy to a 5cm grid, connected components -> cluster dicts."""
    from scipy.ndimage import label
    m = (pts[:, 2] >= z_lo) & (pts[:, 2] < z_hi)
    idx = np.nonzero(m)[0]
    if len(idx) < min_pts:
        return []
    xy = pts[idx, :2]
    gx = np.floor((xy[:, 0] - xy[:, 0].min()) / cell).astype(np.int32)
    gy = np.floor((xy[:, 1] - xy[:, 1].min()) / cell).astype(np.int32)
    G = np.zeros((gx.max() + 1, gy.max() + 1), np.int32)
    np.add.at(G, (gx, gy), 1)
    A = G >= 2
    lab, n = label(A, structure=np.ones((3, 3)))
    if n == 0:
        return []
    cellid = lab[gx, gy]
    keep = np.bincount(cellid.ravel(), minlength=n + 1) >= min_pts
    out = []
    for c in range(1, n + 1):
        if not keep[c]:
            continue
        sel = idx[cellid == c]
        p = pts[sel]
        cx, cy = p[:, 0].mean(), p[:, 1].mean()
        d = np.hypot(p[:, 0] - cx, p[:, 1] - cy)
        r = float(np.percentile(d, 90))
        out.append(dict(ids=sel, cx=float(cx), cy=float(cy), r=r,
                        n=int(len(sel)), zc=float(p[:, 2].mean())))
    return out

def pseudo_trunk(wood, ground_z, slice_h=0.1, max_seeds=4, verbose=False, return_axes=False):
    """Return bool mask of trunk points in `wood` (centered coords, z from ground)."""
    z = wood[:, 2]
    z_top = z.max()
    h = z_top - ground_z
    if h < 0.5 or len(wood) < 800:
        return (np.zeros(len(wood), bool), []) if return_axes else np.zeros(len(wood), bool)
    zs = np.arange(ground_z + 1e-4, z_top + slice_h, slice_h)
    slices = [slice_clusters(wood, lo, lo + slice_h) for lo in zs]
    # seeds: compact, dense clusters in the lower 30%, near the tree axis
    base_xy = np.median(wood[z < ground_z + 0.15 * h][:, :2], axis=0) if (z < ground_z + 0.15 * h).any() else np.median(wood[:, :2], 0)
    seeds = []
    for si in range(len(slices)):
        if zs[si] - ground_z > 0.3 * h:
            break
        for cl in slices[si]:
            if cl["r"] > 0.45 or cl["n"] < 60:
                continue
            if np.hypot(cl["cx"] - base_xy[0], cl["cy"] - base_xy[1]) > 0.6:
                continue
            up = slices[si + 1] if si + 1 < len(slices) else []
            ok = any(np.hypot(c2["cx"] - cl["cx"], c2["cy"] - cl["cy"]) < 0.35
                     and c2["n"] >= 30 for c2 in up)
            if ok:
                seeds.append((si, cl))
    # prefer the LOWEST valid seed (true trunk base), then dedupe by xy
    seeds.sort(key=lambda t: (t[0], -t[1]["n"]))
    keep = []
    for si, cl in seeds:
        if any(np.hypot(cl["cx"] - k["cx"], cl["cy"] - k["cy"]) < 0.5 for k in keep):
            continue
        keep.append(cl)
        if len(keep) == max_seeds:
            break
    if not keep:
        if verbose: print("   no trunk seed")
        return (np.zeros(len(wood), bool), []) if return_axes else np.zeros(len(wood), bool)
    # track one axis per seed: cylinder sampling near the predicted axis,
    # robust to trunk/branch fusion in the crown; leader goes to the top.
    z_all = wood[:, 2]
    axes = []
    for cl0 in keep:
        si = next((x for x in range(len(slices)) if any(c is cl0 for c in slices[x])), None)
        if si is None:
            continue
        a = dict(x=cl0["cx"], y=cl0["cy"], r=max(cl0["r"], 0.05), drift=(0.0, 0.0))
        prof = [(zs[si], a["x"], a["y"], a["r"])]
        miss = 0
        for s2 in range(si + 1, len(slices)):
            zlo = zs[s2]
            pred = (a["x"] + a["drift"][0] * slice_h, a["y"] + a["drift"][1] * slice_h)
            band = (z_all >= zlo) & (z_all < zlo + slice_h)
            wi = np.nonzero(band)[0]
            if len(wi) == 0:
                miss += 1
                if miss >= 3:
                    break
                continue
            d = np.hypot(wood[wi, 0] - pred[0], wood[wi, 1] - pred[1])
            near = wi[d < max(0.16, 1.6 * a["r"])]
            if len(near) < 15:
                miss += 1
                if miss >= 3:
                    break
                continue
            miss = 0
            nx, ny = wood[near, 0].mean(), wood[near, 1].mean()
            step = np.hypot(nx - a["x"], ny - a["y"])
            if step > 0.14:      # leader is near-vertical; refuse big jumps
                break
            dd = np.hypot(wood[near, 0] - nx, wood[near, 1] - ny)
            nr = float(np.percentile(dd, 90))
            a["drift"] = (0.55 * a["drift"][0] + 0.45 * (nx - a["x"]),
                          0.55 * a["drift"][1] + 0.45 * (ny - a["y"]))
            a["x"] = 0.6 * a["x"] + 0.4 * nx
            a["y"] = 0.6 * a["y"] + 0.4 * ny
            a["r"] = 0.8 * a["r"] + 0.2 * max(min(nr, 0.22), 0.02)
            prof.append((zlo, a["x"], a["y"], a["r"]))
        if len(prof) >= 3:
            axes.append(prof)
    if not axes:
        if verbose: print("   tracking failed")
        return (np.zeros(len(wood), bool), []) if return_axes else np.zeros(len(wood), bool)
    trunk = np.zeros(len(wood), bool)
    for prof in axes:
        p = np.array(prof)            # columns: z, x, y, r
        zz = p[:, 0]
        z0 = ground_z
        sel = (wood[:, 2] >= z0 - 0.05) & (wood[:, 2] <= zz.max() + slice_h)
        wi = np.nonzero(sel)[0]
        if not len(wi):
            continue
        zq = np.clip(wood[wi, 2], zz.min(), zz.max())
        ax = np.interp(zq, zz, p[:, 1]); ay = np.interp(zq, zz, p[:, 2]); ar = np.interp(zq, zz, p[:, 3])
        d = np.hypot(wood[wi, 0] - ax, wood[wi, 1] - ay)
        trunk[wi[d < np.minimum(ar, 0.20) * 1.15 + 0.02]] = True
    if verbose:
        print(f"   axes={len(axes)} top={max(a[-1][0] for a in axes):.2f}m trunk={trunk.mean():.2f}")
    return (trunk, axes) if return_axes else trunk

# ---------------- features ----------------

def compute_normals(coord, k=14):
    tree = cKDTree(coord)
    dist, ii = tree.query(coord, k=k, workers=1)
    nb = coord[ii]                                  # (N,k,3)
    nb = nb - nb.mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb) / k
    w, v = np.linalg.eigh(cov.astype(np.float64))
    nrm = v[:, :, 0]
    return nrm.astype(np.float32), w.astype(np.float32), dist[:, 1:].mean(1).astype(np.float32)

def curvature_feats(w, rad):
    # w: ascending eigenvalues (l1<=l2<=l3)
    l1, l2, l3 = w[:, 0], w[:, 1], np.maximum(w[:, 2], 1e-12)
    lin = (l3 - l2) / l3
    pla = (l2 - l1) / l3
    sph = l1 / l3
    return np.stack([lin, pla, sph, np.clip(rad, 0, 0.2)], 1).astype(np.float32)

def orient_outward(coord, nrm):
    c = np.median(coord, axis=0)
    v = coord - c
    s = np.sign(np.einsum("ij,ij->i", v, nrm))
    s[s == 0] = 1
    return (nrm * s[:, None]).astype(np.float32)

# ---------------- main ----------------

def clean_outliers(pts):
    # remove extreme spikes: keep z in [p0.0 - 0.5, p99.9 + 1.0]
    z = pts[:, 2]
    lo, hi = np.percentile(z, 0.01), np.percentile(z, 99.99)
    m = (z >= lo - 0.5) & (z <= hi + 1.0)
    return pts[m]

def process(tid, leaf_pts, wood_pts, path):
    leaf_pts = clean_outliers(leaf_pts) if leaf_pts is not None and len(leaf_pts) else leaf_pts
    wood_pts = clean_outliers(wood_pts)
    if leaf_pts is None:
        pts = wood_pts; lw = np.ones(len(pts), bool)
    else:
        pts = np.concatenate([leaf_pts, wood_pts], 0)
        lw = np.concatenate([np.zeros(len(leaf_pts), bool), np.ones(len(wood_pts), bool)])
    shift = np.array([np.median(pts[:, 0]), np.median(pts[:, 1]), pts[:, 2].min()], np.float32)
    cen = (pts - shift).astype(np.float32)
    nrm_raw, eig, knnd = compute_normals(cen)
    nrm = orient_outward(cen, nrm_raw)
    fx = curvature_feats(eig, knnd)
    lw_mask = lw
    if lw_mask.any():
        trunk = pseudo_trunk(cen[lw_mask], 0.0)
    else:
        trunk = np.zeros(0, bool)
    lab3 = np.where(lw_mask, 1, 0).astype(np.uint8)          # leaf=0 branch=1
    if lw_mask.any():
        lab3[np.nonzero(lw_mask)[0][trunk]] = 2              # trunk=2
    lab2 = lw_mask.astype(np.uint8)
    np.savez_compressed(path, coord=cen, normal=nrm, featx=fx, label3=lab3,
                        label_woodleaf=lab2, shift=shift)
    return dict(n=len(pts), wood=float(lw_mask.mean()),
                trunk=float((lab3 == 2).mean()) if lw_mask.any() else 0.0,
                h=float(cen[:, 2].max()))

def main():
    tr_ids = sorted({int(os.path.basename(f).split("_")[0])
                     for f in glob.glob(os.path.join(ROOT, "训练集", "*_leaf.laz"))}, key=str)
    stats = {}
    for tid in tr_ids:
        lf = laspy.read(os.path.join(ROOT, "训练集", f"{tid}_leaf.laz"))
        wf = laspy.read(os.path.join(ROOT, "训练集", f"{tid}_wood.laz"))
        L = np.asarray(lf.xyz, np.float32)
        W = np.asarray(wf.xyz, np.float32)
        st = process(tid, L, W, os.path.join(OUT, f"train_{tid}.npz"))
        stats[tid] = st; print(tid, st, flush=True)
    te_ids = sorted({int(os.path.splitext(os.path.basename(f))[0])
                     for f in glob.glob(os.path.join(ROOT, "测试集", "*.laz"))}, key=str)
    for tid in te_ids:
        ff = laspy.read(os.path.join(ROOT, "测试集", f"{tid}.laz"))
        F = np.asarray(ff.xyz, np.float32)
        st = process(tid, F[:0], F, os.path.join(OUT, f"test_{tid}.npz"))
        stats["t" + str(tid)] = st; print("test", tid, st, flush=True)
    with open(os.path.join(OUT, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)

if __name__ == "__main__":
    main()
