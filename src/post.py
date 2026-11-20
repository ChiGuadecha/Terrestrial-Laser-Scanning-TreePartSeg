# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# 梁艺萌
# 树干 / 树枝 / 树叶 三维点云分离识别 —— 项目源代码
# 未经本人允许不得商用。third_party/ 内组件许可归其原作者，见 NOTICE.txt。
# ------------------------------------------------------------------
"""Post-processing: KNN voting, connected-component twig rescue, island cleanup."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

def knn_vote(coord, lab, ncls, k=10, sigma=0.03, iters=2, keep_w=0.6):
    tree = cKDTree(coord)
    d, ii = tree.query(coord, k=k + 1, workers=1)
    d = d[:, 1:]; ii = ii[:, 1:]
    wt = np.exp(-(d ** 2) / (2 * sigma ** 2)).astype(np.float32)
    lab = lab.astype(np.int64).copy()
    N = len(coord)
    ar = np.arange(N, dtype=np.int64)
    for _ in range(iters):
        nb = lab[ii]
        flat = nb + (ar[:, None] * ncls)
        votes = np.bincount(flat.ravel(), weights=wt.ravel(), minlength=N * ncls).reshape(N, ncls)
        votes[ar, lab] += keep_w * wt.sum(1)
        new = votes.argmax(1)
        if (new == lab).all():
            break
        lab = new
    return lab

def _components(coord, nodes, eps):
    tree = cKDTree(coord[nodes])
    pairs = tree.query_pairs(r=eps, output_type="ndarray")
    n = len(nodes)
    if len(pairs) == 0:
        return np.arange(n), n
    A = coo_matrix((np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    nc, comp = connected_components(A, directed=False)
    return comp, nc

def twig_rescue(coord, lab, fx, leaf=0, branch=1, trunk=2, lin_th=0.5, rad_th=0.025,
                eps=0.06, min_frac=0.45, min_size=25):
    cand = (lab == leaf) & (fx[:, 0] > lin_th) & (fx[:, 3] < rad_th)
    if not cand.any():
        return lab
    nodes = np.nonzero(cand | (lab == branch))[0]
    comp, nc = _components(coord, nodes, eps)
    br_cnt = np.bincount(comp[(lab[nodes] == branch)], minlength=nc)
    tot = np.bincount(comp, minlength=nc)
    good = (br_cnt >= min_frac * tot) & (tot >= min_size)
    lab_new = lab.copy()
    sel = good[comp] & cand[nodes]
    lab_new[nodes[sel]] = branch
    return lab_new

def remove_islands(coord, lab, leaf=0, branch=1, trunk=2, eps=0.06, min_size=12):
    nodes = np.nonzero(lab != leaf)[0]
    if len(nodes) < min_size:
        return lab
    comp, nc = _components(coord, nodes, eps)
    tot = np.bincount(comp, minlength=nc)
    has_trunk = np.bincount(comp[(lab[nodes] == trunk)], minlength=nc) > 0
    bad = (tot < min_size) & (~has_trunk)
    lab_new = lab.copy()
    lab_new[nodes[bad[comp]]] = leaf
    return lab_new

def tip_growth(coord, lab, fx, leaf=0, branch=1, lin_th=0.5, rad_th=0.02,
               k=8, hops=12, sigma=0.02):
    """Grow branch labels outward along ridge (thin, linear) points from confirmed branches."""
    ridge = (fx[:, 0] > lin_th) & (fx[:, 3] < rad_th)
    if not ridge.any():
        return lab
    tree = cKDTree(coord)
    _, ii = tree.query(coord, k=k + 1, workers=1)
    ii = ii[:, 1:]
    lab = lab.copy()
    cur = (lab == branch) & ridge | ((lab == branch) & ~ridge)
    grown = np.zeros(len(lab), bool)
    frontier = (lab == branch)
    for _ in range(hops):
        nb = frontier[ii]
        spread = nb.any(1) & ridge & ~frontier
        if not spread.any():
            break
        frontier = frontier | spread
        grown = grown | spread
    lab[grown] = branch
    return lab

def postprocess(coord, lab, fx, ncls, trunk_branch=True):
    out = knn_vote(coord, lab, ncls)
    if trunk_branch and ncls == 3:
        out = twig_rescue(coord, out, fx, lin_th=0.45, rad_th=0.03)
        out = twig_rescue(coord, out, fx, lin_th=0.55, rad_th=0.02, eps=0.05, min_frac=0.35, min_size=15)
        out = remove_islands(coord, out)
        out = tip_growth(coord, out, fx)
        out = knn_vote(coord, out, ncls, k=8, sigma=0.02, iters=1, keep_w=1.0)
    return out


def twig_rescue2(coord, lab, fx, prob=None, leaf=0, branch=1, lin_th=0.5, rad_th=0.022,
                 eps=0.035, min_size=10, max_size=8000, attach_frac=0.35, attach_r=0.05,
                 elong=2.2, p_floor=0.10):
    """Local connected-component twig rescue: components are built from candidate points only,
    then kept when they are small, elongated, thin, and physically attached to existing wood."""
    cand = (lab == leaf) & (fx[:, 0] > lin_th) & (fx[:, 3] < rad_th)
    if prob is not None:
        cand &= np.maximum(prob[:, branch], prob[:, 2]) > p_floor
    nodes = np.nonzero(cand)[0]
    if len(nodes) < min_size:
        return lab
    comp, nc = _components(coord, nodes, eps)
    pts = coord[nodes]
    size = np.bincount(comp, minlength=nc).astype(np.float64)
    wood = np.nonzero(lab > leaf)[0]
    if len(wood) == 0:
        return lab
    dd, _ = cKDTree(coord[wood]).query(pts, k=1, workers=1)
    nearfrac = np.bincount(comp, weights=(dd < attach_r).astype(np.float64), minlength=nc) / np.maximum(size, 1)
    linm = np.bincount(comp, weights=fx[nodes, 0].astype(np.float64), minlength=nc) / np.maximum(size, 1)
    radm = np.bincount(comp, weights=fx[nodes, 3].astype(np.float64), minlength=nc) / np.maximum(size, 1)
    mu = np.zeros((nc, 3)); np.add.at(mu, comp, pts)
    mu /= np.maximum(size, 1)[:, None]
    outer = pts[:, :, None] * pts[:, None, :]
    ss = np.zeros((nc, 3, 3)); np.add.at(ss, comp, outer)
    cov = ss / np.maximum(size, 1)[:, None, None] - mu[:, :, None] * mu[:, None, :]
    w, _ = np.linalg.eigh(cov)
    lam = np.maximum(w[:, ::-1], 1e-12)
    ratio = lam[:, 0] / np.maximum(lam[:, 1], 1e-12)
    good = ((size >= min_size) & (size <= max_size) & (nearfrac >= attach_frac) &
            (ratio >= elong) & (radm < rad_th * 1.5) & (linm > lin_th * 0.85))
    out = lab.copy()
    out[nodes[good[comp]]] = branch
    return out


def tip_growth2(coord, lab, fx, prob=None, leaf=0, branch=1, lin_th=0.55, rad_th=0.02,
                k=8, eps=0.035, max_len=0.30, dir_cos=0.55, p_floor=0.08, hops=40):
    """Grow the branch skeleton outward along thin linear ridges, limited by accumulated path
    length and by local direction continuity (principal axis from a kNN covariance)."""
    ridge = (lab == leaf) & (fx[:, 0] > lin_th) & (fx[:, 3] < rad_th)
    if prob is not None:
        ridge &= np.maximum(prob[:, branch], prob[:, 2]) > p_floor
    if not ridge.any():
        return lab
    tree = cKDTree(coord)
    d, ii = tree.query(coord, k=k + 1, workers=1)
    d = d[:, 1:]; ii = ii[:, 1:]
    pd = np.zeros((len(coord), 3), np.float64)
    for s in range(0, len(coord), 150000):
        e = min(s + 150000, len(coord))
        idxs = np.nonzero(ridge[s:e])[0] + s
        if len(idxs) == 0:
            continue
        nb = ii[idxs]
        rel = coord[nb] - coord[idxs][:, None, :]
        cov = np.einsum("nki,nkj->nij", rel, rel) / max(len(nb), 1)
        w, v = np.linalg.eigh(cov)
        pd[idxs] = v[:, :, -1]
    frontier = (lab == branch)
    plen = np.where(frontier, 0.0, np.inf)
    grown = np.zeros(len(lab), bool)
    ar = np.arange(len(ii))
    for _ in range(hops):
        idx = np.nonzero(ridge & ~frontier)[0]
        if len(idx) == 0:
            break
        nb = ii[idx]; nd = d[idx]
        parent = np.where(nd < eps, plen[nb], np.inf)
        best = parent.min(1); abest = parent.argmin(1)
        step = nd[ar[:len(idx)], abest]
        pv = coord[idx] - coord[nb[ar[:len(idx)], abest]]
        cosang = np.abs(np.einsum("ni,ni->n", pv, pd[idx])) / (np.linalg.norm(pv, axis=1) + 1e-12)
        accept = np.isfinite(best) & ((best + step) <= max_len) & (cosang > dir_cos)
        if not accept.any():
            break
        sel = idx[accept]
        newlen = best[accept] + step[accept]
        upd = newlen < plen[sel]
        sel = sel[upd]; newlen = newlen[upd]
        if len(sel) == 0:
            break
        plen[sel] = newlen
        frontier[sel] = True
        grown[sel] = True
    out = lab.copy()
    out[grown] = branch
    return out


def postprocess2(coord, lab, fx, ncls, prob=None, twig_kw=None, grow_kw=None, iters=1):
    out = knn_vote(coord, lab, ncls)
    if ncls == 3:
        for _ in range(iters):
            out = twig_rescue2(coord, out, fx, prob=prob, **(twig_kw or {}))
            out = remove_islands(coord, out)
            out = tip_growth2(coord, out, fx, prob=prob, **(grow_kw or {}))
        out = knn_vote(coord, out, ncls, k=8, sigma=0.02, iters=1, keep_w=1.0)
    return out


def trunk_prior(coord, lab, slice_h=0.1, max_seeds=3, promote=False):
    """Enforce a single ground-to-crown cylinder: reuse the pseudo-labeling axis tracker on the
    predicted wood and demote trunk points that fall outside the tracked cylinder to branch."""
    from data_prep import pseudo_trunk
    wood = np.nonzero(lab > 0)[0]
    if len(wood) < 800:
        return lab
    w = coord[wood].copy()
    w[:, 2] -= coord[:, 2].min()
    tm = pseudo_trunk(w, 0.0, slice_h=slice_h, max_seeds=max_seeds)
    out = lab.copy()
    ti = wood[~tm]
    out[ti[out[ti] == 2]] = 1
    if promote:
        tb = wood[tm]
        out[tb[out[tb] == 1]] = 2
    return out


def trunk_finalize(coord, lab, slice_h=0.1, max_seeds=3, promote=True, bridge=True,
                   bridge_step=0.06, wide_r=0.34, connect_eps=0.10, demote_orphans=True,
                   r_cap=0.20, tol=1.15, return_info=False):
    """Make the trunk a single ground-connected cylinder: keep only wood inside the tracked axis,
    bridge vertical gaps with the nearest in-axis wood, then drop trunk fragments that are not
    connected to the base."""
    from data_prep import pseudo_trunk
    wood = np.nonzero(lab > 0)[0]
    if len(wood) < 800:
        return (lab, None) if return_info else lab
    zmin = float(coord[:, 2].min())
    w = coord[wood].copy(); w[:, 2] -= zmin
    tm, axes = pseudo_trunk(w, 0.0, slice_h=slice_h, max_seeds=max_seeds, return_axes=True)
    out = lab.copy()
    ti = wood[~tm]
    out[ti[out[ti] == 2]] = 1
    if promote:
        tb = wood[tm]
        out[tb[out[tb] == 1]] = 2
    z = coord[:, 2] - zmin
    bands = np.floor(z / bridge_step).astype(np.int64)
    nbridge = 0
    if bridge:
        for prof in axes:
            pa = np.array(prof); zz = pa[:, 0]
            if len(zz) < 3:
                continue
            zq = np.clip(z, zz.min(), zz.max())
            ax = np.interp(zq, zz, pa[:, 1]); ay = np.interp(zq, zz, pa[:, 2]); ar = np.interp(zq, zz, pa[:, 3])
            d = np.hypot(coord[:, 0] - ax, coord[:, 1] - ay)
            span = (z >= zz.min() - 0.02) & (z <= zz.max() + 0.02)
            tight = span & (d < np.minimum(ar, r_cap) * tol + 0.02)
            nb = int(bands.max()) + 2
            ct = np.bincount(bands[tight], weights=(out[tight] == 2).astype(np.float64), minlength=nb)
            cw = np.bincount(bands[tight], weights=(out[tight] == 1).astype(np.float64), minlength=nb)
            gap = (ct == 0) & (cw > 0)
            if gap.any():
                sel = np.nonzero(tight & gap[bands] & (out == 1))[0]
                out[sel] = 2; nbridge += len(sel)
            empty = (ct + cw) == 0
            if empty.any():
                lo_band = int(np.floor(zz.min() / bridge_step))
                wide = span & (d < wide_r) & (bands >= lo_band)
                cand = np.nonzero(wide & empty[bands] & (out > 0))[0]
                if len(cand):
                    order = np.argsort(d[cand])
                    keep = cand[order[: max(1, int(0.02 / bridge_step)) * 400]]
                    out[keep] = 2; nbridge += len(keep)
    if demote_orphans:
        ti = np.nonzero(out == 2)[0]
        if len(ti) > 1:
            comp, nc = _components(coord, ti, connect_eps)
            base = comp[int(np.argmin(coord[ti, 2]))]
            out[ti[comp != base]] = 1
    if return_info:
        return out, dict(axes=[(round(float(a[0][0]), 3), round(float(a[-1][0]), 3), round(float(a[-1][3]), 3))
                               for a in axes], nbridge=int(nbridge),
                         trunk_pct=round(float((out == 2).mean()), 4))
    return out

def trunk_stats(coord, lab, eps=0.10):
    """Connectivity report for the trunk: component count, vertical span, share of tree height."""
    ti = np.nonzero(lab == 2)[0]
    if len(ti) == 0:
        return dict(ncomp=0, zmin=None, zmax=None, ztop=None, reach=None)
    comp, nc = _components(coord, ti, eps) if len(ti) > 1 else (np.zeros(len(ti), np.int64), 1)
    z = coord[ti, 2]
    base = comp[int(np.argmin(z))]
    main = int((comp == base).sum())
    ztop = float(coord[:, 2].max())
    w = coord[lab > 0][:, 2]
    wtop = float(w.max()) if len(w) else ztop
    return dict(ncomp=int(nc), ncomp_main=main, zmin=round(float(z.min()), 3), zmax=round(float(z.max()), 3),
                ztop=round(ztop, 3), wtop=round(wtop, 3), reach=round(float(z.max()) / max(ztop, 1e-6), 4),
                reach_wood=round(float(z.max()) / max(wtop, 1e-6), 4))


def leader_extend(coord, lab, fx, k=14, max_gap=0.14, pad=0.045, tau_rel=0.35, tau_abs=0.02,
                  min_gain=0.3, verbose=False):
    """Extend the trunk along the leader: keep only wood edges whose local radius is thick enough
    (>= tau), walk the maximum spanning forest from the current trunk top to the highest point still
    reachable inside that thick skeleton, and mark that route trunk. Stops before terminal twigs."""
    wi = np.nonzero(lab > 0)[0]
    if len(wi) < 60:
        return lab
    w = coord[wi]
    tree = cKDTree(w)
    kk = min(k, len(w) - 1)
    d, ii = tree.query(w, k=kk + 1, workers=1)
    d = d[:, 1:]; ii = ii[:, 1:]
    rad = fx[wi, 3].astype(np.float64)
    trunk_local = np.nonzero(lab[wi] == 2)[0]
    if len(trunk_local) == 0:
        return lab
    src = int(trunk_local[int(np.argmax(w[trunk_local, 2]))])
    r_trunk = float(np.median(rad[trunk_local]))
    tau = max(tau_abs, tau_rel * r_trunk)
    a = np.repeat(np.arange(len(wi)), d.shape[1])
    b = ii.ravel()
    dd = d.ravel()
    ok = (a < b) & (dd < max_gap)
    a, b = a[ok], b[ok]
    wt = np.minimum(rad[a], rad[b])
    order = np.argsort(-wt)
    parent = np.arange(len(wi)); rank = np.zeros(len(wi), np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    adj = {}
    for e in order:
        if wt[e] < tau:
            break
        ra, rb = find(int(a[e])), find(int(b[e]))
        if ra == rb:
            continue
        adj.setdefault(int(a[e]), []).append((int(b[e]), float(wt[e])))
        adj.setdefault(int(b[e]), []).append((int(a[e]), float(wt[e])))
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra; rank[ra] += 1
    comp = np.array([find(i) for i in range(len(wi))])
    same = np.nonzero(comp == find(src))[0]
    tgt = int(same[int(np.argmax(w[same, 2]))])
    if w[tgt, 2] - w[src, 2] < min_gain:
        if verbose:
            print("   leader gain %.2f < %.2f, skipped" % (w[tgt, 2] - w[src, 2], min_gain))
        return lab
    prev = {src: -1}
    dq = __import__("collections").deque([src])
    while dq:
        u = dq.popleft()
        if u == tgt:
            break
        for v, _ in adj.get(u, ()):
            if v not in prev:
                prev[v] = u
                dq.append(v)
    if tgt not in prev:
        return lab
    path = []
    cur = tgt
    while cur != -1:
        path.append(cur)
        cur = prev[cur]
    path = np.array(path, np.int64)
    thick = float(np.min(rad[path])) if len(path) else 0.0
    gp = w[path]
    if pad > 0:
        near = tree.query_ball_point(gp, r=pad)
        parts = [np.asarray(x, np.int64) for x in near if len(x)]
        sel = np.unique(np.concatenate(parts)) if len(parts) else path
    else:
        sel = path
    out = lab.copy()
    out[wi[sel]] = 2
    if verbose:
        print("   tau=%.3f (trunk_r=%.3f) pts=%d thick=%.3f z %.2f->%.2f added=%d" % (
            tau, r_trunk, len(path), thick, w[src, 2], w[tgt, 2], int((out == 2).sum() - (lab == 2).sum())))
    return out
