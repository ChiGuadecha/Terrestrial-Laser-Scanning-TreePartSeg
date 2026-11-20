# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# 梁艺萌
# 树干 / 树枝 / 树叶 三维点云分离识别 —— 项目源代码
# 未经本人允许不得商用。third_party/ 内组件许可归其原作者，见 NOTICE.txt。
# ------------------------------------------------------------------
"""Eye Dome Lighting (EDL) orthographic renderer for tree point clouds."""
import os, sys
import numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

def render_view(coord, color, azim, elev, size=900, ring_scale=1.0, edl_scale=35.0,
                edl_alpha=0.5, bg=1.0, samples=10):
    az, el = np.deg2rad(azim), np.deg2rad(elev)
    v = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    r = np.array([-np.sin(az), np.cos(az), 0.0])
    u = np.array([-np.sin(el) * np.cos(az), -np.sin(el) * np.sin(az), np.cos(el)])
    c0 = (coord.max(0) + coord.min(0)) * 0.5
    cc = coord - c0
    px = cc @ r; py = cc @ u; d = cc @ v
    ext = max(np.ptp(px), np.ptp(py)) * 0.5 + 1e-6
    px = px / ext; py = py / ext
    dn = (d - d.min()) / (np.ptp(d) + 1e-9)          # 1 = near camera
    W = size
    X = ((px * 0.5 + 0.5) * (W - 1)).astype(np.int32)
    Y = ((0.5 - py * 0.5) * (W - 1)).astype(np.int32)
    order = np.argsort(dn)                            # far -> near, near overwrites
    cbuf = np.full((W, W, 3), bg, np.float32)
    dbuf = np.zeros((W, W), np.float32)
    mbuf = np.zeros((W, W), bool)
    n = len(cc)
    rad = int(min(6, max(1, np.ceil(np.sqrt(3.0 * W * W / max(n, 1)) / 2) - 0.5)))
    offs = [(dx, dy) for dx in range(-rad, rad + 1) for dy in range(-rad, rad + 1)]
    Xo = X[order]; Yo = Y[order]; Co = color[order]; Do = dn[order]
    for dx, dy in offs:
        xs = np.clip(Xo + dx, 0, W - 1); ys = np.clip(Yo + dy, 0, W - 1)
        cbuf[ys, xs] = Co
        dbuf[ys, xs] = Do
        mbuf[ys, xs] = True
    shade = np.zeros((W, W), np.float32)
    cnt = np.zeros((W, W), np.float32)
    for k in (1, 2, 3, 5):
        k = max(1, int(round(k * ring_scale * (rad + 1))))
        for t in range(samples):
            th = 2 * np.pi * t / samples
            ox = np.round(k * np.cos(th)).astype(np.int32)
            oy = np.round(k * np.sin(th)).astype(np.int32)
            xs = np.clip(np.arange(W)[None, :] + ox, 0, W - 1)
            ys = np.clip(np.arange(W)[:, None] + oy, 0, W - 1)
            nd = dbuf[ys, xs]; nm = mbuf[ys, xs]
            shade += np.clip((dbuf - nd) * edl_scale, 0, 1) * nm
            cnt += nm
    shade = shade / np.maximum(cnt, 1)
    lum = np.clip(1.0 - edl_alpha * shade, 0.15, 1.0)
    img = cbuf * lum[..., None]
    img[~mbuf] = bg
    return img

COL3 = np.array([[46, 204, 113], [52, 120, 246], [231, 76, 60]], np.float32) / 255

def render_tree(npz_path, out_png, title=""):
    import matplotlib.pyplot as plt
    d = np.load(npz_path)
    c = d["coord"]; n = len(c)
    rng = np.random.default_rng(1)
    sel = rng.choice(n, min(350000, n), replace=False)
    cc = c[sel]
    if "label3" in d:
        lab = d["label3"][sel]
        col3 = COL3[lab]
        col_wood = COL3[lab]
        col_leaf = np.where((lab == 0)[:, None], COL3[0], 1.0)
        leafm = lab == 0
        woodm = lab > 0
        views = [
            (cc, col3, 0, 6, "3-class  front"),
            (cc, col3, 90, 6, "3-class  side"),
            (cc, col3, 40, 28, "3-class  oblique (EDL)"),
            (cc[leafm], COL3[0][None, :].repeat(leafm.sum(), 0), 0, 8, "leaves only (EDL)"),
            (cc[leafm], COL3[0][None, :].repeat(leafm.sum(), 0), 40, 28, "leaves only (EDL, oblique)"),
            (cc[woodm], COL3[lab[woodm]], 40, 28, "trunk + branches (EDL)"),
        ]
    else:
        col3 = np.full((len(sel), 3), 0.55, np.float32)
        views = [(cc, col3, 0, 6, "front"), (cc, col3, 90, 6, "side"), (cc, col3, 40, 28, "oblique")]
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (cd, cl, az, el, t) in zip(axs.ravel(), views):
        img = render_view(cd, cl, az, el)
        ax.imshow(img); ax.set_title(t, fontsize=10); ax.axis("off")
    fig.suptitle(title or os.path.basename(npz_path))
    fig.tight_layout()
    fig.savefig(out_png, dpi=90, facecolor="white")
    plt.close(fig)
    print("saved", out_png)

if __name__ == "__main__":
    which, tid = sys.argv[1], sys.argv[2]
    render_tree(os.path.join(r"E:\Desktop\场景\treeseg\data", f"{which}_{tid}.npz"),
                os.path.join(r"E:\Desktop\场景\treeseg", f"edl_{which}_{tid}.png"),
                f"tree {tid} — PTv3 pseudo labels, EDL shading")
