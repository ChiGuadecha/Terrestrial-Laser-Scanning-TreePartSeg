# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# 梁艺萌
# 树干 / 树枝 / 树叶 三维点云分离识别 —— 项目源代码
# 未经本人允许不得商用。third_party/ 内组件许可归其原作者，见 NOTICE.txt。
# ------------------------------------------------------------------
import os, sys, time
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, r"E:\Desktop\场景\treeseg")
import laspy
from data_prep import process, OUT, ROOT

tid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
lf = laspy.read(os.path.join(ROOT, "训练集", f"{tid}_leaf.laz"))
wf = laspy.read(os.path.join(ROOT, "训练集", f"{tid}_wood.laz"))
L = np.asarray(lf.xyz, np.float32)
W = np.asarray(wf.xyz, np.float32)
t0 = time.time()
path = os.path.join(OUT, f"train_{tid}.npz")
st = process(tid, L, W, path)
print("done in", round(time.time() - t0, 1), "s", st)

d = np.load(path)
c, l3 = d["coord"], d["label3"]
rng = np.random.default_rng(0)
COL3 = {0: (46, 204, 113), 1: (52, 120, 246), 2: (231, 76, 60)}
def scatter(ax, plane, lab, pal, title):
    sel = rng.choice(len(c), min(60000, len(c)), replace=False)
    for k, v in pal.items():
        m = sel[lab[sel] == k]
        ax.scatter(c[m, plane[0]], c[m, plane[1]], s=0.35, c=[np.array(v) / 255], rasterized=True, linewidths=0)
    ax.set_title(title, fontsize=9); ax.set_aspect("equal"); ax.tick_params(labelsize=6)
fig, axs = plt.subplots(2, 3, figsize=(14, 8.5))
l2 = d["label_woodleaf"]
PAL2 = {0: (46, 204, 113), 1: (160, 160, 160)}
scatter(axs[0, 0], (0, 2), l2, PAL2, "provided annotation: leaf/wood (front)")
scatter(axs[0, 1], (1, 2), l2, PAL2, "leaf/wood (side)")
scatter(axs[0, 2], (0, 1), l2, PAL2, "leaf/wood (top)")
scatter(axs[1, 0], (0, 2), l3, COL3, "pseudo 3-class: trunk red / branch blue / leaf green")
scatter(axs[1, 1], (1, 2), l3, COL3, "3-class (side)")
scatter(axs[1, 2], (0, 1), l3, COL3, "3-class (top)")
fig.suptitle(f"tree {tid}: pseudo trunk/branch labeling QC")
fig.tight_layout()
png = os.path.join(ROOT, "treeseg", f"qc_tree{tid}.png")
fig.savefig(png, dpi=110)
print("png:", png)

# colored PLY (full res, 3-class colors)
rgb = np.array([COL3[k] for k in l3], np.uint8)
hdr = ("ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n" % len(c))
np.savetxt(os.path.join(ROOT, "treeseg", f"qc_tree{tid}.ply"),
           np.hstack([c, rgb]), "%.3f %%.3f %%.3f %%d %%d %%d".replace("%%", "%"), header=hdr, comments="")
print("ply saved")
