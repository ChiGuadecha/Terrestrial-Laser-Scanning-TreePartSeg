# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# 梁艺萌
# 树干 / 树枝 / 树叶 三维点云分离识别 —— 项目源代码
# 未经本人允许不得商用。third_party/ 内组件许可归其原作者，见 NOTICE.txt。
# ------------------------------------------------------------------
"""Export predictions: colored .laz (original frame) + .ply + EDL png per tree."""
import os, sys, glob, json
import numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import laspy
sys.path.insert(0, r"E:\Desktop\场景\treeseg")
from edl_render import render_view, COL3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = r"E:\Desktop\场景\treeseg\data"
OUT = r"E:\Desktop\场景\output"
os.makedirs(os.path.join(OUT, "ply"), exist_ok=True)
os.makedirs(os.path.join(OUT, "laz"), exist_ok=True)
os.makedirs(os.path.join(OUT, "edl"), exist_ok=True)

def load_stats(which):
    try:
        with open(os.path.join(OUT, f"metrics_{which}.json"), encoding="utf-8") as f:
            return json.load(f).get("per_tree", {})
    except Exception:
        return {}

def export(which="test", ids=None):
    stats = load_stats(which)
    if ids is None:
        ids = sorted(int(os.path.basename(f).split("_")[-1].split(".")[0])
                     for f in glob.glob(os.path.join(OUT, f"pred_{which}_*.npz")))
    for tid in ids:
        d = np.load(os.path.join(OUT, f"pred_{which}_{tid}.npz"))
        pred, shift = d["pred"], d["shift"]
        src = os.path.join(DATA, f"train_{tid}.npz") if which == "val" else \
              os.path.join(ROOT_SRC, f"{tid}.laz")
        # colored PLY (centered frame)
        rgb = (COL3[pred] * 255).astype(np.uint8)
        arr = np.hstack([d["coord"], rgb])
        hdr = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\n"
               "property float x\nproperty float y\nproperty float z\n"
               "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n") % len(arr)
        dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
        buf = np.empty(len(arr), dt)
        buf["x"], buf["y"], buf["z"] = arr[:, 0], arr[:, 1], arr[:, 2]
        buf["r"], buf["g"], buf["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        with open(os.path.join(OUT, "ply", f"{which}_{tid}.ply"), "wb") as f:
            f.write(hdr.encode()); f.write(buf.tobytes())
        # LAS in original frame if source exists
        src = os.path.join(ROOT_SRC, f"{tid}.laz")
        if os.path.exists(src):
            p = laspy.read(src)
            if len(p.points) == len(pred):
                p.classification = np.clip(pred.astype(np.uint8), 0, 255)
                p.write(os.path.join(OUT, "laz", f"{which}_{tid}.laz"))
            else:
                print("las skipped (outlier-cleaned)", tid, len(p.points), len(pred))
        # EDL png: 2 views
        rng = np.random.default_rng(tid)
        sel = rng.choice(len(pred), min(300000, len(pred)), replace=False)
        cc, pp = d["coord"][sel], pred[sel]
        fig, axs = plt.subplots(1, 2, figsize=(11, 6))
        for ax, (az, el, t) in zip(axs, [(40, 20, "oblique"), (0, 6, "front")]):
            ax.imshow(render_view(cc, COL3[pp], az, el)); ax.set_title(f"tree {tid} {t}"); ax.axis("off")
        st = stats.get(str(tid), {})
        m3 = st.get("metrics3")
        if m3:
            head = (f"tree {tid} | OA {m3['oa']:.3f} mIoU {m3['miou']:.3f} | "
                    f"recall L/B/T {m3['recall'][0]:.3f}/{m3['recall'][1]:.3f}/{m3['recall'][2]:.3f}")
        else:
            head = (f"tree {tid} | leaf {st.get('leaf_pct',0):.1%} branch {st.get('branch_pct',0):.1%} "
                    f"trunk {st.get('trunk_pct',0):.1%}")
        fig.suptitle(head, y=1.0)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "edl", f"{which}_{tid}.png"), dpi=95)
        plt.close(fig)
        print("exported", which, tid, flush=True)

ROOT_SRC = r"E:\Desktop\场景\测试集"
if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "test"
    if which == "val":
        ROOT_SRC = r"E:\Desktop\场景\训练集"
    ids = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else None
    export(which, ids)
