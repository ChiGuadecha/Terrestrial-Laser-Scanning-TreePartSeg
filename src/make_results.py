# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# 梁艺萌
# 树干 / 树枝 / 树叶 三维点云分离识别 —— 项目源代码
# 未经本人允许不得商用。third_party/ 内组件许可归其原作者，见 NOTICE.txt。
# ------------------------------------------------------------------
"""Build the deliverable folder: per-tree EDL images (segmentation / trunk / trunk+branch /
trunk+branch+leaf), each with the accuracy block printed underneath."""
import os, sys, io, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, r"E:\Desktop\场景\treeseg")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from edl_render import render_view

plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = r"E:\Desktop\场景\output"
DATA = r"E:\Desktop\场景\treeseg\data"
RES = r"E:\Desktop\场景\成果"
LEAF, BRANCH, TRUNK = 0, 1, 2
C = {"leaf": np.array([46, 204, 113]) / 255, "branch": np.array([52, 120, 246]) / 255,
     "trunk": np.array([231, 76, 60]) / 255, "ghost": np.array([222, 226, 232]) / 255,
     "bg": np.ones(3, np.float32)}
VIEWS = [("斜视 oblique", 210, 16), ("正视 front", 0, 6), ("侧视 side", 90, 6)]


def subsample(n, cap=260000, seed=0):
    if n <= cap:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, cap, replace=False))


def render(coord, lab, mode, size=760):
    """'seg'/'full' keep every class; 'trunk' and 'wood' drop leaves entirely."""
    if mode == "trunk":
        keep = lab == TRUNK
    elif mode == "wood":
        keep = lab > LEAF
    else:
        keep = np.ones(len(lab), bool)
    idx = np.nonzero(keep)[0]
    if len(idx) == 0:
        idx = np.arange(len(lab))
    sel = idx[subsample(len(idx))]
    c, l = coord[sel], lab[sel]
    base = np.stack([C["leaf"], C["branch"], C["trunk"]]).astype(np.float32)
    col = base[l]
    if mode == "full":
        col[l == LEAF] = col[l == LEAF] * 0.72 + 0.28
    imgs = [render_view(c, col.astype(np.float32), az, el, size=size) for _, az, el in VIEWS]
    return imgs


def panel(path, title, imgs, lines, sub=""):
    n = len(imgs)
    fig = plt.figure(figsize=(4.1 * n, 5.4), dpi=100)
    gs = fig.add_gridspec(2, n, height_ratios=[3.1, 1.0], hspace=0.02, wspace=0.02)
    for i, im in enumerate(imgs):
        ax = fig.add_subplot(gs[0, i]); ax.imshow(im); ax.axis("off")
        ax.set_title(VIEWS[i][0], fontsize=11, color="#333")
    ax = fig.add_subplot(gs[1, :]); ax.axis("off")
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=9.5,
            linespacing=1.45, bbox=dict(boxstyle="round,pad=0.6", fc="#F7F9FB", ec="#9AA7B4", lw=1.0))
    fig.suptitle(title + (("   " + sub) if sub else ""), fontsize=14, fontweight="bold", y=0.985)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def metric_lines(info, npts):
    m3 = info.get("metrics3"); m2 = info.get("metrics2")
    L = []
    L.append("点数 {:,}   叶 {:.1f}% / 枝 {:.1f}% / 干 {:.1f}%".format(
        npts, 100 * info["leaf_pct"], 100 * info["branch_pct"], 100 * info["trunk_pct"]))
    if m3:
        r, p, f, iou = m3["recall"], m3["precision"], m3["f1"], m3["iou"]
        L.append("三分类  召回 leaf/branch/trunk = {:.3f} / {:.3f} / {:.3f}".format(r[0], r[1], r[2]))
        L.append("         精确率 = {:.3f} / {:.3f} / {:.3f}      F1 = {:.3f} / {:.3f} / {:.3f}".format(
            p[0], p[1], p[2], f[0], f[1], f[2]))
        L.append("         mIoU = {:.4f}   macro-F1 = {:.4f}   总体精度 OA = {:.4f}   末端细枝召回 = {:.4f}".format(
            m3["miou"], m3["macro_f1"], m3["oa"], info.get("tip_branch_recall") or float("nan")))
    if m2:
        L.append("枝叶二分 召回 leaf/wood = {:.3f} / {:.3f}   mIoU = {:.4f}   OA = {:.4f}".format(
            m2["recall"][0], m2["recall"][1], m2["miou"], m2["oa"]))
    else:
        L.append("枝叶二分 该树无地面真值（测试集），精度见验证集统计")
    return L


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "val"
    ids = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else None
    mp = os.path.join(OUT, f"metrics_{which}.json")
    met = json.load(io.open(mp, encoding="utf-8"))
    per = met.get("per_tree", {})
    if ids is None:
        ids = sorted(int(k) for k in per.keys())
    os.makedirs(RES, exist_ok=True)
    src = "train" if which == "val" else "test"
    index = []
    for tid in ids:
        f = os.path.join(OUT, f"pred_{which}_{tid}.npz")
        if not os.path.exists(f):
            print("missing", f); continue
        d = np.load(f)
        coord = d["coord"]; lab = d["pred"].astype(np.int64)
        info = per.get(str(tid), {})
        title = f"树 {which}-{tid}  树干/树枝/树叶 分离结果"
        folder = os.path.join(RES, f"{which}_{tid}")
        os.makedirs(folder, exist_ok=True)
        lines = metric_lines(info, len(coord))
        panel(os.path.join(folder, "01_分割总览.png"), title + "  ① 三分类分割总览",
              render(coord, lab, "seg"), lines)
        panel(os.path.join(folder, "02_仅主干.png"), title + "  ② 主干（地面→梢顶贯通）",
              render(coord, lab, "trunk"), lines)
        panel(os.path.join(folder, "03_主干+树枝.png"), title + "  ③ 主干 + 树枝（木质部）",
              render(coord, lab, "wood"), lines)
        panel(os.path.join(folder, "04_主干+树枝+树叶.png"), title + "  ④ 主干 + 树枝 + 树叶",
              render(coord, lab, "full"), lines)
        with io.open(os.path.join(folder, "metrics.json"), "w", encoding="utf-8") as fh:
            json.dump(info, fh, ensure_ascii=False, indent=1)
        index.append((f"{which}_{tid}", lines))
        print("done", tid, flush=True)
    with io.open(os.path.join(RES, "index.txt"), "a", encoding="utf-8") as fh:
        for name, lines in index:
            fh.write(name + "\n" + "\n".join("  " + x for x in lines) + "\n\n")


if __name__ == "__main__":
    main()
