# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# 梁艺萌
# 树干 / 树枝 / 树叶 三维点云分离识别 —— 项目源代码
# 未经本人允许不得商用。third_party/ 内组件许可归其原作者，见 NOTICE.txt。
# ------------------------------------------------------------------
"""Build the deliverable folder 成果/: per tree four EDL images with an accuracy block,
plus the separated point clouds (wood = trunk+branch, leaf) in the original coordinate frame.
Folder / file names follow the source data names (测试集\13.laz -> 成果\测试集\13\13_wood.laz)."""
import os, sys, io, json, copy, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, r"E:\Desktop\场景\treeseg")
import numpy as np
import laspy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from edl_render import render_view

plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = r"E:\Desktop\场景"
OUT = os.path.join(ROOT, "output")
RES = os.path.join(ROOT, "成果")
LEAF, BRANCH, TRUNK = 0, 1, 2
CLS_CODE = {LEAF: 1, BRANCH: 2, TRUNK: 3}          # written into LAS classification
RGB = {LEAF: (46, 204, 113), BRANCH: (52, 120, 246), TRUNK: (231, 76, 60)}
VIEWS = [("斜视 oblique", 210, 16), ("正视 front", 0, 6), ("侧视 side", 90, 6)]
SETNAME = {"val": "验证集", "test": "测试集"}
SIGN = "梁艺萌 · 未经本人允许不得商用"


def subsample(n, cap=240000, seed=0):
    if n <= cap:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, cap, replace=False))


def render(coord, lab, mode, size=720):
    if mode == "trunk":
        keep = lab == TRUNK
    elif mode == "wood":
        keep = lab > LEAF
    else:
        keep = np.ones(len(lab), bool)
    idx = np.nonzero(keep)[0]
    if len(idx) == 0:
        return [np.ones((size, size, 3), np.float32) for _ in VIEWS]
    sel = idx[subsample(len(idx))]
    c, l = coord[sel], lab[sel]
    base = np.array([RGB[LEAF], RGB[BRANCH], RGB[TRUNK]], np.float32) / 255.0
    col = base[l]
    if mode == "full":
        m = l == LEAF
        col[m] = col[m] * 0.72 + 0.28
    return [render_view(c, col.astype(np.float32), az, el, size=size) for _, az, el in VIEWS]


def panel(path, title, imgs, lines):
    n = len(imgs)
    fig = plt.figure(figsize=(4.0 * n, 5.3), dpi=100)
    gs = fig.add_gridspec(2, n, height_ratios=[3.0, 1.05], hspace=0.02, wspace=0.02)
    axes = []
    for i, im in enumerate(imgs):
        ax = fig.add_subplot(gs[0, i]); ax.imshow(im); ax.axis("off")
        ax.set_title(VIEWS[i][0], fontsize=11, color="#333")
        axes.append(ax)
    ax = fig.add_subplot(gs[1, :]); ax.axis("off")
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=9.5, linespacing=1.42,
            bbox=dict(boxstyle="round,pad=0.55", fc="#F7F9FB", ec="#9AA7B4", lw=1.0))
    ax.text(1.0, 0.0, SIGN, va="bottom", ha="right", fontsize=9.2, color="#40566B",
            fontweight="bold")
    for a in axes:
        a.text(0.985, 0.015, SIGN, transform=a.transAxes, va="bottom", ha="right",
               fontsize=8.4, color="#5A5A5A", alpha=0.40)
    fig.suptitle(title, fontsize=13.5, fontweight="bold", y=0.985)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def trunk_line(info):
    t = info.get("trunk") or {}
    if not t or t.get("ncomp", 0) == 0:
        return "主干贯通: 未识别到主干"
    return ("主干贯通: {:.2f}–{:.2f} m（木质部顶 {:.2f} m，到达 {:.1%}；全树顶 {:.2f} m，到达 {:.1%}），连通块 {} 根".format(
        t["zmin"], t["zmax"], t.get("wtop", 0), t.get("reach_wood", 0), t["ztop"], t["reach"], t["ncomp"]))


def metric_lines(name, info, npts, conf):
    m3 = info.get("metrics3"); m2 = info.get("metrics2")
    L = ["数据 {}   点数 {:,}   叶 {:.1f}% / 枝 {:.1f}% / 干 {:.1f}%   平均置信度 {:.3f}".format(
        name, npts, 100 * info.get("leaf_pct", 0), 100 * info.get("branch_pct", 0),
        100 * info.get("trunk_pct", 0), conf)]
    L.append(trunk_line(info))
    if m3:
        r, p, f = m3["recall"], m3["precision"], m3["f1"]
        L.append("三分类  召回 叶/枝/干 = {:.3f} / {:.3f} / {:.3f}      精确率 = {:.3f} / {:.3f} / {:.3f}".format(
            r[0], r[1], r[2], p[0], p[1], p[2]))
        L.append("         F1 = {:.3f} / {:.3f} / {:.3f}   mIoU = {:.4f}   macro-F1 = {:.4f}   总体精度 OA = {:.4f}".format(
            f[0], f[1], f[2], m3["miou"], m3["macro_f1"], m3["oa"]))
        L.append("         末端细枝召回 = {:.4f}".format(info.get("tip_branch_recall") or float("nan")))
        if m2:
            L.append("枝叶二分 召回 叶/木 = {:.3f} / {:.3f}   mIoU = {:.4f}   OA = {:.4f}".format(
                m2["recall"][0], m2["recall"][1], m2["miou"], m2["oa"]))
    else:
        L.append("三分类  测试集无地面真值，精度指标见验证集（同模型同流程）")
    return L


def write_ply(path, xyz, cls, labels):
    dt = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    buf = np.empty(len(xyz), dt)
    buf["x"], buf["y"], buf["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    col = np.array([RGB[LEAF], RGB[BRANCH], RGB[TRUNK]], np.uint8)
    buf["r"], buf["g"], buf["b"] = col[labels, 0], col[labels, 1], col[labels, 2]
    hdr = ("ply\nformat binary_little_endian 1.0\ncomment classes: 1=leaf 2=branch 3=trunk\n"
           "element vertex %d\nproperty double x\nproperty double y\nproperty double z\n"
           "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n") % len(xyz)
    with open(path, "wb") as f:
        f.write(hdr.encode()); f.write(buf.tobytes())


def write_las(path, xyz, cls, labels, src_header):
    h = copy.deepcopy(src_header) if src_header is not None else laspy.LasHeader(point_format=1, version="1.4")
    h.point_count = len(xyz)
    out = laspy.LasData(h)
    out.xyz = xyz.astype(np.float64)
    out.classification = np.array([CLS_CODE[k] for k in labels], np.uint8)
    if cls is not None and len(cls) == len(xyz):
        out.intensity = cls
    out.write(path)


def full_labels(which, tid, pred, shift, coord):
    """Return points in the ORIGINAL file frame with per-point class. Test trees are mapped back to
    the exact point order of 测试集/<tid>.laz (points dropped by outlier cleaning get the class of the
    nearest kept point); val trees keep the npz order (leaf-file points then wood-file points)."""
    if which != "test":
        return coord + shift, pred, None
    las = laspy.read(os.path.join(ROOT, "测试集", f"{tid}.laz"))
    raw = np.asarray(las.xyz, np.float32)
    cen = raw - shift
    z = cen[:, 2]
    lo, hi = np.percentile(z, 0.01), np.percentile(z, 99.99)
    keep = (z >= lo - 0.5) & (z <= hi + 1.0)
    if bool(keep.all()):
        return raw, pred, las
    lab_full = np.zeros(len(raw), np.int64)
    lab_full[keep] = pred
    drop = np.nonzero(~keep)[0]
    if len(drop):
        from scipy.spatial import cKDTree
        _, jj = cKDTree(cen[keep]).query(cen[drop], k=1, workers=1)
        lab_full[drop] = pred[jj]
    return raw, lab_full, las


def build(which, tag="", ids=None):
    mp = os.path.join(OUT, f"metrics_{which}{('_' + tag) if tag else ''}.json")
    if not os.path.exists(mp):  # 测试集曾按 A/B 两半分别跑，这里自动合并
        parts = [os.path.join(OUT, f"metrics_{which}_{s}.json") for s in ("A", "B")]
        if all(os.path.exists(x) for x in parts):
            met = json.load(io.open(parts[0], encoding="utf-8"))
            other = json.load(io.open(parts[1], encoding="utf-8"))
            met.setdefault("per_tree", {}).update(other.get("per_tree", {}))
            print("merged", os.path.basename(parts[0]), "+", os.path.basename(parts[1]), flush=True)
        else:
            raise FileNotFoundError(mp)
    else:
        met = json.load(io.open(mp, encoding="utf-8"))
    per = met.get("per_tree", {})
    if ids is None:
        ids = sorted(int(k) for k in per.keys())
    setdir = os.path.join(RES, SETNAME[which])
    os.makedirs(setdir, exist_ok=True)
    rows = []
    for tid in ids:
        f = os.path.join(OUT, f"pred_{which}_{tid}.npz")
        if not os.path.exists(f):
            print("missing", f); continue
        d = np.load(f)
        pred, shift = d["pred"].astype(np.int64), d["shift"]
        info = per.get(str(tid), {})
        xyz, labels, las = full_labels(which, tid, pred, shift, d["coord"])
        folder = os.path.join(setdir, str(tid))
        os.makedirs(folder, exist_ok=True)
        name = f"{tid}.laz"
        conf = info.get("conf_mean", float("nan"))
        lines = metric_lines(name, info, len(xyz), conf)
        title = f"{SETNAME[which]} {name}  树干/树枝/树叶 分离结果"
        t0 = time.time()
        panel(os.path.join(folder, "01_分割总览.png"), title + "  ① 三分类分割总览", render(xyz, labels, "seg"), lines)
        panel(os.path.join(folder, "02_仅主干.png"), title + "  ② 主干（地面→梢顶贯通）", render(xyz, labels, "trunk"), lines)
        panel(os.path.join(folder, "03_主干+树枝.png"), title + "  ③ 主干 + 树枝（木质部）", render(xyz, labels, "wood"), lines)
        panel(os.path.join(folder, "04_主干+树枝+树叶.png"), title + "  ④ 主干 + 树枝 + 树叶", render(xyz, labels, "full"), lines)
        if not os.environ.get("SKIP_CLOUD"):  # 只重出图时设 SKIP_CLOUD=1 跳过点云写盘
            hdr = las.header if las is not None else None
            intensity = np.asarray(las.intensity) if las is not None else None
            wm = labels > LEAF; lm = labels == LEAF
            write_las(os.path.join(folder, f"{tid}_wood.laz"), xyz[wm], intensity[wm] if intensity is not None else None, labels[wm], hdr)
            write_ply(os.path.join(folder, f"{tid}_wood.ply"), xyz[wm], labels[wm], labels[wm])
            write_las(os.path.join(folder, f"{tid}_leaf.laz"), xyz[lm], intensity[lm] if intensity is not None else None, labels[lm], hdr)
            write_ply(os.path.join(folder, f"{tid}_leaf.ply"), xyz[lm], labels[lm], labels[lm])
            write_las(os.path.join(folder, f"{tid}_seg.laz"), xyz, intensity, labels, hdr)
            write_ply(os.path.join(folder, f"{tid}_seg.ply"), xyz, labels, labels)
        with io.open(os.path.join(folder, "metrics.json"), "w", encoding="utf-8") as fh:
            rec = dict(info); rec["source"] = name; rec["n_file"] = int(len(xyz))
            json.dump(rec, fh, ensure_ascii=False, indent=1)
        rows.append((tid, info, len(xyz)))
        print("built", which, tid, round(time.time() - t0, 1), "s", flush=True)
    return rows


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "val"
    tag = sys.argv[2] if len(sys.argv) > 2 else ""
    ids = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else None
    build(which, tag, ids)
