# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# 梁艺萌
# 树干 / 树枝 / 树叶 三维点云分离识别 —— 项目源代码
# 未经本人允许不得商用。third_party/ 内组件许可归其原作者，见 NOTICE.txt。
# ------------------------------------------------------------------
"""Per-tree trunk sanity: components, ground connection, top reach, class metrics."""
import os, sys, io, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, r"E:\Desktop\场景\treeseg")
import numpy as np
import post as P
from predict import per_class, TWIG_KW, GROW_KW

DATA = r"E:\Desktop\场景\treeseg\data"
CACHE = r"E:\Desktop\场景\treeseg\cache"


def report(tid, finalize=True):
    d = np.load(os.path.join(DATA, f"train_{tid}.npz"))
    pr = np.load(os.path.join(CACHE, f"probs_{tid}.npz"))["prob"].astype(np.float32)
    coord, fx, y = d["coord"], d["featx"], d["label3"].astype(np.int64)
    lab = P.postprocess2(coord, pr.argmax(1), fx, 3, prob=pr, twig_kw=TWIG_KW, grow_kw=GROW_KW)
    if finalize:
        lab, info = P.trunk_finalize(coord, lab, return_info=True)
    else:
        info = None
    cm = np.zeros((3, 3), np.int64)
    for t in range(3):
        cm[t] = [(lab[y == t] == k).sum() for k in range(3)]
    m = per_class(cm)
    ti = np.nonzero(lab == 2)[0]
    comp, nc = P._components(coord, ti, 0.10) if len(ti) > 1 else (np.zeros(max(len(ti), 1)), 1)
    sizes = np.sort(np.bincount(comp.astype(np.int64), minlength=nc))[::-1]
    tops = []
    for c in np.unique(comp):
        sel = ti[comp == c]
        tops.append((round(float(coord[sel, 2].min()), 2), round(float(coord[sel, 2].max()), 2), int(len(sel))))
    gt_top = float(coord[y == 2][:, 2].max()) if (y == 2).any() else float(coord[:, 2].max())
    print(f"tree {tid} finalize={finalize}")
    print("  trunk P/R/F1 = {:.3f}/{:.3f}/{:.3f}  branch P/R={:.3f}/{:.3f}  mIoU={:.3f} OA={:.4f}".format(
        m["precision"][2], m["recall"][2], m["f1"][2], m["precision"][1], m["recall"][1], m["miou"], m["oa"]))
    print("  n_comp(eps0.1)={} sizes={}  GT trunk top={:.2f}  pred trunk top={:.2f}  pred trunk bottom={:.2f}".format(
        nc, list(sizes[:6]), gt_top, float(coord[ti, 2].max()) if len(ti) else 0,
        float(coord[ti, 2].min()) if len(ti) else 0))
    print("  comps(zmin,zmax,n)=", sorted(tops, key=lambda t: -t[2])[:8], flush=True)
    return lab


if __name__ == "__main__":
    ids = [int(x) for x in sys.argv[1].split(",")]
    for tid in ids:
        report(tid, finalize=False)
        report(tid, finalize=True)
