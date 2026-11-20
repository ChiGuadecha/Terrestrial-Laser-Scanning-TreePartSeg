# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# 梁艺萌
# 树干 / 树枝 / 树叶 三维点云分离识别 —— 项目源代码
# 未经本人允许不得商用。third_party/ 内组件许可归其原作者，见 NOTICE.txt。
# ------------------------------------------------------------------
"""Evaluate post-processing variants on cached probabilities (CPU only)."""
import os, sys, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, r"E:\Desktop\场景\treeseg")
import numpy as np
import post as P
from predict import per_class, VAL_IDS

DATA = r"E:\Desktop\场景\treeseg\data"
CACHE = r"E:\Desktop\场景\treeseg\cache"


def load(tid):
    d = np.load(os.path.join(DATA, f"train_{tid}.npz"))
    pr = np.load(os.path.join(CACHE, f"probs_{tid}.npz"))["prob"].astype(np.float32)
    return d, pr


def tip_mask(d):
    fx = d["featx"]
    return (fx[:, 3] < 0.015) & (fx[:, 0] > 0.5) & (d["label3"] == 1)


def variants(name, coord, lab, fx, prob):
    v = {}
    v["raw"] = lab
    vote = P.knn_vote(coord, lab, 3)
    v["vote"] = vote
    v["old"] = P.postprocess(coord, lab, fx, 3)
    a = P.twig_rescue(coord, vote, fx, lin_th=0.45, rad_th=0.03)
    a = P.twig_rescue(coord, a, fx, lin_th=0.55, rad_th=0.02, eps=0.05, min_frac=0.35, min_size=15)
    a = P.remove_islands(coord, a)
    v["old_twigonly"] = P.knn_vote(coord, a, 3, k=8, sigma=0.02, iters=1, keep_w=1.0)
    b = P.remove_islands(coord, vote)
    b = P.tip_growth(coord, b, fx)
    v["old_growonly"] = P.knn_vote(coord, b, 3, k=8, sigma=0.02, iters=1, keep_w=1.0)
    cfgs = json.loads(name[4:]) if name.startswith("cfg:") else None
    tk, gk = (cfgs or {}).get("twig", {}), (cfgs or {}).get("grow", {})
    v["new"] = P.postprocess2(coord, lab, fx, 3, prob=prob, twig_kw=tk or None, grow_kw=gk or None)
    c = P.twig_rescue2(coord, vote, fx, prob=prob, **(tk or {}))
    c = P.remove_islands(coord, c)
    v["new_twigonly"] = P.knn_vote(coord, c, 3, k=8, sigma=0.02, iters=1, keep_w=1.0)
    e = P.tip_growth2(coord, vote, fx, prob=prob, **(gk or {}))
    v["new_growonly"] = P.knn_vote(coord, e, 3, k=8, sigma=0.02, iters=1, keep_w=1.0)
    return v


def score(y, lab):
    cm = np.zeros((3, 3), np.int64)
    for t in range(3):
        cm[t] = [(lab[y == t] == k).sum() for k in range(3)]
    return per_class(cm), cm


def main():
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else ["raw", "old", "new"]
    ids = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else VAL_IDS
    rows = {}
    for tid in ids:
        d, pr = load(tid)
        y = d["label3"].astype(np.int64)
        tm = tip_mask(d)
        for lab0 in [pr.argmax(1)]:
            vv = variants(",".join(names) if False else "all", d["coord"], lab0, d["featx"], pr)
            for nm in names:
                if nm not in vv:
                    continue
                m, cm = score(y, vv[nm])
                tip = float((vv[nm][tm] == 1).mean())
                flip = int(((vv[nm] == 1) & (y == 0)).sum())
                rows.setdefault(nm, []).append((tid, m["recall"][0], m["precision"][1], m["recall"][1],
                                                tip, m["miou"], m["oa"], flip))
    hdr = f"{'variant':16s} {'tree':>5s} {'leafRec':>7s} {'brPrec':>7s} {'brRec':>7s} {'tipRec':>7s} {'mIoU':>6s} {'OA':>6s} {'L->B':>8s}"
    print(hdr)
    for nm, r in rows.items():
        for t in r:
            print(f"{nm:16s} {t[0]:5d} {t[1]:7.3f} {t[2]:7.3f} {t[3]:7.3f} {t[4]:7.3f} {t[5]:6.3f} {t[6]:6.3f} {t[7]:8d}")
        a = np.array([x[1:] for x in r], float)
        print(f"{nm:16s} {'MEAN':>5s} {a[:,0].mean():7.3f} {a[:,1].mean():7.3f} {a[:,2].mean():7.3f} "
              f"{a[:,3].mean():7.3f} {a[:,4].mean():6.3f} {a[:,5].mean():6.3f} {a[:,6].mean():8.0f}")
        print()


if __name__ == "__main__":
    t0 = time.time()
    main()
    print("elapsed", round(time.time() - t0, 1))
