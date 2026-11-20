# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# 梁艺萌
# 树干 / 树枝 / 树叶 三维点云分离识别 —— 项目源代码
# 未经本人允许不得商用。third_party/ 内组件许可归其原作者，见 NOTICE.txt。
# ------------------------------------------------------------------
"""Sweep post-processing configs on cached probabilities (CPU only)."""
import os, sys, io, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, r"E:\Desktop\场景\treeseg")
import numpy as np
import post as P
from predict import per_class, VAL_IDS

DATA = r"E:\Desktop\场景\treeseg\data"
CACHE = r"E:\Desktop\场景\treeseg\cache"


def score(y, lab):
    cm = np.zeros((3, 3), np.int64)
    for t in range(3):
        cm[t] = [(lab[y == t] == k).sum() for k in range(3)]
    return per_class(cm), cm


def run(cfgs, ids, tag="v5"):
    res = {}
    for tid in ids:
        d = np.load(os.path.join(DATA, f"train_{tid}.npz"))
        pr = np.load(os.path.join(CACHE, f"probs_{tag}_{tid}.npz"))["prob"].astype(np.float32)
        coord, fx, y = d["coord"], d["featx"], d["label3"].astype(np.int64)
        tm = (fx[:, 3] < 0.015) & (fx[:, 0] > 0.5) & (y == 1)
        base = pr.argmax(1)
        for name, spec in cfgs.items():
            if spec.get("kind") == "raw":
                lab = base
            elif spec.get("kind") == "old":
                lab = P.postprocess(coord, base, fx, 3)
            else:
                lab = P.postprocess2(coord, base, fx, 3, prob=pr,
                                     twig_kw=spec.get("twig") or None,
                                     grow_kw=spec.get("grow") or None,
                                     iters=int(spec.get("iters", 1)))
            if spec.get("trunk_prior"):
                lab = P.trunk_prior(coord, lab, promote=bool(spec["trunk_prior"] == "promote"))
            if spec.get("trunk_finalize"):
                kw = spec["trunk_finalize"] if isinstance(spec["trunk_finalize"], dict) else {}
                lab = P.trunk_finalize(coord, lab, **kw)
            m, _ = score(y, lab)
            flip = int(((lab == 1) & (y == 0)).sum())
            lost = int(((lab != 1) & (y == 1)).sum())
            res.setdefault(name, []).append([m["recall"][0], m["precision"][1], m["recall"][1],
                                             float((lab[tm] == 1).mean()), m["miou"], m["oa"], flip, lost])
        print("tree", tid, "done", flush=True)
    print(f"{'variant':14s} {'leafRec':>7s} {'brPrec':>7s} {'brRec':>7s} {'tipRec':>7s} {'mIoU':>6s} {'OA':>6s} {'L->B':>7s} {'Bmiss':>7s}")
    for name, r in res.items():
        a = np.array(r, float)
        print(f"{name:14s} {a[:,0].mean():7.3f} {a[:,1].mean():7.3f} {a[:,2].mean():7.3f} {a[:,3].mean():7.3f} "
              f"{a[:,4].mean():6.3f} {a[:,5].mean():6.3f} {a[:,6].mean():7.0f} {a[:,7].mean():7.0f}")
    return res


if __name__ == "__main__":
    ids = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else VAL_IDS
    t0 = time.time()
    cfgs = json.load(io.open(sys.argv[1], encoding="utf-8"))
    run(cfgs, ids, sys.argv[3] if len(sys.argv) > 3 else "v5")
    print("elapsed", round(time.time() - t0, 1))
