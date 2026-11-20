# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# 梁艺萌
# 树干 / 树枝 / 树叶 三维点云分离识别 —— 项目源代码
# 未经本人允许不得商用。third_party/ 内组件许可归其原作者，见 NOTICE.txt。
# ------------------------------------------------------------------
"""Recompute per-tree stats/metrics from saved predictions (no GPU): counts, confidence,
trunk continuity (incl. wood-top reach), and GT metrics when available."""
import os, sys, io, json, glob
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, r"E:\Desktop\场景\treeseg")
import numpy as np
import post as P
from predict import per_class, VAL_IDS

DATA = r"E:\Desktop\场景\treeseg\data"
OUT = r"E:\Desktop\场景\output"


def refresh(which, tag):
    mp = os.path.join(OUT, f"metrics_{which}{('_' + tag) if tag else ''}.json")
    met = json.load(io.open(mp, encoding="utf-8")) if os.path.exists(mp) else {"which": which, "per_tree": {}}
    per = met.get("per_tree", {})
    src = "train" if which == "val" else which
    files = sorted(glob.glob(os.path.join(OUT, f"pred_{which}_*.npz")))
    for f in files:
        tid = int(os.path.basename(f).split("_")[-1][:-4])
        if which == "val" and tid not in VAL_IDS:
            continue
        if which == "test" and tag and str(tid) not in per:
            continue
        d = np.load(f)
        g = np.load(os.path.join(DATA, f"{src}_{tid}.npz"))
        coord, pred, prob, fx = d["coord"], d["pred"].astype(np.int64), d["prob"].astype(np.float32), g["featx"]
        counts = [int((pred == k).sum()) for k in range(3)]
        info = dict(n=int(len(pred)), leaf=counts[0], branch=counts[1], trunk=counts[2],
                    leaf_pct=round(counts[0] / len(pred), 4), branch_pct=round(counts[1] / len(pred), 4),
                    trunk_pct=round(counts[2] / len(pred), 4),
                    conf_mean=round(float(prob.max(1).mean()), 4))
        info["trunk"] = P.trunk_stats(coord, pred)
        if which == "val":
            y3 = g["label3"].astype(np.int64); y2 = g["label_woodleaf"].astype(np.int64)
            cm = np.zeros((3, 3), np.int64)
            for t in range(3):
                cm[t] = [(pred[y3 == t] == k).sum() for k in range(3)]
            cm2 = np.zeros((2, 2), np.int64)
            lm = pred.copy(); lm[lm > 0] = 1
            for t in range(2):
                cm2[t] = [(lm[y2 == t] == k).sum() for k in range(2)]
            tip = (fx[:, 3] < 0.015) & (fx[:, 0] > 0.5)
            info["metrics3"] = per_class(cm); info["metrics2"] = per_class(cm2)
            info["cm3"] = cm.tolist(); info["cm2"] = cm2.tolist()
            info["tip_branch_recall"] = round(float((pred[tip & (y3 == 1)] == 1).mean()), 4) if (tip & (y3 == 1)).any() else None
        per[str(tid)] = info
    met["per_tree"] = per
    if which == "val":
        agg = np.zeros((3, 3), np.int64); agg2 = np.zeros((2, 2), np.int64)
        for v in per.values():
            if "cm3" in v:
                agg += np.array(v["cm3"], np.int64); agg2 += np.array(v["cm2"], np.int64)
        met["aggregate3"] = per_class(agg); met["aggregate2"] = per_class(agg2)
        met["cm3"] = agg.tolist(); met["cm2"] = agg2.tolist()
        tips = [v.get("tip_branch_recall") for v in per.values() if v.get("tip_branch_recall") is not None]
        met["tip_branch_recall_mean"] = round(float(np.mean(tips)), 4) if tips else None
    with io.open(mp, "w", encoding="utf-8") as fh:
        json.dump(met, fh, ensure_ascii=False, indent=1)
    print("refreshed", mp, len(per), "trees", flush=True)


if __name__ == "__main__":
    refresh(sys.argv[1] if len(sys.argv) > 1 else "val", sys.argv[2] if len(sys.argv) > 2 else "")
