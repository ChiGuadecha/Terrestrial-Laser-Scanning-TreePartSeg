# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# 梁艺萌
# 树干 / 树枝 / 树叶 三维点云分离识别 —— 项目源代码
# 未经本人允许不得商用。third_party/ 内组件许可归其原作者，见 NOTICE.txt。
# ------------------------------------------------------------------
"""Final inference with per-tree metrics, model ensemble, multi-grid TTA, post-processing."""
import os, sys, json, glob, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, r"E:\Desktop\场景\ptv3_repo\src")
sys.path.insert(0, r"E:\Desktop\场景\treeseg")
import numpy as np
import torch
import point_transformer_v3 as ptv3
from point_transformer_v3.presets import PTV3
import post as P

DATA = r"E:\Desktop\场景\treeseg\data"
OUT = r"E:\Desktop\场景\output"
VAL_IDS = [3, 12, 21, 33, 43, 54]
NAMES3 = ["leaf", "branch", "trunk"]
NAMES2 = ["leaf", "wood"]

# tuned twig-recovery parameters (local CC + shape gating + capped tip growth)
TWIG_KW = dict(p_floor=0.02, lin_th=0.45, rad_th=0.030, attach_frac=0.22,
               attach_r=0.12, elong=1.7, max_size=30000)
GROW_KW = dict(p_floor=0.02, lin_th=0.45, rad_th=0.025, max_len=0.6, dir_cos=0.40, eps=0.04)

def load_model(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu")
    model = ptv3.PointTransformerV3(config=ptv3.Config.from_dict(ck["cfg"]), precision="fp32")
    model.to_empty(device="cpu"); model._init_weights(); model.attach_head(ck["ncls"])
    model.load_state_dict(ck["model"], strict=False)
    return model.eval().cuda(), ck

@torch.no_grad()
def infer(model, coord, nrm, fx, grid, flip):
    c = coord.copy()
    if flip:
        c[:, 0] *= -1; nn = nrm.copy(); nn[:, 0] *= -1
    else:
        nn = nrm
    point, inverse = ptv3.prepare(c, normal=nn, grid_size=grid, pick="first",
                                  center=False, feat=("coord", "normal"), device="cpu")
    inv = inverse.numpy()
    first = np.full(point.coord.shape[0], -1, np.int64)
    first[inv[::-1]] = np.arange(len(inv) - 1, -1, -1)
    feat = torch.cat([point.feat, point.coord[:, 2:3], torch.from_numpy(fx[first])], 1)
    data = dict(coord=point.coord.cuda(), grid_coord=point.grid_coord.cuda(),
                feat=feat.cuda(), offset=point.offset.cuda(), grid_size=grid)
    logits = model.seg_head(model(data).feat.float())
    return logits.softmax(1).cpu().numpy(), inv

def predict_tree(models3, model2, d, grids=None, post_proc=True):
    grids = grids or tuple(float(x) for x in os.environ.get("GRIDS", "0.02,0.015").split(","))
    coord, nrm, fx = d["coord"], d["normal"], d["featx"]
    n = len(coord)
    acc3 = np.zeros((n, 3), np.float32); acc2 = np.zeros((n, 2), np.float32); cnt = 0
    for g in grids:
        for flip in (False, True):
            for m in models3:
                p3, inv = infer(m, coord, nrm, fx, g, flip)
                acc3 += p3[inv]; cnt += 1
            if model2 is not None:
                p2, inv2 = infer(model2, coord, nrm, fx, g, flip)
                acc2 += p2[inv2]
    acc3 /= cnt
    if model2 is not None:
        acc2 /= cnt
        fused = acc3.copy()
        fused[:, 1] *= acc2[:, 1]; fused[:, 2] *= acc2[:, 1]
        fused[:, 0] = fused[:, 0] * 0.25 + acc2[:, 0] * 0.75
    else:
        fused = acc3
    lab = fused.argmax(1)
    if post_proc:
        lab = P.postprocess2(coord, lab, fx, 3, prob=fused, twig_kw=TWIG_KW, grow_kw=GROW_KW)
        lab = P.trunk_finalize(coord, lab)
    return lab, fused

def per_class(cm):
    n = len(cm)
    ious, recs, precs, f1s = [], [], [], []
    for t in range(n):
        tp = cm[t, t]; fp = cm[:, t].sum() - tp; fn = cm[t, :].sum() - tp
        iou = tp / (tp + fp + fn + 1e-9); rc = tp / (tp + fn + 1e-9)
        pr = tp / (tp + fp + 1e-9)
        ious.append(round(iou, 4)); recs.append(round(rc, 4)); precs.append(round(pr, 4))
        f1s.append(round(2 * pr * rc / (pr + rc + 1e-9), 4))
    return dict(iou=ious, recall=recs, precision=precs, f1=f1s,
                miou=round(float(np.mean(ious)), 4),
                macro_f1=round(float(np.mean(f1s)), 4),
                oa=round(float(np.trace(cm) / max(cm.sum(), 1)), 4))

def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "val"
    runs3 = (sys.argv[2].split(",") if len(sys.argv) > 2 else ["ptv3_task3_v5", "ptv3_task3_v6f"])
    run2 = sys.argv[3] if len(sys.argv) > 3 else "ptv3_task2_v3"
    models3 = [load_model(os.path.join(r"E:\Desktop\场景\treeseg\runs", r, "best.pth"))[0] for r in runs3]
    try:
        model2 = load_model(os.path.join(r"E:\Desktop\场景\treeseg\runs", run2, "best.pth"))[0]
    except FileNotFoundError:
        print("task2 model missing, no fusion"); model2 = None
    ids = VAL_IDS if which == "val" else sorted(int(os.path.basename(f)[5:-4]) for f in glob.glob(os.path.join(DATA, "test_*.npz")))
    tag = sys.argv[5] if len(sys.argv) > 5 else ""
    if len(sys.argv) > 4 and sys.argv[4] != "all":
        ids = [int(x) for x in sys.argv[4].split(",")]
    mpath = os.path.join(OUT, f"metrics_{which}{('_' + tag) if tag else ''}.json")
    agg = np.zeros((3, 3), np.int64); agg2 = np.zeros((2, 2), np.int64)
    per_tree = {}
    if os.path.exists(mpath):
        try:
            with open(mpath, encoding="utf-8") as f:
                per_tree = dict(json.load(f).get("per_tree") or {})
        except Exception:
            per_tree = {}
    t0 = time.time()
    for tid in ids:
        src = "train" if which == "val" else which
        d = np.load(os.path.join(DATA, f"{src}_{tid}.npz"))
        lab, fused = predict_tree(models3, model2, d)
        np.savez_compressed(os.path.join(OUT, f"pred_{which}_{tid}.npz"), coord=d["coord"],
                            normal=d["normal"], featx=d["featx"], pred=lab.astype(np.uint8),
                            prob=fused.astype(np.float16), shift=d["shift"])
        counts = [int((lab == k).sum()) for k in range(3)]
        conf = float(np.mean(fused.max(1)[lab == 1])) if counts[1] else None
        info = dict(n=int(len(lab)), leaf=counts[0], branch=counts[1], trunk=counts[2],
                    leaf_pct=round(counts[0]/len(lab), 4), branch_pct=round(counts[1]/len(lab), 4),
                    trunk_pct=round(counts[2]/len(lab), 4),
                    conf_mean=round(float(fused.max(1).mean()), 4))
        info["trunk"] = P.trunk_stats(d["coord"], lab)
        if which == "val":
            y3 = d["label3"].astype(np.int64); y2 = d["label_woodleaf"].astype(np.int64)
            cm = np.zeros((3, 3), np.int64)
            for t in range(3):
                cm[t] = [(lab[y3 == t] == k).sum() for k in range(3)]
            agg += cm
            lm = lab.copy(); lm[lm > 0] = 1
            cm2 = np.zeros((2, 2), np.int64)
            for t in range(2):
                cm2[t] = [(lm[y2 == t] == k).sum() for k in range(2)]
            agg2 += cm2
            tip = (d["featx"][:, 3] < 0.015) & (d["featx"][:, 0] > 0.5)
            info["metrics3"] = per_class(cm)
            info["metrics2"] = per_class(cm2)
            info["cm3"] = cm.tolist(); info["cm2"] = cm2.tolist()
            info["tip_branch_recall"] = round(float((lab[tip & (y3==1)] == 1).mean()), 4) if (tip & (y3==1)).any() else None
        per_tree[str(tid)] = info
        print("tree", tid, json.dumps(info.get("metrics3", info), ensure_ascii=False), flush=True)
    res = dict(which=which, runs3=runs3, run2=run2, per_tree=per_tree)
    if which == "val":
        agg = np.zeros((3, 3), np.int64); agg2 = np.zeros((2, 2), np.int64)
        for v in per_tree.values():
            if "cm3" in v:
                agg += np.array(v["cm3"], np.int64); agg2 += np.array(v["cm2"], np.int64)
        res["aggregate3"] = per_class(agg); res["aggregate2"] = per_class(agg2)
        res["cm3"] = agg.tolist(); res["cm2"] = agg2.tolist()
        all_tip_br = [v.get("tip_branch_recall") for v in per_tree.values() if v.get("tip_branch_recall") is not None]
        res["tip_branch_recall_mean"] = round(float(np.mean(all_tip_br)), 4) if all_tip_br else None
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    if which == "val":
        print(json.dumps(dict(aggregate3=res["aggregate3"], aggregate2=res["aggregate2"],
                              tip=res.get("tip_branch_recall_mean")), ensure_ascii=False, indent=1))
    print("elapsed", round(time.time() - t0, 1))

if __name__ == "__main__":
    main()
