# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# 梁艺萌
# 树干 / 树枝 / 树叶 三维点云分离识别 —— 项目源代码
# 未经本人允许不得商用。third_party/ 内组件许可归其原作者，见 NOTICE.txt。
# ------------------------------------------------------------------
"""PTv3 training for tree trunk / branch / leaf segmentation.

task2: leaf vs wood (provided annotations)
task3: leaf vs branch vs trunk (pseudo-labeled wood)
"""
import os, sys, json, time, argparse, random
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, r"E:\Desktop\场景\ptv3_repo\src")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
import numpy as np
import torch
import point_transformer_v3 as ptv3
import post as P
from point_transformer_v3.presets import PTV3

DATA = r"E:\Desktop\场景\treeseg\data"
RUNS = r"E:\Desktop\场景\treeseg\runs"
VAL_IDS = [3, 12, 21, 33, 43, 54]
CLASSES3 = ["leaf", "branch", "trunk"]
CLASSES2 = ["leaf", "wood"]
NAMES = {2: CLASSES2, 3: CLASSES3}

def load_tree(which, tid):
    if which == "val":
        which = "train"
    return np.load(os.path.join(DATA, f"{which}_{tid}.npz"))

def augment(coord, nrm, rng):
    a = rng.uniform(0, 2 * np.pi)
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)
    coord = coord @ R.T; nrm = nrm @ R.T
    if rng.random() < 0.5:
        coord[:, 0] *= -1; nrm[:, 0] *= -1
    if rng.random() < 0.5:
        coord[:, 1] *= -1; nrm[:, 1] *= -1
    sc = rng.uniform(0.95, 1.05)
    coord *= sc
    if rng.random() < 0.8:
        coord += rng.normal(0, 0.004, coord.shape).astype(np.float32)
    return coord.astype(np.float32), nrm.astype(np.float32)

def to_point(coord, nrm, fx, grid, pick, device, gen=None):
    point, inverse = ptv3.prepare(coord, normal=nrm, grid_size=grid, pick=pick,
                                  center=False, feat=("coord", "normal"),
                                  generator=gen, device="cpu")
    inv = inverse.numpy()
    first = np.full(point.coord.shape[0], -1, np.int64)
    first[inv[::-1]] = np.arange(len(inv) - 1, -1, -1)
    fxk = fx[first]
    zrel = point.coord[:, 2:3]
    feat = torch.cat([point.feat, zrel, torch.from_numpy(fxk)], 1)
    return {"point": point, "feat": feat, "inverse": inverse}

def keep_labels(label, inverse, m):
    k = label.max() + 1
    cnt = np.stack([np.bincount(inverse, weights=(label == c).astype(np.float64), minlength=m)
                    for c in range(k)])
    return cnt.argmax(0).astype(np.int64)

# ---- losses ----
def lovasz_softmax(probe, cls, classes="present", ignore=255, lw=None):
    # probe (N,C) probs in [0,1], cls (N,) long
    pos = cls != ignore
    probe, cls = probe[pos], cls[pos]
    if len(cls) == 0:
        return probe.sum() * 0.0
    errors = []
    cl_list = range(probe.shape[1]) if classes == "all" else torch.unique(cls).tolist()
    for c in cl_list:
        fg = (cls == c).float()
        err = (fg - probe[:, c]).abs()
        err_sorted, idx = torch.sort(err, descending=True)
        fg_sorted = fg[idx]
        denom = fg_sorted.sum() + ((1 - fg_sorted) * err_sorted).sum()
        iou = fg_sorted.sum() / (denom + 1e-7)
        errors.append(1 - iou)
    errors = torch.stack(errors)
    if lw is not None:
        return (errors * lw).sum() / lw.sum()
    return errors.mean()

class Loss:
    def __init__(self, w, lam_l=0.5, lw=None):
        self.ce = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32).cuda())
        self.lam = lam_l
        self.lw = torch.tensor(lw, dtype=torch.float32).cuda() if lw is not None else None
    def __call__(self, logits, y):
        return self.ce(logits, y) + self.lam * lovasz_softmax(logits.softmax(1), y, classes="all", lw=self.lw)

# ---- metrics ----
def metrics(y, p, ncls, ignore=-1):
    y, p = np.asarray(y), np.asarray(p)
    m = y >= 0
    y, p = y[m], p[m]
    cm = np.zeros((ncls, ncls), np.int64)
    for t in range(ncls):
        cm[t] = [(p[y == t] == k).sum() for k in range(ncls)]
    out = {"cm": cm.tolist(), "n": int(m.sum())}
    ious, precs, recs, f1s = [], [], [], []
    for t in range(ncls):
        tp = cm[t, t]; fp = cm[:, t].sum() - tp; fn = cm[t, :].sum() - tp
        iu = tp / (tp + fp + fn + 1e-9); pr = tp / (tp + fp + 1e-9); rc = tp / (tp + fn + 1e-9)
        ious.append(iu); precs.append(pr); recs.append(rc); f1s.append(2 * pr * rc / (pr + rc + 1e-9))
    out.update(iou=ious, precision=precs, recall=recs, f1=f1s,
               miou=float(np.mean(ious)), oa=float(np.trace(cm) / (cm.sum() + 1e-9)))
    return out

def merge_wood(p3):
    p2 = p3.copy(); p2[p3 > 0] = 1
    return p2

# ---- EMA ----
class EMA:
    def __init__(self, model, decay=0.999):
        self.d = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.named_parameters()}
    def update(self, model):
        for k, v in model.named_parameters():
            if k in self.shadow:
                self.shadow[k].mul_(self.d).add_(v.detach().float(), alpha=1 - self.d)
    def state(self, model):
        s = {k: v.clone() for k, v in model.state_dict().items()}
        s.update(self.shadow)
        return s

# ---- main ----
def build_model(ncls, cfg, init=None):
    model = ptv3.PointTransformerV3(config=ptv3.Config.from_dict(cfg), precision="auto")
    model.to_empty(device="cpu"); model._init_weights()
    model.attach_head(ncls)
    model.seg_head.reset_parameters()
    if init is not None:
        missing = model.load_state_dict(init, strict=False)
        print("init load:", missing)
    return model.cuda()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=3, choices=[2, 3])
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--grid", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--name", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--oversample", type=float, default=0.6)
    ap.add_argument("--lw-branch", type=float, default=2.0)
    ap.add_argument("--init", default=None, help="backbone state_dict npz/pth to warm start")
    args = ap.parse_args()
    lab = "label3" if args.task == 3 else "label_woodleaf"
    ncls = args.task
    run = args.name or f"task{args.task}_g{args.grid}_lr{args.lr}"
    run_dir = os.path.join(RUNS, run); os.makedirs(run_dir, exist_ok=True)
    tr_ids = sorted(set(int(f[6:-4]) for f in os.listdir(DATA) if f.startswith("train_")) - set(VAL_IDS))
    print("train:", tr_ids, "val:", VAL_IDS)
    rng = np.random.default_rng(0); random.seed(0); torch.manual_seed(0)
    # class weights from training frequency
    freq = np.zeros(ncls)
    for tid in tr_ids:
        y = load_tree("train", tid)[lab]
        for c in range(ncls):
            freq[c] += (y == c).sum()
    exp = 1.0 if ncls == 2 else 0.5
    w = (freq.sum() / (ncls * np.maximum(freq, 1))) ** exp
    w = (w / w.mean()).astype(np.float32)
    print("class freq:", freq, "weights:", w)
    cfg = dict(PTV3); cfg["in_channels"] = 11
    init = torch.load(args.init, map_location="cpu") if args.init else None
    model = build_model(ncls, cfg, init)
    ema = EMA(model)
    crit = Loss(w, lam_l=1.0 if ncls == 2 else 0.5,
                  lw=None if ncls == 2 else [1.0, args.lw_branch, 1.0])
    opt = torch.optim.AdamW([
        {"params": [v for k, v in model.named_parameters() if "seg_head" not in k], "lr": args.lr},
        {"params": model.seg_head.parameters(), "lr": args.lr * 10}], weight_decay=0.02)
    ep = 4 if args.smoke else args.epochs
    steps = max(1, len(tr_ids)) * ep
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.lr * 10],
                                                total_steps=steps, pct_start=0.1, div_factor=10)
    logf = open(os.path.join(run_dir, "log.jsonl"), "a", encoding="utf-8")
    best = {"miou3": -1, "miou2": -1}
    gen = torch.Generator()
    gstep = 0
    t0 = time.time()
    for e in range(ep):
        model.train()
        order = tr_ids.copy(); random.shuffle(order)
        tot = 0.0
        for tid in order:
            d = load_tree("train", tid)
            coord, nrm, y = d["coord"], d["normal"], d[lab].astype(np.int64)
            fx = d["featx"]
            coord, nrm = augment(coord, nrm, rng)
            if args.task == 3 and args.oversample > 0:
                bi = np.nonzero(y == 1)[0]
                if len(bi):
                    extra = rng.choice(bi, int(args.oversample * len(bi)), replace=True)
                    ec = coord[extra] + rng.normal(0, 0.01, (len(extra), 3)).astype(np.float32)
                    coord = np.concatenate([coord, ec]); nrm = np.concatenate([nrm, nrm[extra]])
                    fx = np.concatenate([d["featx"], d["featx"][extra]])
                    y = np.concatenate([y, y[extra]])
            if len(coord) > 600000:
                sel = rng.choice(len(coord), 600000, replace=False)
                coord, nrm, y = coord[sel], nrm[sel], y[sel]
                fx = d["featx"] if len(fx) == len(d["coord"]) else fx[sel]
            s = to_point(coord, nrm, fx, args.grid, "random", "cpu", gen)
            m = s["point"].coord.shape[0]
            yk = torch.from_numpy(keep_labels(y, s["inverse"].numpy(), m)).cuda()
            data = ptv3.Point(dict(coord=s["point"].coord, grid_coord=s["point"].grid_coord,
                                   feat=s["feat"], offset=s["point"].offset, grid_size=args.grid))
            data = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in dict(data).items()}
            logits = model.seg_head(model(data).feat.float())
            loss = crit(logits.float(), yk)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step(); gstep += 1
            ema.update(model)
            tot += float(loss.detach())
        if (e + 1) % (1 if args.smoke else 4) == 0 or e == ep - 1:
            live = {k: v.clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema.state(model))
            rec = {"epoch": e + 1, "loss": tot / len(order), "sec": round(time.time() - t0, 1)}
            model.eval(); model.precision = "fp32"   # eval in fp32: fp16 autocast distorts BN
            with torch.no_grad():
                agg3 = np.zeros((ncls, ncls), np.int64) if ncls == 3 else None
                agg2 = np.zeros((2, 2), np.int64)
                agg3f = np.zeros((ncls, ncls), np.int64) if ncls == 3 else None
                for tid in VAL_IDS:
                    d = load_tree("val", tid)
                    coord, nrm, y3 = d["coord"], d["normal"], d["label3"].astype(np.int64)
                    y2 = d["label_woodleaf"].astype(np.int64)
                    s = to_point(coord, nrm, d["featx"], args.grid, "first", "cpu")
                    m = s["point"].coord.shape[0]
                    yk3 = keep_labels(y3, s["inverse"].numpy(), m)
                    data = ptv3.Point(dict(coord=s["point"].coord, grid_coord=s["point"].grid_coord,
                                           feat=s["feat"], offset=s["point"].offset, grid_size=args.grid))
                    data = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in dict(data).items()}
                    pr = model.seg_head(model(data).feat.float()).argmax(1).cpu().numpy()
                    if ncls == 3:
                        cm = np.zeros((3, 3), np.int64)
                        for t in range(3):
                            cm[t] = [(pr[yk3 == t] == k).sum() for k in range(3)]
                        agg3 += cm
                    pr_full = pr[s["inverse"].numpy()]
                    if ncls == 3:
                        pf = P.postprocess(coord, pr_full, d["featx"], 3)
                        cmf = np.zeros((3, 3), np.int64)
                        for t in range(3):
                            cmf[t] = [(pf[y3 == t] == k).sum() for k in range(3)]
                        agg3f += cmf
                    pr2 = merge_wood(pr) if ncls == 3 else pr
                    yk2 = keep_labels(merge_wood(y2) if ncls == 3 else y2, s["inverse"].numpy(), m)
                    for t in range(2):
                        agg2[t] = [(pr2[yk2 == t] == k).sum() for k in range(2)]
                def rep(cm, names):
                    ious, recs = [], []
                    for t in range(len(names)):
                        tp = cm[t, t]; fp = cm[:, t].sum() - tp; fn = cm[t, :].sum() - tp
                        ious.append(tp / (tp + fp + fn + 1e-9)); recs.append(tp / (tp + fn + 1e-9))
                    return {"iou": [round(x, 4) for x in ious], "recall": [round(x, 4) for x in recs],
                            "miou": round(float(np.mean(ious)), 4),
                            "oa": round(float(np.trace(cm) / cm.sum()), 4)}
                if ncls == 3:
                    rec["val3"] = rep(agg3, CLASSES3)
                    rec["val3p"] = rep(agg3f, CLASSES3)
                    best["miou3"] = max(best["miou3"], rec["val3"]["miou"])
                    best["miou3p"] = max(best.get("miou3p", -1), rec["val3p"]["miou"])
                rec["val2"] = rep(agg2, CLASSES2)
                best["miou2"] = max(best["miou2"], rec["val2"]["miou"])
            rec["best"] = best
            print(json.dumps(rec), flush=True)
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            model.load_state_dict(live)
            model.precision = "auto"; model.train()
            torch.save({"model": ema.state(model), "cfg": dict(cfg), "ncls": ncls, "grid": args.grid,
                        "best": best}, os.path.join(run_dir, "last.pth"))
            key = "miou3" if ncls == 3 else "miou2"
            cur = rec["val3"]["miou"] if (ncls == 3) else rec["val2"]["miou"]
            if cur >= best[key] - 1e-12:
                torch.save({"model": ema.state(model), "cfg": dict(cfg), "ncls": ncls, "grid": args.grid,
                            "best": best}, os.path.join(run_dir, "best.pth"))
    print("done", best, round(time.time() - t0, 1), "s")

if __name__ == "__main__":
    main()
