# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# 梁艺萌
# 树干 / 树枝 / 树叶 三维点云分离识别 —— 项目源代码
# 未经本人允许不得商用。third_party/ 内组件许可归其原作者，见 NOTICE.txt。
# ------------------------------------------------------------------
"""Transfer-learning variants: ptv3_pc (Pointcept init) and sonata (linear probe)."""
import os, sys, json, time, argparse, random
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, r"E:\Desktop\场景\ptv3_repo\src")
sys.path.insert(0, r"E:\Desktop\场景\treeseg")
import numpy as np
import torch
import point_transformer_v3 as ptv3
from point_transformer_v3.presets import PTV3
from point_transformer_v3.convert import convert_state_dict, load_pc_checkpoint
from point_transformer_v3 import upcast
import train as T
import post as P

W = r"E:\Desktop\场景\treeseg\weights\hf"

def find_file(name):
    for root, _, fs in os.walk(W):
        for f in fs:
            if f == name:
                return os.path.join(root, f)
    raise FileNotFoundError(name)

def build_ptv3_pc(ncls, in_ch):
    path = find_file("model_best.pth")
    ck = load_pc_checkpoint(path)
    state = ck["model"] if "model" in ck else ck
    state = {k: v for k, v in state.items() if "embedding" not in k and "seg_head" not in k and "cls_head" not in k}
    cfg = dict(PTV3); cfg["in_channels"] = in_ch
    model = ptv3.PointTransformerV3(config=ptv3.Config.from_dict(cfg), precision="auto")
    model.to_empty(device="cpu"); model._init_weights()
    model.attach_head(ncls)
    model.seg_head.reset_parameters()
    target = dict(model.state_dict())
    tgt_body = {k: v for k, v in target.items() if not k.startswith(("embedding", "seg_head"))}
    new, dropped = convert_state_dict(state, tgt_body)
    merged = dict(target); merged.update(new)
    model.load_state_dict(merged, strict=False)
    print("ptv3_pc init ok, dropped:", len(dropped))
    return model.cuda(), cfg

def build_sonata(ncls):
    path = find_file("sonata.pth")
    model = ptv3.load(path, device="cuda", head=False, warmup=False, precision="auto")
    head = torch.nn.Linear(1232, ncls).cuda()
    return model, head

def to_point9(coord, nrm, grid, pick, gen=None):
    color = np.zeros_like(coord)
    point, inverse = ptv3.prepare(coord, color=color, normal=nrm, grid_size=grid, pick=pick,
                                  center=False, feat=("coord", "color", "normal"),
                                  generator=gen, device="cpu")
    return point, inverse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["ptv3_pc", "sonata"])
    ap.add_argument("--task", type=int, default=3, choices=[2, 3])
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--grid", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    lab = "label3" if args.task == 3 else "label_woodleaf"
    ncls = args.task
    run_dir = os.path.join(T.RUNS, args.name); os.makedirs(run_dir, exist_ok=True)
    tr_ids = sorted(set(int(f[6:-4]) for f in os.listdir(T.DATA) if f.startswith("train_")) - set(T.VAL_IDS))
    rng = np.random.default_rng(7); random.seed(7); torch.manual_seed(7)
    freq = np.zeros(ncls)
    for tid in tr_ids:
        y = T.load_tree("train", tid)[lab]
        for c in range(ncls):
            freq[c] += (y == c).sum()
    w = (freq.sum() / (ncls * np.maximum(freq, 1))) ** 1.0
    w = (w / w.mean()).astype(np.float32)
    lr = args.lr or (2e-4 if args.arch == "sonata" else 5e-4)
    head_lr = lr * 10
    if args.arch == "ptv3_pc":
        model, cfg = build_ptv3_pc(ncls, 11)
        head = model.seg_head
        params = [
            {"params": [v for k, v in model.named_parameters() if "seg_head" not in k], "lr": lr},
            {"params": head.parameters(), "lr": head_lr}]
    else:
        model, head = build_sonata(ncls)
        cfg = {"arch": "sonata", "in_channels": 9}
        params = [
            {"params": model.parameters(), "lr": lr},
            {"params": head.parameters(), "lr": head_lr}]
    ema = T.EMA(model, 0.9995); ema_h = T.EMA(head, 0.9995)
    crit = T.Loss(w, lam_l=1.0)
    opt = torch.optim.AdamW(params, weight_decay=0.02)
    ep = 3 if args.smoke else args.epochs
    steps = max(1, len(tr_ids)) * ep
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr, head_lr], total_steps=steps,
                                                pct_start=0.1, div_factor=10)
    gen = torch.Generator(); t0 = time.time(); gstep = 0
    best = -1; logf = open(os.path.join(run_dir, "log.jsonl"), "a", encoding="utf-8")

    def fwd(coord, nrm, fx, grid, pick, flip=False, aug=False):
        c = coord.copy()
        if flip:
            c[:, 0] *= -1; nn = nrm.copy(); nn[:, 0] *= -1
        else:
            nn = nrm
        if args.arch == "ptv3_pc":
            point, inverse = ptv3.prepare(c, normal=nn, grid_size=grid, pick=pick, center=False,
                                          feat=("coord", "normal"), generator=gen if aug else None,
                                          device="cpu")
            inv = inverse.numpy()
            first = np.full(point.coord.shape[0], -1, np.int64)
            first[inv[::-1]] = np.arange(len(inv) - 1, -1, -1)
            feat = torch.cat([point.feat, point.coord[:, 2:3], torch.from_numpy(fx[first])], 1)
            data = dict(coord=point.coord, grid_coord=point.grid_coord, feat=feat,
                        offset=point.offset, grid_size=grid)
        else:
            point, inverse = to_point9(c, nn, grid, pick, gen if aug else None)
            data = dict(coord=point.coord, grid_coord=point.grid_coord, feat=point.feat,
                        offset=point.offset, grid_size=grid)
        data = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in data.items()}
        if args.arch == "ptv3_pc":
            return model.seg_head(model(data).feat.float()), inverse
        return head(upcast(model(data)).feat.float()), inverse

    for e in range(ep):
        model.train(); head.train()
        order = tr_ids.copy(); random.shuffle(order)
        tot = 0.0
        for tid in order:
            d = T.load_tree("train", tid)
            coord, nrm, fx, y = d["coord"], d["normal"], d["featx"], d[lab].astype(np.int64)
            coord, nrm = T.augment(coord, nrm, rng)
            if rng.random() < 0.5:
                coord = coord.copy(); coord[:, 0] *= -1; nrm = nrm.copy(); nrm[:, 0] *= -1
            if len(coord) > 450000:
                sel = rng.choice(len(coord), 450000, replace=False)
                coord, nrm, fx, y = coord[sel], nrm[sel], fx[sel], y[sel]
            logits, inverse = fwd(coord, nrm, fx, args.grid, "random", aug=True)
            m = logits.shape[0]
            yk = torch.from_numpy(T.keep_labels(y, inverse.numpy(), m)).cuda()
            loss = crit(logits, yk)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 5.0)
            opt.step(); sched.step(); gstep += 1
            ema.update(model); ema_h.update(head)
            tot += float(loss.detach())
        if (e + 1) % (1 if args.smoke else 4) == 0 or e == ep - 1:
            m_state = {k: v.clone() for k, v in model.state_dict().items()}
            h_state = {k: v.clone() for k, v in head.state_dict().items()}
            model.load_state_dict(ema.state(model)); head.load_state_dict(ema_h.state(head))
            model.precision = "fp32"
            model.eval(); head.eval()
            rec = {"epoch": e + 1, "loss": tot / len(order), "sec": round(time.time() - t0, 1)}
            agg = np.zeros((ncls, ncls), np.int64)
            with torch.no_grad():
                for tid in T.VAL_IDS:
                    d = T.load_tree("val", tid)
                    coord, nrm, fx, y = d["coord"], d["normal"], d["featx"], d[lab].astype(np.int64)
                    pr, inv = fwd(coord, nrm, fx, args.grid, "first")
                    labp = pr.argmax(1).cpu().numpy()[inv.numpy()]
                    labp = P.postprocess(coord, labp, fx, ncls, trunk_branch=(ncls == 3))
                    for t in range(ncls):
                        agg[t] = [(labp[y == t] == k).sum() for k in range(ncls)]
            ious, recs = [], []
            for t in range(ncls):
                tp = agg[t, t]; fp = agg[:, t].sum() - tp; fn = agg[t, :].sum() - tp
                ious.append(tp / (tp + fp + fn + 1e-9)); recs.append(tp / (tp + fn + 1e-9))
            rec.update(iou=[round(x, 4) for x in ious], recall=[round(x, 4) for x in recs],
                       miou=round(float(np.mean(ious)), 4),
                       oa=round(float(np.trace(agg) / agg.sum()), 4))
            print(json.dumps(rec), flush=True); logf.write(json.dumps(rec) + "\n"); logf.flush()
            if rec["miou"] > best:
                best = rec["miou"]
                torch.save({"model": ema.state(model), "head": ema_h.state(head), "arch": args.arch,
                            "cfg": cfg, "ncls": ncls, "grid": args.grid, "miou": best},
                           os.path.join(run_dir, "best.pth"))
            model.load_state_dict(m_state); head.load_state_dict(h_state)
            model.precision = "auto"; model.train(); head.train()
    print("done best miou", best, round(time.time() - t0, 1), "s")

if __name__ == "__main__":
    main()
