# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# 梁艺萌
# 树干 / 树枝 / 树叶 三维点云分离识别 —— 项目源代码
# 未经本人允许不得商用。third_party/ 内组件许可归其原作者，见 NOTICE.txt。
# ------------------------------------------------------------------
"""Metrics report: confusion matrices + recall bars from run logs / metrics json."""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r"E:\Desktop\场景\output"
NAMES3 = ["leaf", "branch", "trunk"]
NAMES2 = ["leaf", "wood"]

def cm_plot(cm, names, title, path):
    cm = np.array(cm, float)
    n = len(names)
    fig, ax = plt.subplots(figsize=(1.6 * n + 2.2, 1.6 * n + 1.6))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(n):
        for j in range(n):
            frac = cm[i, j] / max(cm[i].sum(), 1)
            ax.text(j, i, f"{int(cm[i,j])}\n{frac:.1%}", ha="center", va="center",
                    fontsize=9, color="black" if frac < 0.5 else "white")
    ax.set_xticks(range(n), names); ax.set_yticks(range(n), names)
    ax.set_xlabel("predicted"); ax.set_ylabel("ground truth")
    ax.set_title(title); fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)

def bars(res, names, path, title):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    x = np.arange(len(names)); w = 0.27
    ax.bar(x - w, res["recall"], w, label="recall")
    ax.bar(x, res["precision"], w, label="precision")
    ax.bar(x + w, res["f1"], w, label="F1")
    for xi, v in zip(x, res["recall"]):
        ax.text(xi - w, v + 0.015, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, names); ax.set_ylim(0, 1.12)
    ax.set_title(title); ax.legend(ncol=3, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)

def per_tree_table(per_tree, path):
    rows = []
    for tid, v in sorted(per_tree.items(), key=lambda x: int(x[0])):
        m3 = v.get("metrics3"); m2 = v.get("metrics2")
        rows.append((tid, v["n"],
                     f"{m2['recall'][0]:.3f}/{m2['recall'][1]:.3f}" if m2 else "-",
                     f"{m3['recall'][0]:.3f}/{m3['recall'][1]:.3f}/{m3['recall'][2]:.3f}" if m3 else "-",
                     f"{m3['miou']:.3f}" if m3 else "-", f"{m3['oa']:.3f}" if m3 else "-",
                     f"{v.get('tip_branch_recall') or 0:.3f}"))
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(rows) + 2))
    ax.axis("off")
    ax.table(cellText=rows, colLabels=["tree", "n_pts", "rec leaf/wood", "rec L/B/T", "mIoU", "OA", "tip-branch rec"],
             loc="center", cellLoc="center")
    ax.set_title("Per-tree metrics (held-out labeled trees)", pad=12)
    fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)

def main():
    mp = os.path.join(OUT, "metrics_val.json")
    if not os.path.exists(mp):
        print("no metrics json yet"); return
    with open(mp, encoding="utf-8") as f:
        m = json.load(f)
    r3 = m["aggregate3"]
    cm_plot(m["cm3"], NAMES3, "Task 3 (leaf/branch/trunk) — PTv3 ensemble + TTA + post\nheld-out labeled trees", os.path.join(OUT, "cm_task3.png"))
    bars(r3, NAMES3, os.path.join(OUT, "recall_task3.png"), "Per-class recall / precision / F1 (3-class)")
    r2 = m["aggregate2"]
    cm_plot(m["cm2"], NAMES2, "Task 2 (leaf/wood) — provided annotation", os.path.join(OUT, "cm_task2.png"))
    bars(r2, NAMES2, os.path.join(OUT, "recall_task2.png"), "Per-class recall / precision / F1 (leaf vs wood)")
    per_tree_table(m["per_tree"], os.path.join(OUT, "per_tree_table.png"))
    print("plots saved to", OUT)

if __name__ == "__main__":
    main()
