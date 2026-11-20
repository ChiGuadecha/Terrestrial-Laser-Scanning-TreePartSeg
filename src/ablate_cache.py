# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# 梁艺萌
# 树干 / 树枝 / 树叶 三维点云分离识别 —— 项目源代码
# 未经本人允许不得商用。third_party/ 内组件许可归其原作者，见 NOTICE.txt。
# ------------------------------------------------------------------
"""Cache fused probabilities for val trees so post-processing variants can be tuned on CPU."""
import os, sys, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, r"E:\Desktop\场景\treeseg")
import numpy as np
import predict as PR

DATA = PR.DATA
CACHE = r"E:\Desktop\场景\treeseg\cache"
os.makedirs(CACHE, exist_ok=True)
IDS = PR.VAL_IDS


def build(runs3=("ptv3_task3_v5",), run2="ptv3_task2_v3", tag="v5"):
    models3 = [PR.load_model(os.path.join(r"E:\Desktop\场景\treeseg\runs", r, "best.pth"))[0] for r in runs3]
    m2 = PR.load_model(os.path.join(r"E:\Desktop\场景\treeseg\runs", run2, "best.pth"))[0]
    for tid in IDS:
        out = os.path.join(CACHE, f"probs_{tag}_{tid}.npz")
        if os.path.exists(out):
            print("skip", tid, flush=True); continue
        d = np.load(os.path.join(DATA, f"train_{tid}.npz"))
        t0 = time.time()
        _, fused = PR.predict_tree(models3, m2, d, post_proc=False)
        np.savez_compressed(out, prob=fused.astype(np.float16))
        print("cached", tid, len(fused), round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "v5"
    runs = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("ptv3_task3_" + tag,)
    build(runs3=runs, tag=tag)
