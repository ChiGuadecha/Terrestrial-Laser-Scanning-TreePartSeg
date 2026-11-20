# ------------------------------------------------------------------
# 梁艺萌
# 树干 / 树枝 / 树叶 三维点云分离识别 —— 项目源代码
# 未经本人允许不得商用。third_party/ 内组件许可归其原作者，见 NOTICE.txt。
# ------------------------------------------------------------------
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import hf_hub_download
for repo, f in [("facebook/sonata", "sonata.pth"),
                ("Pointcept/PointTransformerV3", "scannet-semseg-pt-v3m1-0-base/model/model_best.pth")]:
    try:
        p = hf_hub_download(repo, f, cache_dir=r"E:\Desktop\场景\treeseg\weights\hf")
        print("saved:", p, round(os.path.getsize(p)/1e6,1), "MB", flush=True)
    except Exception as e:
        print("FAIL", repo, f, repr(e)[:200], flush=True)
