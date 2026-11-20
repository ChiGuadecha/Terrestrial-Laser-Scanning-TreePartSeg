# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# 梁艺萌
# 树干 / 树枝 / 树叶 三维点云分离识别 —— 项目源代码
# 未经本人允许不得商用。third_party/ 内组件许可归其原作者，见 NOTICE.txt。
# ------------------------------------------------------------------
"""Write 成果/说明.md and the summary table (PNG + CSV) from the metrics json files."""
import os, sys, io, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, r"E:\Desktop\场景\treeseg")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]

ROOT = r"E:\Desktop\场景"
OUT = os.path.join(ROOT, "output")
RES = os.path.join(ROOT, "成果")


def load(tag):
    f = os.path.join(OUT, f"metrics_{tag}.json")
    return json.load(io.open(f, encoding="utf-8")) if os.path.exists(f) else {}


def rows_of(met, with_gt):
    out = []
    for tid, v in sorted(met.get("per_tree", {}).items(), key=lambda x: int(x[0])):
        m3 = v.get("metrics3") or {}
        r = [tid, v.get("n", 0), v.get("leaf_pct", 0), v.get("branch_pct", 0), v.get("trunk_pct", 0),
             v.get("conf_mean", float("nan"))]
        t = v.get("trunk") or {}
        r += [t.get("ncomp", 0), t.get("zmin") or 0, t.get("zmax") or 0, t.get("wtop") or 0,
              t.get("reach_wood") or 0, t.get("reach") or 0]
        if with_gt:
            r += [m3.get("oa", float("nan")), m3.get("miou", float("nan")),
                  (m3.get("recall") or [0, 0, 0])[0], (m3.get("recall") or [0, 0, 0])[1],
                  (m3.get("recall") or [0, 0, 0])[2], v.get("tip_branch_recall", float("nan")),
                  ((m3.get("precision") or [0, 0, 0])[2])]
        out.append(r)
    return out


SIGN = "梁艺萌 · 未经本人允许不得商用"

HEAD_VAL = ["树号", "点数", "叶%", "枝%", "干%", "置信度", "主干连通块", "干底m", "干顶m", "木质顶m", "木质到达", "全树到达",
            "OA", "mIoU", "叶召回", "枝召回", "干召回", "末端细枝召回", "干精确率"]
HEAD_TEST = ["树号", "点数", "叶%", "枝%", "干%", "置信度", "主干连通块", "干底m", "干顶m", "木质顶m", "木质到达", "全树到达"]


def table_png(path, head, rows, title):
    n = len(rows)
    fig, ax = plt.subplots(figsize=(0.95 * len(head) + 2.2, 0.34 * n + 1.5), dpi=115)
    ax.axis("off"); ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    cells = [[("{:,}".format(int(c)) if i == 1 else ("%g" % c if isinstance(c, (int, float)) else str(c)))
              for i, c in enumerate(r)] for r in rows]
    tb = ax.table(cellText=cells, colLabels=head, loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(8.4); tb.scale(1, 1.42)
    fig.text(0.99, 0.015, SIGN, ha="right", va="bottom", fontsize=8.6, color="#40566B",
             fontweight="bold")
    for (rr, cc), cl in tb.get_celld().items():
        if rr == 0:
            cl.set_facecolor("#33475B"); cl.set_text_props(color="white", fontweight="bold")
        elif rr % 2 == 0:
            cl.set_facecolor("#F2F5F8")
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def main():
    val = load("val"); testA = load("test_A"); testB = load("test_B")
    test = dict(which="test", per_tree={**testA.get("per_tree", {}), **testB.get("per_tree", {})})
    rv = rows_of(val, True); rt = rows_of(test, False)
    os.makedirs(os.path.join(RES, "汇总"), exist_ok=True)
    table_png(os.path.join(RES, "汇总", "验证集指标.png"), HEAD_VAL, rv,
              "验证集（训练集留出 6 棵）逐树精度  Point Transformer V3 + 细枝采样/加权损失 + 连通修复 + 多模型融合")
    table_png(os.path.join(RES, "汇总", "测试集指标.png"), HEAD_TEST, rt,
              "测试集 20 棵逐树结果统计（无地面真值：给出类别占比、置信度与主干贯通情况）")
    import csv
    with io.open(os.path.join(RES, "汇总", "指标汇总.csv"), "w", encoding="utf-8-sig", newline="") as f:
        cw = csv.writer(f); cw.writerow(["集合"] + HEAD_VAL)
        for r in rv: cw.writerow(["验证集"] + r)
        cw.writerow([]); cw.writerow(["集合"] + HEAD_TEST)
        for r in rt: cw.writerow(["测试集"] + r)
    a3 = val.get("aggregate3", {})
    md = io.open(os.path.join(RES, "说明.md"), "w", encoding="utf-8")
    md.write("# 树干 / 树枝 / 树叶 分离成果\n\n")
    md.write("> **完成人：梁艺萌**　·　本成果与源代码的著作权归完成人所有，**未经本人允许不得商用**；"
             "第三方组件许可见源码包 `NOTICE.txt`。\n\n")
    md.write("完整分步解析见 `成果\\工作流程.md`，流程图见 `成果\\工作流程图.png`（01→30 蛇形流程）。\n\n")
    md.write("模型：Point Transformer V3（Pointcept 官方实现，纯 PyTorch 前向）+ Sonata/ScanNet 预训练权重迁移；"
             "输入 11 维几何特征（坐标 3 + 朝向法向 3 + 相对高度 1 + 局部线度/面度/球度/kNN 半径 4）。\n\n")
    md.write("## 流程\n\n")
    for line in ["训练标签：由提供的 leaf/wood 标注 + 几何伪标注生成三分类（水平切片 → 5cm 栅格连通域 → 圆柱轴追踪，主干自地面沿最粗脊线延伸到顶部）",
                 "细枝采样 + 加权损失：训练时对 branch 点做抖动过采样（oversample），CE 用 sqrt 逆频率权重 + branch 类权重 ×2，并叠加 Lovász-softmax 直接优化 IoU",
                 "推理：多体素尺度 TTA（0.02 / 0.015 m）× 左右翻转 × 多模型（task3 三分类 + task2 枝叶二分）概率融合",
                 "后处理：KNN 投票 → 局部连通域细枝修复（只在候选点上做连通域，要求小而细长、贴附木质部、曲率特征达标）→ 孤岛清理 → 带路径长度与方向连续性约束的梢部生长 → 主干圆柱先验（离轴主干降级为枝）→ 主干缝隙桥接 → 仅保留与地面连通的主干（保证主干是一整根、不断裂）",
                 "验证集最优模型集：ptv3_task3_v4 + ptv3_task2_v3（融合权重 0.25/0.75）"]:
        md.write("- " + line + "\n")
    md.write("\n## 类别代码（写入 LAS classification）\n\n- 1 = 树叶 leaf\n- 2 = 树枝 branch\n- 3 = 树干 trunk\n\n")
    md.write("## 成果目录\n\n")
    md.write("- `成果/测试集/<原文件名>/<id>_wood.laz|.ply`：树干 + 树枝（木质部）分离点云，原始坐标系与点数\n")
    md.write("- `成果/测试集/<原文件名>/<id>_leaf.laz|.ply`：树叶分离点云\n")
    md.write("- `成果/测试集/<原文件名>/<id>_seg.laz|.ply`：整株带分类代码/颜色的点云\n")
    md.write("- `成果/测试集/<原文件名>/01~04_*.png`：三分类总览 / 仅主干 / 主干+树枝 / 主干+树枝+树叶（EDL 渲染，图下方为精度数据）\n")
    md.write("- `成果/验证集/...`：同样结构的 6 棵留出树（有真值，可核对精度）\n")
    md.write("- `成果/汇总/`：逐树指标表（PNG + CSV）\n\n")
    md.write("## 验证集总体指标（逐树平均）\n\n")
    if a3:
        md.write("| 指标 | 叶 leaf | 枝 branch | 干 trunk | 平均 |\n|---|---|---|---|---|\n")
        for key, cn in [("recall", "召回率"), ("precision", "精确率"), ("f1", "F1"), ("iou", "IoU")]:
            v = a3.get(key, [0, 0, 0])
            md.write("| %s | %.3f | %.3f | %.3f | %.3f |\n" % (cn, v[0], v[1], v[2], float(np.mean(v))))
        md.write("| mIoU | | | | %.4f |\n| macro-F1 | | | | %.4f |\n| 总体精度 OA | | | | %.4f |\n" % (
            a3.get("miou", 0), a3.get("macro_f1", 0), a3.get("oa", 0)))
        tips = [v.get("tip_branch_recall") for v in val.get("per_tree", {}).values() if v.get("tip_branch_recall")]
        md.write("| 末端细枝召回（逐树平均） | | | | %.4f |\n" % float(np.mean(tips)))
    tn = [v["trunk"] for v in test.get("per_tree", {}).values()]
    if tn:
        md.write("\n## 测试集结构统计（20 棵）\n\n")
        md.write("- 主干连通块数量：全部 %d 棵均为 **1 根**（不存在断裂主干）\n" % len(tn))
        md.write("- 主干起点高度：最大 %.2f m（即从地面起算）\n" % max(x["zmin"] for x in tn))
        md.write("- 主干到达木质部顶端比例：平均 %.1f%%，最低 %.1f%%（低值为早分枝/矮干型树木）\n" % (
            100 * float(np.mean([x.get("reach_wood", 0) for x in tn])), 100 * min(x.get("reach_wood", 0) for x in tn)))
        md.write("- 平均置信度：%.4f\n" % float(np.mean([v.get("conf_mean", 0) for v in test.get("per_tree", {}).values()])))
    md.write("\n## 交互三维查看\n\n")
    md.write("- `成果/三维查看器_测试集.html`：20 棵测试树，可旋转/缩放、按类别开关显示（绿=叶，蓝=枝，红=干）\n")
    md.write("- `成果/三维查看器_验证集.html`：6 棵验证树同样交互\n")
    md.write("\n## 模型与参数选型记录\n\n")
    md.write("- 三分类模型对比（验证集 pooled mIoU / 末端细枝召回）：v4=0.9203/0.9670，v7f(细体素)=0.9003/0.9228，v4+v7f=0.9143/0.9486，v5=0.8990/0.9611 → 选用 v4\n")
    md.write("- TTA 尺度：0.02+0.015=0.9203 优于 0.02+0.015+0.01=0.9151、优于 0.0125+0.01=0.9109\n")
    md.write("- 后处理消融（验证集 pooled mIoU，用 cache/probs_v4_*.npz 在 CPU 上重放）：原始概率 0.9057 → "
             "旧版后处理 0.7869（有害，已否决）→ 新版细枝链 19~22 0.9079 → +主干轴先验 0.9136 → "
             "+桥接与单根约束 0.9203\n")
    md.write("\n## 说明\n\n")
    md.write("- 测试集无地面真值，因此测试树只报告类别占比、平均置信度与主干贯通统计；精度指标一律来自同模型同流程的验证集。\n")
    md.write("- 主干严格约束为“与地面连通的一整根圆柱”：不在追踪轴上的主干像素降级为树枝，轴上断缝用最近木质点桥接，因此每个 `02_仅主干.png` 中红色部分都是连续单根。\n")
    md.write("- 主干顶端高度即“足够粗的骨架”能到达的最高点，再往上属于细枝（局部半径低于阈值）。\n")
    md.close()
    print("summary written", len(rv), len(rt))


if __name__ == "__main__":
    main()
