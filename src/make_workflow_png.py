# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# 梁艺萌
# 树干 / 树枝 / 树叶 三维点云分离识别 —— 项目源代码
# 未经本人允许不得商用。third_party/ 内组件许可归其原作者，见 NOTICE.txt。
# ------------------------------------------------------------------
"""Trunk/branch/leaf workflow: strict serpentine grid, every arrow joins two boxes."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FW, FH = 20.0, 11.8
UX, UY = FW * 72.0 / 100.0, FH * 72.0 / 100.0
FS_T, FS_B = 9.6, 8.5
LINE = 1.42 * FS_B / UY
HEAD = 1.15 + FS_T / UY + 0.85
TAIL = 1.40
PADX = 1.0
NCOL, X0, XEND, GAP, GY = 6, 4.2, 99.2, 1.5, 3.4

LANE = {"data": "#2C3E50", "train": "#1F6F5C", "infer": "#96601F",
        "post": "#4B3B8F", "out": "#8E2F4A"}
FC = {"data": "#EAF1F7", "train": "#E8F5F0", "infer": "#FBF1E3",
      "post": "#EEEAF9", "out": "#FAE9EE"}

fig = plt.figure(figsize=(FW, FH), dpi=125)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
warn = []


def tw(s, fs):
    w = 0.0
    for ch in s:
        o = ord(ch)
        if o > 0x2E00:
            w += 1.0
        elif ch == " ":
            w += 0.30
        elif ch.isdigit() or ch.isalpha():
            w += 0.53
        else:
            w += 0.50
    return w * fs / UX


def box(x, y, w, h, title, lines, lane):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.9",
                                fc=FC[lane], ec=LANE[lane], lw=1.5, zorder=2))
    if tw(title, FS_T) > w - 0.6:
        warn.append(("TITLE " + title, round(tw(title, FS_T), 1), round(w, 1), ""))
    ax.text(x + w / 2.0, y + h - 1.15, title, ha="center", va="top", fontsize=FS_T,
            fontweight="bold", color=LANE[lane], zorder=3)
    ty = y + h - HEAD
    for ln in lines:
        if tw(ln, FS_B) + PADX > w - PADX:
            warn.append((title, round(tw(ln, FS_B) + 2 * PADX, 1), round(w, 1), ln))
        ax.text(x + PADX, ty, ln, ha="left", va="top", fontsize=FS_B,
                color="#232323", zorder=3)
        ty -= LINE


def arrow(x1, y1, x2, y2, lane):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                 lw=1.8, color=LANE[lane], zorder=5,
                                 connectionstyle="arc3,rad=0.0"))


def lane_label(y, h, text, lane):
    ax.add_patch(FancyBboxPatch((0.6, y), 2.3, h, boxstyle="round,pad=0.0,rounding_size=0.6",
                                fc=LANE[lane], ec="none", zorder=2))
    ax.text(1.75, y + h / 2.0, text, ha="center", va="center", fontsize=9.4,
            color="white", fontweight="bold", rotation=90, zorder=3)


def row_h(items):
    return HEAD + max(len(it[1]) for it in items) * LINE + TAIL


COLW = (XEND - X0 - GAP * (NCOL - 1)) / NCOL


def cols(direction):
    if direction == "ltr":
        return [X0 + i * (COLW + GAP) for i in range(NCOL)]
    return [X0 + (NCOL - 1 - i) * (COLW + GAP) for i in range(NCOL)]

ROWS = [
 ("数据", "data", "ltr", [
  ("01 原始点云读取", ["laspy.read(f).xyz 必须 .xyz",
   "直接取 X/Y/Z 是未缩放整数", "（曾致切片循环 10^7 次卡死）",
   "训练集 N.laz / _leaf / _wood"]),
  ("02 清洗与归一", ["clean_outliers：z 分位去飞点",
   "shift = (med x, med y, min z)", "coord = raw - shift",
   "z 从地面 0 起算"]),
  ("03 几何特征（11 维）", ["kNN 协方差 → 法向 + 特征值",
   "orient_outward 统一朝向", "线度/面度/球度/kNN 半径",
   "intensity 全 0，只能靠几何"]),
  ("04 伪标注：切片聚类", ["水平切片 slice_h = 0.1 m",
   "5 cm 栅格 → 连通域", "每片质心 + 90% 分位半径",
   "min_pts = 15 去噪"]),
  ("05 伪标注：主干轴追踪", ["底部 30% 取紧凑种子",
   "（最低层优先，最多 4 个）", "沿圆柱轴向上：位置+漂移预测",
   "干=轴内木质，枝=轴外木质"]),
  ("06 划分与落盘", ["30 训练 / 6 验证留出 / 20 测试",
   "data/{train,test}_*.npz", "coord / normal / featx",
   "label3 / label_woodleaf / shift"]),
 ]),
 ("训练", "train", "rtl", [
  ("07 PTv3 网络搭建", ["Pointcept 纯 PyTorch 实现",
   "免编译 CUDA 扩展", "triton 快路径，否则 padded SDPA",
   "seg_head 手工接（64 ch）"]),
  ("08 预训练权重迁移", ["weights/hf/sonata.pth",
   "PointTransformerV3 ScanNet ckpt",
   "train_transfer.py 做迁移探针"]),
  ("09 细枝过采样", ["branch 点抖动复制 ×0.6（σ=0.01 m）",
   "枝 : 叶 ≈ 1 : 3.2（0.186 vs 0.642）", "只作用于训练集，验证/测试不采样"]),
  ("10 加权损失", ["CE 权重 = sqrt(逆类频)",
   "0.642 / 1.152 / 1.206", "branch 类权重 ×2（--lw-branch）",
   "+ λ·Lovász-softmax 优化 IoU"]),
  ("11 训练稳定性", ["OneCycle 5e-4 / 120 epoch / bf16",
   "评估必须 fp32", "（fp16 曾使枝召回 0.22→0.024）",
   "EMA 只平滑参数，BN 用实时值"]),
  ("12 产出权重", ["runs/ptv3_task3_v4/", "runs/ptv3_task2_v3/",
   "三分类 + 枝叶二分", "grid 0.02 / 0.015"]),
 ]),
 ("推理", "infer", "ltr", [
  ("13 载入模型集", ["load_model(best.pth) fp32",
   "models3 = [v4]，model2 = v3", "task2 缺失则自动退化为单模型"]),
  ("14 多尺度 TTA", ["GRIDS = 0.02, 0.015（实测最优）",
   "再加 0.01 反而 0.9203→0.9151", "× 左右翻转（x 与法向取反）"]),
  ("15 概率累加", ["pick=first + first[inv] 散射回原点",
   "acc3 += p3[inv]（不是 np.add.at，", "广播会报错）；acc2 同理",
   "按次数取均值"]),
  ("16 双任务融合", ["p(枝)·p(木)，p(干)·p(木)",
   "叶 = 0.25·三分类 + 0.75·枝叶二分", "选型 v4 0.9203 > v4+v7f 0.9143",
   "> v7f 0.9003 > v5 0.8990"]),
  ("17 初判 + 保留概率", ["lab = fused.argmax(1)",
   "fused 概率交给后处理做门控", "（没有概率就没有细枝修复）"]),
  ("18 概率缓存", ["cache/probs_{tag}_{id}.npz",
   "CPU 秒级调参 sweep", "不再占用 GPU"]),
 ]),
 ("后处理", "post", "rtl", [
  ("19 KNN 投票", ["k=10，σ=0.03，两轮", "keep_w = 0.6",
   "去孤立误分类与块状噪声"]),
  ("20 局部连通域细枝修复", ["候选 = 叶 & 线度>0.45 & 半径<0.030",
   "& 模型枝概率>0.02（门控）", "连通域只在候选点上构建",
   "旧版并入骨架→12 号叶召回 0.50", "尺寸 10~30000，细长比≥1.7",
   "贴附木质≥22% @ 0.12 m"]),
  ("21 孤岛清理", ["<12 点且不含干的木质", "小连通块 → 回退为叶",
   "避免零碎木点污染"]),
  ("22 梢部生长 tip_growth2", ["从已确认枝沿“细而直”脊点外扩",
   "约束 1：累计路径长度 ≤ 0.6 m", "约束 2：kNN 主方向与行进方向",
   "|cos| > 0.40（局部曲率连续）", "→ 只修真漏梢，不糊成一片"]),
  ("23 主干圆柱先验", ["对预测木质重跑 pseudo_trunk",
   "轴外却判成干 → 降级为枝", "轴内木质 → 升级为干",
   "树 43 干精确率 0.955→0.999"]),
  ("24 断缝桥接 + 单根约束", ["沿轴每 6 cm 一层：无干有枝→补干",
   "整层无点 → 0.34 m 内就近补木质", "eps=0.10 只保留含最低点的一块",
   "主干 = 从地面连续向上的一根", "26/26 棵树主干连通块 = 1"]),
 ]),
 ("交付", "out", "ltr", [
  ("25 输出预测 npz", ["output/pred_{val|test}_{id}.npz",
   "coord / normal / featx / pred", "prob / shift（可还原原始坐标）"]),
  ("26 指标评估", ["逐树混淆矩阵 / mIoU / OA",
   "分类别 P-R-F1 + macro-F1", "末端细枝召回 0.9670",
   "主干贯通与置信度统计"]),
  ("27 分离点云（原始坐标）", ["<id>_wood.laz/.ply 干 + 枝",
   "<id>_leaf.laz/.ply 叶", "<id>_seg.laz 整株 class 1/2/3",
   "wood + leaf = 原始点数"]),
  ("28 EDL 图 ×4 / 树", ["01 分割总览 · 02 仅主干",
   "03 主干+枝 · 04 主干+枝+叶", "三视角 + 图下方该树精度",
   "没分出叶的树不出树叶图"]),
  ("29 汇总与查看器", ["成果\\汇总\\*.png + 指标汇总.csv",
   "说明.md / 工作流程.md", "三维查看器_*.html 可旋转"]),
  ("30 校验（全部通过）", ["20 棵测试树无缺树",
   "坐标 / intensity 与原始一致", "验证集 mIoU 0.9203 / OA 0.9682",
   "主干连通块全部 = 1"]),
 ]),
]

hs = [row_h(items) for _, _, _, items in ROWS]
FOOT_H = 4.9
total = sum(hs) + GY * (len(ROWS) - 1) + 2 * (FOOT_H + GY)
print("content height = %.1f  (top 91.0 -> bottom %.1f)" % (total, 91.0 - total))

ax.text(50, 98.6, "树干 / 树枝 / 树叶 分离识别 —— 完整工作流", ha="center", va="center",
        fontsize=17.0, fontweight="bold", color="#111111")
ax.text(50, 95.7, "Point Transformer V3（三分类 + 枝叶二分双模型融合）＋ 几何伪标注 ＋ 结构后处理",
        ha="center", va="center", fontsize=11.0, color="#333333")
ax.text(50, 93.2, "编号 01→30 即执行顺序；蛇形排布：数据/推理/交付行从左到右，训练/后处理行从右到左，每个箭头只连接相邻两框的边      "
        "数据：训练集 30 棵（含 leaf/wood 标注）/ 验证留出 6 棵 / 测试集 20 棵（无标注）      "
        "代码：E:\\Desktop\\场景\\treeseg      成果：E:\\Desktop\\场景\\成果",
        ha="center", va="center", fontsize=8.6, color="#666666")

ytop = 91.0
for ri, (label, lane, direction, items) in enumerate(ROWS):
    h = hs[ri]
    ybot = ytop - h
    lane_label(ybot, h, label, lane)
    xs = cols(direction)
    for (title, lines), x in zip(items, xs):
        box(x, ybot, COLW, h, title, lines, lane)
    yc = ybot + h / 2.0
    for i in range(len(items) - 1):
        if direction == "ltr":
            arrow(xs[i] + COLW, yc, xs[i + 1], yc, lane)
        else:
            arrow(xs[i], yc, xs[i + 1] + COLW, yc, lane)
    if ri + 1 < len(ROWS):
        nxt = ROWS[ri + 1]
        xn = cols(nxt[2])[0]
        arrow(xs[-1] + COLW / 2.0, ybot, xn + COLW / 2.0, ybot - GY - row_h(nxt[3]), nxt[1])
    ytop = ybot - GY

# footer strip 1: measured gains
ybot = ytop - FOOT_H
ax.add_patch(FancyBboxPatch((X0, ybot), XEND - X0, FOOT_H,
                            boxstyle="round,pad=0.0,rounding_size=0.9",
                            fc="#FFF6D9", ec="#B58900", lw=1.6, zorder=2))
ax.text(X0 + 1.4, ybot + FOOT_H - 1.5,
        "实测增益（验证集 pooled mIoU，CPU 重放 cache/probs_v4_*.npz）：原始概率 0.9057 → 旧版后处理 0.7869（有害，已否决） "
        "→ 19~22 细枝链 0.9079 → +23 主干轴先验 0.9136 → +24 断缝桥接与单根约束 0.9203",
        fontsize=8.8, va="center", color="#8E2F4A", fontweight="bold")
ax.text(X0 + 1.4, ybot + 1.6,
        "验证集总体：mIoU 0.9203 / OA 0.9682 / macro-F1 0.9568；召回 叶 0.981、枝 0.886、干 0.991；"
        "末端细枝召回 0.9670；主干连通块 26/26 = 1 根",
        fontsize=8.8, va="center", color="#2b2b2b")
ytop = ybot - GY

# footer strip 2: rejected variants
ybot = ytop - FOOT_H
ax.add_patch(FancyBboxPatch((X0, ybot), XEND - X0, FOOT_H,
                            boxstyle="round,pad=0.0,rounding_size=0.9",
                            fc="#F4F4F4", ec="#999999", lw=1.4, zorder=2))
ax.text(X0 + 1.4, ybot + FOOT_H - 1.5,
        "试过但被否掉的分支：leader_extend 主干延长（默认关闭，无脑拉到木质顶 mIoU 0.920→0.741）；"
        "旧版 twig_rescue 把骨架并入连通域（针叶整片误判为枝）；fp16 评估；TTA 加 0.01 尺度；v5 / v7f 单模型",
        fontsize=8.6, va="center", color="#444444")
ax.text(X0 + 1.4, ybot + 1.6,
        "复现命令：python data_prep.py → python train.py --task 3 --name ptv3_task3_v4 → "
        "python predict.py val ptv3_task3_v4 ptv3_task2_v3 all → python predict.py test ... → "
        "python make_deliverables.py → python make_summary.py",
        fontsize=8.6, va="center", color="#444444")

ax.text(99.2, 0.6, "梁艺萌  ·  未经本人允许不得商用",
        ha="right", fontsize=8.2, va="bottom", color="#777777")
ax.text(1.0, 0.6, "类别码：LAS classification 1=叶 2=枝 3=干；成果文件夹名沿用测试集原始文件名（13/14/18/2/…）",
        fontsize=8.2, va="bottom", color="#777777")

outp = r"E:\Desktop\场景\成果\工作流程图.png"
fig.savefig(outp, dpi=125, facecolor="white")
print("saved", outp, round(os.path.getsize(outp) / 1024), "KB")
for t, need, w, ln in warn:
    print("OVERFLOW %-38s need=%.1f box=%.1f | %s" % (t[:38], need, w, ln))
print("overflow lines:", len(warn))
