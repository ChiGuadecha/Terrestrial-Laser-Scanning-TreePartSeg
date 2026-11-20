# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# 梁艺萌
# 树干 / 树枝 / 树叶 三维点云分离识别 —— 项目源代码
# 未经本人允许不得商用。third_party/ 内组件许可归其原作者，见 NOTICE.txt。
# ------------------------------------------------------------------
"""Build the signed source package: stage tree + zip + MANIFEST (sha256)."""
import os, io, sys, glob, shutil, zipfile, hashlib, datetime

ROOT = r"E:\Desktop\场景"
SRC = os.path.join(ROOT, "treeseg")
RES = os.path.join(ROOT, "成果")
PTV3 = os.path.join(ROOT, "ptv3_repo")
PKG = os.path.join(ROOT, "源代码包")
STAGE = os.path.join(PKG, "treeseg_src")
INNER = "treeseg_src"

SRC_FILES = ["data_prep.py", "train.py", "train_transfer.py", "post.py", "predict.py",
             "ablate_cache.py", "ablate_eval.py", "sweep.py", "check_trunk.py",
             "refresh_stats.py", "edl_render.py", "make_deliverables.py", "make_summary.py",
             "make_viewer.py", "make_workflow_png.py", "make_report.py", "make_results.py",
             "export_outputs.py", "qc_one.py", "dl_sonata.py", "add_banner.py",
             "relocate.py", "pack_source.py"]
CONFIGS = sorted(os.path.basename(p) for p in glob.glob(os.path.join(SRC, "cfg*.json")))
LOGS = ["run_task3_v4.log", "run_task3_v5.log", "run_task3_v7f.log", "run_task3_v6f.log",
        "run_task3_v3.log", "run_task3_v2.log", "run_task3.log", "run_task2_v3.log",
        "run_task2_v2.log", "run_task2.log", "run_test_A.log", "run_test_B.log",
        "sweep5.log", "build_A.log", "build_B.log", "build_val.log"]
DOCS = ["工作流程图.png", "工作流程.md", "说明.md"]
PTV3_TOP = ["LICENSE", "NOTICE", "README.md", "pyproject.toml"]
SIG = "梁艺萌（北京市园林古建设计研究院 规划所） · 未经本人允许不得商用"


def put(src_abs, rel, encoding=None):
    """copy (or re-encode) a file into the staging tree"""
    dst = os.path.join(STAGE, *rel.split("/"))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if encoding:
        text = io.open(src_abs, encoding="utf-8").read()
        io.open(dst, "w", encoding=encoding, newline="\n").write(text)
    else:
        shutil.copy2(src_abs, dst)
    return rel.replace("\\", "/")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def build():
    if os.path.exists(STAGE):
        shutil.rmtree(STAGE)
    os.makedirs(STAGE)
    entries = []
    # 1. first-party code
    for name in SRC_FILES:
        entries.append(put(os.path.join(SRC, name), "src/" + name))
    io.open(os.path.join(STAGE, "src", ".root"), "w", encoding="utf-8").write(ROOT + "\n")
    entries.append("src/.root")
    # 2. configs / docs / logs
    for name in CONFIGS:
        entries.append(put(os.path.join(SRC, name), "configs/" + name))
    for name in DOCS:
        entries.append(put(os.path.join(RES, name), "docs/" + name))
    for name in LOGS:
        p = os.path.join(SRC, name)
        if os.path.exists(p):
            entries.append(put(p, "logs/" + name))
    # 3. package meta (utf-8-sig so Windows editors / PowerShell read Chinese correctly)
    entries.append(put(os.path.join(SRC, "PKG_README.md"), "README.md", "utf-8-sig"))
    entries.append(put(os.path.join(SRC, "PKG_NOTICE.txt"), "NOTICE.txt", "utf-8-sig"))
    entries.append(put(os.path.join(SRC, "requirements.txt"), "requirements.txt", "utf-8-sig"))
    entries.append(put(os.path.join(SRC, "run_pipeline.ps1"), "run_pipeline.ps1", "utf-8-sig"))
    # 4. third party (authorship / licence stays with upstream)
    for name in PTV3_TOP:
        p = os.path.join(PTV3, name)
        if os.path.exists(p):
            entries.append(put(p, "third_party/point-transformer-v3/" + name))
    base = os.path.join(PTV3, "src", "point_transformer_v3")
    for path in sorted(glob.glob(os.path.join(base, "**", "*.py"), recursive=True)):
        rel = os.path.relpath(path, PTV3).replace("\\", "/")
        entries.append(put(path, "third_party/" + rel))
    entries.append(put(os.path.join(SRC, "three.min.js"), "third_party/three.js/three.min.js"))
    # 5. manifest
    lines = ["# 树干 / 树枝 / 树叶 三维点云分离识别 —— 源代码包清单",
             "# " + SIG,
             "# 生成时间：%s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
             "# 条目：%d    sha256(前 16 位)" % len(entries), ""]
    for rel in sorted(entries):
        fp = os.path.join(STAGE, *rel.split("/"))
        lines.append("%10d  %s  %s" % (os.path.getsize(fp), sha(fp), rel))
    io.open(os.path.join(STAGE, "MANIFEST.txt"), "w", encoding="utf-8-sig", newline="\n").write(
        "\n".join(lines) + "\n")
    return sorted(entries)


def zipout(all_entries):
    date = datetime.date.today().strftime("%Y%m%d")
    zp = os.path.join(PKG, "树干树枝树叶分离_源代码包_梁艺萌_%s.zip" % date)
    if os.path.exists(zp):
        os.remove(zp)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for rel in all_entries:
            fp = os.path.join(STAGE, *rel.split("/"))
            z.write(fp, INNER + "/" + rel)
    return zp


def main():
    entries = build()
    all_entries = entries + ["MANIFEST.txt"]
    zp = zipout(all_entries)
    print("staged %d files -> %s" % (len(all_entries), STAGE))
    print("zip    %s  (%.2f MB)" % (zp, os.path.getsize(zp) / 1048576.0))
    bad = [e for e in all_entries if not os.path.exists(os.path.join(STAGE, *e.split("/")))]
    print("missing after build:", bad)
    print("third-party entries:", sum(1 for e in all_entries if e.startswith("third_party/")))


if __name__ == "__main__":
    main()
