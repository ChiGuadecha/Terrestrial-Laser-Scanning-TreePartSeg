# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# 梁艺萌
# 树干 / 树枝 / 树叶 三维点云分离识别 —— 项目源代码
# 未经本人允许不得商用。third_party/ 内组件许可归其原作者，见 NOTICE.txt。
# ------------------------------------------------------------------
"""Rewrite the hard-coded project root in every first-party source file.

usage:
    python relocate.py D:\\work\\trees           # point the whole package at a new root
    python relocate.py <原根目录>                  # restore (原根目录记在 src/.root 里)
    python relocate.py D:\\work\\trees --ptv3-local  # 同时改用包内 third_party 的 PTv3
"""
import os, sys, glob

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OLD = r"E:\Desktop\场景"
MARKER = os.path.join(HERE, ".root")


def variants(root):
    """the three spellings used in the sources: raw, python-escaped, forward slash"""
    esc = root.replace("\\", "\\\\")
    fwd = root.replace("\\", "/")
    return root, esc, fwd


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    new = os.path.abspath(sys.argv[1])
    ptv3_local = "--ptv3-local" in sys.argv[2:]
    old = DEFAULT_OLD
    if os.path.exists(MARKER):
        old = open(MARKER, encoding="utf-8").read().strip()
    o_raw, o_esc, o_fwd = variants(old)
    n_raw, n_esc, n_fwd = variants(new)
    pkg_ptv3 = os.path.abspath(os.path.join(HERE, "..", "third_party",
                                           "point-transformer-v3", "src"))
    changed = []
    for path in sorted(glob.glob(os.path.join(HERE, "*.py")) +
                       [os.path.join(HERE, "run_pipeline.ps1")]):
        if not os.path.exists(path):
            continue
        text = open(path, encoding="utf-8").read()
        new_text = (text.replace(o_esc, n_esc).replace(o_raw, n_raw).replace(o_fwd, n_fwd))
        if ptv3_local:
            for a in (os.path.join(o_raw, "ptv3_repo", "src"),
                      os.path.join(o_esc, "ptv3_repo", "src"),
                      os.path.join(o_fwd, "ptv3_repo", "src")):
                new_text = new_text.replace(a, pkg_ptv3)
        if new_text != text:
            open(path, "w", encoding="utf-8").write(new_text)
            changed.append(os.path.basename(path))
    open(MARKER, "w", encoding="utf-8").write(new)
    print("root: %s -> %s" % (old, new))
    print("ptv3: %s" % ("bundled third_party" if ptv3_local else "keep <root>/ptv3_repo/src"))
    print("rewrote %d file(s): %s" % (len(changed), ", ".join(changed)))
    print("note: 目录约定 <root>\\{训练集,测试集,treeseg\\{data,runs,cache,weights},output,成果}")


if __name__ == "__main__":
    main()
