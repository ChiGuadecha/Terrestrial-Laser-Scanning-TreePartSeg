# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# 梁艺萌
# 树干 / 树枝 / 树叶 三维点云分离识别 —— 项目源代码
# 未经本人允许不得商用。third_party/ 内组件许可归其原作者，见 NOTICE.txt。
# ------------------------------------------------------------------
"""Set / replace the authorship banner in every first-party source file (idempotent)."""
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DASH = "# " + "-" * 66
OWNER = "梁艺萌"
MARKS = ("梁艺萌", "北京市园林古建设计研究院")

BANNER = "\n".join([
    DASH,
    "# 梁艺萌",
    "# 树干 / 树枝 / 树叶 三维点云分离识别 —— 项目源代码",
    "# 未经本人允许不得商用。third_party/ 内组件许可归其原作者，见 NOTICE.txt。",
    DASH,
])

SRC = ["data_prep.py", "train.py", "train_transfer.py", "post.py", "predict.py",
       "ablate_cache.py", "ablate_eval.py", "sweep.py", "check_trunk.py",
       "refresh_stats.py", "edl_render.py", "make_deliverables.py", "make_summary.py",
       "make_viewer.py", "make_workflow_png.py", "make_report.py", "make_results.py",
       "export_outputs.py", "qc_one.py", "dl_sonata.py", "add_banner.py", "pack_source.py", "relocate.py"]

RE_DASH = re.compile(r"^# -{60,}$")


def strip_old(lines):
    """Remove an existing banner block (two dash rules whose body mentions the author)."""
    idx = [i for i, l in enumerate(lines[:30]) if RE_DASH.match(l.rstrip())]
    for a, b in zip(idx, idx[1:]):
        body = "\n".join(lines[a + 1:b])
        if any(m in body for m in MARKS):
            return lines[:a] + lines[b + 1:], True
    return lines, False


def patch(path):
    text = open(path, encoding="utf-8").read()
    lines = text.split("\n")
    lines, removed = strip_old(lines)
    want = BANNER.split("\n")
    if lines[0].startswith("# -*- coding:") or lines[0].startswith("#!"):
        at = 1
    else:
        at = 0
    head = "\n".join(lines[:12])
    if all(w in head for w in ("梁艺萌", "未经本人允许不得商用")) and not removed:
        return "kept"
    lines[at:at] = want
    open(path, "w", encoding="utf-8").write("\n".join(lines))
    return "replaced" if removed else "added"


def main():
    tally = {"added": 0, "replaced": 0, "kept": 0, "missing": 0}
    for name in SRC:
        p = os.path.join(HERE, name)
        if not os.path.exists(p):
            print("%-24s missing" % name); tally["missing"] += 1; continue
        r = patch(p)
        tally[r] += 1
        print("%-24s %s" % (name, r))
    print(tally)


if __name__ == "__main__":
    main()
