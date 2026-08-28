"""vision/rsweep_collect.py — 跨半径扫描结果收集（docs/246）。

读 logs/rsw_*.log，用纯 python 正则只抽取 R_RSW_* 摘要行（ASCII 标签 + 数字），
打印紧凑汇总：每格一行（含 SC1/SC2/churn/ratio/bridge 全套与格通过）+ 工作点表 +
主/变体工作点 + 判定 + 守卫 + 缺失清单。绝不打印原始日志内容（只回显 R_ 元数据行）。
编码自动识别（UTF-8 / UTF-16LE BOM）。

判据与判定规则（预注册冻结，docs/246 §1.4/§1.5，本脚本只实现冻结规则）：
  格通过 = BRIDGE_FIX(bridge_sw>=0.5) 且 HELD_OUT(holdout_sw>=0.5) 且
    CHURN(R0 churn<=0.5 且 R1 churn<=0.5) 且 STABLE(R0 ratio<=1.5 且 R1 ratio<=1.5)。
  主网格（k=3）工作点 = 满足全部四判据的半径乘子集合；最小/最优 = 最小 M。
  判定：主网格工作点非空 = TENSION_RESOLVED；空 = TENSION_PERSISTS（k=2 变体单独报告）。

用法：
  python vision/rsweep_collect.py [logdir]
"""
import glob
import os
import re
import sys

LINE = re.compile(r"^(R_RSW_[A-Z0-9_]+)=(.*)$")


def read_text(path):
    with open(path, "rb") as f:
        raw = f.read()
    if raw.startswith(b"\xff\xfe"):
        return raw.decode("utf-16-le").lstrip("\ufeff")
    if raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16-be").lstrip("\ufeff")
    return raw.decode("utf-8", errors="replace").lstrip("\ufeff")


def parse_log(path):
    d = {}
    for line in read_text(path).splitlines():
        m = LINE.match(line)
        if m:
            d[m.group(1)] = m.group(2).strip()
    return d


def fnum(s, default=None):
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def main():
    logdir = sys.argv[1] if len(sys.argv) > 1 else "logs"
    # 预注册网格（docs/246 §1.2/§1.3，冻结）
    mults = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
    ks = (3, 2)

    def mtag(m):
        s = ("%.2f" % m).rstrip("0")
        if s.endswith("."):
            s += "0"
        return "m" + s.replace(".", "p")

    present = {}
    for path in sorted(glob.glob(os.path.join(logdir, "rsw_*.log"))):
        d = parse_log(path)
        if "R_RSW_MULT" not in d or "R_RSW_K" not in d:
            continue
        m = fnum(d["R_RSW_MULT"])
        k = int(fnum(d["R_RSW_K"], -1))
        if m is None or k not in ks:
            continue
        present.setdefault(k, {})[m] = (os.path.basename(path), d)

    print("R_RSW_COL_GRID=9x2")
    # ---- 每格一行 ----
    for k in ks:
        for m in mults:
            ent = present.get(k, {}).get(m)
            if not ent:
                print("R_RSW_COL_CELL=%d,%.2f,MISSING" % (k, m))
                continue
            _, d = ent
            cells = [str(k), "%.2f" % m, d.get("R_RSW_RADIUS", "NA"),
                     d.get("R_RSW_0_SC1", "NA"), d.get("R_RSW_0_SC2", "NA"),
                     d.get("R_RSW_0_CHURN", "NA"),
                     d.get("R_RSW_1_SC1", "NA"), d.get("R_RSW_1_SC2", "NA"),
                     d.get("R_RSW_1_CHURN", "NA"),
                     d.get("R_RSW_0_RATIO", "NA"), d.get("R_RSW_1_RATIO", "NA"),
                     d.get("R_RSW_BRIDGE_SW", "NA"), d.get("R_RSW_CALIB_SW", "NA"),
                     d.get("R_RSW_HOLDOUT_SW", "NA"), d.get("R_RSW_BRIDGE_VID", "NA"),
                     d.get("R_RSW_SPURIOUS", "NA"), d.get("R_RSW_SW_ENC", "NA"),
                     d.get("R_RSW_CRIT_BRIDGE", "NA"), d.get("R_RSW_CRIT_HELDOUT", "NA"),
                     d.get("R_RSW_CRIT_CHURN", "NA"), d.get("R_RSW_CRIT_STABLE", "NA"),
                     d.get("R_RSW_CELL_PASS", "NA"), d.get("R_RSW_GUARD_D245", "NA"),
                     d.get("R_RSW_ELAPSED", "NA")]
            print("R_RSW_COL_CELL=%s" % ",".join(cells))

    # ---- 工作点表（每通过格一行，含余量） ----
    def wp_table(k):
        out = []
        for m in mults:
            ent = present.get(k, {}).get(m)
            if not ent or ent[1].get("R_RSW_CELL_PASS") != "1":
                continue
            d = ent[1]
            out.append(",".join([str(k), "%.2f" % m, d.get("R_RSW_RADIUS", "NA"),
                                 d.get("R_RSW_BRIDGE_SW", "NA"),
                                 d.get("R_RSW_HOLDOUT_SW", "NA"),
                                 d.get("R_RSW_0_CHURN", "NA"),
                                 d.get("R_RSW_1_CHURN", "NA"),
                                 d.get("R_RSW_0_RATIO", "NA"),
                                 d.get("R_RSW_1_RATIO", "NA"),
                                 d.get("R_RSW_MARGIN_BRIDGE", "NA"),
                                 d.get("R_RSW_MARGIN_HELDOUT", "NA"),
                                 d.get("R_RSW_MARGIN_CHURN", "NA"),
                                 d.get("R_RSW_MARGIN_STABLE", "NA")]))
        return out

    for k in ks:
        rows = wp_table(k)
        if rows:
            print("R_RSW_COL_WP_TABLE_K%d=%s" % (k, ";".join(rows)))
        else:
            print("R_RSW_COL_WP_TABLE_K%d=NA" % k)

    # ---- 工作点与判定（冻结规则，docs/246 §1.5） ----
    def wps(k):
        out = []
        for m in mults:
            ent = present.get(k, {}).get(m)
            if ent and ent[1].get("R_RSW_CELL_PASS") == "1":
                out.append(m)
        return out

    wp3 = wps(3)
    wp2 = wps(2)
    if wp3:
        mmin = min(wp3)
        d = present[3][mmin][1]
        margins = ",".join([d.get("R_RSW_MARGIN_BRIDGE", "NA"),
                            d.get("R_RSW_MARGIN_HELDOUT", "NA"),
                            d.get("R_RSW_MARGIN_CHURN", "NA"),
                            d.get("R_RSW_MARGIN_STABLE", "NA")])
        print("R_RSW_COL_WP_K3=%s" % ",".join("%.2f" % m for m in wp3))
        print("R_RSW_COL_WP_MIN_K3=%.2f" % mmin)
        print("R_RSW_COL_WP_MIN_MARGINS=%s" % margins)
        print("R_RSW_COL_VERDICT=TENSION_RESOLVED")
    else:
        print("R_RSW_COL_WP_K3=NA")
        print("R_RSW_COL_WP_MIN_K3=NA")
        print("R_RSW_COL_WP_MIN_MARGINS=NA")
        print("R_RSW_COL_VERDICT=TENSION_PERSISTS")
    if wp2:
        mmin2 = min(wp2)
        d = present[2][mmin2][1]
        margins2 = ",".join([d.get("R_RSW_MARGIN_BRIDGE", "NA"),
                             d.get("R_RSW_MARGIN_HELDOUT", "NA"),
                             d.get("R_RSW_MARGIN_CHURN", "NA"),
                             d.get("R_RSW_MARGIN_STABLE", "NA")])
        print("R_RSW_COL_WP_K2=%s" % ",".join("%.2f" % m for m in wp2))
        print("R_RSW_COL_WP_MIN_K2=%.2f" % mmin2)
        print("R_RSW_COL_WP_MIN_K2_MARGINS=%s" % margins2)
    else:
        print("R_RSW_COL_WP_K2=NA")
        print("R_RSW_COL_WP_MIN_K2=NA")
        print("R_RSW_COL_WP_MIN_K2_MARGINS=NA")

    # ---- 守卫与缺失 ----
    g = present.get(3, {}).get(1.0)
    if g:
        print("R_RSW_COL_GUARD_D245=%s" % g[1].get("R_RSW_GUARD_D245", "NA"))
        print("R_RSW_COL_RBASE=%s" % g[1].get("R_RSW_R_BASE", "NA"))
    else:
        print("R_RSW_COL_GUARD_D245=NA")
        print("R_RSW_COL_RBASE=NA")
    missing = []
    for k in ks:
        for m in mults:
            if m not in present.get(k, {}):
                missing.append("rsw_%s_k%d.log" % (mtag(m), k))
    print("R_RSW_COL_MISSING=%s" % (",".join(sorted(missing)) if missing else "NA"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
