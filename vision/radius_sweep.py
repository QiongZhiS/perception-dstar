"""vision/radius_sweep.py — docs/246 跨半径工作点扫描（docs/245 半径旋钮的下一步预注册实验）。

把 docs/245 的软匹配"单半径点"升级为"预注册半径网格 x 创建滞回 k_consist"的二维工作点
扫描，寻找"判别 + 稳定同时泛化"的工作点。

预注册（docs/246 §一，冻结；判据与网格先于最终运行写入 docs/246，docs/63 纪律）：
  主旋钮：半径乘子 M ∈ {0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0}，绝对半径
    r = M x r_base；r_base = 校准集参与窗口最近邻距离中位数（对数 (E,U) 域，
    soft_match_test.calibrate_radius 复用，docs/245 实测 0.2659）。
  第二旋钮（启用）：k_consist ∈ {2, 3}（创建滞回；k=2 使原型更快固化，docs/245 §六4
    短场景尾部创建问题的机制变体）。二维网格 = 9 x 2 = 18 格。主分析 k=3，变体 k=2。
  判据（每格，冻结）：
    1. BRIDGE_FIX : bridge_corr_switch >= 0.5
    2. HELD_OUT   : holdout_switch >= 0.5（calib_switch 分开报告，不进判据）
    3. CHURN      : R0 churn <= 0.5 且 R1 churn <= 0.5
    4. STABLE     : R0 末/首四分之一 MAE 比 <= 1.5 且 R1 比 <= 1.5（控制项：
       预测路径未动，MAE 与 docs/243/244/245 逐位一致，与半径/k 无关）
  工作点 = 主网格（k=3）满足全部四判据的半径（集）；最小/最优工作点 = 满足全部判据的
    最小 M（保留最多判别分辨率）。余量 = 各判据距阈值距离
    （bridge_sw-0.5, holdout_sw-0.5, 0.5-max(churn), 1.5-max(ratio)）。
  判定：主网格存在工作点 = TENSION_RESOLVED；无 = TENSION_PERSISTS（k=2 变体单独报告，
    不进主判定）。
  回归守卫：格 (1.0, 3) 必须逐位复现 docs/245 smt_main（r_base 0.2659、R0 SC2=3/churn 0/
    ratio 0.907701、R1 SC2=3/churn 0.7692/ratio 0.951261、bridge_sw 1.0、calib_sw 1.0、
    holdout_sw 1.0、bridge_vid 1.0、spurious 0）——R_RSW_GUARD_D245。

安全纪律（docs/243/244/245 同款）：stdout 只输出 ASCII 标签 + 每行一个数字的 R_RSW_*
摘要块；运行经 powershell 包装重定向到 logs/；数字用纯 python 正则（extract_r /
rsweep_collect）抽取；禁止读日志/JSON 原文；davis 目录是数据。禁止修改既有脚本——
只 import 复用（soft_match_test / real_stream_test / real_recalib / stream_test /
compose_test / critical_point）。

用法（每 (半径, k) 一次运行）：
  python vision/radius_sweep.py --radius-mult 1.0 --k 3 --tag rsw_m1p0_k3
  python vision/radius_sweep.py --radius-mult 2.0 --k 2 --tag rsw_m2p0_k2
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

from critical_point import mean_sd, bootstrap_ci
from stream_test import LOOP_CFG
from real_stream_test import load_video_frames, VIDEOS, WINDOW, LEVEL_NAMES
from real_recalib import CALIB_VIDEOS, HOLDOUT_VIDEOS, bridge_metrics
from soft_match_test import SoftLoop, calibrate_radius, ALPHA, HITS_MIN

warnings.filterwarnings("ignore")

DEFAULT_OUT = os.path.join("vision", "out", "results")

# 预注册网格（docs/246 §1.2，冻结；先于任何结果）
RADIUS_MULTS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
K_GRID = (3, 2)                       # 主分析 k=3；变体 k=2（docs/246 §1.3）

# docs/245 smt_main 回归守卫期望（冻结，docs/245 §三/§四）
D245 = {"r_base": 0.2659, "r0_sc2": 3, "r0_churn": 0.0, "r0_ratio": 0.907701,
        "r1_sc2": 3, "r1_churn": 0.7692, "r1_ratio": 0.951261,
        "bridge_sw": 1.0, "calib_sw": 1.0, "holdout_sw": 1.0,
        "bridge_vid": 1.0, "spurious": 0}


def run_level_soft(lvcode, frames, spans, radius, k, alpha):
    """一个 (r, k) 格的一个级：SoftLoop 独立冷启动跑一遍（每格一次运行，确定性流）。"""
    lc = dict(LOOP_CFG)
    lc["k_consist"] = k
    loop = SoftLoop(radius=radius, alpha=alpha, window=WINDOW, **lc)
    for g in frames:
        loop.step(g)
    base = loop.finalize(max(1, len(frames) // WINDOW), labels=None)

    mae_arr = np.asarray(loop.mae, float)
    q = max(1, len(mae_arr) // 4)
    q1 = float(mae_arr[:q].mean()) if len(mae_arr) >= q else 0.0
    q4 = float(mae_arr[-q:].mean()) if len(mae_arr) >= q else 0.0
    ratio = (q4 / q1) if q1 > 0 else 0.0
    mae_m, mae_sd = mean_sd(list(mae_arr))
    mae_lo, mae_hi = bootstrap_ci(list(mae_arr))

    out = dict(base)
    out.update({"level": lvcode, "name": LEVEL_NAMES[lvcode], "frames": len(frames),
                "n_windows": len(mae_arr),
                "mae_mean_win": round(mae_m, 6), "mae_sd_win": round(mae_sd, 6),
                "mae_ci95": [round(mae_lo, 6), round(mae_hi, 6)],
                "mae_q1": round(q1, 6), "mae_q4": round(q4, 6),
                "ratio": round(ratio, 6), "pin_frac": base["pin_frac"],
                "theta_mean": base["theta_mean"]})
    if lvcode == 31:
        out["bridge"] = bridge_metrics(base, spans)
    return out


def guard_vs_d245(r_base, r0r, r1r):
    """格 (1.0, 3) 与 docs/245 smt_main 逐项比对（容差 1e-4）。"""
    b = r1r["bridge"]
    items = [
        ("r_base", abs(r_base - D245["r_base"]) < 1e-4),
        ("r0_sc2", r0r["sc2"] == D245["r0_sc2"]),
        ("r0_churn", abs(r0r["churn_frac"] - D245["r0_churn"]) < 1e-4),
        ("r0_ratio", abs(r0r["ratio"] - D245["r0_ratio"]) < 1e-4),
        ("r1_sc2", r1r["sc2"] == D245["r1_sc2"]),
        ("r1_churn", abs(r1r["churn_frac"] - D245["r1_churn"]) < 1e-4),
        ("r1_ratio", abs(r1r["ratio"] - D245["r1_ratio"]) < 1e-4),
        ("bridge_sw", abs(b["bridge_corr_switch"] - D245["bridge_sw"]) < 1e-4),
        ("calib_sw", abs(b["calib_switch"] - D245["calib_sw"]) < 1e-4),
        ("holdout_sw", abs(b["holdout_switch"] - D245["holdout_sw"]) < 1e-4),
        ("bridge_vid", abs(b["bridge_corr_video"] - D245["bridge_vid"]) < 1e-4),
        ("spurious", b["spurious"] == D245["spurious"]),
    ]
    ok = all(v for _, v in items)
    detail = ",".join("%s:%d" % (name, int(flag)) for name, flag in items)
    return int(ok), detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius-mult", type=float, required=True,
                    help="半径乘子 M（预注册网格之一：0.5/0.75/1.0/1.5/2.0/3.0/4.0/6.0/8.0）")
    ap.add_argument("--k", type=int, required=True, choices=[2, 3],
                    help="创建滞回 k_consist（预注册 {2, 3}）")
    ap.add_argument("--tag", default="rsw")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    args = ap.parse_args()

    mult = float(args.radius_mult)
    k = int(args.k)
    if mult not in RADIUS_MULTS:
        sys.stderr.write("mult %.2f not in frozen grid %s\n"
                         % (mult, ",".join(str(m) for m in RADIUS_MULTS)))
        return 2
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    # ---- 数据（160x120 灰度，docs/243 预处理冻结；确定性流） ----
    allv = {v: load_video_frames(v) for v in VIDEOS}
    r0 = allv["flamingo"] * 5                      # R0：flamingo x5 = 400 帧
    r1, spans, start = [], [], 0
    for v in VIDEOS:
        fr = allv[v]
        r1.extend(fr)
        spans.append((start, start + len(fr)))
        start += len(fr)
    calib_end = sum(len(allv[v]) for v in CALIB_VIDEOS)
    calib_frames = r1[:calib_end]                  # 校准段 = R1 前 367 帧

    # ---- 半径校准（只依赖校准集；单次确定性 pass；r = M x r_base） ----
    r_base, calib_stats = calibrate_radius(calib_frames)
    radius = float(mult) * float(r_base)

    # ---- 主运行（每格一次）：R0 + R1 ----
    r0r = run_level_soft(30, r0, None, radius, k, ALPHA)
    r1r = run_level_soft(31, r1, spans, radius, k, ALPHA)

    # ---- 判据（预注册冻结，docs/246 §1.4）+ 余量（§1.5） ----
    b31 = r1r["bridge"]
    crit_bridge = int(b31["bridge_corr_switch"] >= 0.5)
    crit_holdout = int(b31["holdout_switch"] >= 0.5)
    crit_churn = int(r0r["churn_frac"] <= 0.5 and r1r["churn_frac"] <= 0.5)
    crit_stable = int(r0r["ratio"] <= 1.5 and r1r["ratio"] <= 1.5)
    cell_pass = int(crit_bridge and crit_holdout and crit_churn and crit_stable)
    m_bridge = round(b31["bridge_corr_switch"] - 0.5, 4)
    m_holdout = round(b31["holdout_switch"] - 0.5, 4)
    m_churn = round(0.5 - max(r0r["churn_frac"], r1r["churn_frac"]), 4)
    m_stable = round(1.5 - max(r0r["ratio"], r1r["ratio"]), 4)

    # ---- 回归守卫（格 (1.0, 3) 复现 docs/245；不进判据） ----
    if abs(mult - 1.0) < 1e-9 and k == 3:
        guard_ok, guard_detail = guard_vs_d245(r_base, r0r, r1r)
    else:
        guard_ok, guard_detail = -1, "NA"

    # ---- 归档 ----
    cfg = {"tag": args.tag, "size": [160, 120], "window": WINDOW,
           "videos": VIDEOS, "calib_videos": CALIB_VIDEOS,
           "holdout_videos": HOLDOUT_VIDEOS,
           "grid": {"radius_mults": list(RADIUS_MULTS), "k": [2, 3],
                    "doc": "docs/246 sec 1.2/1.3"},
           "cell": {"mult": mult, "k": k, "radius": round(radius, 6),
                    "alpha": ALPHA},
           "loop": LOOP_CFG,
           "seed_protocol": "none (deterministic real-pixel stream; window-level stats)"}
    out = {
        "artifact": "radius_sweep",
        "doc_ref": "docs/241, docs/243, docs/244, docs/245, docs/246",
        "config": cfg,
        "calib": calib_stats,
        "levels": {"30": r0r, "31": r1r},
        "criteria": {"bridge": crit_bridge, "heldout": crit_holdout,
                     "churn": crit_churn, "stable": crit_stable,
                     "cell_pass": cell_pass},
        "margins": {"bridge": m_bridge, "heldout": m_holdout,
                    "churn": m_churn, "stable": m_stable},
        "guard": {"d245": guard_ok, "detail": guard_detail},
        "timing": {"elapsed_sec": round(time.time() - t0, 2)},
    }
    res_path = os.path.join(args.out_dir, "%s.json" % args.tag)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 摘要块：ASCII 标签，每行一个数字（顺序固定） ----
    print("R_RSW_ARTIFACT=radius_sweep")
    print("R_RSW_DOC=246")
    print("R_RSW_MULT=%.2f" % mult)
    print("R_RSW_K=%d" % k)
    print("R_RSW_R_BASE=%.4f" % r_base)
    print("R_RSW_RADIUS=%.4f" % radius)
    print("R_RSW_ALPHA=%.4f" % ALPHA)
    for i, r in ((0, r0r), (1, r1r)):
        print("R_RSW_%d_FRAMES=%d" % (i, r["frames"]))
        print("R_RSW_%d_WINDOWS=%d" % (i, r["n_windows"]))
        print("R_RSW_%d_MAE=%.6f" % (i, r["mae_mean_win"]))
        print("R_RSW_%d_MAE_SD=%.6f" % (i, r["mae_sd_win"]))
        print("R_RSW_%d_MAE_LO=%.6f" % (i, r["mae_ci95"][0]))
        print("R_RSW_%d_MAE_HI=%.6f" % (i, r["mae_ci95"][1]))
        print("R_RSW_%d_Q1=%.6f" % (i, r["mae_q1"]))
        print("R_RSW_%d_Q4=%.6f" % (i, r["mae_q4"]))
        print("R_RSW_%d_RATIO=%.6f" % (i, r["ratio"]))
        print("R_RSW_%d_SC1=%d" % (i, r["sc1"]))
        print("R_RSW_%d_SC2=%d" % (i, r["sc2"]))
        print("R_RSW_%d_CHURN=%.4f" % (i, r["churn_frac"]))
        print("R_RSW_%d_PIN=%.4f" % (i, r["pin_frac"]))
        print("R_RSW_%d_THETA=%.6f" % (i, r["theta_mean"]))
    print("R_RSW_BRIDGE_SW=%.4f" % b31["bridge_corr_switch"])
    print("R_RSW_BRIDGE_VID=%.4f" % b31["bridge_corr_video"])
    print("R_RSW_CALIB_SW=%.4f" % b31["calib_switch"])
    print("R_RSW_HOLDOUT_SW=%.4f" % b31["holdout_switch"])
    print("R_RSW_SPURIOUS=%d" % b31["spurious"])
    print("R_RSW_SW_ENC=%s" % ",".join(str(v) for v in b31["sw_enc_flags"]))
    print("R_RSW_CRIT_BRIDGE=%d" % crit_bridge)
    print("R_RSW_CRIT_HELDOUT=%d" % crit_holdout)
    print("R_RSW_CRIT_CHURN=%d" % crit_churn)
    print("R_RSW_CRIT_STABLE=%d" % crit_stable)
    print("R_RSW_CELL_PASS=%d" % cell_pass)
    print("R_RSW_MARGIN_BRIDGE=%.4f" % m_bridge)
    print("R_RSW_MARGIN_HELDOUT=%.4f" % m_holdout)
    print("R_RSW_MARGIN_CHURN=%.4f" % m_churn)
    print("R_RSW_MARGIN_STABLE=%.4f" % m_stable)
    print("R_RSW_GUARD_D245=%d" % guard_ok)
    print("R_RSW_GUARD_DETAIL=%s" % guard_detail)
    print("R_RSW_ELAPSED=%.2f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
