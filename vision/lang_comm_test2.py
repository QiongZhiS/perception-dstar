"""vision/lang_comm_test2.py — 语言线第二格：环境带重调（测 COMM_EMERGES 本体；
交付 docs/269）。

docs/269 §一 预注册冻结，运行后不改。机制/信道/采纳/判据/守卫逐字沿用 docs/268
（import lang_comm_test 复用逐字，零改写）；唯一变化 = 环境常量 m1=2.6/m2=4.2
（docs/269 §1.1 纯 numpy 能量带预计算冻结：B 单目标两态中位能量 481-530/697-746
同落 c0 档 2、比值 1.392-1.449、设计窗口 8/10）。lvcode 沿用 {40: G0, 41: G1,
42: G1n, 43: G2s}、LV_WORLD=41（世界 rng 与 docs/268 同流 -> 预计算锚点逐位成立 +
R_L2H_CELL1_REPRO 旧环境复现锚可行）。

新增（相对 docs/268 实现）：
- R_L2H_CELL1_REPRO：旧环境 (m1=2.2, m2=3.2) 下 G0/G1G/G1T 三臂 10 种子与 docs/268
  §三/§四 冻结数字逐位一致（复现锚：本格机制代码 ≡ docs/268 机制代码）。
- --precompute：纯 numpy 能量带预计算核对（锚点校验 + 冻结候选设计窗口 + 预计算 vs
  回路能量逐位比对 + 网格节选）。机制无关（预测路径 = docs/232/235 既有冻结基础设施，
  无模式表/采纳/判据）。

流（docs/269 §1.7 逐字沿用 docs/268）：G1-T（双回路两阶段）、G1-G（单阶段）、C
（校准前缀 [0,140)）、G0（off）、G1n（null）、G2s（scrambled，(seed+5)%10）。同一
世界种子的四臂共享同一 B 帧。

度量（§1.3 沿用）：M1 预测 MAE；M2 结构；M3 联合残差 JR；M4 信号质量诊断；M5 信号
留出归因（N_C=14）。判据（§1.4 沿用）：C1a JR 配对比值 <=0.85、C1b MAE(G1)==MAE(G0)
abs<1e-9、C2 adopt_frac>=0.6 且采纳种子 compound>=0.5 且 G0/G1n 零采纳、C3 结构保持、
C4 transfer>=0.10 且 >=0.5*calib_baseline。判定映射：COMM_EMERGES/COMM_FLAT（含
COMM_NO_GAIN）/PARTIAL/GUARD_FAIL/LANG_BLOCKED。

守卫（§1.5 沿用 + 新增）：R_L2H_GUARD_D232、R_L2H_GUARD_D235（import lang_comm_test
逐字）、R_L2H_CONSTRUCTION、R_L2H_PREFIX_EQ、R_L2H_TWO_PHASE_EQ、R_L2H_REPRO_MAE、
R_L2H_DETERM（timing/main 逐位一致，外部核对）、R_L2H_SMOKE、R_L2H_CELL1_REPRO。

安全纪律（§1.10）：新文件仅本文件；stdout 只输出 ASCII 标签 + 每行一个数字的
R_L2H_* 摘要块；运行经 powershell 包装重定向到 logs/；数字用 vision/extract_r.py
抽取；禁止读日志/JSON 原文；本格不读 DAVIS。

用法：
  python vision/lang_comm_test2.py --smoke
  python vision/lang_comm_test2.py --precompute
  python vision/lang_comm_test2.py --tag timing
  python vision/lang_comm_test2.py --tag main
"""
import argparse
import hashlib
import json
import os
import sys
import time
import warnings

import numpy as np

import lang_comm_test as l2g
from critical_point import mean_sd, bootstrap_ci, JITTER, N_BOOT, BOOT_SEED
from stream_test import LOOP_CFG

warnings.filterwarnings("ignore")

DEFAULT_OUT = os.path.join("vision", "out", "results")

# ---------------- 冻结常量（docs/269 §1.1/§1.6；运行后不改） ----------------
LVCODES = {40: "G0", 41: "G1", 42: "G1n", 43: "G2s"}   # 沿用 docs/268 同款（§1.7 注明）
LV_WORLD = 41                                            # 世界 rng 的 lvcode 项（同 docs/268）
B_M1 = 2.6                 # 环境常量（docs/269 §1.1 预计算冻结主候选；docs/268 2.20 -> 2.6）
B_M2 = 4.2                 # 环境常量（docs/269 §1.1 预计算冻结主候选；docs/268 3.2 -> 4.2）
BACKUP_M1 = 2.8            # 备用候选（仅诊断轮证明主候选能量带不足时启用，§二 记录）
BACKUP_M2 = 4.8
OLD_M1 = 2.2               # docs/268 旧环境（R_L2H_CELL1_REPRO 复现锚）
OLD_M2 = 3.2
N_C = 14                   # 信号留出校准前缀窗数（帧 [0,140)）
TRANSFER_FLOOR = 0.10      # C4 绝对下界（docs/264 冻结值逐字移植）
TRANSFER_REL = 0.5         # C4 相对保持系数（docs/264 冻结值逐字移植）
JR_RATIO_MAX = 0.85        # C1a 阈值（冻结）
ADOPT_FRAC_MIN = 0.6       # C2 阈值（冻结）
COMPOUND_MIN = 0.5         # C2 阈值（冻结）
N_FRAMES = l2g.N_FRAMES
WINDOW = l2g.WINDOW
ENERGY_BINS = l2g.ENERGY_BINS
DESIGN_MIN_E = 450.0       # 设计窗口：c0 档 2 下界（energy_bins[1]，docs/269 §1.1）
DESIGN_RATIO = 1.30        # 设计窗口：两态中位比下界（= 1+delta_rel，冻结）
K_CONSIST = 3              # 设计窗口：两态窗口数下界（k_consist 代理，冻结）

# ---------------- 纯 numpy 能量带预计算（docs/269 §1.1 冻结协议；机制无关） ----------------
# 预计算器 = 世界几何（docs/268 §1.1 B 视域帧生成）+ 预测路径（docs/232/235 CPLoop
# 的 bg_fast EWMA + sigma_hat 自适应阈值 + 事件掩码窗口并集）。不含模式表/采纳/判据
# 任何机制。逐窗口事件能量由预测路径唯一决定 -> 与真实回路能量逐位等价（锚点证明）。


def precompute_world(seed, m1, m2, n_frames=N_FRAMES):
    """(种子, m1, m2) 的 B 视域帧 + 逐窗口真值 ctx 标签（纯 numpy；docs/268 §1.1
    世界构造的机制无关复制：rng 同 docs/268，噪声场/A 帧与 docs/268 逐位相同）。"""
    rng = np.random.default_rng(seed * 7919 + LV_WORLD * 104729 + 13)
    noise_mult = rng.uniform(1 - JITTER, 1 + JITTER) if JITTER > 0 else 1.0
    sigma = l2g.NOISE_SIGMA * noise_mult
    bg = np.zeros((120, 160), np.float32)
    step = 24
    for y in range(0, 120, step):
        for x in range(0, 160, step):
            bg[y:y + step, x:x + step] = 64 if ((x // step) + (y // step)) % 2 == 0 else 96
    th_a = rng.uniform(0, 2 * np.pi)
    th_b = rng.uniform(0, 2 * np.pi)
    ctx_prev = 1
    frames_b = []
    per_frame = []
    yy, xx = np.ogrid[:120, :160]
    for t in range(n_frames):
        th_a += 2 * np.pi * l2g.A_FREQ / 30.0
        ax = l2g.A_CENTER[0] + l2g.A_ORBIT * np.cos(th_a)
        ay = l2g.A_CENTER[1] + l2g.A_ORBIT * np.sin(th_a)
        m = m1 if ctx_prev == 0 else m2
        th_b += 2 * np.pi * l2g.B_FREQ * m / 30.0
        bx = l2g.B_CENTER[0] + l2g.B_ORBIT * np.cos(th_b)
        by = l2g.B_CENTER[1] + l2g.B_ORBIT * np.sin(th_b)
        noise = rng.normal(0, sigma, (120, 160)).astype(np.float32)
        img = bg.copy()
        mask = (xx - int(bx)) ** 2 + (yy - int(by)) ** 2 <= l2g.DISK_R * l2g.DISK_R
        img[mask] = l2g.OBJ_GRAY
        frames_b.append(np.clip(img + noise, 0, 255).astype(np.uint8))
        ctx = 0 if (ax - bx) < 0 else 1
        per_frame.append((ctx, m))
        ctx_prev = ctx
    wl = []
    for i in range(n_frames // WINDOW):
        seg = per_frame[i * WINDOW:(i + 1) * WINDOW]
        ctxs = [s[0] for s in seg]
        n0 = sum(1 for c in ctxs if c == 0)
        wl.append(0 if n0 > len(ctxs) - n0 else 1)
    return frames_b, wl


def precompute_energies(frames_b):
    """预测路径事件能量（CPLoop.step 逐字复制；finalize 冲刷残留缓冲）。"""
    a_fast = 0.5
    thresh = 0.15
    deadband = 0.015
    k_theta = 6.0
    k_db = 1.5
    thresh_max = 0.6
    db_max = 0.15
    bg_fast = None
    prev_L = None
    sigma_hat = 0.0
    ev_win = None
    buf = 0
    wins = []
    for g in frames_b:
        L = np.log(np.maximum(g.astype(np.float32), 1.0))
        if bg_fast is None:
            bg_fast = L.copy()
            prev_L = L.copy()
            sigma_hat = 0.0
            continue
        d = L - prev_L
        sig = float(np.median(np.abs(d - np.median(d))) / 0.6745)
        sigma_hat = 0.15 * sig + 0.85 * sigma_hat
        theta = float(np.clip(max(thresh, k_theta * sigma_hat), thresh, thresh_max))
        db = float(np.clip(max(deadband, k_db * sigma_hat), deadband, db_max))
        prev_L = L
        bg_fast = a_fast * L + (1 - a_fast) * bg_fast
        c = L - bg_fast
        r = np.abs(c)
        rd = np.maximum(r - db, 0.0)
        ev = rd > theta
        if ev_win is None:
            ev_win = ev.copy()
        else:
            ev_win |= ev
        buf += 1
        if buf >= WINDOW:
            wins.append(int(ev_win.sum()))
            ev_win = None
            buf = 0
    if ev_win is not None and buf > 0:
        wins.append(int(ev_win.sum()))
    return wins


def precompute_diag(seed, m1, m2):
    """(种子, m1, m2) 的两态中位能量/档位/比值/停留诊断（机制无关）。"""
    frames_b, wl = precompute_world(seed, m1, m2)
    E = precompute_energies(frames_b)
    by = {0: [], 1: []}
    for w, c in enumerate(wl):
        if w < len(E):
            by[c].append(float(E[w]))
    meds = {c: (float(np.median(v)) if v else None) for c, v in by.items()}
    bands = {c: (None if meds[c] is None else l2g._band(meds[c], ENERGY_BINS))
             for c in (0, 1)}
    ratio = None
    if meds[0] and meds[1]:
        ratio = max(meds[0], meds[1]) / max(1e-9, min(meds[0], meds[1]))
    seq = wl
    flips = 0
    dwell = {0: [], 1: []}
    prev = None
    run = 0
    for c in seq:
        if c == prev:
            run += 1
        else:
            if prev is not None:
                dwell[prev].append(run)
                flips += 1
            prev, run = c, 1
    if prev is not None:
        dwell[prev].append(run)
    n0 = sum(1 for c in seq if c == 0)
    n1 = sum(1 for c in seq if c == 1)
    seg_ok = (max(dwell[0], default=0) >= K_CONSIST
              and max(dwell[1], default=0) >= K_CONSIST)
    return dict(med0=meds[0], med1=meds[1], band0=bands[0], band1=bands[1],
                ratio=ratio, flips=flips, seg_ok=int(seg_ok), n0=n0, n1=n1,
                E=E, wl=wl)


def design_window(d):
    """设计窗口（docs/269 §1.1 冻结判据）：band0==2 且 band1==2 且比值 >=1.30 且
    两态窗口数均 >=3。"""
    return int(d["band0"] == 2 and d["band1"] == 2
               and d["ratio"] is not None and d["ratio"] >= DESIGN_RATIO
               and d["n0"] >= K_CONSIST and d["n1"] >= K_CONSIST)


def precompute_main():
    """--precompute：能量带预计算核对（§1.1 协议 + §二 诊断）。
    ① 锚点校验（docs/268 §二 实测逐位）；② 冻结主/备候选设计窗口；③ 预计算 vs
    回路能量逐位比对（run_b_only off，机制能量 = 预测路径能量）；④ 网格节选。"""
    # ① 锚点校验
    anchors = {
        (2.0, 3.2): [223.5, 292.0, 389.0, 405.0, 405.0, 408.0, 417.0, 423.0, 423.0, 440.5],
        (2.2, 3.2): [264.0, 308.0, 422.5, 436.0, 441.0, 449.5, 450.0, 451.0, 459.0, 460.0],
    }
    anchor_ok = 1
    for (m1, m2), exp_e0 in anchors.items():
        got = sorted(round(precompute_diag(s, m1, m2)["med0"], 1) for s in range(10))
        ok = int(all(abs(g - e) < 0.15 for g, e in zip(got, exp_e0)))
        anchor_ok &= ok
        print("R_L2H_PRECOMPUTE_ANCHOR_M1=%.2f_M2=%.2f" % (m1, m2))
        print("R_L2H_PRECOMPUTE_ANCHOR_OK=%d" % ok)
        print("R_L2H_PRECOMPUTE_ANCHOR_E0=%s" % ",".join("%.1f" % v for v in got))
    # ② 冻结主/备候选设计窗口（预计算）
    for (m1, m2, tag) in ((B_M1, B_M2, "MAIN"), (BACKUP_M1, BACKUP_M2, "BACKUP")):
        rows = [precompute_diag(s, m1, m2) for s in range(10)]
        wins = [design_window(d) for d in rows]
        print("R_L2H_PRECOMPUTE_%s_M1=%.2f" % (tag, m1))
        print("R_L2H_PRECOMPUTE_%s_M2=%.2f" % (tag, m2))
        print("R_L2H_PRECOMPUTE_%s_WIN=%d" % (tag, int(sum(wins))))
        print("R_L2H_PRECOMPUTE_%s_PER=%s" % (tag, ",".join(str(v) for v in wins)))
        for s in range(10):
            d = rows[s]
            print("R_L2H_PRECOMPUTE_%s_S%d_E0=%.1f" % (tag, s, d["med0"]))
            print("R_L2H_PRECOMPUTE_%s_S%d_E1=%.1f" % (tag, s, d["med1"]))
            print("R_L2H_PRECOMPUTE_%s_S%d_B0=%s" % (tag, s, d["band0"]))
            print("R_L2H_PRECOMPUTE_%s_S%d_B1=%s" % (tag, s, d["band1"]))
            print("R_L2H_PRECOMPUTE_%s_S%d_RATIO=%.4f" % (tag, s, d["ratio"]))
            print("R_L2H_PRECOMPUTE_%s_S%d_FLIPS=%d" % (tag, s, d["flips"]))
    # ③ 预计算 vs 回路能量逐位比对（主候选；run_b_only off 的 energy_trace）
    eq_oks = []
    for s in range(10):
        frames_b, wl = precompute_world(s, B_M1, B_M2)
        pE = precompute_energies(frames_b)
        labels = [dict(ctx=v, b_mult=1.0, a_regime=None) for v in wl]
        _, loop_b = l2g.run_b_only(frames_b, labels, "off", signal_fn=None)
        lE = loop_b.energy_trace
        same = (len(pE) == len(lE) and all(int(a) == int(b) for a, b in zip(pE, lE)))
        eq_oks.append(int(same))
        print("R_L2H_PRECOMPUTE_EQ_S%d=%d" % (s, int(same)))
    print("R_L2H_PRECOMPUTE_EQ=%d" % int(all(eq_oks)))
    # ④ 网格节选（m1 x m2 -> 设计窗口计数；全 48 组合快扫）
    grid_rows = []
    for m1 in (2.6, 2.8, 3.0, 3.2):
        counts = []
        for m2 in (3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0):
            n = int(sum(design_window(precompute_diag(s, m1, m2)) for s in range(10)))
            counts.append(n)
        grid_rows.append("m1=%.1f:%s" % (m1, ",".join(str(v) for v in counts)))
    print("R_L2H_PRECOMPUTE_GRID=%s" % ";".join(grid_rows))
    return 0 if (anchor_ok == 1 and all(eq_oks)) else 1


# ---------------- 构造冒烟（docs/268 §1.5-8 同款；合成帧，非数据） ----------------
def smoke_main2():
    """构造冒烟：各信道模式构造运行正常；off 与 null 逐窗一致；G0 无提升；
    DEGENERATE_OK；归因不变量；baseline=0 不崩。"""
    results = {}
    fb = l2g._synth_frames(30)
    fa = l2g._synth_frames(30, y0=26)
    labels = [dict(ctx=1, b_mult=1.0, a_regime=None)] * 3
    outs = {}
    loops = {}
    for mode in ("off", "comm", "null", "scrambled"):
        if mode in ("comm", "scrambled"):
            out, loop_b, _, _ = l2g.run_dual(0, fa, fb, labels, mode=mode,
                                             two_phase=False, n_c=N_C)
        else:
            sig_fn = (lambda w: l2g.NULL_SIGNAL) if mode == "null" else None
            out, loop_b = l2g.run_b_only(fb, labels, mode, signal_fn=sig_fn)
        outs[mode] = out
        loops[mode] = loop_b
        results["construct_" + mode] = int(
            isinstance(out, dict) and len(out.get("mae_trace", [])) >= 1
            and isinstance(out.get("sc1"), int))
    off, nul = loops["off"], loops["null"]
    results["off_null_eq"] = int(
        off.energy_trace == nul.energy_trace
        and off.up_trace == nul.up_trace
        and [s[0] for s in off.sig_trace] == [s[0] for s in nul.sig_trace]
        and [s[1] for s in off.sig_trace] == [s[1] for s in nul.sig_trace]
        and off.match_trace == nul.match_trace)
    results["g0_no_promo"] = int(outs["off"]["n_promo"] == 0)
    results["degenerate_ok"] = int(outs["off"]["n_promo"] == 0
                                   and outs["off"]["compound_frac"] == 0.0)
    fb2 = l2g._synth_frames(100)
    fa2 = l2g._synth_frames(100, y0=26)
    lab2 = [dict(ctx=1, b_mult=1.0, a_regime=None)] * 10
    _, loop2, _, _ = l2g.run_dual(0, fa2, fb2, lab2, mode="comm",
                                  two_phase=False, n_c=5)
    att = l2g.attribution(loop2, n_c=5)
    rates = [att["calib_baseline"], att["transfer_adopted_hit_rate"]]
    results["attr_invariants"] = int(
        all(0.0 <= r <= 1.0 for r in rates)
        and att["transfer_adopted_hits"] >= 0
        and att["n_heldout_eligible"] >= 0
        and att["transfer_adopted_hit_rate"] <= 1.0)
    _, loop3 = l2g.run_b_only(fb2, lab2, "off", signal_fn=None)
    att3 = l2g.attribution(loop3, n_c=5)
    results["baseline_zero_ok"] = int(att3["calib_baseline"] == 0.0
                                      and att3["transfer_adopted_hits"] == 0
                                      and att3["transfer_adopted_hit_rate"] == 0.0)
    for k in sorted(results):
        print("R_L2H_SMOKE_%s=%d" % (k.upper(), results[k]))
    return 0 if all(results.values()) else 1


# ---------------- R_L2H_CELL1_REPRO（docs/269 §1.5 新增复现锚） ----------------
# 期望数字 = docs/268 §三/§四 冻结值（旧环境 m1=2.2/m2=3.2 实测打印；同代码路径 ->
# 期望位精确，容差取打印精度 + 余量）。来源行：docs/268 §3.1（逐种子 JR）、§3.3
# （聚合 MAE/JR/adopt/comp/transfer/calib/fid）。
CELL1_EXP = {
    "jr_ratio": [1.0000, 0.8602, 0.6736, 1.0000, 1.0000, 0.5482,
                 1.0000, 0.9979, 1.0000, 0.5437],
    "jr_g0": [0.0556, 0.0860, 0.0832, 0.0513, 0.1233, 0.0905, 0.0930,
              0.2024, 0.0779, 0.1163],
    "jr_g1": [0.0556, 0.0739, 0.0560, 0.0513, 0.1233, 0.0496, 0.0930,
              0.2020, 0.0779, 0.0632],
    "adopt_frac": 0.5000,
    "comp_adopted": 0.9000,
    "mae": 0.022355,
    "mae_sd": 0.002223,
    "jr_ratio_mean": 0.862361,
    "transfer_mean": 0.260,
    "calib_mean": 0.150,
    "fid_mean": 0.9542,
    "sc2_g0": 1.2,
    "sc2_g1": 1.6,
    "promo_g1": 0.5,
}


def repro_cell1():
    """旧环境 (2.2, 3.2)：G0/G1G/G1T 三臂 10 种子，与 docs/268 §三/§四 冻结数字
    逐位一致（浮点容差 1e-4/1e-3 按打印精度；离散精确）。返回 (ok, detail)。"""
    seeds = list(range(10))
    jr_ratios, jr_g0s, jr_g1s = [], [], []
    mae_g0s, mae_g1s, fids = [], [], []
    trans, calibs = [], []
    n_promos, sc2_g1s, sc2_g0s, comps = [], [], [], []
    for s in seeds:
        fa, fb, wl = l2g.make_world(s, m1=OLD_M1, m2=OLD_M2)
        out_t, loop_t, _, _ = l2g.run_dual(s, fa, fb, wl, mode="comm",
                                           two_phase=True, n_c=N_C)
        jr1 = l2g.jr_b(loop_t)[0]
        att = l2g.attribution(loop_t, N_C)
        out_g, loop_g, _, _ = l2g.run_dual(s, fa, fb, wl, mode="comm",
                                           two_phase=False)
        out0, loop0 = l2g.run_b_only(fb, wl, "off", signal_fn=None)
        jr0 = l2g.jr_b(loop0)[0]
        jr_ratios.append(jr1 / max(jr0, 1e-12))
        jr_g0s.append(jr0)
        jr_g1s.append(jr1)
        mae_g0s.append(out0["mae_mean"])
        mae_g1s.append(out_t["mae_mean"])
        fids.append(out_t["ctx_fidelity"])
        trans.append(att["transfer_adopted_hit_rate"])
        calibs.append(att["calib_baseline"])
        n_promos.append(out_t["n_promo"])
        sc2_g1s.append(out_t["sc2"])
        sc2_g0s.append(out0["sc2"])
        comps.append(out_t["compound_frac"])
    adopt_frac = float(np.mean([v >= 1 for v in n_promos]))
    adopted_comp = [c for c, n in zip(comps, n_promos) if n >= 1]
    comp_adopted = float(np.mean(adopted_comp)) if adopted_comp else 0.0
    mae_m, mae_sd = mean_sd(mae_g0s)
    jr_ratio_m, _ = mean_sd(jr_ratios)
    trans_m, _ = mean_sd(trans)
    calib_m, _ = mean_sd(calibs)
    fid_m, _ = mean_sd(fids)
    sc2_g0_m, _ = mean_sd(sc2_g0s)
    sc2_g1_m, _ = mean_sd(sc2_g1s)
    promo_m, _ = mean_sd(n_promos)

    def chk(name, got, exp, tol=1e-4):
        return name, int(abs(got - exp) < tol), got

    checks = []
    for s in seeds:
        checks.append(chk("JR_RATIO_S%d" % s, jr_ratios[s],
                          CELL1_EXP["jr_ratio"][s], 1e-4))
        checks.append(chk("JR_G0_S%d" % s, jr_g0s[s], CELL1_EXP["jr_g0"][s], 1e-4))
        checks.append(chk("JR_G1_S%d" % s, jr_g1s[s], CELL1_EXP["jr_g1"][s], 1e-4))
    checks.append(chk("ADOPT_FRAC", adopt_frac, CELL1_EXP["adopt_frac"], 1e-6))
    checks.append(chk("COMP_ADOPTED", comp_adopted, CELL1_EXP["comp_adopted"], 1e-4))
    checks.append(chk("MAE_G0", mae_m, CELL1_EXP["mae"], 1e-5))
    checks.append(chk("MAE_SD", mae_sd, CELL1_EXP["mae_sd"], 1e-5))
    checks.append(chk("JR_RATIO_MEAN", jr_ratio_m, CELL1_EXP["jr_ratio_mean"], 1e-4))
    checks.append(chk("TRANSFER_MEAN", trans_m, CELL1_EXP["transfer_mean"], 1e-3))
    checks.append(chk("CALIB_MEAN", calib_m, CELL1_EXP["calib_mean"], 1e-3))
    checks.append(chk("FID_MEAN", fid_m, CELL1_EXP["fid_mean"], 1e-4))
    checks.append(chk("SC2_G0", sc2_g0_m, CELL1_EXP["sc2_g0"], 1e-4))
    checks.append(chk("SC2_G1", sc2_g1_m, CELL1_EXP["sc2_g1"], 1e-4))
    checks.append(chk("PROMO_G1", promo_m, CELL1_EXP["promo_g1"], 1e-4))
    ok = int(all(c[1] == 1 for c in checks))
    return ok, dict(checks=checks, adopt_frac=adopt_frac,
                    comp_adopted=comp_adopted, mae=mae_m, mae_sd=mae_sd,
                    jr_ratio_mean=jr_ratio_m, transfer=trans_m,
                    calib=calib_m, fid=fid_m, sc2_g0=sc2_g0_m,
                    sc2_g1=sc2_g1_m, promo=promo_m)


# ---------------- 单元记录（checkpoint/resume 用；自包含） ----------------
def unit_record2(arm, seed, out, loop_b, loop_a=None, snap=None, jr=None, att=None):
    return {
        "arm": arm, "seed": seed,
        "mae": list(loop_b.mae),
        "E": [int(v) for v in loop_b.energy_trace],
        "U": [int(v) for v in loop_b.up_trace],
        "sig": [[None if x is None else int(x) for x in s]
                for s in loop_b.sig_trace],
        "matched": [[w, None if k is None else [int(v) for v in k]]
                    for w, k in loop_b.match_trace],
        "finalize": {k: out[k] for k in (
            "mae_mean", "sc1", "sc2", "sc_late", "sc4", "arity2_stable",
            "arity3_stable", "compound_frac", "churn_frac", "n_promo",
            "ctx_fidelity", "ctx_purity", "n_created_total", "n_retired")},
        "ratio": l2g.mae_ratio(loop_b),
        "pop_hash": l2g.pop_hash(loop_b),
        "snapshot": snap,
        "jr": None if jr is None else list(jr),
        "att": att,
        "a_sc2": (None if loop_a is None
                  else loop_a.finalize(max(1, len(loop_a.energy_trace)),
                                       None)["sc2"]),
        "entry_log": [{"key": [int(k) for k in e["key"]], "arity": e["arity"],
                       "hits": e["hits"], "created": e["created"],
                       "retired": bool(e.get("retired")),
                       "retired_at": e.get("retired_at")}
                      for e in out["entry_log"]],
    }


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="l2h")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--precompute", action="store_true",
                    help="能量带预计算核对（§1.1 协议 + §二 诊断；机制无关）")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.smoke:
        return smoke_main2()
    if args.precompute:
        return precompute_main()
    t0 = time.time()
    seeds = list(range(10))

    cfg = {"tag": args.tag, "n_seeds": len(seeds), "frames": N_FRAMES,
           "window": WINDOW, "n_c": N_C, "jitter": JITTER,
           "b_m1": B_M1, "b_m2": B_M2, "backup": {"m1": BACKUP_M1, "m2": BACKUP_M2},
           "old_env": {"m1": OLD_M1, "m2": OLD_M2},
           "noise_sigma": l2g.NOISE_SIGMA,
           "world": {"a_center": list(l2g.A_CENTER), "a_orbit": l2g.A_ORBIT,
                     "a_freq": l2g.A_FREQ, "b_center": list(l2g.B_CENTER),
                     "b_orbit": l2g.B_ORBIT, "b_freq": l2g.B_FREQ,
                     "rng_lvcode": LV_WORLD},
           "channel": {"sparse_px": l2g.SIG_SPARSE_PX,
                       "null_signal": l2g.NULL_SIGNAL},
           "criteria": {"jr_ratio_max": JR_RATIO_MAX,
                        "adopt_frac_min": ADOPT_FRAC_MIN,
                        "compound_min": COMPOUND_MIN,
                        "transfer_floor": TRANSFER_FLOOR,
                        "transfer_rel": TRANSFER_REL},
           "loop": LOOP_CFG}
    ck_tag = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    ckpt_path = os.path.join(args.out_dir, "ckpt_l2h_%s.json" % ck_tag)

    done = {}
    if args.resume and not args.no_resume and os.path.exists(ckpt_path):
        with open(ckpt_path, encoding="utf-8") as f:
            done = json.load(f).get("per_unit", {})

    per_unit = dict(done)
    worlds = {s: l2g.make_world(s, m1=B_M1, m2=B_M2) for s in seeds}

    def need(arm, s):
        return "%s_%d" % (arm, s) not in per_unit

    for s in seeds:
        fa, fb, wl = worlds[s]
        if need("G1T", s):
            out, loop_b, loop_a, snap = l2g.run_dual(s, fa, fb, wl, mode="comm",
                                                     two_phase=True, n_c=N_C)
            per_unit["G1T_%d" % s] = unit_record2(
                "G1T", s, out, loop_b, loop_a=loop_a, snap=snap,
                jr=l2g.jr_b(loop_b), att=l2g.attribution(loop_b, N_C))
            print("PROGRESS", flush=True)
        if need("G1G", s):
            out, loop_b, loop_a, _ = l2g.run_dual(s, fa, fb, wl, mode="comm",
                                                  two_phase=False)
            per_unit["G1G_%d" % s] = unit_record2(
                "G1G", s, out, loop_b, loop_a=loop_a)
            print("PROGRESS", flush=True)
        if need("C", s):
            out, loop_b, _, snap = l2g.run_dual(s, fa[:140], fb[:140], wl[:14],
                                                mode="comm", two_phase=False,
                                                want_end_snap=True)
            per_unit["C_%d" % s] = unit_record2("C", s, out, loop_b, snap=snap)
            print("PROGRESS", flush=True)
        if need("G0", s):
            out, loop_b = l2g.run_b_only(fb, wl, "off", signal_fn=None)
            per_unit["G0_%d" % s] = unit_record2("G0", s, out, loop_b,
                                                 jr=l2g.jr_b(loop_b))
            print("PROGRESS", flush=True)
        if need("G1N", s):
            out, loop_b = l2g.run_b_only(fb, wl, "null", signal_fn=None)
            per_unit["G1N_%d" % s] = unit_record2("G1N", s, out, loop_b)
            print("PROGRESS", flush=True)
        if need("G2S", s):
            other = (s + 5) % 10
            a_other = l2g.run_a_signal(worlds[other][0])
            out, loop_b = l2g.run_b_only(fb, wl, "scrambled",
                                         signal_fn=lambda w: a_other.sA_trace[w])
            per_unit["G2S_%d" % s] = unit_record2("G2S", s, out, loop_b)
            print("PROGRESS", flush=True)
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump({"config": cfg, "per_unit": per_unit},
                      f, ensure_ascii=False, indent=1)

    # ---- 守卫 ----
    g232_ok, g232 = l2g.guard_d232()
    g235_ok, g235 = l2g.guard_d235()
    c1_ok, c1_detail = repro_cell1()

    # ---- 跨单元核对（PREFIX_EQ / TWO_PHASE_EQ / REPRO_MAE / CONSTRUCTION） ----
    prefix_oks = []
    two_phase_oks = []
    repro_oks = []
    cons_g0 = []
    cons_g1n = []
    seed_rows = []
    for s in seeds:
        t = per_unit["G1T_%d" % s]
        g = per_unit["G1G_%d" % s]
        c = per_unit["C_%d" % s]
        g0 = per_unit["G0_%d" % s]
        gn = per_unit["G1N_%d" % s]
        g2 = per_unit["G2S_%d" % s]
        prefix_oks.append(l2g.prefix_eq(c["snapshot"], t["snapshot"]))
        two_phase_oks.append(l2g.two_phase_eq(g, t))
        repro_oks.append(l2g.repro_mae(g0, g))
        cons_g0.append(int(g0["finalize"]["compound_frac"] == 0.0))
        cons_g1n.append(int(gn["finalize"]["compound_frac"] == 0.0))
        jr_g0 = g0["jr"][0]
        jr_g1 = t["jr"][0]
        jr_ratio = jr_g1 / max(jr_g0, 1e-12)
        att = t["att"]
        fd = l2g.flip_diag(worlds[s][2])
        ed = l2g.energy_diag(g["E"], worlds[s][2])
        seed_rows.append(dict(
            seed=s, g0=g0, g1t=t, g1g=g, c=c, g1n=gn, g2s=g2,
            jr_g0=jr_g0, jr_g1=jr_g1, jr_ratio=jr_ratio,
            transfer_rate=att["transfer_adopted_hit_rate"],
            calib_baseline=att["calib_baseline"],
            transfer_hits=att["transfer_adopted_hits"],
            held_elig=att["n_heldout_eligible"],
            first_promo=att["first_promo_win"],
            promo_wins=att["promo_wins"],
            flip=fd, energy=ed,
            design_win=design_window(dict(
                band0=ed["band0"], band1=ed["band1"], ratio=ed["ratio"],
                n0=fd["n0"], n1=fd["n1"]))))

    prefix_ok = int(all(prefix_oks))
    two_phase_ok = int(all(two_phase_oks))
    repro_ok = int(all(repro_oks))
    construction_ok = int(all(cons_g0) and all(cons_g1n))

    # ---- 聚合（G1 臂主数字 = G1-G 单阶段流，§1.7；T 两阶段与其逐窗一致） ----
    UNIT_ARM = {"G0": "G0", "G1": "G1G", "G1N": "G1N", "G2S": "G2S"}

    def col(arm, key):
        ua = UNIT_ARM[arm]
        return [per_unit["%s_%d" % (ua, s)]["finalize"][key] for s in seeds]

    agg = {}
    for arm in ("G0", "G1", "G1N", "G2S"):
        mae_m, mae_sd = mean_sd(col(arm, "mae_mean"))
        sc2_m, _ = mean_sd(col(arm, "sc2"))
        comp_m, comp_sd = mean_sd(col(arm, "compound_frac"))
        churn_m, _ = mean_sd(col(arm, "churn_frac"))
        promo_m, _ = mean_sd(col(arm, "n_promo"))
        agg[arm] = dict(mae_mean=mae_m, mae_sd=mae_sd, sc2_mean=sc2_m,
                        comp_mean=comp_m, comp_sd=comp_sd,
                        churn_mean=churn_m, promo_mean=promo_m)
    agg["G1"]["ratio_mean"] = float(np.mean([per_unit["G1G_%d" % s]["ratio"]
                                             for s in seeds]))
    agg["G0"]["ratio_mean"] = float(np.mean([per_unit["G0_%d" % s]["ratio"]
                                             for s in seeds]))
    agg["G1"]["fid_mean"] = float(np.mean(col("G1", "ctx_fidelity")))
    agg["G1"]["mae_ci95"] = list(bootstrap_ci(col("G1", "mae_mean")))
    agg["G1"]["comp_ci95"] = list(bootstrap_ci(col("G1", "compound_frac")))
    jr_g0s = [r["jr_g0"] for r in seed_rows]
    jr_g1s = [r["jr_g1"] for r in seed_rows]
    jr_ratios = [r["jr_ratio"] for r in seed_rows]
    jr_g0_m, jr_g0_sd = mean_sd(jr_g0s)
    jr_g1_m, jr_g1_sd = mean_sd(jr_g1s)
    jr_ratio_m, jr_ratio_sd = mean_sd(jr_ratios)
    transfer_rates = [r["transfer_rate"] for r in seed_rows]
    calib_bases = [r["calib_baseline"] for r in seed_rows]
    transfer_m, transfer_sd = mean_sd(transfer_rates)
    calib_m, calib_sd = mean_sd(calib_bases)
    adopt_frac = float(np.mean([r["g1t"]["finalize"]["n_promo"] >= 1
                                for r in seed_rows]))
    adopted_comp = [r["g1t"]["finalize"]["compound_frac"] for r in seed_rows
                    if r["g1t"]["finalize"]["n_promo"] >= 1]
    comp_adopted = float(np.mean(adopted_comp)) if adopted_comp else 0.0
    a_sc2s = [r["g1t"]["a_sc2"] for r in seed_rows]
    a_sc2_m = float(np.mean([v for v in a_sc2s if v is not None]))
    fid_m = agg["G1"]["fid_mean"]
    eratios = [r["energy"]["ratio"] for r in seed_rows
               if r["energy"]["ratio"] is not None]
    eratio_m = float(np.mean(eratios)) if eratios else 0.0
    band0_ok = float(np.mean([r["energy"]["band0"] == 2 for r in seed_rows]))
    band1_ok = float(np.mean([r["energy"]["band1"] == 2 for r in seed_rows]))
    seg_ok_frac = float(np.mean([r["flip"]["seg_ok"] for r in seed_rows]))
    flips_m = float(np.mean([r["flip"]["flips"] for r in seed_rows]))
    design_win_frac = float(np.mean([r["design_win"] for r in seed_rows]))

    # ---- 判据（docs/269 §1.4 冻结，docs/268 逐字） ----
    c1a = int(jr_ratio_m <= JR_RATIO_MAX)
    c1b = repro_ok
    c2 = int(adopt_frac >= ADOPT_FRAC_MIN and comp_adopted >= COMPOUND_MIN
             and all(cons_g0) and all(cons_g1n))
    c3_g0 = int(agg["G0"]["sc2_mean"] >= 1 and agg["G0"]["churn_mean"] < 0.3
                and agg["G0"]["ratio_mean"] <= 1.5)
    c3_g1 = int(agg["G1"]["sc2_mean"] >= 1 and agg["G1"]["churn_mean"] < 0.3
                and agg["G1"]["ratio_mean"] <= 1.5)
    c3_a = int(a_sc2_m >= 1)
    c3 = int(c3_g0 and c3_g1 and c3_a)
    c4 = int(transfer_m >= TRANSFER_FLOOR
             and transfer_m >= TRANSFER_REL * calib_m)

    # 数据可用性（LANG_BLOCKED 预防）：逐种子 JR 有窗口、留出 eligible 非空
    blocked = int(not (all(r["g0"]["jr"][1] >= 1 for r in seed_rows)
                       and all(r["g1t"]["jr"][1] >= 1 for r in seed_rows)
                       and all(r["held_elig"] >= 1 for r in seed_rows)
                       and all(r["transfer_rate"] == r["transfer_rate"]
                               for r in seed_rows)))

    guards_ok = (g232_ok == 1 and g235_ok == 1 and construction_ok == 1
                 and prefix_ok == 1 and two_phase_ok == 1 and repro_ok == 1
                 and c1_ok == 1)
    if not guards_ok:
        verdict = "GUARD_FAIL"
        vnote = ("guard(s) failed: D232=%d, D235=%d, CONSTRUCTION=%d, "
                 "PREFIX_EQ=%d, TWO_PHASE_EQ=%d, REPRO_MAE=%d, CELL1_REPRO=%d "
                 "-> implementation drift; fix implementation, do not judge "
                 "mechanism" % (g232_ok, g235_ok, construction_ok, prefix_ok,
                                two_phase_ok, repro_ok, c1_ok))
    elif blocked:
        verdict = "LANG_BLOCKED"
        vnote = ("synthetic environment unavailable (per-seed eligible/JR "
                 "windows missing); see per-seed numbers")
    elif c1a and c1b and c2 and c3 and c4:
        verdict = "COMM_EMERGES"
        vnote = ("criteria C1-C4 all pass and all guards pass: other-agent "
                 "signal spontaneously adopted (ledger-driven, no switch) and "
                 "adoption lowers joint residual; structural behavior kept; "
                 "holdout transfer holds -> minimal measurable precursor of "
                 "language emergence (docs/269 sec 1.4)")
    elif not c2:
        verdict = "COMM_FLAT"
        vnote = ("C2 fails (signal not adopted): adopt_frac=%.4f < 0.6 or "
                 "G0/G1n non-zero adoption -> honest negative: no information "
                 "gain or channel ineffective; see decomposition" % adopt_frac)
    elif not (c1a and c1b):
        verdict = "COMM_FLAT"
        vnote = ("C2 passes but C1 fails (adopted but joint residual not "
                 "lowered) -> COMM_NO_GAIN sub-form: JR_ratio=%.4f > 0.85; "
                 "mechanism has no rollback (docs/268 sec 5.6)" % jr_ratio_m)
    else:
        why = []
        if not c3:
            why.append("C3 STRUCTURE_KEEP fails (see SC2/churn/ratio/A-SC2)")
        if not c4:
            why.append("C4 SIG_HOLDOUT fails (transfer_rate=%.4f < floor 0.10 "
                       "or < 0.5*calib %.4f)" % (transfer_m, calib_m))
        verdict = "PARTIAL"
        vnote = "; ".join(why) + " (see R_L2H_CRIT* numbers)"

    # ---- 工件（自描述 JSON） ----
    out = {
        "artifact": "lang_comm_test2",
        "doc_ref": "docs/63, docs/228, docs/232, docs/235, docs/247, docs/258, "
                   "docs/264, docs/266, docs/268, docs/269",
        "config": cfg,
        "guards": {"d232": {"ok": g232_ok, "detail": g232},
                   "d235": {"ok": g235_ok, "detail": g235},
                   "construction": {"ok": construction_ok,
                                    "g0_zero": int(sum(cons_g0)),
                                    "g1n_zero": int(sum(cons_g1n))},
                   "prefix_eq": prefix_ok, "two_phase_eq": two_phase_ok,
                   "repro_mae": repro_ok, "cell1_repro": {"ok": c1_ok},
                   "prefix_per_seed": prefix_oks,
                   "two_phase_per_seed": two_phase_oks,
                   "repro_per_seed": repro_oks},
        "per_seed": seed_rows,
        "arms": {k: {"name": LVCODES[40] if k == "G0" else
                     (LVCODES[41] if k == "G1" else
                      (LVCODES[42] if k == "G1N" else LVCODES[43])),
                     "mean_sd": agg[k],
                     "mae_ci95": agg[k].get("mae_ci95"),
                     "comp_ci95": agg[k].get("comp_ci95")}
                 for k in ("G0", "G1", "G1N", "G2S")},
        "jr": {"g0_mean": jr_g0_m, "g0_sd": jr_g0_sd, "g1_mean": jr_g1_m,
               "g1_sd": jr_g1_sd, "ratio_mean": jr_ratio_m,
               "ratio_sd": jr_ratio_sd, "per_seed_ratios": jr_ratios,
               "ci95": list(bootstrap_ci(jr_g1s))},
        "adoption": {"adopt_frac": adopt_frac, "comp_adopted": comp_adopted,
                     "adopted_comp_per_seed": adopted_comp,
                     "a_sc2_mean": a_sc2_m},
        "holdout": {"transfer_rate_mean": transfer_m,
                    "transfer_rate_sd": transfer_sd,
                    "calib_baseline_mean": calib_m,
                    "calib_baseline_sd": calib_sd,
                    "per_seed_rates": transfer_rates},
        "diag": {"fidelity_mean": fid_m, "energy_ratio_mean": eratio_m,
                 "band0_frac": band0_ok, "band1_frac": band1_ok,
                 "seg_ok_frac": seg_ok_frac, "flips_mean": flips_m,
                 "design_win_frac": design_win_frac},
        "criteria": {"c1a_comm_value": c1a, "c1b_mae_eq": c1b,
                     "c2_adoption_emerges": c2, "c3_structure_keep": c3,
                     "c4_sig_holdout": c4,
                     "jr_ratio": jr_ratio_m, "adopt_frac": adopt_frac,
                     "comp_adopted": comp_adopted, "transfer_rate": transfer_m,
                     "calib_baseline": calib_m,
                     "c3_g0": c3_g0, "c3_g1": c3_g1, "c3_a": c3_a},
        "verdict": {"verdict": verdict, "note": vnote},
        "timing": {"elapsed_sec": round(time.time() - t0, 2)},
    }
    res_path = os.path.join(args.out_dir, "l2h_%s.json" % args.tag)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 摘要块：ASCII 标签，每行一个数字（顺序固定；无中文/日志/JSON） ----
    print("R_L2H_TAG=%s" % args.tag)
    print("R_L2H_SEEDS=%d" % len(seeds))
    print("R_L2H_FRAMES=%d" % N_FRAMES)
    print("R_L2H_WINDOWS=%d" % (N_FRAMES // WINDOW))
    print("R_L2H_NC=%d" % N_C)
    print("R_L2H_M1=%.1f" % B_M1)
    print("R_L2H_M2=%.1f" % B_M2)
    print("R_L2H_GUARD_D232=%d" % g232_ok)
    print("R_L2H_GUARD_D232_SC2=%s" % ",".join(str(v) for v in g232["sc2"]))
    print("R_L2H_GUARD_D232_SCLATE_FRAC=%.4f" % g232["sc_late_frac"])
    print("R_L2H_GUARD_D232_SC4=%.4f" % g232["sc4"])
    print("R_L2H_GUARD_D232_MAE=%.6f" % g232["mae"])
    print("R_L2H_GUARD_D232_MAE_SD=%.6f" % g232["mae_sd"])
    print("R_L2H_GUARD_D232_PIN=%.4f" % g232["pin"])
    print("R_L2H_GUARD_D232_CLASS=%s" % g232["cls"])
    print("R_L2H_GUARD_D235=%d" % g235_ok)
    for lv in (21, 22):
        d = g235[lv]
        print("R_L2H_GUARD_D235_C%d_OK=%d" % (lv, d["ok"]))
        print("R_L2H_GUARD_D235_C%d_MAE=%.6f" % (lv, d["mae"]))
        print("R_L2H_GUARD_D235_C%d_MAE_SD=%.6f" % (lv, d["mae_sd"]))
        print("R_L2H_GUARD_D235_C%d_SC2=%.4f" % (lv, d["sc2"]))
        print("R_L2H_GUARD_D235_C%d_SC2_SD=%.4f" % (lv, d["sc2_sd"]))
        print("R_L2H_GUARD_D235_C%d_COMP=%.4f" % (lv, d["comp"]))
        print("R_L2H_GUARD_D235_C%d_CHURN=%.4f" % (lv, d["churn"]))
        print("R_L2H_GUARD_D235_C%d_FID=%.4f" % (lv, d["fid"]))
    print("R_L2H_CONSTRUCTION=%d" % construction_ok)
    print("R_L2H_CONSTRUCTION_G0=%d" % int(sum(cons_g0)))
    print("R_L2H_CONSTRUCTION_G1N=%d" % int(sum(cons_g1n)))
    print("R_L2H_PREFIX_EQ=%d" % prefix_ok)
    print("R_L2H_TWO_PHASE_EQ=%d" % two_phase_ok)
    print("R_L2H_REPRO_MAE=%d" % repro_ok)
    print("R_L2H_REPRO_MAE_SEEDS=%d" % int(sum(repro_oks)))
    print("R_L2H_CELL1_REPRO=%d" % c1_ok)
    for (name, ok, got) in c1_detail["checks"]:
        print("R_L2H_CELL1_%s=%d" % (name, ok))
        print("R_L2H_CELL1_%s_VAL=%.6f" % (name, got))
    print("R_L2H_CELL1_ADOPT=%.4f" % c1_detail["adopt_frac"])
    print("R_L2H_CELL1_COMP_ADOPTED=%.4f" % c1_detail["comp_adopted"])
    print("R_L2H_CELL1_MAE=%.6f" % c1_detail["mae"])
    print("R_L2H_CELL1_TRANSFER=%.4f" % c1_detail["transfer"])
    print("R_L2H_CELL1_CALIB=%.4f" % c1_detail["calib"])
    print("R_L2H_CELL1_FID=%.4f" % c1_detail["fid"])
    for r in seed_rows:
        s = r["seed"]
        print("R_L2H_SEED=%d" % s)
        g0f = r["g0"]["finalize"]
        g1f = r["g1t"]["finalize"]
        gnf = r["g1n"]["finalize"]
        g2f = r["g2s"]["finalize"]
        print("R_L2H_S%d_G0_MAE=%.6f" % (s, g0f["mae_mean"]))
        print("R_L2H_S%d_G0_SC2=%d" % (s, g0f["sc2"]))
        print("R_L2H_S%d_G0_COMP=%.4f" % (s, g0f["compound_frac"]))
        print("R_L2H_S%d_G0_CHURN=%.4f" % (s, g0f["churn_frac"]))
        print("R_L2H_S%d_G0_PROMO=%d" % (s, g0f["n_promo"]))
        print("R_L2H_S%d_G0_RATIO=%.6f" % (s, r["g0"]["ratio"]))
        print("R_L2H_S%d_G1_MAE=%.6f" % (s, g1f["mae_mean"]))
        print("R_L2H_S%d_G1_SC2=%d" % (s, g1f["sc2"]))
        print("R_L2H_S%d_G1_COMP=%.4f" % (s, g1f["compound_frac"]))
        print("R_L2H_S%d_G1_CHURN=%.4f" % (s, g1f["churn_frac"]))
        print("R_L2H_S%d_G1_PROMO=%d" % (s, g1f["n_promo"]))
        print("R_L2H_S%d_G1_RATIO=%.6f" % (s, r["g1g"]["ratio"]))
        print("R_L2H_S%d_G1_FID=%.4f" % (s, g1f["ctx_fidelity"]))
        print("R_L2H_S%d_G1N_COMP=%.4f" % (s, gnf["compound_frac"]))
        print("R_L2H_S%d_G1N_PROMO=%d" % (s, gnf["n_promo"]))
        print("R_L2H_S%d_G2S_COMP=%.4f" % (s, g2f["compound_frac"]))
        print("R_L2H_S%d_G2S_PROMO=%d" % (s, g2f["n_promo"]))
        print("R_L2H_S%d_A_SC2=%s" % (s, ("NA" if r["g1t"]["a_sc2"] is None
                                          else str(r["g1t"]["a_sc2"]))))
        print("R_L2H_S%d_JR_G0=%.6f" % (s, r["jr_g0"]))
        print("R_L2H_S%d_JR_G1=%.6f" % (s, r["jr_g1"]))
        print("R_L2H_S%d_JR_RATIO=%.6f" % (s, r["jr_ratio"]))
        print("R_L2H_S%d_TRANSFER_RATE=%.6f" % (s, r["transfer_rate"]))
        print("R_L2H_S%d_CALIB_BASE=%.6f" % (s, r["calib_baseline"]))
        print("R_L2H_S%d_TRANSFER_HITS=%d" % (s, r["transfer_hits"]))
        print("R_L2H_S%d_HELD_ELIG=%d" % (s, r["held_elig"]))
        print("R_L2H_S%d_FIRST_PROMO=%s" % (s, ("NA" if r["first_promo"] is None
                                                else str(r["first_promo"]))))
        print("R_L2H_S%d_DESIGN_WIN=%d" % (s, r["design_win"]))
        ed = r["energy"]
        print("R_L2H_S%d_DIAG_E0_MED=%s" % (s, ("NA" if ed["med0"] is None
                                                else "%.1f" % ed["med0"])))
        print("R_L2H_S%d_DIAG_E1_MED=%s" % (s, ("NA" if ed["med1"] is None
                                                else "%.1f" % ed["med1"])))
        print("R_L2H_S%d_DIAG_EBAND0=%s" % (s, ("NA" if ed["band0"] is None
                                                else str(ed["band0"]))))
        print("R_L2H_S%d_DIAG_EBAND1=%s" % (s, ("NA" if ed["band1"] is None
                                                else str(ed["band1"]))))
        print("R_L2H_S%d_DIAG_ERATIO=%s" % (s, ("NA" if ed["ratio"] is None
                                                else "%.4f" % ed["ratio"])))
        print("R_L2H_S%d_DIAG_FLIPS=%d" % (s, r["flip"]["flips"]))
        print("R_L2H_S%d_DIAG_SEGOK=%d" % (s, r["flip"]["seg_ok"]))
    print("R_L2H_MAE_G0=%.6f" % agg["G0"]["mae_mean"])
    print("R_L2H_MAE_G0_SD=%.6f" % agg["G0"]["mae_sd"])
    print("R_L2H_MAE_G1=%.6f" % agg["G1"]["mae_mean"])
    print("R_L2H_MAE_G1_SD=%.6f" % agg["G1"]["mae_sd"])
    print("R_L2H_MAE_G1N=%.6f" % agg["G1N"]["mae_mean"])
    print("R_L2H_MAE_G1N_SD=%.6f" % agg["G1N"]["mae_sd"])
    print("R_L2H_MAE_G2S=%.6f" % agg["G2S"]["mae_mean"])
    print("R_L2H_MAE_G2S_SD=%.6f" % agg["G2S"]["mae_sd"])
    print("R_L2H_SC2_G0=%.4f" % agg["G0"]["sc2_mean"])
    print("R_L2H_SC2_G1=%.4f" % agg["G1"]["sc2_mean"])
    print("R_L2H_SC2_G1N=%.4f" % agg["G1N"]["sc2_mean"])
    print("R_L2H_SC2_G2S=%.4f" % agg["G2S"]["sc2_mean"])
    print("R_L2H_COMP_G0=%.4f" % agg["G0"]["comp_mean"])
    print("R_L2H_COMP_G1=%.4f" % agg["G1"]["comp_mean"])
    print("R_L2H_COMP_G1_SD=%.4f" % agg["G1"]["comp_sd"])
    print("R_L2H_COMP_G1N=%.4f" % agg["G1N"]["comp_mean"])
    print("R_L2H_COMP_G2S=%.4f" % agg["G2S"]["comp_mean"])
    print("R_L2H_CHURN_G0=%.4f" % agg["G0"]["churn_mean"])
    print("R_L2H_CHURN_G1=%.4f" % agg["G1"]["churn_mean"])
    print("R_L2H_PROMO_G1=%.4f" % agg["G1"]["promo_mean"])
    print("R_L2H_FID_G1=%.4f" % agg["G1"]["fid_mean"])
    print("R_L2H_MAE_CI95_G1_LO=%.6f" % agg["G1"]["mae_ci95"][0])
    print("R_L2H_MAE_CI95_G1_HI=%.6f" % agg["G1"]["mae_ci95"][1])
    print("R_L2H_COMP_CI95_G1_LO=%.4f" % agg["G1"]["comp_ci95"][0])
    print("R_L2H_COMP_CI95_G1_HI=%.4f" % agg["G1"]["comp_ci95"][1])
    print("R_L2H_JR_G0=%.6f" % jr_g0_m)
    print("R_L2H_JR_G0_SD=%.6f" % jr_g0_sd)
    print("R_L2H_JR_G1=%.6f" % jr_g1_m)
    print("R_L2H_JR_G1_SD=%.6f" % jr_g1_sd)
    print("R_L2H_JR_RATIO=%.6f" % jr_ratio_m)
    print("R_L2H_JR_RATIO_SD=%.6f" % jr_ratio_sd)
    print("R_L2H_ADOPT_FRAC=%.4f" % adopt_frac)
    print("R_L2H_COMP_ADOPTED=%.4f" % comp_adopted)
    print("R_L2H_A_SC2_MEAN=%.4f" % a_sc2_m)
    print("R_L2H_TRANSFER_RATE_MEAN=%.6f" % transfer_m)
    print("R_L2H_TRANSFER_RATE_SD=%.6f" % transfer_sd)
    print("R_L2H_CALIB_BASE_MEAN=%.6f" % calib_m)
    print("R_L2H_DIAG_CTXFID=%.4f" % fid_m)
    print("R_L2H_DIAG_ERATIO_MEAN=%.4f" % eratio_m)
    print("R_L2H_DIAG_BAND0_FRAC=%.4f" % band0_ok)
    print("R_L2H_DIAG_BAND1_FRAC=%.4f" % band1_ok)
    print("R_L2H_DIAG_SEGOK_FRAC=%.4f" % seg_ok_frac)
    print("R_L2H_DIAG_FLIPS_MEAN=%.2f" % flips_m)
    print("R_L2H_DESIGN_WIN_FRAC=%.4f" % design_win_frac)
    print("R_L2H_CRIT_C1A=%d" % c1a)
    print("R_L2H_CRIT_C1B=%d" % c1b)
    print("R_L2H_CRIT_C2=%d" % c2)
    print("R_L2H_CRIT_C3=%d" % c3)
    print("R_L2H_CRIT_C4=%d" % c4)
    print("R_L2H_CRIT_C3_G0=%d" % c3_g0)
    print("R_L2H_CRIT_C3_G1=%d" % c3_g1)
    print("R_L2H_CRIT_C3_A=%d" % c3_a)
    print("R_L2H_CRIT1_JR_RATIO=%.6f" % jr_ratio_m)
    print("R_L2H_CRIT2_ADOPT_FRAC=%.4f" % adopt_frac)
    print("R_L2H_CRIT2_COMP_ADOPTED=%.4f" % comp_adopted)
    print("R_L2H_CRIT2_G0_ZERO=%d" % int(sum(cons_g0)))
    print("R_L2H_CRIT2_G1N_ZERO=%d" % int(sum(cons_g1n)))
    print("R_L2H_CRIT4_TRANSFER_RATE=%.6f" % transfer_m)
    print("R_L2H_CRIT4_CALIB_BASE=%.6f" % calib_m)
    print("R_L2H_CRIT4_FLOOR_OK=%d" % int(transfer_m >= TRANSFER_FLOOR))
    print("R_L2H_CRIT4_REL_OK=%d" % int(transfer_m >= TRANSFER_REL * calib_m))
    print("R_L2H_VERDICT=%s" % verdict)
    print("R_L2H_VERDICT_NOTE=%s" % vnote)
    print("R_L2H_ELAPSED=%.2f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
