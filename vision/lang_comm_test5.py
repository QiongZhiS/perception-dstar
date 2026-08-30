"""vision/lang_comm_test5.py — 语言线第五格：双向同窗递送（CH2 同窗：B 先闭窗管线、
CH1 转滞后——攻"一窗滞后"根因；交付 docs/273）。

docs/273 §一 预注册冻结，运行后不改。机制基座 = docs/271 逐字继承（import
lang_comm_test4/lang_comm_test3/lang_comm_test2/lang_comm_test 复用，零改写）：
BiLangLoop（双向信道 CH1+CH2、对称世界 a1=1.6/a2=2.6、门条件④ GATE_PURITY_MIN=0.80
双回路）、环境 m1=2.6/m2=4.2、判据阈值、留出划分（N_C=14）全沿用（五格同尺可比）。
**本格唯一机制改动 = 递送时序重设计（docs/271 §五13 预告）**：

1. 帧同步管线改 **B 先闭窗**（run_bidi 的 timing 参数）：
   - timing="b_first"（本格默认）：每帧 B.step ->（B 闭窗发布 s_B(w) -> A.set_signal）->
     A.step ->（A 闭窗发布 s_A(w) -> B.set_signal）；窗口收尾冲刷顺序镜像（B 先冲刷）。
     **CH2 同窗**（A 的窗口 w 读到 s_B(w)——攻 docs/271 根因：ctx_A 保真度 0.7739 ->
     0.9292、A 采纳 5/10 -> 8/10，§二 C1 设计期实测）+ **CH1 转一窗滞后**（B 的窗口 w
     读到 s_A(w-1)——B 承担滞后：ctx_B 保真度 0.9292 -> 0.5913、B 采纳 2/10，§二 C1
     设计期实测；预期 verdict = ONE_WAY（A 侧采纳成立的镜像形态），如实预注册）。
   - timing="a_first"（= docs/271 run_bidi 逐字；CELL4_REPRO 用）。
2. 守卫锚定重建：**R_L2K_CELL4_REPRO**（旧时序（a_first）判据相关臂 ≡ docs/271 逐位：
   adopt_A 0.5/adopt_B 0.8/JR 比值 A 0.895134/B 0.606402/transfer A 0.4/B 0.8/
   spurious A {S8}/B {S1,S3,S8}/逐种子采纳集/首提升窗/门纯度——证明本格唯一差异是递送
   时序）；**R_L2K_CELL4_REPRO_B**（新时序单向关闭诊断：B 侧继承数字 ≠ docs/270 的
   诚实量化——CH1 转滞后是机制改动，不进判据）。

流（docs/273 §1.8）：双向世界（B 先闭窗）M-T（两阶段）/M-G（单阶段）/C（前缀）/G0A
（双 off）/G1n（双 null）/G2s（双 scrambled，(seed+5)%10 + 随机二元注入）/W（均匀世界
双 ON）；CELL4_REPRO-A（旧时序 12 臂）+ CELL4_REPRO-B（新时序单向关闭诊断）。同一世界
种子的双向四臂共享同一世界帧。

度量（§1.4 双侧化，docs/271 逐字）：M1 预测 MAE；M2 结构；M3 联合残差 JR；M4 信号质量
诊断 + 门诊断（双侧）；M5 信号留出归因（双侧，N_C=14）。判据（§1.5）：C1 MUTUAL_VALUE
（双侧 JR 配对比值 <=0.85 + 双侧 MAE==G0A abs<1e-9）、C2 MUTUAL_ADOPTION（双侧 adopt
>=0.6 且采纳 compound>=0.5 且 G0A/G1n 双侧零采纳）、C3 CLEAN_KEEP（双侧
spurious(G2s)==0）、C4 STRUCTURE_KEEP/SIG_HOLDOUT（双侧结构 + 双侧 transfer>=0.10 且
>=0.5*calib）。判定映射：MUTUAL_EMERGES/ONE_WAY/MUTUAL_FLAT（含 MUTUAL_NO_GAIN）/
PARTIAL/GUARD_FAIL/LANG_BLOCKED。

守卫（§1.6）：R_L2K_GUARD_D232、R_L2K_GUARD_D235、R_L2K_CELL2_REPRO（import docs/270
逐字）、R_L2K_CONSTRUCTION、R_L2K_PREFIX_EQ（双侧）、R_L2K_TWO_PHASE_EQ（双侧）、
R_L2K_REPRO_MAE（双侧）、R_L2K_DETERM（timing/main 逐位一致，外部核对）、R_L2K_SMOKE
（含双侧门语义 + CH2 同窗/CH1 滞后语义）、R_L2K_CELL4_REPRO（旧时序 ≡ docs/271 逐位）、
R_L2K_CELL4_REPRO_B（新时序单向关闭诊断）、R_L2K_WORLD_EQ（世界零改动，直接沿用）、
R_L2K_PRECOMPUTE（环境预计算核对）。

安全纪律（§1.11）：新文件仅本文件；stdout 只输出 ASCII 标签 + 每行一个数字的 R_L2K_*
摘要块；运行经 powershell 包装重定向到 logs/；数字用 vision/extract_r.py 抽取；禁止读
日志/JSON 原文；本格不读 DAVIS。

用法：
  python vision/lang_comm_test5.py --smoke
  python vision/lang_comm_test5.py --precompute
  python vision/lang_comm_test5.py --tag timing
  python vision/lang_comm_test5.py --tag main
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
import lang_comm_test2 as l2h
import lang_comm_test3 as l2i
import lang_comm_test4 as l2j
from critical_point import mean_sd, bootstrap_ci, JITTER, N_BOOT, BOOT_SEED
from stream_test import LOOP_CFG

warnings.filterwarnings("ignore")

DEFAULT_OUT = os.path.join("vision", "out", "results")

# ---------------- 冻结常量（docs/273 §1.1/§1.7；运行后不改；docs/271 逐字沿用） ----------------
LVCODES = l2h.LVCODES
LV_WORLD = l2h.LV_WORLD
B_M1 = l2h.B_M1                 # 2.6（B 环境常量，docs/269 冻结）
B_M2 = l2h.B_M2                 # 4.2
A_M1 = l2j.A_M1                 # 1.6（A 环境常量，docs/271 §1.1 冻结）
A_M2 = l2j.A_M2                 # 2.6
A_MIRROR = l2j.A_MIRROR
N_C = l2h.N_C                   # 14
TRANSFER_FLOOR = l2h.TRANSFER_FLOOR
TRANSFER_REL = l2h.TRANSFER_REL
JR_RATIO_MAX = l2h.JR_RATIO_MAX
ADOPT_FRAC_MIN = l2h.ADOPT_FRAC_MIN
COMPOUND_MIN = l2h.COMPOUND_MIN
N_FRAMES = l2h.N_FRAMES
WINDOW = l2h.WINDOW
ENERGY_BINS = l2h.ENERGY_BINS
GATE_PURITY_MIN = l2i.GATE_PURITY_MIN      # 0.80（docs/270 冻结，双侧生效）
TIMING = "b_first"                         # 本格唯一机制改动：B 先闭窗（CH2 同窗/CH1 滞后）


# ---------------- 双向流运行（docs/273 §1.1/§1.8 冻结；唯一机制改动 = 递送时序） ----------------
def run_bidi(fa, fb, wl, ch1="comm", ch2="comm", a_mode=None, two_phase=False,
             n_c=N_C, gate=GATE_PURITY_MIN, want_end_snap=False,
             sig1_fn=None, sig2_fn=None, timing=TIMING):
    """双向世界双回路（A/B 帧同步步进）。timing：
    - "b_first"（本格默认）：B 先闭窗——CH2 同窗（A 的窗口 w 读到 s_B(w)）、CH1 一窗滞后
      （B 的窗口 w 读到 s_A(w-1)）；窗口收尾冲刷顺序镜像（B 先冲刷发布 s_B -> A 末窗同窗）。
    - "a_first"（= docs/271 run_bidi 逐字）：A 先闭窗——CH1 同窗（docs/268 逐字）、CH2
      一窗滞后；CELL4_REPRO 用。
    ch1 = B 侧信道模式、ch2 = A 侧信道模式（off/null/scrambled 在窗起始预置信号；
    comm 由对方闭窗发布设置）；a_mode：A 回路模式（默认 = ch2；CELL4_REPRO_B 用
    a_mode='pixel' 复现 docs/270 世界）。sig1_fn(w)/sig2_fn(w) = scrambled 信号源。
    返回 (out_a, loop_a, out_b, loop_b, snap_a, snap_b)。"""
    loop_a = l2j.BiLangLoop(mode=(a_mode if a_mode is not None else ch2),
                            self_side="upper", publish_side="upper",
                            gate_purity_min=gate, window=WINDOW, **LOOP_CFG)
    loop_b = l2j.BiLangLoop(mode=ch1, self_side="lower", publish_side="lower",
                            gate_purity_min=gate, window=WINDOW, **LOOP_CFG)
    n_frames = len(fb)
    n_w = n_frames // WINDOW
    phases = ([(0, n_c * WINDOW), (n_c * WINDOW, n_frames)] if two_phase
              else [(0, n_frames)])
    snap_a = snap_b = None
    a_closed = b_closed = 0
    for (f0, f1) in phases:
        for k in range(f0, f1):
            if timing == "b_first":
                # ---- B 先闭窗（本格冻结）：CH2 同窗 / CH1 一窗滞后 ----
                if k > 0 and len(loop_b._frame_buf) == 0:
                    if ch1 == "off":
                        loop_b.set_signal(None)
                    elif ch1 == "null":
                        loop_b.set_signal(l2g.NULL_SIGNAL)
                    elif ch1 == "scrambled":
                        loop_b.set_signal(sig1_fn(a_closed) if sig1_fn is not None
                                          else None)
                prev_b = len(loop_b.sA_trace)
                loop_b.step(fb[k])
                if len(loop_b.sA_trace) > prev_b:
                    if ch2 == "comm":        # CH2 发布 -> A 同窗读取（B 先闭窗）
                        loop_a.set_signal(loop_b.sA_trace[b_closed])
                    b_closed += 1
                if k > 0 and len(loop_a._frame_buf) == 0:
                    if ch2 == "off":
                        loop_a.set_signal(None)
                    elif ch2 == "null":
                        loop_a.set_signal(l2g.NULL_SIGNAL)
                    elif ch2 == "scrambled":
                        loop_a.set_signal(sig2_fn(b_closed) if sig2_fn is not None
                                          else None)
                prev_a = len(loop_a.sA_trace)
                loop_a.step(fa[k])
                if len(loop_a.sA_trace) > prev_a:
                    if ch1 == "comm":        # CH1 发布 -> B 下一窗读取（一窗滞后）
                        loop_b.set_signal(loop_a.sA_trace[a_closed])
                    a_closed += 1
            else:
                # ---- A 先闭窗（docs/271 run_bidi 逐字；CELL4_REPRO 用）----
                if k > 0 and len(loop_a._frame_buf) == 0:
                    if ch2 == "off":
                        loop_a.set_signal(None)
                    elif ch2 == "null":
                        loop_a.set_signal(l2g.NULL_SIGNAL)
                    elif ch2 == "scrambled":
                        loop_a.set_signal(sig2_fn(b_closed) if sig2_fn is not None
                                          else None)
                prev_a = len(loop_a.sA_trace)
                loop_a.step(fa[k])
                if len(loop_a.sA_trace) > prev_a:
                    if ch1 == "comm":        # CH1 同窗递送（docs/268 逐字）
                        loop_b.set_signal(loop_a.sA_trace[a_closed])
                    a_closed += 1
                if k > 0 and len(loop_b._frame_buf) == 0:
                    if ch1 == "off":
                        loop_b.set_signal(None)
                    elif ch1 == "null":
                        loop_b.set_signal(l2g.NULL_SIGNAL)
                    elif ch1 == "scrambled":
                        loop_b.set_signal(sig1_fn(a_closed) if sig1_fn is not None
                                          else None)
                prev_b = len(loop_b.sA_trace)
                loop_b.step(fb[k])
                if len(loop_b.sA_trace) > prev_b:
                    if ch2 == "comm":        # CH2 发布 -> A 下一窗读取（一窗滞后）
                        loop_a.set_signal(loop_b.sA_trace[b_closed])
                    b_closed += 1
        if two_phase and f0 == 0:
            snap_a = l2g.snapshot_b(loop_a)
            snap_b = l2g.snapshot_b(loop_b)
    if want_end_snap and snap_a is None:
        snap_a = l2g.snapshot_b(loop_a)
        snap_b = l2g.snapshot_b(loop_b)
    # 收尾冲刷（镜像）：b_first = B 先冲刷（发布 s_B -> A 末窗同窗读取）；a_first = A 先
    # 冲刷（docs/271 逐字：A 发布 s_A -> B 的 finalize 冲刷窗读取）
    if timing == "b_first":
        if len(loop_b._frame_buf):
            loop_b.finalize(n_w, None)
            if len(loop_b.sA_trace) > b_closed and ch2 == "comm":
                loop_a.set_signal(loop_b.sA_trace[-1])
        if len(loop_a._frame_buf):
            loop_a.finalize(n_w, None)
            if len(loop_a.sA_trace) > a_closed and ch1 == "comm":
                loop_b.set_signal(loop_a.sA_trace[-1])
    else:
        if len(loop_a._frame_buf):
            loop_a.finalize(n_w, None)
            if len(loop_a.sA_trace) > a_closed and ch1 == "comm":
                loop_b.set_signal(loop_a.sA_trace[-1])
        if len(loop_b._frame_buf):
            loop_b.finalize(n_w, None)
            if len(loop_b.sA_trace) > b_closed and ch2 == "comm":
                loop_a.set_signal(loop_b.sA_trace[-1])
    labels_a = [dict(ctx=1 - lb["ctx"], b_mult=lb["b_mult"],
                     a_regime=lb["a_regime"]) for lb in wl]   # A 侧真值 = 1-ctx（对称）
    out_a = loop_a.finalize(n_w, labels_a)
    out_b = loop_b.finalize(n_w, wl)
    return out_a, loop_a, out_b, loop_b, snap_a, snap_b


# ---------------- 环境预计算（docs/273 §1.6；世界零改动，直接沿用方法模板） ----------------
def precompute_ok_main():
    """主运行内的环境预计算核对（R_L2K_PRECOMPUTE）：① 锚点：a_ctx_dep=False 世界 B 侧
    中位能量 ≡ docs/269 预计算表（10/10 逐位）；② 冻结候选 (1.6, 2.6, F)：双向世界 A/B
    双侧设计窗口 8/10 + 设计种子窗口级能量双侧零跌破 450；③ 预计算 vs 回路能量逐位比对
    （本格 run_bidi b_first 双 off 的 energy_trace，双侧）。"""
    seeds = list(range(10))
    exp_e0 = [481.0, 512.0, 513.5, 510.0, 367.0, 510.0, 520.5, 310.0, 517.0, 530.0]
    exp_e1 = [697.0, 724.0, 725.0, 712.0, 538.0, 710.0, 740.0, 452.0, 728.0, 746.0]
    anchor_ok = 1
    for s in seeds:
        _, fb, wl = l2j.make_bidi_world(s, a_ctx_dep=False)
        E = l2h.precompute_energies(fb)
        ctxs = [lb["ctx"] for lb in wl]
        d = l2j.bidi_diag(ctxs, E)
        ok = int(abs(d["med0"] - exp_e0[s]) < 0.15
                 and abs(d["med1"] - exp_e1[s]) < 0.15)
        anchor_ok &= ok
    dw_a = dw_b = 0
    under_a = under_b = 0
    eq_oks = []
    per_a = []
    per_b = []
    for s in seeds:
        fa, fb, wl = l2j.make_bidi_world(s, a1=A_M1, a2=A_M2, mirror=A_MIRROR)
        ctxs = [lb["ctx"] for lb in wl]
        Ea = l2h.precompute_energies(fa)
        Eb = l2h.precompute_energies(fb)
        da_ = l2j.bidi_diag(ctxs, Ea)
        db_ = l2j.bidi_diag(ctxs, Eb)
        wa = l2j.design_window_a(da_)
        wb = l2j.design_window_b(db_)
        dw_a += wa
        dw_b += wb
        per_a.append(wa)
        per_b.append(wb)
        if wa and wb:               # 设计种子（双侧）窗口级零跌破核对
            under_a += sum(1 for e in Ea if e < l2j.BAND2_LO)
            under_b += sum(1 for e in Eb if e < l2j.BAND2_LO)
        # 预计算 vs 回路能量（本格 run_bidi b_first 双 off；能量为预测路径，与时序无关）
        out_a, loop_a, out_b, loop_b, _, _ = run_bidi(
            fa, fb, wl, ch1="off", ch2="off", a_mode="off", timing="b_first")
        eq_oks.append(int(len(Ea) == len(loop_a.energy_trace)
                          and all(int(x) == int(y) for x, y in
                                  zip(Ea, loop_a.energy_trace))
                          and len(Eb) == len(loop_b.energy_trace)
                          and all(int(x) == int(y) for x, y in
                                  zip(Eb, loop_b.energy_trace))))
    ok = int(anchor_ok == 1 and dw_a == 8 and dw_b == 8
             and under_a == 0 and under_b == 0 and all(eq_oks))
    return ok, dict(anchor=anchor_ok, dw_a=dw_a, dw_b=dw_b,
                    per_a=per_a, per_b=per_b, under_a=under_a,
                    under_b=under_b, eq=int(all(eq_oks)))


def precompute_main():
    """--precompute：完整环境预计算核对（§1.1 协议复现 + 网格节选 + 锚点；世界零改动）。"""
    seeds = list(range(10))
    exp_e0 = [481.0, 512.0, 513.5, 510.0, 367.0, 510.0, 520.5, 310.0, 517.0, 530.0]
    exp_e1 = [697.0, 724.0, 725.0, 712.0, 538.0, 710.0, 740.0, 452.0, 728.0, 746.0]
    anch = 1
    for s in seeds:
        _, fb, wl = l2j.make_bidi_world(s, a_ctx_dep=False)
        E = l2h.precompute_energies(fb)
        ctxs = [lb["ctx"] for lb in wl]
        d = l2j.bidi_diag(ctxs, E)
        ok = int(abs(d["med0"] - exp_e0[s]) < 0.15 and abs(d["med1"] - exp_e1[s]) < 0.15)
        anch &= ok
        print("R_L2K_PRECOMPUTE_ANCHOR_S%d=%d" % (s, ok))
        print("R_L2K_PRECOMPUTE_ANCHOR_S%d_E0=%.1f" % (s, d["med0"]))
        print("R_L2K_PRECOMPUTE_ANCHOR_S%d_E1=%.1f" % (s, d["med1"]))
    print("R_L2K_PRECOMPUTE_ANCHOR=%d" % anch)
    dw_a = dw_b = 0
    under_a = under_b = 0
    for s in seeds:
        fa, fb, wl = l2j.make_bidi_world(s, a1=A_M1, a2=A_M2, mirror=A_MIRROR)
        ctxs = [lb["ctx"] for lb in wl]
        Ea = l2h.precompute_energies(fa)
        Eb = l2h.precompute_energies(fb)
        da_ = l2j.bidi_diag(ctxs, Ea)
        db_ = l2j.bidi_diag(ctxs, Eb)
        wa = l2j.design_window_a(da_)
        wb = l2j.design_window_b(db_)
        dw_a += wa
        dw_b += wb
        if wa and wb:
            under_a += sum(1 for e in Ea if e < l2j.BAND2_LO)
            under_b += sum(1 for e in Eb if e < l2j.BAND2_LO)
        print("R_L2K_PRECOMPUTE_S%d_A_E0=%.1f" % (s, da_["med0"]))
        print("R_L2K_PRECOMPUTE_S%d_A_E1=%.1f" % (s, da_["med1"]))
        print("R_L2K_PRECOMPUTE_S%d_A_RATIO=%.4f" % (s, da_["ratio"]))
        print("R_L2K_PRECOMPUTE_S%d_A_DW=%d" % (s, wa))
        print("R_L2K_PRECOMPUTE_S%d_B_E0=%.1f" % (s, db_["med0"]))
        print("R_L2K_PRECOMPUTE_S%d_B_E1=%.1f" % (s, db_["med1"]))
        print("R_L2K_PRECOMPUTE_S%d_B_RATIO=%.4f" % (s, db_["ratio"]))
        print("R_L2K_PRECOMPUTE_S%d_B_DW=%d" % (s, wb))
    print("R_L2K_PRECOMPUTE_DW_A=%d" % dw_a)
    print("R_L2K_PRECOMPUTE_DW_B=%d" % dw_b)
    print("R_L2K_PRECOMPUTE_UNDER450_A=%d" % under_a)
    print("R_L2K_PRECOMPUTE_UNDER450_B=%d" % under_b)
    grid_rows = []
    for (a1, a2) in ((1.4, 2.2), (1.4, 2.4), (1.4, 2.6), (1.6, 2.4), (1.6, 2.6)):
        row = []
        for mirror in (False, True):
            da = db = 0
            for s in seeds:
                fa, fb, wl = l2j.make_bidi_world(s, a1=a1, a2=a2, mirror=mirror)
                ctxs = [lb["ctx"] for lb in wl]
                da += l2j.design_window_a(l2j.bidi_diag(ctxs, l2h.precompute_energies(fa)))
                db += l2j.design_window_b(l2j.bidi_diag(ctxs, l2h.precompute_energies(fb)))
            row.append("m%d:%d/%d" % (int(mirror), da, db))
        grid_rows.append("a1=%.1f,a2=%.1f:%s" % (a1, a2, ";".join(row)))
    print("R_L2K_PRECOMPUTE_GRID=%s" % "|".join(grid_rows))
    return 0 if (anch == 1 and dw_a == 8 and dw_b == 8
                 and under_a == 0 and under_b == 0) else 1


# ---------------- R_L2K_CELL4_REPRO（docs/273 §1.6 复现锚：旧时序 ≡ docs/271 逐位） ----------------
# 期望数字 = docs/271 §三/§四 冻结值（timing="a_first" = docs/271 run_bidi 逐字；同代码
# 路径 -> 期望位精确，容差取打印精度 + 余量）。来源行：docs/271 §3.1（逐种子 JR/首提升窗/
# 门纯度/保真度）、§3.2（逐种子 transfer/calib）、§3.3（聚合）、§3.4（判据）、§3.5-3.6。
CELL4_EXP = {
    "adopt_a": 0.5000,
    "adopt_b": 0.8000,
    "comp_a": 1.0000,
    "comp_b": 1.0000,
    "jr_ratio_a": 0.895134,
    "jr_ratio_b": 0.606402,
    "fid_a": 0.7739,
    "fid_b": 0.9292,
    "transfer_a": 0.400,
    "transfer_b": 0.800,
    "calib_a": 0.312,
    "calib_b": 0.661,
    "jr_a0": [0.1508, 0.1481, 0.1579, 0.1260, 0.0831, 0.1389, 0.1527,
              0.2234, 0.1344, 0.1418],
    "jr_a1": [0.1508, 0.1481, 0.1326, 0.1260, 0.0831, 0.1219, 0.1201,
              0.1783, 0.1344, 0.0921],
    "jr_b0": [0.1472, 0.1478, 0.1583, 0.1305, 0.0951, 0.1363, 0.1626,
              0.1672, 0.1343, 0.1355],
    "jr_b1": [0.0685, 0.0720, 0.0638, 0.0637, 0.0951, 0.0914, 0.0848,
              0.1672, 0.0701, 0.0688],
    "ratio_a": [1.0000, 1.0000, 0.8396, 1.0000, 1.0000, 0.8776, 0.7866,
                0.7980, 1.0000, 0.6495],
    "ratio_b": [0.4653, 0.4868, 0.4027, 0.4880, 1.0000, 0.6703, 0.5215,
                1.0000, 0.5217, 0.5077],
    "fp_a": [None, None, 11, None, None, 7, 9, 15, None, 9],
    "fp_b": [6, 6, 7, 9, None, 10, 9, None, 7, 9],
    "gate_a": [None, None, 0.8571, None, None, 1.0, 0.8, 0.875, None, 1.0],
    "gate_b": [1.0, 1.0, 1.0, 0.8, None, 0.8333, 0.8, None, 1.0, 1.0],
    "fid_a_per": [0.7391, 0.6957, 0.7826, 0.7826, 0.8261, 0.7826, 0.8261,
                  0.6522, 0.7826, 0.8696],
    "fid_b_per": [0.9583, 0.9167, 0.9583, 0.9583, 0.8750, 1.0000, 0.9167,
                  0.7500, 0.9583, 1.0000],
    "transfer_a_per": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
    "transfer_b_per": [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
    "calib_a_per": [0.0, 0.0, 0.667, 0.0, 0.0, 0.857, 0.8, 0.0, 0.0, 0.8],
    "calib_b_per": [0.875, 0.875, 0.857, 0.8, 0.0, 0.75, 0.8, 0.0, 0.857, 0.8],
    "spurious_a": [8],
    "spurious_b": [1, 3, 8],
    "w_a": 0.0,
    "w_b": 0.8,
}


def repro_cell4():
    """旧时序（timing="a_first" = docs/271 run_bidi 逐字）：本格判据相关臂（M-T/M-G/G0A/
    G1n/G2s/W 双侧，12 臂 × 10 种子，双向世界 a_ctx_dep=True）与 docs/271 §三/§四 冻结
    数字逐位一致。返回 (ok, detail)。"""
    seeds = list(range(10))
    jr_a0s, jr_a1s, jr_b0s, jr_b1s = [], [], [], []
    ratio_as, ratio_bs = [], []
    fids_a, fids_b = [], []
    trans_a, trans_b, calibs_a, calibs_b = [], [], [], []
    fp_as, fp_bs, gate_as, gate_bs = [], [], [], []
    comp_a, comp_b = [], []
    g2a_promos, g2b_promos = [], []
    w_a, w_b = [], []
    two_phase_oks_a, two_phase_oks_b = [], []
    prefix_oks_a, prefix_oks_b = [], []
    cons_g0a, cons_g0b, cons_g1na, cons_g1nb = [], [], [], []
    repro_oks_a, repro_oks_b = [], []
    for s in seeds:
        fa, fb, wl = l2j.make_bidi_world(s, a1=A_M1, a2=A_M2, mirror=A_MIRROR)
        # M-T（两阶段，旧时序）
        out_ta, loop_ta, out_tb, loop_tb, snap_ta, snap_tb = run_bidi(
            fa, fb, wl, ch1="comm", ch2="comm", two_phase=True, n_c=N_C,
            timing="a_first")
        jr_a1 = l2g.jr_b(loop_ta)[0]
        jr_b1 = l2g.jr_b(loop_tb)[0]
        att_a = l2g.attribution(loop_ta, N_C)
        att_b = l2g.attribution(loop_tb, N_C)
        # M-G（单阶段，旧时序）
        out_ga, loop_ga, out_gb, loop_gb, _, _ = run_bidi(
            fa, fb, wl, ch1="comm", ch2="comm", two_phase=False, timing="a_first")
        # C（前缀，旧时序）-> PREFIX_EQ
        out_ca, _, out_cb, _, snap_ca, snap_cb = run_bidi(
            fa[:140], fb[:140], wl[:14], ch1="comm", ch2="comm",
            two_phase=False, want_end_snap=True, timing="a_first")
        # G0A（双 off，旧时序）
        out_0a, loop_0a, out_0b, loop_0b, _, _ = run_bidi(
            fa, fb, wl, ch1="off", ch2="off", a_mode="off", two_phase=False,
            timing="a_first")
        jr_a0 = l2g.jr_b(loop_0a)[0]
        jr_b0 = l2g.jr_b(loop_0b)[0]
        # G1n（双 null，旧时序）
        out_na, _, out_nb, _, _, _ = run_bidi(
            fa, fb, wl, ch1="null", ch2="null", a_mode="null", two_phase=False,
            timing="a_first")
        # G2s（双 scrambled，旧时序）
        other = (s + 5) % 10
        bidi_sig_other = (list(l2g.run_a_signal(
            l2j.make_bidi_world(other, a1=A_M1, a2=A_M2, mirror=A_MIRROR)[0]
        ).sA_trace), None)
        rng_bits = np.random.default_rng(s * 99991 + 12345)
        rand_bits = [int(rng_bits.random() < 0.5) for _ in range(len(wl))]
        sig2 = [131.0 if b else 31.0 for b in rand_bits]
        out_2a, _, out_2b, _, _, _ = run_bidi(
            fa, fb, wl, ch1="scrambled", ch2="scrambled", a_mode="scrambled",
            two_phase=False, timing="a_first",
            sig1_fn=lambda w, sa=bidi_sig_other[0]: sa[w],
            sig2_fn=lambda w, sv=sig2: sv[w])
        # W（单向世界，双 ON，旧时序）
        fa_u, fb_u, wl_u = l2g.make_world(s, m1=B_M1, m2=B_M2)
        out_wa, _, out_wb, _, _, _ = run_bidi(
            fa_u, fb_u, wl_u, ch1="comm", ch2="comm", a_mode="comm",
            two_phase=False, timing="a_first")
        # 收集
        ratio_as.append(jr_a1 / max(jr_a0, 1e-12))
        ratio_bs.append(jr_b1 / max(jr_b0, 1e-12))
        jr_a0s.append(jr_a0); jr_a1s.append(jr_a1)
        jr_b0s.append(jr_b0); jr_b1s.append(jr_b1)
        fids_a.append(out_ta["ctx_fidelity"])
        fids_b.append(out_tb["ctx_fidelity"])
        trans_a.append(att_a["transfer_adopted_hit_rate"])
        trans_b.append(att_b["transfer_adopted_hit_rate"])
        calibs_a.append(att_a["calib_baseline"])
        calibs_b.append(att_b["calib_baseline"])
        fp_as.append(att_a["first_promo_win"])
        fp_bs.append(att_b["first_promo_win"])
        comp_a.append(out_ta["compound_frac"])
        comp_b.append(out_tb["compound_frac"])
        g2a_promos.append(out_2a["n_promo"])
        g2b_promos.append(out_2b["n_promo"])
        w_a.append(out_wa["n_promo"])
        w_b.append(out_wb["n_promo"])
        # 门纯度（首提升窗处）
        ga_ = None
        if att_a["first_promo_win"] is not None:
            for rec in loop_ta.gate_attempts:
                if rec[0] == att_a["first_promo_win"]:
                    ga_ = rec[2]
                    break
        gb_ = None
        if att_b["first_promo_win"] is not None:
            for rec in loop_tb.gate_attempts:
                if rec[0] == att_b["first_promo_win"]:
                    gb_ = rec[2]
                    break
        gate_as.append(ga_)
        gate_bs.append(gb_)
        # 跨单元核对（旧时序自洽）
        two_phase_oks_a.append(l2g.two_phase_eq(
            l2h.unit_record2("MGA", s, out_ga, loop_ga),
            l2h.unit_record2("MTA", s, out_ta, loop_ta, snap=snap_ta)))
        two_phase_oks_b.append(l2g.two_phase_eq(
            l2h.unit_record2("MGB", s, out_gb, loop_gb),
            l2h.unit_record2("MTB", s, out_tb, loop_tb, snap=snap_tb)))
        prefix_oks_a.append(l2g.prefix_eq(snap_ca, snap_ta))
        prefix_oks_b.append(l2g.prefix_eq(snap_cb, snap_tb))
        cons_g0a.append(int(out_0a["compound_frac"] == 0.0))
        cons_g0b.append(int(out_0b["compound_frac"] == 0.0))
        cons_g1na.append(int(out_na["compound_frac"] == 0.0))
        cons_g1nb.append(int(out_nb["compound_frac"] == 0.0))
        repro_oks_a.append(l2g.repro_mae(
            l2h.unit_record2("G0AA", s, out_0a, loop_0a),
            l2h.unit_record2("MTA", s, out_ta, loop_ta)))
        repro_oks_b.append(l2g.repro_mae(
            l2h.unit_record2("G0AB", s, out_0b, loop_0b),
            l2h.unit_record2("MTB", s, out_tb, loop_tb)))
    adopt_a = float(np.mean([fp is not None for fp in fp_as]))
    adopt_b = float(np.mean([fp is not None for fp in fp_bs]))
    comp_adopted_a = float(np.mean([c for s, c in enumerate(comp_a)
                                    if fp_as[s] is not None])) if any(fp_as) else 0.0
    comp_adopted_b = float(np.mean([c for s, c in enumerate(comp_b)
                                    if fp_bs[s] is not None])) if any(fp_bs) else 0.0
    fid_a_m = float(np.mean(fids_a))
    fid_b_m = float(np.mean(fids_b))
    trans_a_m, _ = mean_sd(trans_a)
    trans_b_m, _ = mean_sd(trans_b)
    calib_a_m, _ = mean_sd(calibs_a)
    calib_b_m, _ = mean_sd(calibs_b)
    jr_ratio_a_m, _ = mean_sd(ratio_as)
    jr_ratio_b_m, _ = mean_sd(ratio_bs)
    w_a_m = float(np.mean(w_a))
    w_b_m = float(np.mean(w_b))
    g2a_spur = [s for s in seeds if g2a_promos[s] >= 1]
    g2b_spur = [s for s in seeds if g2b_promos[s] >= 1]

    def chk(name, got, exp, tol=1e-4):
        return name, int(abs(got - exp) < tol), got

    checks = []
    for s in seeds:
        checks.append(chk("JR_RATIO_A_S%d" % s, ratio_as[s],
                          CELL4_EXP["ratio_a"][s], 1e-4))
        checks.append(chk("JR_RATIO_B_S%d" % s, ratio_bs[s],
                          CELL4_EXP["ratio_b"][s], 1e-4))
        checks.append(chk("JR_A0_S%d" % s, jr_a0s[s], CELL4_EXP["jr_a0"][s], 1e-4))
        checks.append(chk("JR_A1_S%d" % s, jr_a1s[s], CELL4_EXP["jr_a1"][s], 1e-4))
        checks.append(chk("JR_B0_S%d" % s, jr_b0s[s], CELL4_EXP["jr_b0"][s], 1e-4))
        checks.append(chk("JR_B1_S%d" % s, jr_b1s[s], CELL4_EXP["jr_b1"][s], 1e-4))
        checks.append(chk("FID_A_S%d" % s, fids_a[s], CELL4_EXP["fid_a_per"][s], 1e-4))
        checks.append(chk("FID_B_S%d" % s, fids_b[s], CELL4_EXP["fid_b_per"][s], 1e-4))
        checks.append(chk("TRANSFER_A_S%d" % s, trans_a[s],
                          CELL4_EXP["transfer_a_per"][s], 1e-3))
        checks.append(chk("TRANSFER_B_S%d" % s, trans_b[s],
                          CELL4_EXP["transfer_b_per"][s], 1e-3))
        checks.append(chk("CALIB_A_S%d" % s, calibs_a[s],
                          CELL4_EXP["calib_a_per"][s], 1e-3))
        checks.append(chk("CALIB_B_S%d" % s, calibs_b[s],
                          CELL4_EXP["calib_b_per"][s], 1e-3))
        exp_fpa = CELL4_EXP["fp_a"][s]
        checks.append(("FP_A_S%d" % s, int(fp_as[s] == exp_fpa),
                       (fp_as[s] if fp_as[s] is not None else -1)))
        exp_fpb = CELL4_EXP["fp_b"][s]
        checks.append(("FP_B_S%d" % s, int(fp_bs[s] == exp_fpb),
                       (fp_bs[s] if fp_bs[s] is not None else -1)))
        exp_ga = CELL4_EXP["gate_a"][s]
        checks.append(("GATE_A_S%d" % s,
                       int((gate_as[s] is None and exp_ga is None)
                           or (gate_as[s] is not None and exp_ga is not None
                               and abs(gate_as[s] - exp_ga) < 1e-4)),
                       (gate_as[s] if gate_as[s] is not None else -1)))
        exp_gb = CELL4_EXP["gate_b"][s]
        checks.append(("GATE_B_S%d" % s,
                       int((gate_bs[s] is None and exp_gb is None)
                           or (gate_bs[s] is not None and exp_gb is not None
                               and abs(gate_bs[s] - exp_gb) < 1e-4)),
                       (gate_bs[s] if gate_bs[s] is not None else -1)))
    checks.append(chk("ADOPT_A", adopt_a, CELL4_EXP["adopt_a"], 1e-6))
    checks.append(chk("ADOPT_B", adopt_b, CELL4_EXP["adopt_b"], 1e-6))
    checks.append(chk("COMP_ADOPTED_A", comp_adopted_a, CELL4_EXP["comp_a"], 1e-4))
    checks.append(chk("COMP_ADOPTED_B", comp_adopted_b, CELL4_EXP["comp_b"], 1e-4))
    checks.append(chk("JR_RATIO_A_MEAN", jr_ratio_a_m, CELL4_EXP["jr_ratio_a"], 1e-3))
    checks.append(chk("JR_RATIO_B_MEAN", jr_ratio_b_m, CELL4_EXP["jr_ratio_b"], 1e-3))
    checks.append(chk("FID_A_MEAN", fid_a_m, CELL4_EXP["fid_a"], 1e-4))
    checks.append(chk("FID_B_MEAN", fid_b_m, CELL4_EXP["fid_b"], 1e-4))
    checks.append(chk("TRANSFER_A_MEAN", trans_a_m, CELL4_EXP["transfer_a"], 1e-3))
    checks.append(chk("TRANSFER_B_MEAN", trans_b_m, CELL4_EXP["transfer_b"], 1e-3))
    checks.append(chk("CALIB_A_MEAN", calib_a_m, CELL4_EXP["calib_a"], 1e-3))
    checks.append(chk("CALIB_B_MEAN", calib_b_m, CELL4_EXP["calib_b"], 1e-3))
    checks.append(chk("W_ADOPT_A", w_a_m, CELL4_EXP["w_a"], 1e-6))
    checks.append(chk("W_ADOPT_B", w_b_m, CELL4_EXP["w_b"], 1e-3))
    checks.append(("SPURIOUS_A", int(g2a_spur == CELL4_EXP["spurious_a"]), g2a_spur))
    checks.append(("SPURIOUS_B", int(g2b_spur == CELL4_EXP["spurious_b"]), g2b_spur))
    checks.append(("TWO_PHASE_EQ_A", int(all(two_phase_oks_a)),
                   int(sum(two_phase_oks_a))))
    checks.append(("TWO_PHASE_EQ_B", int(all(two_phase_oks_b)),
                   int(sum(two_phase_oks_b))))
    checks.append(("PREFIX_EQ_A", int(all(prefix_oks_a)), int(sum(prefix_oks_a))))
    checks.append(("PREFIX_EQ_B", int(all(prefix_oks_b)), int(sum(prefix_oks_b))))
    checks.append(("CONSTRUCTION_G0A", int(all(cons_g0a)), int(sum(cons_g0a))))
    checks.append(("CONSTRUCTION_G0B", int(all(cons_g0b)), int(sum(cons_g0b))))
    checks.append(("CONSTRUCTION_G1NA", int(all(cons_g1na)), int(sum(cons_g1na))))
    checks.append(("CONSTRUCTION_G1NB", int(all(cons_g1nb)), int(sum(cons_g1nb))))
    checks.append(("REPRO_MAE_A", int(all(repro_oks_a)), int(sum(repro_oks_a))))
    checks.append(("REPRO_MAE_B", int(all(repro_oks_b)), int(sum(repro_oks_b))))
    ok = int(all(c[1] == 1 for c in checks))
    return ok, dict(checks=checks, adopt_a=adopt_a, adopt_b=adopt_b,
                    comp_a=comp_adopted_a, comp_b=comp_adopted_b,
                    jr_ratio_a=jr_ratio_a_m, jr_ratio_b=jr_ratio_b_m,
                    fid_a=fid_a_m, fid_b=fid_b_m, transfer_a=trans_a_m,
                    transfer_b=trans_b_m, calib_a=calib_a_m, calib_b=calib_b_m,
                    w_a=w_a_m, w_b=w_b_m, spurious_a=g2a_spur,
                    spurious_b=g2b_spur, fp_a=fp_as, fp_b=fp_bs)


def repro_cell4_b():
    """新时序（timing="b_first"）单向关闭（CH2 off、A pixel、均匀世界 = docs/270 世界）
    B 侧行为量化（诊断，不进判据）：adopt_B/保真度/JR 配对比值——CH1 转滞后使 B 侧继承
    数字不再逐位 ≡ docs/270（0.8/0.9292/0.587），诚实报告。"""
    seeds = list(range(10))
    b_adopt = 0
    fids = []
    jr_ratios = []
    fp_bs = []
    for s in seeds:
        fa, fb, wl = l2g.make_world(s, m1=B_M1, m2=B_M2)
        _, _, out_t, loop_t, _, _ = run_bidi(fa, fb, wl, ch1="comm", ch2="off",
                                             a_mode="pixel", two_phase=True,
                                             n_c=N_C, timing="b_first")
        _, _, out0, loop0, _, _ = run_bidi(fa, fb, wl, ch1="off", ch2="off",
                                           a_mode="pixel", two_phase=False,
                                           timing="b_first")
        jr1 = l2g.jr_b(loop_t)[0]
        jr0 = l2g.jr_b(loop0)[0]
        jr_ratios.append(jr1 / max(jr0, 1e-12))
        b_adopt += out_t["n_promo"] >= 1
        fids.append(out_t["ctx_fidelity"])
        att = l2g.attribution(loop_t, N_C)
        fp_bs.append(att["first_promo_win"])
    return dict(adopt=float(b_adopt) / len(seeds),
                fid=float(np.mean(fids)),
                jr_ratio=float(np.mean(jr_ratios)),
                fp_b=fp_bs)


# ---------------- 构造冒烟（docs/273 §1.6-8；合成帧，非数据） ----------------
def smoke_main5():
    """构造冒烟：双侧四模式构造运行正常；off 与 null 逐窗一致；G0 无提升；退化不崩；
    归因不变量；baseline=0 不崩；**门语义双侧单元测试 + CH2 同窗/CH1 滞后语义**。"""
    results = {}
    fb = l2g._synth_frames(30)
    fa = l2g._synth_frames(30, y0=26)
    labels = [dict(ctx=1, b_mult=1.0, a_regime=None)] * 3
    outs = {}
    loops = {}
    for ch1, ch2 in (("off", "off"), ("comm", "comm"), ("null", "null"),
                     ("scrambled", "scrambled")):
        out_a, loop_a, out_b, loop_b, _, _ = run_bidi(
            fa, fb, labels, ch1=ch1, ch2=ch2, two_phase=False, n_c=3,
            timing="b_first")
        outs[(ch1, ch2)] = (out_a, out_b)
        loops[(ch1, ch2)] = (loop_a, loop_b)
        results["construct_%s_%s" % (ch1, ch2)] = int(
            isinstance(out_a, dict) and isinstance(out_b, dict)
            and len(out_a.get("mae_trace", [])) >= 1
            and len(out_b.get("mae_trace", [])) >= 1
            and isinstance(out_a.get("sc1"), int)
            and isinstance(out_b.get("sc1"), int))
    offa, offb = loops[("off", "off")]
    nula, nulb = loops[("null", "null")]
    results["off_null_eq_a"] = int(
        offa.energy_trace == nula.energy_trace
        and offa.up_trace == nula.up_trace
        and [s[0] for s in offa.sig_trace] == [s[0] for s in nula.sig_trace]
        and [s[1] for s in offa.sig_trace] == [s[1] for s in nula.sig_trace]
        and offa.match_trace == nula.match_trace)
    results["off_null_eq_b"] = int(
        offb.energy_trace == nulb.energy_trace
        and offb.up_trace == nulb.up_trace
        and [s[0] for s in offb.sig_trace] == [s[0] for s in nulb.sig_trace]
        and [s[1] for s in offb.sig_trace] == [s[1] for s in nulb.sig_trace]
        and offb.match_trace == nulb.match_trace)
    results["g0_no_promo_a"] = int(outs[("off", "off")][0]["n_promo"] == 0)
    results["g0_no_promo_b"] = int(outs[("off", "off")][1]["n_promo"] == 0)
    results["degenerate_ok"] = int(
        outs[("off", "off")][0]["n_promo"] == 0
        and outs[("off", "off")][0]["compound_frac"] == 0.0
        and outs[("off", "off")][1]["n_promo"] == 0
        and outs[("off", "off")][1]["compound_frac"] == 0.0)
    # 归因不变量（100 帧合成 comm 流；双侧）
    fb2 = l2g._synth_frames(100)
    fa2 = l2g._synth_frames(100, y0=26)
    lab2 = [dict(ctx=1, b_mult=1.0, a_regime=None)] * 10
    _, la, _, lb, _, _ = run_bidi(fa2, fb2, lab2, ch1="comm", ch2="comm",
                                  two_phase=False, n_c=5, timing="b_first")
    att_a = l2g.attribution(la, n_c=5)
    att_b = l2g.attribution(lb, n_c=5)
    results["attr_invariants"] = int(
        all(0.0 <= r <= 1.0 for r in
            (att_a["calib_baseline"], att_a["transfer_adopted_hit_rate"],
             att_b["calib_baseline"], att_b["transfer_adopted_hit_rate"]))
        and att_a["transfer_adopted_hits"] >= 0
        and att_b["transfer_adopted_hits"] >= 0)
    # baseline=0 不崩（off 长流双侧）
    _, la3, _, lb3, _, _ = run_bidi(fa2, fb2, lab2, ch1="off", ch2="off",
                                    two_phase=False, n_c=5, timing="b_first")
    att_a3 = l2g.attribution(la3, n_c=5)
    att_b3 = l2g.attribution(lb3, n_c=5)
    results["baseline_zero_ok"] = int(
        att_a3["calib_baseline"] == 0.0 and att_a3["transfer_adopted_hits"] == 0
        and att_b3["calib_baseline"] == 0.0 and att_b3["transfer_adopted_hits"] == 0)
    # ---- 门语义双侧单元测试（docs/270 §1.1 门定义；合成账本组，非数据） ----
    loop_on_a = l2j.BiLangLoop(mode="comm", self_side="upper", publish_side="upper",
                               gate_purity_min=GATE_PURITY_MIN, window=10, **LOOP_CFG)
    loop_on_b = l2j.BiLangLoop(mode="comm", self_side="lower", publish_side="lower",
                               gate_purity_min=GATE_PURITY_MIN, window=10, **LOOP_CFG)
    loop_off = l2j.BiLangLoop(mode="comm", self_side="lower", publish_side="lower",
                              gate_purity_min=None, window=10, **LOOP_CFG)
    pure_nd = {"c2_ene": {0: [500.0, 510.0, 520.0, 515.0],
                          1: [700.0, 710.0, 720.0, 705.0]}}
    mixed_nd = {"c2_ene": {0: [500.0, 510.0, 700.0, 520.0],
                           1: [720.0, 705.0, 710.0, 690.0]}}
    ok_p, r_p, p_p = loop_on_a._gate_eval(pure_nd)
    results["gate_pure_pass_a"] = int(ok_p and p_p >= GATE_PURITY_MIN)
    ok_pb, _, p_pb = loop_on_b._gate_eval(pure_nd)
    results["gate_pure_pass_b"] = int(ok_pb and p_pb >= GATE_PURITY_MIN)
    ok_m, r_m, p_m = loop_on_a._gate_eval(mixed_nd)
    results["gate_mixed_block_a"] = int((not ok_m) and p_m < GATE_PURITY_MIN)
    ok_mb, _, p_mb = loop_on_b._gate_eval(mixed_nd)
    results["gate_mixed_block_b"] = int((not ok_mb) and p_mb < GATE_PURITY_MIN)
    ok_o, r_o, p_o = loop_off._gate_eval(pure_nd)
    results["gate_off_passthrough"] = int(ok_o and len(loop_off.gate_attempts) == 0)
    # ---- CH2 同窗 / CH1 滞后语义（docs/273 §1.1 冻结） ----
    # b_first：A 首窗 c2 非 None（B 先闭窗发布 s_B(0) -> A 同窗读取）；B 首窗 c2=None
    _, la4, _, lb4, _, _ = run_bidi(fa2, fb2, lab2, ch1="comm", ch2="comm",
                                    two_phase=False, n_c=5, timing="b_first")
    results["ch2_same_win_first_win_non_none"] = int(
        la4.sig_trace[0][2] is not None)
    results["ch1_lag_first_win_none"] = int(lb4.sig_trace[0][2] is None)
    # a_first（271 语义，CELL4_REPRO 前提）：A 首窗 c2=None（CH2 一窗滞后）
    _, la5, _, _, _, _ = run_bidi(fa2, fb2, lab2, ch1="comm", ch2="comm",
                                  two_phase=False, n_c=5, timing="a_first")
    results["a_first_ch2_lag_first_win_none"] = int(la5.sig_trace[0][2] is None)
    # null 模式双侧 c2 恒 1（单值 -> 无采纳条件②）
    _, la6, _, lb6, _, _ = run_bidi(fa2, fb2, lab2, ch1="null", ch2="null",
                                    two_phase=False, n_c=5, timing="b_first")
    results["null_both_single_ctx"] = int(
        all(s[2] == 1 for s in la6.sig_trace if s[2] is not None)
        and all(s[2] == 1 for s in lb6.sig_trace if s[2] is not None))
    for k in sorted(results):
        print("R_L2K_SMOKE_%s=%d" % (k.upper(), results[k]))
    return 0 if all(results.values()) else 1


# ---------------- 单元记录（checkpoint/resume 用；自包含；复用 docs/271） ----------------
def unit_record5(arm, seed, out, loop, side, snap=None, jr=None, att=None):
    return l2j.unit_record4(arm, seed, out, loop, side, snap=snap, jr=jr, att=att)


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="l2k")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--precompute", action="store_true",
                    help="环境预计算完整核对（§1.1 协议复现；机制无关）")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.smoke:
        return smoke_main5()
    if args.precompute:
        return precompute_main()
    t0 = time.time()
    seeds = list(range(10))

    cfg = {"tag": args.tag, "n_seeds": len(seeds), "frames": N_FRAMES,
           "window": WINDOW, "n_c": N_C, "jitter": JITTER,
           "b_m1": B_M1, "b_m2": B_M2, "a_m1": A_M1, "a_m2": A_M2,
           "a_mirror": A_MIRROR, "noise_sigma": l2g.NOISE_SIGMA,
           "timing": TIMING,
           "world": {"a_center": list(l2g.A_CENTER), "a_orbit": l2g.A_ORBIT,
                     "a_freq": l2g.A_FREQ, "b_center": list(l2g.B_CENTER),
                     "b_orbit": l2g.B_ORBIT, "b_freq": l2g.B_FREQ,
                     "rng_lvcode": LV_WORLD},
           "channel": {"sparse_px": l2g.SIG_SPARSE_PX,
                       "null_signal": l2g.NULL_SIGNAL,
                       "ch2_same_win": 1, "ch1_lag_windows": 1},
           "gate": {"purity_min": GATE_PURITY_MIN},
           "criteria": {"jr_ratio_max": JR_RATIO_MAX,
                        "adopt_frac_min": ADOPT_FRAC_MIN,
                        "compound_min": COMPOUND_MIN,
                        "transfer_floor": TRANSFER_FLOOR,
                        "transfer_rel": TRANSFER_REL},
           "loop": LOOP_CFG}
    ck_tag = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    ckpt_path = os.path.join(args.out_dir, "ckpt_l2k_%s.json" % ck_tag)

    done = {}
    if args.resume and not args.no_resume and os.path.exists(ckpt_path):
        with open(ckpt_path, encoding="utf-8") as f:
            done = json.load(f).get("per_unit", {})

    per_unit = dict(done)
    worlds_bidi = {s: l2j.make_bidi_world(s, a1=A_M1, a2=A_M2, mirror=A_MIRROR)
                   for s in seeds}
    worlds_uniform = {s: l2g.make_world(s, m1=B_M1, m2=B_M2) for s in seeds}

    # G2s 错乱信号源（双向世界，每种子一次）：s_A 序列（A 回路单独跑）+ s_B 序列
    bidi_sig = {}
    for s in seeds:
        fa, fb, _ = worlds_bidi[s]
        la = l2g.run_a_signal(fa)
        lb = l2j.run_b_signal_bidi(fb)
        bidi_sig[s] = (list(la.sA_trace), list(lb.sA_trace))

    def need(arm, s):
        return "%s_%d" % (arm, s) not in per_unit

    for s in seeds:
        fa, fb, wl = worlds_bidi[s]
        if need("MTA", s) or need("MTB", s):
            out_a, loop_a, out_b, loop_b, snap_a, snap_b = run_bidi(
                fa, fb, wl, ch1="comm", ch2="comm", two_phase=True, n_c=N_C,
                timing=TIMING)
            per_unit["MTA_%d" % s] = unit_record5(
                "MTA", s, out_a, loop_a, "A", snap=snap_a,
                jr=l2g.jr_b(loop_a), att=l2g.attribution(loop_a, N_C))
            per_unit["MTB_%d" % s] = unit_record5(
                "MTB", s, out_b, loop_b, "B", snap=snap_b,
                jr=l2g.jr_b(loop_b), att=l2g.attribution(loop_b, N_C))
            print("PROGRESS", flush=True)
        if need("MGA", s) or need("MGB", s):
            out_a, loop_a, out_b, loop_b, _, _ = run_bidi(
                fa, fb, wl, ch1="comm", ch2="comm", two_phase=False,
                timing=TIMING)
            per_unit["MGA_%d" % s] = unit_record5(
                "MGA", s, out_a, loop_a, "A")
            per_unit["MGB_%d" % s] = unit_record5(
                "MGB", s, out_b, loop_b, "B")
            print("PROGRESS", flush=True)
        if need("CA", s) or need("CB", s):
            out_a, loop_a, out_b, loop_b, snap_a, snap_b = run_bidi(
                fa[:140], fb[:140], wl[:14], ch1="comm", ch2="comm",
                two_phase=False, want_end_snap=True, timing=TIMING)
            per_unit["CA_%d" % s] = unit_record5(
                "CA", s, out_a, loop_a, "A", snap=snap_a)
            per_unit["CB_%d" % s] = unit_record5(
                "CB", s, out_b, loop_b, "B", snap=snap_b)
            print("PROGRESS", flush=True)
        if need("G0AA", s) or need("G0AB", s):
            out_a, loop_a, out_b, loop_b, _, _ = run_bidi(
                fa, fb, wl, ch1="off", ch2="off", a_mode="off",
                timing=TIMING)
            per_unit["G0AA_%d" % s] = unit_record5(
                "G0AA", s, out_a, loop_a, "A", jr=l2g.jr_b(loop_a))
            per_unit["G0AB_%d" % s] = unit_record5(
                "G0AB", s, out_b, loop_b, "B", jr=l2g.jr_b(loop_b))
            print("PROGRESS", flush=True)
        if need("G1NA", s) or need("G1NB", s):
            out_a, loop_a, out_b, loop_b, _, _ = run_bidi(
                fa, fb, wl, ch1="null", ch2="null", a_mode="null",
                timing=TIMING)
            per_unit["G1NA_%d" % s] = unit_record5(
                "G1NA", s, out_a, loop_a, "A")
            per_unit["G1NB_%d" % s] = unit_record5(
                "G1NB", s, out_b, loop_b, "B")
            print("PROGRESS", flush=True)
        if need("G2SA", s) or need("G2SB", s):
            other = (s + 5) % 10
            sigA = bidi_sig[other][0]
            # A 侧错乱源（docs/271 §1.2 逐字）：确定性每种子随机二元 ctx 注入（rng 派生
            # 0/1 经远阈值信号 v=131/31 实现 sign(v-x_A)=bit；docs/270 §二 C1 蒙特卡洛模型）
            rng_bits = np.random.default_rng(s * 99991 + 12345)
            rand_bits = [int(rng_bits.random() < 0.5) for _ in range(
                len(wl))]
            sig2 = [131.0 if b else 31.0 for b in rand_bits]
            out_a, loop_a, out_b, loop_b, _, _ = run_bidi(
                fa, fb, wl, ch1="scrambled", ch2="scrambled",
                a_mode="scrambled", timing=TIMING,
                sig1_fn=lambda w, sa=sigA: sa[w],
                sig2_fn=lambda w, sv=sig2: sv[w])
            per_unit["G2SA_%d" % s] = unit_record5(
                "G2SA", s, out_a, loop_a, "A")
            per_unit["G2SB_%d" % s] = unit_record5(
                "G2SB", s, out_b, loop_b, "B")
            print("PROGRESS", flush=True)
        if need("WA", s) or need("WB", s):
            fa_u, fb_u, wl_u = worlds_uniform[s]
            out_a, loop_a, out_b, loop_b, _, _ = run_bidi(
                fa_u, fb_u, wl_u, ch1="comm", ch2="comm", a_mode="comm",
                two_phase=False, timing=TIMING)
            per_unit["WA_%d" % s] = unit_record5(
                "WA", s, out_a, loop_a, "A")
            per_unit["WB_%d" % s] = unit_record5(
                "WB", s, out_b, loop_b, "B")
            print("PROGRESS", flush=True)
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump({"config": cfg, "per_unit": per_unit},
                      f, ensure_ascii=False, indent=1)

    # ---- 守卫 ----
    g232_ok, g232 = l2g.guard_d232()
    g235_ok, g235 = l2g.guard_d235()
    c2_ok, c2_detail = l2i.repro_cell2()
    c4_ok, c4_detail = repro_cell4()
    c4b_detail = repro_cell4_b()
    world_ok, world_oks = l2j.world_eq()
    pre_ok, pre_detail = precompute_ok_main()

    # ---- 跨单元核对（双侧，新时序） ----
    prefix_oks_a, prefix_oks_b = [], []
    two_phase_oks_a, two_phase_oks_b = [], []
    repro_oks_a, repro_oks_b = [], []
    cons_g0a, cons_g0b = [], []
    cons_g1na, cons_g1nb = [], []
    seed_rows = []
    for s in seeds:
        ta = per_unit["MTA_%d" % s]
        tb = per_unit["MTB_%d" % s]
        ga = per_unit["MGA_%d" % s]
        gb = per_unit["MGB_%d" % s]
        ca = per_unit["CA_%d" % s]
        cb = per_unit["CB_%d" % s]
        g0a = per_unit["G0AA_%d" % s]
        g0b = per_unit["G0AB_%d" % s]
        na = per_unit["G1NA_%d" % s]
        nb = per_unit["G1NB_%d" % s]
        g2a = per_unit["G2SA_%d" % s]
        g2b = per_unit["G2SB_%d" % s]
        wa = per_unit["WA_%d" % s]
        wb = per_unit["WB_%d" % s]
        prefix_oks_a.append(l2g.prefix_eq(ca["snapshot"], ta["snapshot"]))
        prefix_oks_b.append(l2g.prefix_eq(cb["snapshot"], tb["snapshot"]))
        two_phase_oks_a.append(l2g.two_phase_eq(ga, ta))
        two_phase_oks_b.append(l2g.two_phase_eq(gb, tb))
        repro_oks_a.append(l2g.repro_mae(g0a, ta))
        repro_oks_b.append(l2g.repro_mae(g0b, tb))
        cons_g0a.append(int(g0a["finalize"]["compound_frac"] == 0.0))
        cons_g0b.append(int(g0b["finalize"]["compound_frac"] == 0.0))
        cons_g1na.append(int(na["finalize"]["compound_frac"] == 0.0))
        cons_g1nb.append(int(nb["finalize"]["compound_frac"] == 0.0))
        jr_a0 = g0a["jr"][0]
        jr_b0 = g0b["jr"][0]
        jr_a1 = ta["jr"][0]
        jr_b1 = tb["jr"][0]
        att_a = ta["att"]
        att_b = tb["att"]
        gate_a = None
        gate_b = None
        if att_a["first_promo_win"] is not None:
            for rec in ta["gate"]:
                if rec[0] == att_a["first_promo_win"]:
                    gate_a = dict(win=rec[0], ratio=rec[1], purity=rec[2],
                                  ok=rec[3])
                    break
        if att_b["first_promo_win"] is not None:
            for rec in tb["gate"]:
                if rec[0] == att_b["first_promo_win"]:
                    gate_b = dict(win=rec[0], ratio=rec[1], purity=rec[2],
                                  ok=rec[3])
                    break
        seed_rows.append(dict(
            seed=s, ta=ta, tb=tb, ga=ga, gb=gb, ca=ca, cb=cb,
            g0a=g0a, g0b=g0b, na=na, nb=nb, g2a=g2a, g2b=g2b,
            wa=wa, wb=wb,
            jr_a0=jr_a0, jr_b0=jr_b0, jr_a1=jr_a1, jr_b1=jr_b1,
            jr_ratio_a=jr_a1 / max(jr_a0, 1e-12),
            jr_ratio_b=jr_b1 / max(jr_b0, 1e-12),
            transfer_a=att_a["transfer_adopted_hit_rate"],
            calib_a=att_a["calib_baseline"],
            transfer_b=att_b["transfer_adopted_hit_rate"],
            calib_b=att_b["calib_baseline"],
            held_elig_a=att_a["n_heldout_eligible"],
            held_elig_b=att_b["n_heldout_eligible"],
            first_promo_a=att_a["first_promo_win"],
            first_promo_b=att_b["first_promo_win"],
            gate_a=gate_a, gate_b=gate_b))

    prefix_ok_a = int(all(prefix_oks_a))
    prefix_ok_b = int(all(prefix_oks_b))
    two_phase_ok_a = int(all(two_phase_oks_a))
    two_phase_ok_b = int(all(two_phase_oks_b))
    repro_ok_a = int(all(repro_oks_a))
    repro_ok_b = int(all(repro_oks_b))
    construction_ok = int(all(cons_g0a) and all(cons_g0b)
                          and all(cons_g1na) and all(cons_g1nb))

    # ---- 聚合（双侧；M 臂主数字 = M-G 单阶段流，§1.8；T 两阶段与其逐窗一致） ----
    UNIT_ARM = {"A": "MGA", "B": "MGB"}

    def col(side, key, arm=None):
        ua = UNIT_ARM[side] if arm is None else arm
        if key == "ratio":
            return [per_unit["%s_%d" % (ua, s)]["ratio"] for s in seeds]
        return [per_unit["%s_%d" % (ua, s)]["finalize"][key] for s in seeds]

    agg = {}
    for side in ("A", "B"):
        mae_m, mae_sd = mean_sd(col(side, "mae_mean"))
        sc2_m, _ = mean_sd(col(side, "sc2"))
        comp_m, comp_sd = mean_sd(col(side, "compound_frac"))
        churn_m, _ = mean_sd(col(side, "churn_frac"))
        promo_m, _ = mean_sd(col(side, "n_promo"))
        agg[side] = dict(mae_mean=mae_m, mae_sd=mae_sd, sc2_mean=sc2_m,
                         comp_mean=comp_m, comp_sd=comp_sd,
                         churn_mean=churn_m, promo_mean=promo_m,
                         ratio_mean=float(np.mean(col(side, "ratio", "MGA"))))
        agg[side]["mae_ci95"] = list(bootstrap_ci(col(side, "mae_mean")))
        agg[side]["comp_ci95"] = list(bootstrap_ci(col(side, "compound_frac")))
    agg_g0 = {}
    for side in ("A", "B"):
        arm = "G0AA" if side == "A" else "G0AB"
        agg_g0[side] = dict(
            sc2_mean=float(np.mean(col(side, "sc2", arm))),
            churn_mean=float(np.mean(col(side, "churn_frac", arm))),
            ratio_mean=float(np.mean([per_unit["%s_%d" % (arm, s)]["ratio"]
                                      for s in seeds])))
    jr_a0s = [r["jr_a0"] for r in seed_rows]
    jr_b0s = [r["jr_b0"] for r in seed_rows]
    jr_a1s = [r["jr_a1"] for r in seed_rows]
    jr_b1s = [r["jr_b1"] for r in seed_rows]
    jr_ratio_as = [r["jr_ratio_a"] for r in seed_rows]
    jr_ratio_bs = [r["jr_ratio_b"] for r in seed_rows]
    jr_a0_m, jr_a0_sd = mean_sd(jr_a0s)
    jr_b0_m, jr_b0_sd = mean_sd(jr_b0s)
    jr_a1_m, jr_a1_sd = mean_sd(jr_a1s)
    jr_b1_m, jr_b1_sd = mean_sd(jr_b1s)
    jr_ratio_a_m, jr_ratio_a_sd = mean_sd(jr_ratio_as)
    jr_ratio_b_m, jr_ratio_b_sd = mean_sd(jr_ratio_bs)
    transfer_as = [r["transfer_a"] for r in seed_rows]
    transfer_bs = [r["transfer_b"] for r in seed_rows]
    calib_as = [r["calib_a"] for r in seed_rows]
    calib_bs = [r["calib_b"] for r in seed_rows]
    transfer_a_m, transfer_a_sd = mean_sd(transfer_as)
    transfer_b_m, transfer_b_sd = mean_sd(transfer_bs)
    calib_a_m, calib_a_sd = mean_sd(calib_as)
    calib_b_m, calib_b_sd = mean_sd(calib_bs)
    adopt_a = float(np.mean([r["ta"]["finalize"]["n_promo"] >= 1
                             for r in seed_rows]))
    adopt_b = float(np.mean([r["tb"]["finalize"]["n_promo"] >= 1
                             for r in seed_rows]))
    adopted_comp_a = [r["ta"]["finalize"]["compound_frac"] for r in seed_rows
                      if r["ta"]["finalize"]["n_promo"] >= 1]
    adopted_comp_b = [r["tb"]["finalize"]["compound_frac"] for r in seed_rows
                      if r["tb"]["finalize"]["n_promo"] >= 1]
    comp_adopted_a = float(np.mean(adopted_comp_a)) if adopted_comp_a else 0.0
    comp_adopted_b = float(np.mean(adopted_comp_b)) if adopted_comp_b else 0.0
    fid_a_m = float(np.mean(col("A", "ctx_fidelity")))
    fid_b_m = float(np.mean(col("B", "ctx_fidelity")))
    adopt_a_w = float(np.mean([r["wa"]["finalize"]["n_promo"] >= 1
                               for r in seed_rows]))
    adopt_b_w = float(np.mean([r["wb"]["finalize"]["n_promo"] >= 1
                               for r in seed_rows]))
    # 双向世界诊断（M-G A/B 能量带）
    eratios_a = []
    eratios_b = []
    for s in seeds:
        fa, fb, wl = worlds_bidi[s]
        ctxs = [lb["ctx"] for lb in wl]
        ga_E = per_unit["MGA_%d" % s]["E"]
        gb_E = per_unit["MGB_%d" % s]["E"]
        da_ = l2j.bidi_diag(ctxs, ga_E)
        db_ = l2j.bidi_diag(ctxs, gb_E)
        if da_["ratio"] is not None:
            eratios_a.append(da_["ratio"])
        if db_["ratio"] is not None:
            eratios_b.append(db_["ratio"])
    eratio_a_m = float(np.mean(eratios_a)) if eratios_a else 0.0
    eratio_b_m = float(np.mean(eratios_b)) if eratios_b else 0.0
    design_win_a_frac = float(np.mean([l2j.design_window_a(l2j.bidi_diag(
        [lb["ctx"] for lb in worlds_bidi[s][2]], per_unit["MGA_%d" % s]["E"]))
        for s in seeds]))
    design_win_b_frac = float(np.mean([l2j.design_window_b(l2j.bidi_diag(
        [lb["ctx"] for lb in worlds_bidi[s][2]], per_unit["MGB_%d" % s]["E"]))
        for s in seeds]))

    # ---- 判据（docs/273 §1.5 冻结：C1-C4 双侧化，docs/271 逐字） ----
    c1a = int(jr_ratio_a_m <= JR_RATIO_MAX and jr_ratio_b_m <= JR_RATIO_MAX)
    c1b = repro_ok_a == 1 and repro_ok_b == 1
    c2 = int(adopt_a >= ADOPT_FRAC_MIN and adopt_b >= ADOPT_FRAC_MIN
             and comp_adopted_a >= COMPOUND_MIN
             and comp_adopted_b >= COMPOUND_MIN
             and all(cons_g0a) and all(cons_g0b)
             and all(cons_g1na) and all(cons_g1nb))
    spurious_a = [r["seed"] for r in seed_rows
                  if r["g2a"]["finalize"]["n_promo"] >= 1]
    spurious_b = [r["seed"] for r in seed_rows
                  if r["g2b"]["finalize"]["n_promo"] >= 1]
    c3 = int(len(spurious_a) == 0 and len(spurious_b) == 0)
    c4a_a = int(agg_g0["A"]["sc2_mean"] >= 1
                and agg_g0["A"]["churn_mean"] < 0.3
                and agg_g0["A"]["ratio_mean"] <= 1.5
                and agg["A"]["sc2_mean"] >= 1
                and agg["A"]["churn_mean"] < 0.3
                and agg["A"]["ratio_mean"] <= 1.5)
    c4a_b = int(agg_g0["B"]["sc2_mean"] >= 1
                and agg_g0["B"]["churn_mean"] < 0.3
                and agg_g0["B"]["ratio_mean"] <= 1.5
                and agg["B"]["sc2_mean"] >= 1
                and agg["B"]["churn_mean"] < 0.3
                and agg["B"]["ratio_mean"] <= 1.5)
    c4a = int(c4a_a == 1 and c4a_b == 1)
    c4b = int(transfer_a_m >= TRANSFER_FLOOR
              and transfer_a_m >= TRANSFER_REL * calib_a_m
              and transfer_b_m >= TRANSFER_FLOOR
              and transfer_b_m >= TRANSFER_REL * calib_b_m)
    c4 = int(c4a == 1 and c4b == 1)

    # 数据可用性（LANG_BLOCKED 预防）：双侧逐种子 JR 有窗口、留出 eligible 非空
    blocked = int(not (all(r["g0a"]["jr"][1] >= 1 for r in seed_rows)
                       and all(r["g0b"]["jr"][1] >= 1 for r in seed_rows)
                       and all(r["ta"]["jr"][1] >= 1 for r in seed_rows)
                       and all(r["tb"]["jr"][1] >= 1 for r in seed_rows)
                       and all(r["held_elig_a"] >= 1 for r in seed_rows)
                       and all(r["held_elig_b"] >= 1 for r in seed_rows)))

    guards_ok = (g232_ok == 1 and g235_ok == 1 and construction_ok == 1
                 and prefix_ok_a == 1 and prefix_ok_b == 1
                 and two_phase_ok_a == 1 and two_phase_ok_b == 1
                 and repro_ok_a == 1 and repro_ok_b == 1
                 and c2_ok == 1 and c4_ok == 1 and world_ok == 1
                 and pre_ok == 1)
    if not guards_ok:
        verdict = "GUARD_FAIL"
        vnote = ("guard(s) failed: D232=%d, D235=%d, CONSTRUCTION=%d, "
                 "PREFIX_EQ_A=%d, PREFIX_EQ_B=%d, TWO_PHASE_EQ_A=%d, "
                 "TWO_PHASE_EQ_B=%d, REPRO_MAE_A=%d, REPRO_MAE_B=%d, "
                 "CELL2_REPRO=%d, CELL4_REPRO=%d, WORLD_EQ=%d, PRECOMPUTE=%d "
                 "-> implementation drift; fix implementation, do not judge "
                 "mechanism" % (g232_ok, g235_ok, construction_ok, prefix_ok_a,
                                prefix_ok_b, two_phase_ok_a, two_phase_ok_b,
                                repro_ok_a, repro_ok_b, c2_ok, c4_ok,
                                world_ok, pre_ok))
    elif blocked:
        verdict = "LANG_BLOCKED"
        vnote = ("synthetic environment unavailable (per-seed eligible/JR "
                 "windows missing on A/B sides); see per-seed numbers")
    elif c1a and c1b and c2 and c3 and c4:
        verdict = "MUTUAL_EMERGES"
        vnote = ("criteria C1-C4 all pass and all guards pass: A and B both "
                 "spontaneously adopt the other's signal (ledger-driven, no "
                 "switch, gate on both loops) and both sides' joint residual "
                 "drops; gate cleanliness both sides (spurious(G2s)=0 both); "
                 "holdout transfer holds both sides -> minimal measurable "
                 "closed-loop evidence (docs/273 sec 1.5)")
    elif not c2:
        if (adopt_a >= ADOPT_FRAC_MIN) != (adopt_b >= ADOPT_FRAC_MIN):
            verdict = "ONE_WAY"
            vnote = ("C2 fails with exactly one side adopting: adopt_A=%.4f, "
                     "adopt_B=%.4f -> one-way round-trip only, closed loop "
                     "not achieved; honest report of which side and why "
                     "(docs/273 sec 1.5)" % (adopt_a, adopt_b))
        else:
            verdict = "MUTUAL_FLAT"
            vnote = ("C2 fails (signal not mutually adopted): adopt_A=%.4f, "
                     "adopt_B=%.4f (<0.6 or G0A/G1n non-zero adoption) -> "
                     "honest negative; see bilateral decomposition" %
                     (adopt_a, adopt_b))
    elif not (c1a and c1b):
        verdict = "MUTUAL_FLAT"
        vnote = ("C2 passes but C1 fails (adopted but joint residual not "
                 "lowered on at least one side) -> MUTUAL_NO_GAIN sub-form: "
                 "JR_ratio_A=%.4f, JR_ratio_B=%.4f > 0.85; mechanism has no "
                 "rollback (docs/268 sec 5.6)" % (jr_ratio_a_m, jr_ratio_b_m))
    else:
        why = []
        if not c3:
            why.append("C3 CLEAN_KEEP fails: spurious_A=%d, spurious_B=%d "
                       "(gate must hold on both loops)" % (len(spurious_a),
                                                           len(spurious_b)))
        if not c4a:
            why.append("C4a STRUCTURE_KEEP fails (see SC2/churn/ratio per side)")
        if not c4b:
            why.append("C4b SIG_HOLDOUT fails (transfer_A=%.4f/calib_A=%.4f, "
                       "transfer_B=%.4f/calib_B=%.4f)" % (transfer_a_m,
                                                          calib_a_m,
                                                          transfer_b_m,
                                                          calib_b_m))
        verdict = "PARTIAL"
        vnote = "; ".join(why) + " (see R_L2K_CRIT* numbers)"

    # ---- 工件（自描述 JSON） ----
    out = {
        "artifact": "lang_comm_test5",
        "doc_ref": "docs/63, docs/228, docs/232, docs/235, docs/247, docs/258, "
                   "docs/264, docs/266, docs/268, docs/269, docs/270, docs/271, "
                   "docs/273",
        "config": cfg,
        "guards": {"d232": {"ok": g232_ok, "detail": g232},
                   "d235": {"ok": g235_ok, "detail": g235},
                   "construction": {"ok": construction_ok,
                                    "g0a_zero": int(sum(cons_g0a)),
                                    "g0b_zero": int(sum(cons_g0b)),
                                    "g1na_zero": int(sum(cons_g1na)),
                                    "g1nb_zero": int(sum(cons_g1nb))},
                   "prefix_eq_a": prefix_ok_a, "prefix_eq_b": prefix_ok_b,
                   "two_phase_eq_a": two_phase_ok_a,
                   "two_phase_eq_b": two_phase_ok_b,
                   "repro_mae_a": repro_ok_a, "repro_mae_b": repro_ok_b,
                   "cell2_repro": {"ok": c2_ok},
                   "cell4_repro": {"ok": c4_ok},
                   "cell4_repro_b": c4b_detail,
                   "world_eq": {"ok": world_ok, "per_seed": world_oks},
                   "precompute": pre_detail,
                   "prefix_per_seed_a": prefix_oks_a,
                   "prefix_per_seed_b": prefix_oks_b,
                   "two_phase_per_seed_a": two_phase_oks_a,
                   "two_phase_per_seed_b": two_phase_oks_b,
                   "repro_per_seed_a": repro_oks_a,
                   "repro_per_seed_b": repro_oks_b},
        "per_seed": seed_rows,
        "arms": {k: {"mean_sd": agg[k], "mae_ci95": agg[k]["mae_ci95"],
                     "comp_ci95": agg[k]["comp_ci95"]}
                 for k in ("A", "B")},
        "arms_g0a": agg_g0,
        "jr": {"a0_mean": jr_a0_m, "a0_sd": jr_a0_sd, "b0_mean": jr_b0_m,
               "b0_sd": jr_b0_sd, "a1_mean": jr_a1_m, "a1_sd": jr_a1_sd,
               "b1_mean": jr_b1_m, "b1_sd": jr_b1_sd,
               "ratio_a_mean": jr_ratio_a_m, "ratio_a_sd": jr_ratio_a_sd,
               "ratio_b_mean": jr_ratio_b_m, "ratio_b_sd": jr_ratio_b_sd,
               "per_seed_ratio_a": jr_ratio_as,
               "per_seed_ratio_b": jr_ratio_bs},
        "adoption": {"adopt_a": adopt_a, "adopt_b": adopt_b,
                     "comp_adopted_a": comp_adopted_a,
                     "comp_adopted_b": comp_adopted_b,
                     "adopt_a_w": adopt_a_w, "adopt_b_w": adopt_b_w},
        "holdout": {"transfer_a_mean": transfer_a_m,
                    "transfer_a_sd": transfer_a_sd,
                    "calib_a_mean": calib_a_m,
                    "transfer_b_mean": transfer_b_m,
                    "transfer_b_sd": transfer_b_sd,
                    "calib_b_mean": calib_b_m,
                    "per_seed_a": transfer_as, "per_seed_b": transfer_bs},
        "diag": {"fid_a_mean": fid_a_m, "fid_b_mean": fid_b_m,
                 "energy_ratio_a_mean": eratio_a_m,
                 "energy_ratio_b_mean": eratio_b_m,
                 "design_win_a_frac": design_win_a_frac,
                 "design_win_b_frac": design_win_b_frac},
        "criteria": {"c1a_mutual_value": c1a, "c1b_mae_eq": c1b,
                     "c2_mutual_adoption": c2, "c3_clean_keep": c3,
                     "c4_structure_holdout": c4, "c4a": c4a, "c4b": c4b,
                     "jr_ratio_a": jr_ratio_a_m, "jr_ratio_b": jr_ratio_b_m,
                     "adopt_a": adopt_a, "adopt_b": adopt_b,
                     "comp_adopted_a": comp_adopted_a,
                     "comp_adopted_b": comp_adopted_b,
                     "transfer_a": transfer_a_m, "calib_a": calib_a_m,
                     "transfer_b": transfer_b_m, "calib_b": calib_b_m,
                     "spurious_a_seeds": spurious_a,
                     "spurious_b_seeds": spurious_b},
        "verdict": {"verdict": verdict, "note": vnote},
        "timing": {"elapsed_sec": round(time.time() - t0, 2)},
    }
    res_path = os.path.join(args.out_dir, "l2k_%s.json" % args.tag)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 摘要块：ASCII 标签，每行一个数字（顺序固定；无中文/日志/JSON） ----
    print("R_L2K_TAG=%s" % args.tag)
    print("R_L2K_SEEDS=%d" % len(seeds))
    print("R_L2K_FRAMES=%d" % N_FRAMES)
    print("R_L2K_WINDOWS=%d" % (N_FRAMES // WINDOW))
    print("R_L2K_NC=%d" % N_C)
    print("R_L2K_TIMING=%s" % TIMING)
    print("R_L2K_M1=%.1f" % B_M1)
    print("R_L2K_M2=%.1f" % B_M2)
    print("R_L2K_A1=%.1f" % A_M1)
    print("R_L2K_A2=%.1f" % A_M2)
    print("R_L2K_GATE_PURITY_MIN=%.2f" % GATE_PURITY_MIN)
    print("R_L2K_GUARD_D232=%d" % g232_ok)
    print("R_L2K_GUARD_D232_SC2=%s" % ",".join(str(v) for v in g232["sc2"]))
    print("R_L2K_GUARD_D232_SCLATE_FRAC=%.4f" % g232["sc_late_frac"])
    print("R_L2K_GUARD_D232_SC4=%.4f" % g232["sc4"])
    print("R_L2K_GUARD_D232_MAE=%.6f" % g232["mae"])
    print("R_L2K_GUARD_D232_MAE_SD=%.6f" % g232["mae_sd"])
    print("R_L2K_GUARD_D232_PIN=%.4f" % g232["pin"])
    print("R_L2K_GUARD_D232_CLASS=%s" % g232["cls"])
    print("R_L2K_GUARD_D235=%d" % g235_ok)
    for lv in (21, 22):
        d = g235[lv]
        print("R_L2K_GUARD_D235_C%d_OK=%d" % (lv, d["ok"]))
        print("R_L2K_GUARD_D235_C%d_MAE=%.6f" % (lv, d["mae"]))
        print("R_L2K_GUARD_D235_C%d_MAE_SD=%.6f" % (lv, d["mae_sd"]))
        print("R_L2K_GUARD_D235_C%d_SC2=%.4f" % (lv, d["sc2"]))
        print("R_L2K_GUARD_D235_C%d_SC2_SD=%.4f" % (lv, d["sc2_sd"]))
        print("R_L2K_GUARD_D235_C%d_COMP=%.4f" % (lv, d["comp"]))
        print("R_L2K_GUARD_D235_C%d_CHURN=%.4f" % (lv, d["churn"]))
        print("R_L2K_GUARD_D235_C%d_FID=%.4f" % (lv, d["fid"]))
    print("R_L2K_CONSTRUCTION=%d" % construction_ok)
    print("R_L2K_CONSTRUCTION_G0A=%d" % int(sum(cons_g0a)))
    print("R_L2K_CONSTRUCTION_G0B=%d" % int(sum(cons_g0b)))
    print("R_L2K_CONSTRUCTION_G1NA=%d" % int(sum(cons_g1na)))
    print("R_L2K_CONSTRUCTION_G1NB=%d" % int(sum(cons_g1nb)))
    print("R_L2K_PREFIX_EQ_A=%d" % prefix_ok_a)
    print("R_L2K_PREFIX_EQ_B=%d" % prefix_ok_b)
    print("R_L2K_TWO_PHASE_EQ_A=%d" % two_phase_ok_a)
    print("R_L2K_TWO_PHASE_EQ_B=%d" % two_phase_ok_b)
    print("R_L2K_REPRO_MAE_A=%d" % repro_ok_a)
    print("R_L2K_REPRO_MAE_B=%d" % repro_ok_b)
    print("R_L2K_CELL2_REPRO=%d" % c2_ok)
    print("R_L2K_CELL4_REPRO=%d" % c4_ok)
    for (name, ok, got) in c4_detail["checks"]:
        print("R_L2K_CELL4_%s=%d" % (name, ok))
        print("R_L2K_CELL4_%s_VAL=%s" % (name,
              (",".join(str(v) for v in got) if isinstance(got, list)
               else "%.6f" % got)))
    print("R_L2K_CELL4_ADOPT_A=%.4f" % c4_detail["adopt_a"])
    print("R_L2K_CELL4_ADOPT_B=%.4f" % c4_detail["adopt_b"])
    print("R_L2K_CELL4_COMP_A=%.4f" % c4_detail["comp_a"])
    print("R_L2K_CELL4_COMP_B=%.4f" % c4_detail["comp_b"])
    print("R_L2K_CELL4_JR_RATIO_A=%.6f" % c4_detail["jr_ratio_a"])
    print("R_L2K_CELL4_JR_RATIO_B=%.6f" % c4_detail["jr_ratio_b"])
    print("R_L2K_CELL4_FID_A=%.4f" % c4_detail["fid_a"])
    print("R_L2K_CELL4_FID_B=%.4f" % c4_detail["fid_b"])
    print("R_L2K_CELL4_TRANSFER_A=%.4f" % c4_detail["transfer_a"])
    print("R_L2K_CELL4_TRANSFER_B=%.4f" % c4_detail["transfer_b"])
    print("R_L2K_CELL4_CALIB_A=%.4f" % c4_detail["calib_a"])
    print("R_L2K_CELL4_CALIB_B=%.4f" % c4_detail["calib_b"])
    print("R_L2K_CELL4_W_ADOPT_A=%.4f" % c4_detail["w_a"])
    print("R_L2K_CELL4_W_ADOPT_B=%.4f" % c4_detail["w_b"])
    print("R_L2K_CELL4_SPURIOUS_A=%d" % len(c4_detail["spurious_a"]))
    print("R_L2K_CELL4_SPURIOUS_B=%d" % len(c4_detail["spurious_b"]))
    print("R_L2K_CELL4B_ADOPT=%.4f" % c4b_detail["adopt"])
    print("R_L2K_CELL4B_FID=%.4f" % c4b_detail["fid"])
    print("R_L2K_CELL4B_JR_RATIO=%.6f" % c4b_detail["jr_ratio"])
    print("R_L2K_CELL4B_FP_B=%s" % ",".join(
        ("NA" if v is None else str(v)) for v in c4b_detail["fp_b"]))
    print("R_L2K_WORLD_EQ=%d" % world_ok)
    print("R_L2K_WORLD_EQ_SEEDS=%d" % int(sum(world_oks)))
    print("R_L2K_PRECOMPUTE=%d" % pre_ok)
    print("R_L2K_PRECOMPUTE_DW_A=%d" % pre_detail["dw_a"])
    print("R_L2K_PRECOMPUTE_DW_B=%d" % pre_detail["dw_b"])
    print("R_L2K_PRECOMPUTE_UNDER450_A=%d" % pre_detail["under_a"])
    print("R_L2K_PRECOMPUTE_UNDER450_B=%d" % pre_detail["under_b"])
    for r in seed_rows:
        s = r["seed"]
        print("R_L2K_SEED=%d" % s)
        for side, tag in (("A", "G0AA"), ("B", "G0AB")):
            g0f = r["g0a" if side == "A" else "g0b"]["finalize"]
            print("R_L2K_S%d_%s_G0_MAE=%.6f" % (s, side, g0f["mae_mean"]))
            print("R_L2K_S%d_%s_G0_SC2=%d" % (s, side, g0f["sc2"]))
            print("R_L2K_S%d_%s_G0_COMP=%.4f" % (s, side, g0f["compound_frac"]))
            print("R_L2K_S%d_%s_G0_CHURN=%.4f" % (s, side, g0f["churn_frac"]))
            print("R_L2K_S%d_%s_G0_PROMO=%d" % (s, side, g0f["n_promo"]))
        for side, key in (("A", "ta"), ("B", "tb")):
            mf = r[key]["finalize"]
            print("R_L2K_S%d_%s_M_MAE=%.6f" % (s, side, mf["mae_mean"]))
            print("R_L2K_S%d_%s_M_SC2=%d" % (s, side, mf["sc2"]))
            print("R_L2K_S%d_%s_M_COMP=%.4f" % (s, side, mf["compound_frac"]))
            print("R_L2K_S%d_%s_M_CHURN=%.4f" % (s, side, mf["churn_frac"]))
            print("R_L2K_S%d_%s_M_PROMO=%d" % (s, side, mf["n_promo"]))
            print("R_L2K_S%d_%s_M_FID=%.4f" % (s, side, mf["ctx_fidelity"]))
        for side, key in (("A", "g2a"), ("B", "g2b")):
            g2f = r[key]["finalize"]
            print("R_L2K_S%d_%s_G2S_COMP=%.4f" % (s, side, g2f["compound_frac"]))
            print("R_L2K_S%d_%s_G2S_PROMO=%d" % (s, side, g2f["n_promo"]))
        print("R_L2K_S%d_A_W_PROMO=%d" % (s, r["wa"]["finalize"]["n_promo"]))
        print("R_L2K_S%d_B_W_PROMO=%d" % (s, r["wb"]["finalize"]["n_promo"]))
        print("R_L2K_S%d_JR_A0=%.6f" % (s, r["jr_a0"]))
        print("R_L2K_S%d_JR_B0=%.6f" % (s, r["jr_b0"]))
        print("R_L2K_S%d_JR_A1=%.6f" % (s, r["jr_a1"]))
        print("R_L2K_S%d_JR_B1=%.6f" % (s, r["jr_b1"]))
        print("R_L2K_S%d_JR_RATIO_A=%.6f" % (s, r["jr_ratio_a"]))
        print("R_L2K_S%d_JR_RATIO_B=%.6f" % (s, r["jr_ratio_b"]))
        print("R_L2K_S%d_TRANSFER_A=%.6f" % (s, r["transfer_a"]))
        print("R_L2K_S%d_CALIB_A=%.6f" % (s, r["calib_a"]))
        print("R_L2K_S%d_TRANSFER_B=%.6f" % (s, r["transfer_b"]))
        print("R_L2K_S%d_CALIB_B=%.6f" % (s, r["calib_b"]))
        print("R_L2K_S%d_FIRST_PROMO_A=%s" % (s, ("NA" if r["first_promo_a"] is None
                                                  else str(r["first_promo_a"]))))
        print("R_L2K_S%d_FIRST_PROMO_B=%s" % (s, ("NA" if r["first_promo_b"] is None
                                                  else str(r["first_promo_b"]))))
        if r["gate_a"] is None:
            print("R_L2K_S%d_GATE_PURITY_A=NA" % s)
        else:
            print("R_L2K_S%d_GATE_PURITY_A=%.4f" % (s, r["gate_a"]["purity"]))
        if r["gate_b"] is None:
            print("R_L2K_S%d_GATE_PURITY_B=NA" % s)
        else:
            print("R_L2K_S%d_GATE_PURITY_B=%.4f" % (s, r["gate_b"]["purity"]))
    print("R_L2K_MAE_A=%.6f" % agg["A"]["mae_mean"])
    print("R_L2K_MAE_A_SD=%.6f" % agg["A"]["mae_sd"])
    print("R_L2K_MAE_B=%.6f" % agg["B"]["mae_mean"])
    print("R_L2K_MAE_B_SD=%.6f" % agg["B"]["mae_sd"])
    print("R_L2K_SC2_A=%.4f" % agg["A"]["sc2_mean"])
    print("R_L2K_SC2_B=%.4f" % agg["B"]["sc2_mean"])
    print("R_L2K_COMP_A=%.4f" % agg["A"]["comp_mean"])
    print("R_L2K_COMP_B=%.4f" % agg["B"]["comp_mean"])
    print("R_L2K_CHURN_A=%.4f" % agg["A"]["churn_mean"])
    print("R_L2K_CHURN_B=%.4f" % agg["B"]["churn_mean"])
    print("R_L2K_PROMO_A=%.4f" % agg["A"]["promo_mean"])
    print("R_L2K_PROMO_B=%.4f" % agg["B"]["promo_mean"])
    print("R_L2K_FID_A=%.4f" % fid_a_m)
    print("R_L2K_FID_B=%.4f" % fid_b_m)
    print("R_L2K_JR_A0=%.6f" % jr_a0_m)
    print("R_L2K_JR_B0=%.6f" % jr_b0_m)
    print("R_L2K_JR_A1=%.6f" % jr_a1_m)
    print("R_L2K_JR_B1=%.6f" % jr_b1_m)
    print("R_L2K_JR_RATIO_A=%.6f" % jr_ratio_a_m)
    print("R_L2K_JR_RATIO_B=%.6f" % jr_ratio_b_m)
    print("R_L2K_ADOPT_A=%.4f" % adopt_a)
    print("R_L2K_ADOPT_B=%.4f" % adopt_b)
    print("R_L2K_COMP_ADOPTED_A=%.4f" % comp_adopted_a)
    print("R_L2K_COMP_ADOPTED_B=%.4f" % comp_adopted_b)
    print("R_L2K_ADOPT_A_W=%.4f" % adopt_a_w)
    print("R_L2K_ADOPT_B_W=%.4f" % adopt_b_w)
    print("R_L2K_TRANSFER_A_MEAN=%.6f" % transfer_a_m)
    print("R_L2K_CALIB_A_MEAN=%.6f" % calib_a_m)
    print("R_L2K_TRANSFER_B_MEAN=%.6f" % transfer_b_m)
    print("R_L2K_CALIB_B_MEAN=%.6f" % calib_b_m)
    print("R_L2K_DIAG_ERATIO_A=%.4f" % eratio_a_m)
    print("R_L2K_DIAG_ERATIO_B=%.4f" % eratio_b_m)
    print("R_L2K_DESIGN_WIN_A_FRAC=%.4f" % design_win_a_frac)
    print("R_L2K_DESIGN_WIN_B_FRAC=%.4f" % design_win_b_frac)
    print("R_L2K_CRIT_C1A=%d" % c1a)
    print("R_L2K_CRIT_C1B=%d" % c1b)
    print("R_L2K_CRIT_C2=%d" % c2)
    print("R_L2K_CRIT_C3=%d" % c3)
    print("R_L2K_CRIT_C4=%d" % c4)
    print("R_L2K_CRIT_C4A=%d" % c4a)
    print("R_L2K_CRIT_C4B=%d" % c4b)
    print("R_L2K_CRIT1_JR_RATIO_A=%.6f" % jr_ratio_a_m)
    print("R_L2K_CRIT1_JR_RATIO_B=%.6f" % jr_ratio_b_m)
    print("R_L2K_CRIT2_ADOPT_A=%.4f" % adopt_a)
    print("R_L2K_CRIT2_ADOPT_B=%.4f" % adopt_b)
    print("R_L2K_CRIT2_COMP_ADOPTED_A=%.4f" % comp_adopted_a)
    print("R_L2K_CRIT2_COMP_ADOPTED_B=%.4f" % comp_adopted_b)
    print("R_L2K_CRIT2_G0A_ZERO=%d" % int(sum(cons_g0a)))
    print("R_L2K_CRIT2_G0B_ZERO=%d" % int(sum(cons_g0b)))
    print("R_L2K_CRIT2_G1NA_ZERO=%d" % int(sum(cons_g1na)))
    print("R_L2K_CRIT2_G1NB_ZERO=%d" % int(sum(cons_g1nb)))
    print("R_L2K_CRIT3_SPURIOUS_A=%d" % len(spurious_a))
    print("R_L2K_CRIT3_SPURIOUS_B=%d" % len(spurious_b))
    print("R_L2K_CRIT3_SPURIOUS_A_SEEDS=%s" % (",".join(str(v) for v in spurious_a)
                                               if spurious_a else "NONE"))
    print("R_L2K_CRIT3_SPURIOUS_B_SEEDS=%s" % (",".join(str(v) for v in spurious_b)
                                               if spurious_b else "NONE"))
    print("R_L2K_CRIT4_TRANSFER_A=%.6f" % transfer_a_m)
    print("R_L2K_CRIT4_CALIB_A=%.6f" % calib_a_m)
    print("R_L2K_CRIT4_TRANSFER_B=%.6f" % transfer_b_m)
    print("R_L2K_CRIT4_CALIB_B=%.6f" % calib_b_m)
    print("R_L2K_VERDICT=%s" % verdict)
    print("R_L2K_VERDICT_NOTE=%s" % vnote)
    print("R_L2K_ELAPSED=%.2f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
