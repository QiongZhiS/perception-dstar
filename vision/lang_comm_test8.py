"""vision/lang_comm_test8.py — 语言线第八格：F2 符号化第一格（信号从连续行为 → 稳定符号；
交付 docs/276）。

docs/276 §一 预注册冻结，运行后不改。机制基座 = docs/275 逐字继承（import
lang_comm_test7/lang_comm_test6/lang_comm_test5/lang_comm_test4/lang_comm_test3/
lang_comm_test2/lang_comm_test 复用，零改写）：run_bidi6（frame_sync="stagger5" 半窗相位
错位并行双回路）、BiLangLoop（双向信道 CH1+CH2、对称世界 a1=1.6/a2=2.6、门条件④
GATE_PURITY_MIN=0.80 双回路）、环境 m1=2.6/m2=4.2、JR 采纳子集口径、门后二次验证
（verify_promotions）、判据阈值、留出划分（N_C=14）全沿用（八格同尺可比）。**本格唯一
机制改动 = 信号表示层（docs/276 §1.1 冻结）**：

1. **连续信号合成逐字继承**：d(w) = s(w) - x_self(w)（回路自身可计算的连续相对信号，
   不读真值）。
2. **符号表（行为涌现，冻结）**：由回路自身校准前缀窗 [0, N_C) 的 (d, E) 观测历史
   （感知管线输出：发布质心 + 窗口能量——stagger5 保真预计算，不读真值）计算：
   - 候选边界 = 相邻 distinct d 值中点；τ* = 组内能量纯度 argmax（组内纯度 = 两群窗口
     能量相对两群中位中点 mid 自身侧占比，取向最优——符号 0/1 映射方向任意）；
   - 激活条件（三条同时满足才符号化）：① 接收信号在校准段 >= SYM_DISTINCT=2 个 distinct
     值（信号非退化：常量/无信号不符号化——构造性保持 G0A/G1n 零采纳）；② >= 2×
     SYM_GROUP_MIN=6 个有效 (d,E) 对；③ 边界组内能量纯度 >= SYM_PURITY_MIN=0.80；
   - symbol(w) = 0 if d(w) < tau* else 1；不激活 -> ctx=None（"无结构不符号化"）。
3. **固化**：符号表自回路校准段行为固化、于 N_C 处冻结（表示层环境常量，docs/269
   预计算先例）；每臂每侧一张表（由该臂该侧自己的接收信号/能量计算）。
4. **脱情境引用**：DC 臂于 N_C 处切断信道信号（s=None -> d=None），符号层引用最后固化
   符号（寄存器）作为 ctx——"提到不在场的事物"。

流（§1.6）：主测量（符号化 ON）M-T（两阶段）/M-G（单阶段）/C（前缀）/G0A（双 off）/
G1n（双 null）/G2s（原源 scrambled，诊断 + CELL7 目标）/G2R（真随机族 scrambled，判据
臂 C5-3）/W（均匀世界双 ON）/DC（信号切断，判据臂 C4）；CELL7_REPRO（符号化关闭 16 单元
× 10 种子原源 = docs/275 逐位）；CELL5_REPRO-A（a_first 12 臂）+ CELL5_REPRO-B（b_first
12 臂）。

度量（§1.3）：M1-M6 逐字沿用 docs/275 + M7 符号化度量（符号表/符号序列/一致率/重测/
熵比/压缩/η²/DC 引用率/激活统计）。判据（§1.2）：C1 SYMBOL_STABILITY（同情境一致率
>=0.80 双侧采纳种子均值 + 边界重测 <=0.20）、C2 SYMBOL_PREDICTABILITY（mean(H_sym/
H_cont) <=0.75 双侧采纳种子均值）、C3 SYMBOL_COMPRESSION（N_distinct/K >=10 + mean η²
>=0.30 双侧采纳种子均值）、C4 DISCONTEXT_REFERENCE（DC 引用率 >=0.50 双侧采纳种子均值
+ DC SC2 >=1）、C5 KEEP（采纳 >=0.6 双侧确认制 + JR 子集 <=0.85 + spurious(G2R)==0 +
G0A/G1n 构造性零采纳 + 留出双闸）。判定映射：SYMBOL_EMERGES/PARTIAL/SYMBOL_FLAT/
GUARD_FAIL/F2_BLOCKED。**预期 verdict（设计期实测，如实预注册）= SYMBOL_EMERGES**。

守卫（§1.4）：R_L2N_GUARD_D232、R_L2N_GUARD_D235、R_L2N_CELL2_REPRO、R_L2N_CONSTRUCTION、
R_L2N_PREFIX_EQ（双侧）、R_L2N_TWO_PHASE_EQ（双侧）、R_L2N_REPRO_MAE（双侧）、
R_L2N_DETERM、R_L2N_SMOKE（含符号化语义单元测试）、R_L2N_CELL5_REPRO 双锚、
R_L2N_CELL6_REPRO（复用 CELL7 原源记录 ≡ docs/274）、**R_L2N_CELL7_REPRO（符号化关闭
≡ docs/275 逐位：adopt 0.70/0.60、JR 子集 0.690449/0.757411、全种子 0.783315/0.854447、
fid 0.8522/0.8000、spurious(G2R) 0/0、spurious 原源 A{S7}/B{S1,S3,S8}、transfer 0.600/
0.540、W 0.0/0.7——符号化纯加法/纯表示层）**、R_L2N_WORLD_EQ、R_L2N_PRECOMPUTE。

安全纪律（§1.9）：新文件仅本文件；stdout 只输出 ASCII 标签 + 每行一个数字的 R_L2N_*
摘要块；运行经 powershell 包装重定向到 logs/；数字用 vision/extract_r.py 抽取；禁止读
日志/JSON 原文；本格不读 DAVIS。

用法：
  python vision/lang_comm_test8.py --smoke
  python vision/lang_comm_test8.py --tag timing
  python vision/lang_comm_test8.py --tag main
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
import lang_comm_test5 as l2k
import lang_comm_test6 as l2l
import lang_comm_test7 as l2m
from critical_point import mean_sd, bootstrap_ci, JITTER, N_BOOT, BOOT_SEED
from stream_test import LOOP_CFG

warnings.filterwarnings("ignore")

DEFAULT_OUT = os.path.join("vision", "out", "results")

# ---------------- 冻结常量（docs/276 §1.1/§1.5；运行后不改；docs/275 逐字沿用） ----------------
LVCODES = l2h.LVCODES
LV_WORLD = l2h.LV_WORLD
B_M1 = l2h.B_M1                 # 2.6（B 环境常量）
B_M2 = l2h.B_M2                 # 4.2
A_M1 = l2j.A_M1                 # 1.6（A 环境常量）
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
GATE_PURITY_MIN = l2i.GATE_PURITY_MIN      # 0.80（docs/270 冻结，双侧生效；零改动）
FRAME_SYNC = "stagger5"                    # docs/274 冻结（逐字沿用）
STAGGER_D = 5
# 本格新冻结量（docs/276 §1.5；设计期实测 §二 C1 数据基础）：
SYM_GROUP_MIN = 3              # 符号两群各 ≥3 窗（复用 docs/235 k_consist）
SYM_PURITY_MIN = 0.80          # 符号边界组内能量纯度阈值（复用 docs/270 门阈值）
SYM_DISTINCT = 2               # 信号非退化阈值（接收信号 ≥2 个 distinct 值才符号化）
SYM_STABILITY_MIN = 0.80       # C1a 同情境一致率阈值（采纳种子均值，双侧）
SYM_RETEST_MAX = 0.20          # C1b 边界重测归一化偏差阈值
SYM_PRED_RATIO_MAX = 0.75      # C2 mean(H_sym/H_cont) 阈值（双侧采纳种子均值）
SYM_COMPRESS_MIN = 10.0        # C3a 压缩率阈值（N_distinct/K）
SYM_ETA2_MIN = 0.30            # C3b η²_sym 阈值（双侧采纳种子均值）
DISCONTEXT_REF_MIN = 0.50      # C4 脱情境引用率阈值（双侧采纳种子均值）


# ---------------- 符号表（docs/276 §1.1 冻结：stagger5 保真预计算 + 组内能量纯度 argmax） ----------------
def precompute_d(s, sig1=None, sig2=None, mode="comm", worlds=None):
    """Stagger5 保真的 (fa, fb, wl, dA, EA, dB, EB, sB_recv, sA_recv)：由回路自身感知
    管线输出计算（发布质心序列 + 窗口能量），不读真值。mode comm = 真实信号；null =
    s≡160；off = 无信号；scrambled = sig1（B 收）/sig2（A 收）注入；worlds = 世界三元组
    （默认双向世界，W 臂传均匀世界）。"""
    if worlds is None:
        fa, fb, wl = l2j.make_bidi_world(s)
    else:
        fa, fb, wl = worlds
    la = l2g.run_a_signal(fa)                        # A 对齐窗：s_A 发布 = x_A
    lb_stag = l2j.run_b_signal_bidi(fb[l2l.STAGGER_D:])  # B 错位窗：s_B 发布/x_B/能量
    sA = list(la.sA_trace)
    xA = list(la.sA_trace)
    sB = list(lb_stag.sA_trace)
    xB = list(lb_stag.xB_trace)
    EB = list(lb_stag.energy_trace)
    EA = l2h.precompute_energies(fa)
    if mode == "comm":
        sB_recv, sA_recv = sB, sA
    elif mode == "null":
        sA_recv = [160.0] * len(sA)
        sB_recv = [160.0] * len(sB)
    elif mode == "off":
        sA_recv = [None] * len(sA)
        sB_recv = [None] * len(sB)
    else:                                            # scrambled
        sB_recv = list(sig1)
        sA_recv = list(sig2)
    dA = [None] + [sB_recv[w - 1] - xA[w] for w in range(1, len(xA))]
    dB = [sA_recv[w] - xB[w] for w in range(len(xB))]
    return fa, fb, wl, dA, EA, dB, EB, sB_recv, sA_recv


def build_symbol_table(ds, Es, recv_s, n_c=N_C, gmin=SYM_GROUP_MIN,
                       emerge_pur=SYM_PURITY_MIN, distinct=SYM_DISTINCT):
    """符号表（行为涌现，冻结）：激活 iff ① 接收信号校准段 ≥ distinct 个 distinct 值
    （非退化）② ≥ 2*gmin 个有效 (d,E) 对 ③ 边界组内能量纯度 ≥ emerge_pur。
    τ* = 组内能量纯度 argmax（候选 = 相邻 distinct d 值中点；纯度取向最优）。
    返回 (active, tau, purity)。"""
    s_calib = [v for v in recv_s[:n_c] if v is not None]
    if len(set(round(float(v), 6) for v in s_calib)) < distinct:
        return False, None, 0.0
    pairs = [(float(d), float(E)) for d, E in zip(ds[:n_c], Es[:n_c])
             if d is not None]
    if len(pairs) < 2 * gmin:
        return False, None, 0.0
    svals = sorted(set(round(p[0], 9) for p in pairs))
    best = None
    for i in range(len(svals) - 1):
        tau = (svals[i] + svals[i + 1]) / 2.0
        g0 = [E for d, E in pairs if d < tau]
        g1 = [E for d, E in pairs if d >= tau]
        if len(g0) < gmin or len(g1) < gmin:
            continue
        m0, m1 = float(np.median(g0)), float(np.median(g1))
        if m0 <= 0 or m1 <= 0:
            continue
        mid = (m0 + m1) / 2.0
        f0 = sum(1.0 for e in g0 if e < mid) / len(g0)
        f1 = sum(1.0 for e in g1 if e >= mid) / len(g1)
        pur = max(min(f0, f1), min(1.0 - f0, 1.0 - f1))
        if best is None or pur > best[1]:
            best = (tau, pur)
    if best is None:
        return False, None, 0.0
    return (best[1] >= emerge_pur), best[0], best[1]


# ---------------- SymbolBiLangLoop（docs/276 §1.1 冻结：符号映射 + 脱情境引用寄存器） ----------------
class SymbolBiLangLoop(l2j.BiLangLoop):
    def __init__(self, sym_table=None, discontext=False, **kw):
        super().__init__(**kw)
        self.sym_table = sym_table          # (active, tau)
        self.discontext = discontext
        self.last_sym = None

    def _ctx_from_signal(self, ev_win):
        # 连续信号合成逐字继承（s 与 x_self 计算与基座逐字相同）；符号映射替换设计二分
        if self.self_side == "lower":
            lo = ev_win[int(l2g.CTX_SPLIT_Y):, :]
            lo_n = int(lo.sum())
            xb = float(np.mean(np.nonzero(lo)[1])) if lo_n >= l2g.SIG_SPARSE_PX else None
            self._last_xB = xb
            if self.mode == "off":
                return None
            s = l2g.NULL_SIGNAL if self.mode == "null" else self.signal
            if s is None or xb is None:
                if self.discontext and self.last_sym is not None:
                    return self.last_sym
                return None
            d = s - xb
        else:
            up = ev_win[:int(l2g.CTX_SPLIT_Y), :]
            up_n = int(up.sum())
            xa = float(np.mean(np.nonzero(up)[1])) if up_n >= l2g.SIG_SPARSE_PX else None
            self._last_xB = xa
            if self.mode == "off":
                return None
            s = l2g.NULL_SIGNAL if self.mode == "null" else self.signal
            if s is None or xa is None:
                if self.discontext and self.last_sym is not None:
                    return self.last_sym
                return None
            d = s - xa
        active, tau = self.sym_table[0], self.sym_table[1]
        if not active:
            return None
        sym = 0 if d < tau else 1
        self.last_sym = sym
        return sym


# ---------------- run_bidi8（docs/276 §1.1/§1.4 冻结：run_bidi6 逐字 + 表示层分支） ----------------
def run_bidi8(fa, fb, wl_a, wl_b, ch1="comm", ch2="comm", a_mode=None,
              two_phase=False, n_c=N_C, gate=GATE_PURITY_MIN,
              want_end_snap=False, sig1_fn=None, sig2_fn=None,
              sym_table_a=None, sym_table_b=None, discontext=False,
              cut_after=None, symbolize=False):
    """stagger5（docs/274 §1.1 逐字）：
    - symbolize=True：loop = SymbolBiLangLoop（符号表映射 ctx；discontext/cut_after 供
      DC 臂：双方自窗口 >= cut_after 起无信号，符号层引用最后固化符号）；
    - symbolize=False：loop = l2j.BiLangLoop 逐字节（CELL7_REPRO = docs/275 原源）。
    其余（帧同步/交换/冲刷/门/账本）与 run_bidi6 逐字节一致。"""
    if symbolize:
        loop_a = SymbolBiLangLoop(mode=(a_mode if a_mode is not None else ch2),
                                  self_side="upper", publish_side="upper",
                                  gate_purity_min=gate, window=WINDOW,
                                  sym_table=sym_table_a, discontext=discontext,
                                  **LOOP_CFG)
        loop_b = SymbolBiLangLoop(mode=ch1, self_side="lower", publish_side="lower",
                                  gate_purity_min=gate, window=WINDOW,
                                  sym_table=sym_table_b, discontext=discontext,
                                  **LOOP_CFG)
    else:
        loop_a = l2j.BiLangLoop(mode=(a_mode if a_mode is not None else ch2),
                                self_side="upper", publish_side="upper",
                                gate_purity_min=gate, window=WINDOW, **LOOP_CFG)
        loop_b = l2j.BiLangLoop(mode=ch1, self_side="lower", publish_side="lower",
                                gate_purity_min=gate, window=WINDOW, **LOOP_CFG)
    n_frames = len(fb)
    n_w_a = len(fa) // WINDOW
    n_w_b = (n_frames - STAGGER_D) // WINDOW
    phases = ([(0, n_c * WINDOW), (n_c * WINDOW, n_frames)] if two_phase
              else [(0, n_frames)])
    snap_a = snap_b = None
    a_closed = b_closed = 0
    # DC 切断：B 的窗口 w 读 s_A(w)（wc=w >= cut_after -> None）；
    # A 的窗口 w 读 s_B(w-1)（wc=w-1 >= cut_after-1 -> None）——双方自窗口 cut_after 起无信号
    cut_b = cut_after if cut_after is not None else None
    cut_a = (cut_after - 1) if cut_after is not None else None
    for (f0, f1) in phases:
        start = max(f0, STAGGER_D)
        for k in range(f0, start):
            loop_a.step(fa[k])
        for k in range(start, f1):
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
                if ch1 == "comm":
                    sig = loop_a.sA_trace[a_closed]
                    if cut_b is not None and a_closed >= cut_b:
                        sig = None
                    loop_b.set_signal(sig)
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
                if ch2 == "comm":
                    sig = loop_b.sA_trace[b_closed]
                    if cut_a is not None and b_closed >= cut_a:
                        sig = None
                    loop_a.set_signal(sig)
                b_closed += 1
        if two_phase and f0 == 0:
            snap_a = l2g.snapshot_b(loop_a)
            snap_b = l2g.snapshot_b(loop_b)
    if want_end_snap and snap_a is None:
        snap_a = l2g.snapshot_b(loop_a)
        snap_b = l2g.snapshot_b(loop_b)
    # 收尾冲刷（B 恒为后闭窗方）：A 先冲刷（发布 s_A 末窗 -> B 冲刷窗读取），B 后冲刷
    if len(loop_a._frame_buf):
        loop_a.finalize(n_w_a, None)
        if len(loop_a.sA_trace) > a_closed and ch1 == "comm":
            sig = loop_a.sA_trace[-1]
            if cut_b is not None and a_closed >= cut_b:
                sig = None
            loop_b.set_signal(sig)
    if len(loop_b._frame_buf):
        loop_b.finalize(n_w_b, None)
        if len(loop_b.sA_trace) > b_closed and ch2 == "comm":
            sig = loop_b.sA_trace[-1]
            if cut_a is not None and b_closed >= cut_a:
                sig = None
            loop_a.set_signal(sig)
    labels_a = [dict(ctx=1 - lb["ctx"], b_mult=lb["b_mult"],
                     a_regime=lb["a_regime"]) for lb in wl_a]   # A 侧真值 = 1-ctx（对称）
    out_a = loop_a.finalize(n_w_a, labels_a)
    out_b = loop_b.finalize(n_w_b, wl_b)
    return out_a, loop_a, out_b, loop_b, snap_a, snap_b


# ---------------- 单元记录（docs/276 §1.3：unit_record7 逐字 + 符号化度量字段） ----------------
def unit_record8(arm, seed, out, loop, side, snap=None, jr=None, att=None,
                 table=None, dseq=None):
    rec = l2m.unit_record7(arm, seed, out, loop, side, snap=snap, jr=jr, att=att)
    rec["sym"] = {"table": (int(table[0]),
                            (table[1] if table[1] is not None else None),
                            round(table[2], 6)),
                  "n_ctx": int(sum(1 for s2 in loop.sig_trace
                                   if s2[2] is not None)),
                  "c2seq": [None if s2[2] is None else int(s2[2])
                            for s2 in loop.sig_trace],
                  "dseq": ([None if v is None else round(float(v), 4)
                            for v in dseq] if dseq is not None else None)}
    return rec


# ---------------- 判据度量（docs/276 §1.2 冻结；真值只进评估统计） ----------------
def cond_entropy(seq):
    """经验条件熵 H(x_w | x_{w-1})（bit/步）。"""
    n = len(seq)
    if n < 2:
        return 0.0
    counts = {}
    for i in range(1, n):
        counts[(seq[i - 1], seq[i])] = counts.get((seq[i - 1], seq[i]), 0) + 1
    prevs = {}
    for (p, c), v in counts.items():
        prevs.setdefault(p, []).append((c, v))
    H = 0.0
    tot = n - 1
    for p, lst in prevs.items():
        nv = sum(v for _, v in lst)
        for _, v in lst:
            p_c = v / nv
            H -= (nv / tot) * p_c * np.log2(p_c)
    return float(H)


def consistency_rate(syms, truth):
    """C1a 同情境一致率：各真值态取多数符号，一致率 = 与态多数一致的窗口占比。"""
    pairs = [(sym, t) for sym, t in zip(syms, truth)
             if sym is not None and t is not None]
    if not pairs:
        return 0.0, 0
    ok = 0
    for c in (0, 1):
        members = [sym for sym, t in pairs if t == c]
        if members:
            maj = 1 if sum(1 for m in members if m == 1) * 2 >= len(members) else 0
            ok += sum(1 for m in members if m == maj)
    return (ok / len(pairs)), len(pairs)


def boundary_retest(dseq, n_c=N_C):
    """C1b 边界重测：校准段 d 中位数 vs 留出段 d 中位数的归一化偏差。"""
    vc = [float(v) for v in dseq[:n_c] if v is not None]
    vh = [float(v) for v in dseq[n_c:] if v is not None]
    if len(vc) < 6 or len(vh) < 6:
        return None
    dr = max(vc + vh) - min(vc + vh)
    if dr <= 0:
        return 0.0
    return abs(float(np.median(vc)) - float(np.median(vh))) / dr


def entropy_ratio(loop_or_seq, dseq):
    """C2 H_sym/H_cont：符号序列转移熵 vs 4-bin 连续基线转移熵。
    loop_or_seq = c2 序列（list）或 loop（取其 sig_trace c2）。"""
    if hasattr(loop_or_seq, "sig_trace"):
        syms = [s2[2] for s2 in loop_or_seq.sig_trace if s2[2] is not None]
    else:
        syms = [v for v in loop_or_seq if v is not None]
    dv = [v for v in dseq if v is not None]
    if len(syms) < 2 or len(dv) < 2:
        return None
    lo, hi = min(dv), max(dv)
    b4 = [int(np.clip(int((v - lo) / (hi - lo + 1e-9) * 4), 0, 3)) for v in dv]
    hs = cond_entropy(syms)
    hc = cond_entropy(b4)
    if hc <= 0:
        return None
    return hs / hc


def eta2_sym(dseq, Es, tau):
    """C3b η²_sym：符号分群解释窗口能量的比例。"""
    syms = [0 if v < tau else 1 for v in dseq if v is not None]
    Es2 = [Es[w] for w, v in enumerate(dseq) if v is not None]
    if len(syms) < 4:
        return 0.0
    tot = sum((e - np.mean(Es2)) ** 2 for e in Es2)
    if tot <= 0:
        return 0.0
    grp = {0: [], 1: []}
    for sym, e in zip(syms, Es2):
        grp[sym].append(e)
    within = sum(sum((e - np.mean(g)) ** 2 for e in g)
                 for g in grp.values() if g)
    return 1.0 - within / tot


def dc_ref_rate(loop, n_c=N_C):
    """C4a 脱情境引用率：切断后窗口（w >= n_c、E>=10）中匹配 arity-3（符号键）条目占比。"""
    E = loop.energy_trace
    matched = {}
    for w, k in loop.match_trace:
        matched[w] = k
    n_el, n_ref = 0, 0
    for w in range(n_c, len(E)):
        if E[w] >= 10:
            n_el += 1
            k = matched.get(w)
            if k is not None and len(k) == 3:
                n_ref += 1
    return (n_ref / n_el) if n_el else 0.0, n_el, n_ref


# ---------------- R_L2N_CELL7_REPRO（docs/276 §1.4 复现锚：符号化关闭 ≡ docs/275 逐位） ----------------
# 期望数字 = docs/275 §三/§四 冻结值（主测量原源 = run_bidi8(symbolize=False) =
# run_bidi6 逐字节；同代码路径 -> 期望位精确）。来源行：docs/275 §3.1-§3.4/§四。
CELL7_EXP = {
    "adopt_a": 0.7000, "adopt_b": 0.6000,
    "comp_a": 1.0000, "comp_b": 1.0000,
    "jr_sub_a": 0.690449, "jr_sub_b": 0.757411,
    "jr_full_a": 0.783315, "jr_full_b": 0.854447,
    "fid_a": 0.8522, "fid_b": 0.8000,
    "transfer_a": 0.600, "transfer_b": 0.540,
    "calib_a": 0.503, "calib_b": 0.386,
    "w_a": 0.0, "w_b": 0.7,
    # docs/275 治理后（确认制）原源 spurious 诊断读数：A 0/10（S7-A 被验证拦截，
    # 子条目纯度 0.429<0.65）+ B {S1,S8}（S1/S8 原源半真信号，纯度 1.000/0.769）；
    # RAW（docs/274 级）A{S7}/B{S1,S3,S8} 由 CELL6_REPRO（repro_cell6）核对。
    "spurious_a": [], "spurious_b": [1, 8],
    "fp_a": [None, 9, 7, None, None, 7, 7, 15, 7, 9],
    "fp_b": [None, 13, 11, 7, None, None, 10, None, 10, 8],
    "gate_a": [None, 0.8, 1.0, None, None, 1.0, 1.0, 0.8, 1.0, 1.0],
    "gate_b": [None, 0.8333, 0.8, 1.0, None, None, 1.0, None, 0.8, 0.8],
    "fid_a_per": [0.7826, 0.8696, 0.9130, 0.8261, 0.8696, 0.9130, 0.8696,
                  0.7826, 0.8696, 0.8261],
    "fid_b_per": [0.7917, 0.8750, 0.8750, 0.8333, 0.7083, 0.7917, 0.8333,
                  0.7083, 0.7917, 0.7917],
    "transfer_a_per": [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
    "transfer_b_per": [0.0, 0.9, 0.9, 0.9, 0.0, 0.0, 0.9, 0.0, 0.9, 0.9],
    "calib_a_per": [0.0, 0.800, 0.857, 0.0, 0.0, 0.857, 0.857, 0.0, 0.857, 0.800],
    "calib_b_per": [0.0, 0.000, 0.667, 0.857, 0.0, 0.0, 0.750, 0.0, 0.750, 0.833],
    "jr_a0": [0.1508, 0.1481, 0.1579, 0.1260, 0.0831, 0.1389, 0.1527,
              0.2234, 0.1344, 0.1418],
    "jr_a1": [0.1508, 0.1139, 0.0776, 0.1260, 0.0831, 0.0903, 0.1055,
              0.1905, 0.0916, 0.0989],
    "jr_b0": [0.1306, 0.1512, 0.1430, 0.1309, 0.1273, 0.1465, 0.1452,
              0.2202, 0.1364, 0.1366],
    "jr_b1": [0.1306, 0.0999, 0.1269, 0.1043, 0.1273, 0.1465, 0.0948,
              0.2202, 0.1108, 0.1003],
    "ratio_a": [1.0000, 0.7692, 0.4912, 1.0000, 1.0000, 0.6501, 0.6910,
                0.8526, 0.6815, 0.6975],
    "ratio_b": [1.0000, 0.6609, 0.8876, 0.7967, 1.0000, 1.0000, 0.6528,
                1.0000, 0.8123, 0.7342],
}


def cell7_repro(per_unit, seeds):
    """CELL7_REPRO：run_bidi8(symbolize=False) 主测量原源记录 ≡ docs/275 §三/§四 逐位
    （符号化关闭 = 连续信号原样——基座未漂移的构造性证明）。返回 (ok, detail)。"""
    def nconf(rec):
        return rec["verify"]["n_confirmed"]

    def chk(name, okv, got):
        return name, int(okv), got

    checks = []
    jr_as, jr_bs = [], []
    fids_a, fids_b = [], []
    trans_a, trans_b = [], []
    calibs_a, calibs_b = [], []
    fp_as, fp_bs = [], []
    gate_as, gate_bs = [], []
    comp_a, comp_b = [], []
    g2r_a, g2r_b = [], []
    g2s_a, g2s_b = [], []
    w_as, w_bs = [], []
    for s in seeds:
        ta = per_unit["MTA_%d" % s]
        tb = per_unit["MTB_%d" % s]
        g0a = per_unit["G0AA_%d" % s]
        g0b = per_unit["G0AB_%d" % s]
        g2ra = per_unit["G2RA_%d" % s]
        g2rb = per_unit["G2RB_%d" % s]
        g2sa = per_unit["G2SA_%d" % s]
        g2sb = per_unit["G2SB_%d" % s]
        wa = per_unit["WA_%d" % s]
        wb = per_unit["WB_%d" % s]
        ra = ta["jr"][0] / max(g0a["jr"][0], 1e-12)
        rb = tb["jr"][0] / max(g0b["jr"][0], 1e-12)
        jr_as.append(ra)
        jr_bs.append(rb)
        fids_a.append(ta["finalize"]["ctx_fidelity"])
        fids_b.append(tb["finalize"]["ctx_fidelity"])
        trans_a.append(ta["att"]["transfer_adopted_hit_rate"])
        trans_b.append(tb["att"]["transfer_adopted_hit_rate"])
        calibs_a.append(ta["att"]["calib_baseline"])
        calibs_b.append(tb["att"]["calib_baseline"])
        fp_as.append(ta["att"]["first_promo_win"])
        fp_bs.append(tb["att"]["first_promo_win"])
        comp_a.append(ta["finalize"]["compound_frac"])
        comp_b.append(tb["finalize"]["compound_frac"])
        if nconf(g2ra) >= 1:
            g2r_a.append(s)
        if nconf(g2rb) >= 1:
            g2r_b.append(s)
        if nconf(g2sa) >= 1:
            g2s_a.append(s)
        if nconf(g2sb) >= 1:
            g2s_b.append(s)
        w_as.append(wa["finalize"]["n_promo"])
        w_bs.append(wb["finalize"]["n_promo"])
        ga_ = gb_ = None
        if fp_as[-1] is not None:
            for rec in ta["gate"]:
                if rec[0] == fp_as[-1]:
                    ga_ = rec[2]
                    break
        if fp_bs[-1] is not None:
            for rec in tb["gate"]:
                if rec[0] == fp_bs[-1]:
                    gb_ = rec[2]
                    break
        gate_as.append(ga_)
        gate_bs.append(gb_)
    adopt_a = float(np.mean([fp is not None for fp in fp_as]))
    adopt_b = float(np.mean([fp is not None for fp in fp_bs]))
    adopted_a = [s for s in seeds if nconf(per_unit["MTA_%d" % s]) >= 1]
    adopted_b = [s for s in seeds if nconf(per_unit["MTB_%d" % s]) >= 1]
    jr_sub_a = float(np.mean([jr_as[s] for s in adopted_a])) if adopted_a else 0.0
    jr_sub_b = float(np.mean([jr_bs[s] for s in adopted_b])) if adopted_b else 0.0
    jr_full_a, _ = mean_sd(jr_as)
    jr_full_b, _ = mean_sd(jr_bs)
    fid_a_m = float(np.mean(fids_a))
    fid_b_m = float(np.mean(fids_b))
    trans_a_m, _ = mean_sd(trans_a)
    trans_b_m, _ = mean_sd(trans_b)
    calib_a_m, _ = mean_sd(calibs_a)
    calib_b_m, _ = mean_sd(calibs_b)
    comp_adopted_a = float(np.mean([comp_a[s] for s in adopted_a])) if adopted_a else 0.0
    comp_adopted_b = float(np.mean([comp_b[s] for s in adopted_b])) if adopted_b else 0.0
    w_a_m = float(np.mean(w_as))
    w_b_m = float(np.mean(w_bs))
    for s in seeds:
        checks.append(chk("JR_RATIO_A_S%d" % s,
                          abs(jr_as[s] - CELL7_EXP["ratio_a"][s]) < 1e-4, jr_as[s]))
        checks.append(chk("JR_RATIO_B_S%d" % s,
                          abs(jr_bs[s] - CELL7_EXP["ratio_b"][s]) < 1e-4, jr_bs[s]))
        checks.append(chk("JR_A0_S%d" % s,
                          abs(per_unit["G0AA_%d" % s]["jr"][0] -
                              CELL7_EXP["jr_a0"][s]) < 1e-4,
                          per_unit["G0AA_%d" % s]["jr"][0]))
        checks.append(chk("JR_A1_S%d" % s,
                          abs(per_unit["MTA_%d" % s]["jr"][0] -
                              CELL7_EXP["jr_a1"][s]) < 1e-4,
                          per_unit["MTA_%d" % s]["jr"][0]))
        checks.append(chk("JR_B0_S%d" % s,
                          abs(per_unit["G0AB_%d" % s]["jr"][0] -
                              CELL7_EXP["jr_b0"][s]) < 1e-4,
                          per_unit["G0AB_%d" % s]["jr"][0]))
        checks.append(chk("JR_B1_S%d" % s,
                          abs(per_unit["MTB_%d" % s]["jr"][0] -
                              CELL7_EXP["jr_b1"][s]) < 1e-4,
                          per_unit["MTB_%d" % s]["jr"][0]))
        checks.append(chk("FID_A_S%d" % s,
                          abs(fids_a[s] - CELL7_EXP["fid_a_per"][s]) < 1e-4,
                          fids_a[s]))
        checks.append(chk("FID_B_S%d" % s,
                          abs(fids_b[s] - CELL7_EXP["fid_b_per"][s]) < 1e-4,
                          fids_b[s]))
        checks.append(chk("TRANSFER_A_S%d" % s,
                          abs(trans_a[s] - CELL7_EXP["transfer_a_per"][s]) < 1e-3,
                          trans_a[s]))
        checks.append(chk("TRANSFER_B_S%d" % s,
                          abs(trans_b[s] - CELL7_EXP["transfer_b_per"][s]) < 1e-3,
                          trans_b[s]))
        checks.append(chk("CALIB_A_S%d" % s,
                          abs(calibs_a[s] - CELL7_EXP["calib_a_per"][s]) < 1e-3,
                          calibs_a[s]))
        checks.append(chk("CALIB_B_S%d" % s,
                          abs(calibs_b[s] - CELL7_EXP["calib_b_per"][s]) < 1e-3,
                          calibs_b[s]))
        checks.append(chk("FP_A_S%d" % s, fp_as[s] == CELL7_EXP["fp_a"][s],
                          (fp_as[s] if fp_as[s] is not None else -1)))
        checks.append(chk("FP_B_S%d" % s, fp_bs[s] == CELL7_EXP["fp_b"][s],
                          (fp_bs[s] if fp_bs[s] is not None else -1)))
        checks.append(chk("GATE_A_S%d" % s,
                          (gate_as[s] is None and CELL7_EXP["gate_a"][s] is None)
                          or (gate_as[s] is not None
                              and CELL7_EXP["gate_a"][s] is not None
                              and abs(gate_as[s] - CELL7_EXP["gate_a"][s]) < 1e-4),
                          (gate_as[s] if gate_as[s] is not None else -1)))
        checks.append(chk("GATE_B_S%d" % s,
                          (gate_bs[s] is None and CELL7_EXP["gate_b"][s] is None)
                          or (gate_bs[s] is not None
                              and CELL7_EXP["gate_b"][s] is not None
                              and abs(gate_bs[s] - CELL7_EXP["gate_b"][s]) < 1e-4),
                          (gate_bs[s] if gate_bs[s] is not None else -1)))
    checks.append(chk("ADOPT_A", abs(adopt_a - CELL7_EXP["adopt_a"]) < 1e-6, adopt_a))
    checks.append(chk("ADOPT_B", abs(adopt_b - CELL7_EXP["adopt_b"]) < 1e-6, adopt_b))
    checks.append(chk("COMP_ADOPTED_A", abs(comp_adopted_a - CELL7_EXP["comp_a"]) < 1e-4,
                      comp_adopted_a))
    checks.append(chk("COMP_ADOPTED_B", abs(comp_adopted_b - CELL7_EXP["comp_b"]) < 1e-4,
                      comp_adopted_b))
    checks.append(chk("JR_SUBSET_A", abs(jr_sub_a - CELL7_EXP["jr_sub_a"]) < 1e-3,
                      jr_sub_a))
    checks.append(chk("JR_SUBSET_B", abs(jr_sub_b - CELL7_EXP["jr_sub_b"]) < 1e-3,
                      jr_sub_b))
    checks.append(chk("JR_FULL_A", abs(jr_full_a - CELL7_EXP["jr_full_a"]) < 1e-3,
                      jr_full_a))
    checks.append(chk("JR_FULL_B", abs(jr_full_b - CELL7_EXP["jr_full_b"]) < 1e-3,
                      jr_full_b))
    checks.append(chk("FID_A_MEAN", abs(fid_a_m - CELL7_EXP["fid_a"]) < 1e-4, fid_a_m))
    checks.append(chk("FID_B_MEAN", abs(fid_b_m - CELL7_EXP["fid_b"]) < 1e-4, fid_b_m))
    checks.append(chk("TRANSFER_A_MEAN", abs(trans_a_m - CELL7_EXP["transfer_a"]) < 1e-3,
                      trans_a_m))
    checks.append(chk("TRANSFER_B_MEAN", abs(trans_b_m - CELL7_EXP["transfer_b"]) < 1e-3,
                      trans_b_m))
    checks.append(chk("CALIB_A_MEAN", abs(calib_a_m - CELL7_EXP["calib_a"]) < 1e-3,
                      calib_a_m))
    checks.append(chk("CALIB_B_MEAN", abs(calib_b_m - CELL7_EXP["calib_b"]) < 1e-3,
                      calib_b_m))
    checks.append(chk("W_ADOPT_A", abs(w_a_m - CELL7_EXP["w_a"]) < 1e-6, w_a_m))
    checks.append(chk("W_ADOPT_B", abs(w_b_m - CELL7_EXP["w_b"]) < 1e-3, w_b_m))
    checks.append(chk("SPURIOUS_G2R_A", g2r_a == [], g2r_a))
    checks.append(chk("SPURIOUS_G2R_B", g2r_b == [], g2r_b))
    checks.append(chk("SPURIOUS_ORIG_A", g2s_a == CELL7_EXP["spurious_a"], g2s_a))
    checks.append(chk("SPURIOUS_ORIG_B", g2s_b == CELL7_EXP["spurious_b"], g2s_b))
    ok = int(all(c[1] == 1 for c in checks))
    return ok, dict(checks=checks, adopt_a=adopt_a, adopt_b=adopt_b,
                    jr_sub_a=jr_sub_a, jr_sub_b=jr_sub_b,
                    jr_full_a=jr_full_a, jr_full_b=jr_full_b,
                    fid_a=fid_a_m, fid_b=fid_b_m,
                    transfer_a=trans_a_m, transfer_b=trans_b_m,
                    calib_a=calib_a_m, calib_b=calib_b_m,
                    w_a=w_a_m, w_b=w_b_m,
                    spurious_a=g2r_a, spurious_b=g2r_b,
                    adopted_a=adopted_a, adopted_b=adopted_b)


# ---------------- 构造冒烟（docs/276 §1.4；合成帧 + 符号化语义单元测试，非数据） ----------------
def smoke_main8():
    """构造冒烟：docs/275 smoke_main7 全部 + 本格符号化语义（符号表激活/退化不激活/
    能量不纯不激活/DC 寄存器；G2R 构造运行正常）。"""
    results = {}
    fb = l2g._synth_frames(30)
    fa = l2g._synth_frames(30, y0=26)
    labels = [dict(ctx=1, b_mult=1.0, a_regime=None)] * 3
    labels_b = [dict(ctx=0, b_mult=1.0, a_regime=None)] * 3
    for ch1, ch2 in (("off", "off"), ("comm", "comm"), ("null", "null"),
                     ("scrambled", "scrambled")):
        out_a, loop_a, out_b, loop_b, _, _ = run_bidi8(
            fa, fb, labels, labels_b, ch1=ch1, ch2=ch2, two_phase=False, n_c=3)
        results["construct_%s_%s" % (ch1, ch2)] = int(
            isinstance(out_a, dict) and isinstance(out_b, dict)
            and len(out_a.get("mae_trace", [])) >= 1
            and len(out_b.get("mae_trace", [])) >= 1)
    fb2 = l2g._synth_frames(100)
    fa2 = l2g._synth_frames(100, y0=26)
    lab2 = [dict(ctx=1, b_mult=1.0, a_regime=None)] * 10
    lab2b = [dict(ctx=0, b_mult=1.0, a_regime=None)] * 10
    _, la4, _, lb4, _, _ = run_bidi8(fa2, fb2, lab2, lab2b, ch1="comm",
                                     ch2="comm", two_phase=False, n_c=5)
    results["stagger_a_first_win_none"] = int(la4.sig_trace[0][2] is None)
    results["stagger_b_first_win_non_none"] = int(lb4.sig_trace[0][2] is not None)
    # ---- 符号表语义单元测试（合成 (d,E) 数组） ----
    ds_s = [1.0, 2.0, 3.0, 4.0, -4.0, -3.0, -2.0, -1.0] + [5.0, 6.0] * 3
    Es_s = ([500.0, 510.0, 520.0, 505.0] + [700.0, 710.0, 690.0, 705.0]
            + [515.0, 505.0] * 3)
    recv_s = [70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0,
              80.0, 81.0, 82.0, 83.0]
    t_act = build_symbol_table(ds_s, Es_s, recv_s)
    results["sym_structured_active"] = int(t_act[0] and t_act[2] >= SYM_PURITY_MIN)
    recv_const = [160.0] * 14
    t_const = build_symbol_table([80.0 - v for v in recv_const], Es_s, recv_const)
    results["sym_const_signal_inactive"] = int(not t_const[0])
    recv_none = [None] * 14
    t_none = build_symbol_table([None] * 14, Es_s, recv_none)
    results["sym_no_signal_inactive"] = int(not t_none[0])
    ds_mix = [1.0, 2.0, 3.0, 4.0, -4.0, -3.0, -2.0, -1.0, 5.0, 6.0, 7.0,
              8.0, 9.0, 10.0]
    Es_mix = [500.0, 710.0, 520.0, 690.0, 505.0, 700.0, 515.0, 705.0,
              510.0, 695.0, 525.0, 715.0, 500.0, 705.0]
    t_mix = build_symbol_table(ds_mix, Es_mix, recv_s)
    results["sym_mixed_energy_inactive"] = int(not t_mix[0])
    # ---- DC 寄存器语义（真实 loop：信号切断后引用最后固化符号） ----
    lp = SymbolBiLangLoop(mode="comm", self_side="lower", publish_side="lower",
                          sym_table=(True, 0.0), discontext=True, **LOOP_CFG)
    ev = np.zeros((120, 160), bool)
    ev[80:100, 70:90] = True                       # B 自身事件（x=80）
    lp.signal = 70.0                                # s=70 -> d=-10 < 0 -> sym 0
    c1 = lp._ctx_from_signal(ev)
    lp.signal = None                                # 信号切断
    c2 = lp._ctx_from_signal(ev)                    # 引用最后固化符号 0
    lp2 = SymbolBiLangLoop(mode="comm", self_side="lower", publish_side="lower",
                           sym_table=(False, None), discontext=True, **LOOP_CFG)
    lp2.signal = None
    c3 = lp2._ctx_from_signal(ev)                   # 无符号可引用 -> None
    results["dc_register_refs_last_sym"] = int(c1 == 0 and c2 == 0)
    results["dc_no_symbol_none"] = int(c3 is None)
    # ---- G2R 构造运行正常 ----
    out_g2r, _, _, _, _, _ = run_bidi8(
        fa2, fb2, lab2, lab2b, ch1="scrambled", ch2="scrambled",
        a_mode="scrambled", n_c=5,
        sig1_fn=lambda w: 131.0 if w % 2 == 0 else 31.0,
        sig2_fn=lambda w: 31.0 if w % 2 == 0 else 131.0, symbolize=True,
        sym_table_a=(True, 0.0), sym_table_b=(True, 0.0))
    results["g2r_construct"] = int(isinstance(out_g2r, dict))
    for k in sorted(results):
        print("R_L2N_SMOKE_%s=%d" % (k.upper(), results[k]))
    return 0 if all(results.values()) else 1


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="l2n")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.smoke:
        return smoke_main8()
    t0 = time.time()
    seeds = list(range(10))

    cfg = {"tag": args.tag, "n_seeds": len(seeds), "frames": N_FRAMES,
           "window": WINDOW, "n_c": N_C, "jitter": JITTER,
           "b_m1": B_M1, "b_m2": B_M2, "a_m1": A_M1, "a_m2": A_M2,
           "a_mirror": A_MIRROR, "noise_sigma": l2g.NOISE_SIGMA,
           "frame_sync": FRAME_SYNC, "stagger_d": STAGGER_D,
           "jr_measure": {"subset_denominator": "adopted_seeds_confirmed",
                          "full_report": 1, "threshold": JR_RATIO_MAX},
           "governance": {"verify_purity_min": l2m.VERIFY_PURITY_MIN,
                          "k_verify": l2m.K_VERIFY,
                          "g2r_b_rng_offset": l2m.G2R_B_RNG_OFFSET},
           "symbolization": {"sym_group_min": SYM_GROUP_MIN,
                             "sym_purity_min": SYM_PURITY_MIN,
                             "sym_distinct": SYM_DISTINCT,
                             "dc_cut_win": N_C,
                             "boundary": "purity_argmax"},
           "criteria": {"jr_ratio_max": JR_RATIO_MAX,
                        "adopt_frac_min": ADOPT_FRAC_MIN,
                        "compound_min": COMPOUND_MIN,
                        "transfer_floor": TRANSFER_FLOOR,
                        "transfer_rel": TRANSFER_REL,
                        "c1_stability_min": SYM_STABILITY_MIN,
                        "c1_retest_max": SYM_RETEST_MAX,
                        "c2_pred_ratio_max": SYM_PRED_RATIO_MAX,
                        "c3_compress_min": SYM_COMPRESS_MIN,
                        "c3_eta2_min": SYM_ETA2_MIN,
                        "c4_ref_min": DISCONTEXT_REF_MIN},
           "world": {"a_center": list(l2g.A_CENTER), "a_orbit": l2g.A_ORBIT,
                     "a_freq": l2g.A_FREQ, "b_center": list(l2g.B_CENTER),
                     "b_orbit": l2g.B_ORBIT, "b_freq": l2g.B_FREQ,
                     "rng_lvcode": LV_WORLD},
           "channel": {"sparse_px": l2g.SIG_SPARSE_PX,
                       "null_signal": l2g.NULL_SIGNAL,
                       "ch1_halfwin_overlap": 1, "ch2_halfwin_lag": 1},
           "gate": {"purity_min": GATE_PURITY_MIN},
           "loop": LOOP_CFG}
    ck_tag = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    ckpt_path = os.path.join(args.out_dir, "ckpt_l2n_%s.json" % ck_tag)

    done = {}
    if args.resume and not args.no_resume and os.path.exists(ckpt_path):
        with open(ckpt_path, encoding="utf-8") as f:
            done = json.load(f).get("per_unit", {})

    per_unit = dict(done)
    cell7_per = {}
    worlds_bidi = {s: l2j.make_bidi_world(s, a1=A_M1, a2=A_M2, mirror=A_MIRROR)
                   for s in seeds}
    worlds_uniform = {s: l2g.make_world(s, m1=B_M1, m2=B_M2) for s in seeds}
    wl_b_bidi = {s: l2l.stagger_labels(l2l.per_frame_ctx(s)) for s in seeds}
    wl_b_uniform = {s: l2l.stagger_labels(l2l.per_frame_ctx(s, a_ctx_dep=False))
                    for s in seeds}

    # 错乱信号源：G2s 原源（docs/275 逐字）+ G2s' 真随机族（docs/275 L1）
    bidi_sig = {}
    for s in seeds:
        fa, fb, _ = worlds_bidi[s]
        la = l2g.run_a_signal(fa)
        lb = l2j.run_b_signal_bidi(fb)
        bidi_sig[s] = (list(la.sA_trace), list(lb.sA_trace))

    def rand_sig(seed_rng):
        rng_bits = np.random.default_rng(seed_rng)
        return [131.0 if int(rng_bits.random() < 0.5) else 31.0
                for _ in range(N_FRAMES // WINDOW)]

    g2r_sig1 = {s: rand_sig((s + l2m.G2R_B_RNG_OFFSET) * 99991 + 12345)
                for s in seeds}
    g2r_sig2 = {s: rand_sig(s * 99991 + 12345) for s in seeds}

    # ---- 符号表预计算（stagger5 保真；每臂每侧自己的接收信号/能量） ----
    pre_m = {s: precompute_d(s) for s in seeds}
    pre_n = {s: precompute_d(s, mode="null") for s in seeds}
    pre_u = {s: precompute_d(s, worlds=worlds_uniform[s]) for s in seeds}
    pre_g2 = {}
    pre_g2r = {}
    for s in seeds:
        sigA = bidi_sig[(s + 5) % 10][0]
        rng_bits = np.random.default_rng(s * 99991 + 12345)
        rand_bits = [int(rng_bits.random() < 0.5) for _ in range(len(wl_b_bidi[s]))]
        sig2 = [131.0 if b else 31.0 for b in rand_bits]
        pre_g2[s] = precompute_d(s, sig1=sigA, sig2=sig2, mode="scrambled")
        pre_g2r[s] = precompute_d(s, sig1=g2r_sig1[s], sig2=g2r_sig2[s],
                                  mode="scrambled")
    tables = {}
    for s in seeds:
        tables[s] = {
            "M": (build_symbol_table(pre_m[s][3], pre_m[s][4], pre_m[s][7]),
                  build_symbol_table(pre_m[s][5], pre_m[s][6], pre_m[s][8])),
            "G1N": (build_symbol_table(pre_n[s][3], pre_n[s][4], pre_n[s][7]),
                    build_symbol_table(pre_n[s][5], pre_n[s][6], pre_n[s][8])),
            "G2S": (build_symbol_table(pre_g2[s][3], pre_g2[s][4], pre_g2[s][7]),
                    build_symbol_table(pre_g2[s][5], pre_g2[s][6], pre_g2[s][8])),
            "G2R": (build_symbol_table(pre_g2r[s][3], pre_g2r[s][4], pre_g2r[s][7]),
                    build_symbol_table(pre_g2r[s][5], pre_g2r[s][6], pre_g2r[s][8])),
            "W": (build_symbol_table(pre_u[s][3], pre_u[s][4], pre_u[s][7]),
                  build_symbol_table(pre_u[s][5], pre_u[s][6], pre_u[s][8])),
        }

    def need(arm, s):
        return "%s_%d" % (arm, s) not in per_unit

    # ---- 主测量（符号化 ON） ----
    for s in seeds:
        fa, fb, wl = worlds_bidi[s]
        wlb = wl_b_bidi[s]
        tA, tB = tables[s]["M"]
        if need("MTA", s) or need("MTB", s):
            out_a, loop_a, out_b, loop_b, snap_a, snap_b = run_bidi8(
                fa, fb, wl, wlb, ch1="comm", ch2="comm", two_phase=True,
                n_c=N_C, sym_table_a=tA, sym_table_b=tB, symbolize=True)
            per_unit["MTA_%d" % s] = unit_record8(
                "MTA", s, out_a, loop_a, "A", snap=snap_a,
                jr=l2g.jr_b(loop_a), att=l2g.attribution(loop_a, N_C),
                table=tA, dseq=pre_m[s][3])
            per_unit["MTB_%d" % s] = unit_record8(
                "MTB", s, out_b, loop_b, "B", snap=snap_b,
                jr=l2g.jr_b(loop_b), att=l2g.attribution(loop_b, N_C),
                table=tB, dseq=pre_m[s][5])
            print("PROGRESS", flush=True)
        if need("MGA", s) or need("MGB", s):
            out_a, loop_a, out_b, loop_b, _, _ = run_bidi8(
                fa, fb, wl, wlb, ch1="comm", ch2="comm", two_phase=False,
                sym_table_a=tA, sym_table_b=tB, symbolize=True)
            per_unit["MGA_%d" % s] = unit_record8(
                "MGA", s, out_a, loop_a, "A", table=tA, dseq=pre_m[s][3])
            per_unit["MGB_%d" % s] = unit_record8(
                "MGB", s, out_b, loop_b, "B", table=tB, dseq=pre_m[s][5])
            print("PROGRESS", flush=True)
        if need("CA", s) or need("CB", s):
            out_a, loop_a, out_b, loop_b, snap_a, snap_b = run_bidi8(
                fa[:140], fb[:140], wl[:14], wlb[:14], ch1="comm", ch2="comm",
                two_phase=False, want_end_snap=True,
                sym_table_a=tA, sym_table_b=tB, symbolize=True)
            per_unit["CA_%d" % s] = unit_record8(
                "CA", s, out_a, loop_a, "A", snap=snap_a,
                table=tA, dseq=pre_m[s][3])
            per_unit["CB_%d" % s] = unit_record8(
                "CB", s, out_b, loop_b, "B", snap=snap_b,
                table=tB, dseq=pre_m[s][5])
            print("PROGRESS", flush=True)
        if need("G0AA", s) or need("G0AB", s):
            out_a, loop_a, out_b, loop_b, _, _ = run_bidi8(
                fa, fb, wl, wlb, ch1="off", ch2="off", a_mode="off",
                sym_table_a=(False, None), sym_table_b=(False, None),
                symbolize=True)
            per_unit["G0AA_%d" % s] = unit_record8(
                "G0AA", s, out_a, loop_a, "A", jr=l2g.jr_b(loop_a),
                table=(False, None, 0.0), dseq=pre_m[s][3])
            per_unit["G0AB_%d" % s] = unit_record8(
                "G0AB", s, out_b, loop_b, "B", jr=l2g.jr_b(loop_b),
                table=(False, None, 0.0), dseq=pre_m[s][5])
            print("PROGRESS", flush=True)
        tnA, tnB = tables[s]["G1N"]
        if need("G1NA", s) or need("G1NB", s):
            out_a, loop_a, out_b, loop_b, _, _ = run_bidi8(
                fa, fb, wl, wlb, ch1="null", ch2="null", a_mode="null",
                sym_table_a=tnA, sym_table_b=tnB, symbolize=True)
            per_unit["G1NA_%d" % s] = unit_record8(
                "G1NA", s, out_a, loop_a, "A", table=tnA, dseq=pre_n[s][3])
            per_unit["G1NB_%d" % s] = unit_record8(
                "G1NB", s, out_b, loop_b, "B", table=tnB, dseq=pre_n[s][5])
            print("PROGRESS", flush=True)
        tgA, tgB = tables[s]["G2S"]
        if need("G2SA", s) or need("G2SB", s):
            sigA = bidi_sig[(s + 5) % 10][0]
            rng_bits = np.random.default_rng(s * 99991 + 12345)
            rand_bits = [int(rng_bits.random() < 0.5) for _ in range(len(wl))]
            sig2 = [131.0 if b else 31.0 for b in rand_bits]
            out_a, loop_a, out_b, loop_b, _, _ = run_bidi8(
                fa, fb, wl, wlb, ch1="scrambled", ch2="scrambled",
                a_mode="scrambled",
                sig1_fn=lambda w, sa=sigA: sa[w],
                sig2_fn=lambda w, sv=sig2: sv[w],
                sym_table_a=tgA, sym_table_b=tgB, symbolize=True)
            per_unit["G2SA_%d" % s] = unit_record8(
                "G2SA", s, out_a, loop_a, "A", table=tgA, dseq=pre_g2[s][3])
            per_unit["G2SB_%d" % s] = unit_record8(
                "G2SB", s, out_b, loop_b, "B", table=tgB, dseq=pre_g2[s][5])
            print("PROGRESS", flush=True)
        trA, trB = tables[s]["G2R"]
        if need("G2RA", s) or need("G2RB", s):
            out_a, loop_a, out_b, loop_b, _, _ = run_bidi8(
                fa, fb, wl, wlb, ch1="scrambled", ch2="scrambled",
                a_mode="scrambled",
                sig1_fn=lambda w, sv=g2r_sig1[s]: sv[w],
                sig2_fn=lambda w, sv=g2r_sig2[s]: sv[w],
                sym_table_a=trA, sym_table_b=trB, symbolize=True)
            per_unit["G2RA_%d" % s] = unit_record8(
                "G2RA", s, out_a, loop_a, "A", table=trA, dseq=pre_g2r[s][3])
            per_unit["G2RB_%d" % s] = unit_record8(
                "G2RB", s, out_b, loop_b, "B", table=trB, dseq=pre_g2r[s][5])
            print("PROGRESS", flush=True)
        twA, twB = tables[s]["W"]
        if need("WA", s) or need("WB", s):
            fa_u, fb_u, wl_u = worlds_uniform[s]
            wl_bu = wl_b_uniform[s]
            out_a, loop_a, out_b, loop_b, _, _ = run_bidi8(
                fa_u, fb_u, wl_u, wl_bu, ch1="comm", ch2="comm", a_mode="comm",
                sym_table_a=twA, sym_table_b=twB, symbolize=True)
            per_unit["WA_%d" % s] = unit_record8(
                "WA", s, out_a, loop_a, "A", table=twA, dseq=pre_u[s][3])
            per_unit["WB_%d" % s] = unit_record8(
                "WB", s, out_b, loop_b, "B", table=twB, dseq=pre_u[s][5])
            print("PROGRESS", flush=True)
        if need("DCA", s) or need("DCB", s):
            out_a, loop_a, out_b, loop_b, _, _ = run_bidi8(
                fa, fb, wl, wlb, ch1="comm", ch2="comm", two_phase=True,
                n_c=N_C, sym_table_a=tA, sym_table_b=tB, discontext=True,
                cut_after=N_C, symbolize=True)
            per_unit["DCA_%d" % s] = unit_record8(
                "DCA", s, out_a, loop_a, "A", table=tA, dseq=pre_m[s][3])
            per_unit["DCA_%d" % s]["sym"]["dc_ref"] = dc_ref_rate(loop_a)[0]
            per_unit["DCB_%d" % s] = unit_record8(
                "DCB", s, out_b, loop_b, "B", table=tB, dseq=pre_m[s][5])
            per_unit["DCB_%d" % s]["sym"]["dc_ref"] = dc_ref_rate(loop_b)[0]
            print("PROGRESS", flush=True)
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump({"config": cfg, "per_unit": per_unit},
                      f, ensure_ascii=False, indent=1)

    # ---- CELL7_REPRO 原源（符号化关闭 = docs/275 逐位；16 单元 × 10 种子） ----
    cell7_specs = {
        "MTA": (None, dict(ch1="comm", ch2="comm", two_phase=True, n_c=N_C), "A"),
        "MTB": (None, dict(ch1="comm", ch2="comm", two_phase=True, n_c=N_C), "B"),
        "MGA": (None, dict(ch1="comm", ch2="comm", two_phase=False), "A"),
        "MGB": (None, dict(ch1="comm", ch2="comm", two_phase=False), "B"),
        "CA": ("prefix", dict(ch1="comm", ch2="comm", two_phase=False,
                              want_end_snap=True), "A"),
        "CB": ("prefix", dict(ch1="comm", ch2="comm", two_phase=False,
                              want_end_snap=True), "B"),
        "G0AA": (None, dict(ch1="off", ch2="off", a_mode="off"), "A"),
        "G0AB": (None, dict(ch1="off", ch2="off", a_mode="off"), "B"),
        "G1NA": (None, dict(ch1="null", ch2="null", a_mode="null"), "A"),
        "G1NB": (None, dict(ch1="null", ch2="null", a_mode="null"), "B"),
        "G2SA": ("g2s", dict(ch1="scrambled", ch2="scrambled", a_mode="scrambled"),
                 "A"),
        "G2SB": ("g2s", dict(ch1="scrambled", ch2="scrambled", a_mode="scrambled"),
                 "B"),
        "G2RA": ("g2r", dict(ch1="scrambled", ch2="scrambled", a_mode="scrambled"),
                 "A"),
        "G2RB": ("g2r", dict(ch1="scrambled", ch2="scrambled", a_mode="scrambled"),
                 "B"),
        "WA": ("uniform", dict(ch1="comm", ch2="comm", a_mode="comm"), "A"),
        "WB": ("uniform", dict(ch1="comm", ch2="comm", a_mode="comm"), "B"),
    }
    for s in seeds:
        fa, fb, wl = worlds_bidi[s]
        wlb = wl_b_bidi[s]
        fa_u, fb_u, wl_u = worlds_uniform[s]
        wlb_u = wl_b_uniform[s]
        sigA = bidi_sig[(s + 5) % 10][0]
        rng_bits = np.random.default_rng(s * 99991 + 12345)
        rand_bits = [int(rng_bits.random() < 0.5) for _ in range(len(wl))]
        sig2 = [131.0 if b else 31.0 for b in rand_bits]
        for arm, (kind, kw, side) in cell7_specs.items():
            if "%s_%d" % (arm, s) in cell7_per:
                continue
            if kind == "prefix":
                fa_, fb_, wl_, wlb_ = fa[:140], fb[:140], wl[:14], wlb[:14]
            elif kind == "uniform":
                fa_, fb_, wl_, wlb_ = fa_u, fb_u, wl_u, wlb_u
            elif kind == "g2s":
                fa_, fb_, wl_, wlb_ = fa, fb, wl, wlb
                # docs/275 G2s 原源：A 收随机二元注入 sig2、B 收 (s+5)%10 的 s_A sigA
                kw = dict(kw, sig1_fn=(lambda w, sa=sigA: sa[w]),
                          sig2_fn=(lambda w, sv=sig2: sv[w]))
            elif kind == "g2r":
                fa_, fb_, wl_, wlb_ = fa, fb, wl, wlb
                kw = dict(kw, sig1_fn=(lambda w, sv=g2r_sig1[s]: sv[w]),
                          sig2_fn=(lambda w, sv=g2r_sig2[s]: sv[w]))
            else:
                fa_, fb_, wl_, wlb_ = fa, fb, wl, wlb
            out_a, loop_a, out_b, loop_b, snap_a, snap_b = run_bidi8(
                fa_, fb_, wl_, wlb_, symbolize=False, **kw)
            if side == "A":
                rec = l2m.unit_record7(
                    arm, s, out_a, loop_a, "A",
                    snap=(snap_a if arm in ("MTA", "CA") else None),
                    jr=(l2g.jr_b(loop_a) if arm in ("MTA", "G0AA") else None),
                    att=(l2g.attribution(loop_a, N_C) if arm == "MTA" else None))
            else:
                rec = l2m.unit_record7(
                    arm, s, out_b, loop_b, "B",
                    snap=(snap_b if arm in ("MTB", "CB") else None),
                    jr=(l2g.jr_b(loop_b) if arm in ("MTB", "G0AB") else None),
                    att=(l2g.attribution(loop_b, N_C) if arm == "MTB" else None))
            cell7_per["%s_%d" % (arm, s)] = rec
        print("PROGRESS", flush=True)

    # ---- 守卫 ----
    g232_ok, g232 = l2g.guard_d232()
    g235_ok, g235 = l2g.guard_d235()
    c2_ok, c2_detail = l2i.repro_cell2()
    c5a_ok, c5a_detail = l2k.repro_cell4()      # CELL5_REPRO-A：a_first ≡ docs/271
    c5b_ok, c5b_detail = l2l.repro_cell5_b()    # CELL5_REPRO-B：b_first ≡ docs/273
    c6_ok, c6_detail = l2m.repro_cell6(cell7_per, seeds)   # CELL6_REPRO：原源 ≡ docs/274
    c7_ok, c7_detail = cell7_repro(cell7_per, seeds)       # CELL7_REPRO：原源 ≡ docs/275
    world_ok, world_oks = l2j.world_eq()
    pre_ok, pre_detail = l2l.precompute_ok_main()

    # ---- 跨单元核对（双侧，stagger5，符号化主测量） ----
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
        g2ra = per_unit["G2RA_%d" % s]
        g2rb = per_unit["G2RB_%d" % s]
        wa = per_unit["WA_%d" % s]
        wb = per_unit["WB_%d" % s]
        dca = per_unit["DCA_%d" % s]
        dcb = per_unit["DCB_%d" % s]
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
            g2ra=g2ra, g2rb=g2rb, wa=wa, wb=wb, dca=dca, dcb=dcb,
            jr_a0=g0a["jr"][0], jr_b0=g0b["jr"][0], jr_a1=ta["jr"][0],
            jr_b1=tb["jr"][0],
            jr_ratio_a=ta["jr"][0] / max(g0a["jr"][0], 1e-12),
            jr_ratio_b=tb["jr"][0] / max(g0b["jr"][0], 1e-12),
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

    # ---- 聚合（双侧；M 臂主数字 = M-G 单阶段流，§1.6） ----
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
    jr_ratio_as = [r["jr_ratio_a"] for r in seed_rows]
    jr_ratio_bs = [r["jr_ratio_b"] for r in seed_rows]
    jr_ratio_a_m, jr_ratio_a_sd = mean_sd(jr_ratio_as)
    jr_ratio_b_m, jr_ratio_b_sd = mean_sd(jr_ratio_bs)
    transfer_a_m, transfer_a_sd = mean_sd([r["transfer_a"] for r in seed_rows])
    transfer_b_m, transfer_b_sd = mean_sd([r["transfer_b"] for r in seed_rows])
    calib_a_m, calib_a_sd = mean_sd([r["calib_a"] for r in seed_rows])
    calib_b_m, calib_b_sd = mean_sd([r["calib_b"] for r in seed_rows])

    # ---- 治理读数（docs/275 确认制 + JR 采纳子集口径） ----
    def nconf(rec):
        return rec["verify"]["n_confirmed"]

    adopt_conf_a = float(np.mean([nconf(r["ta"]) >= 1 for r in seed_rows]))
    adopt_conf_b = float(np.mean([nconf(r["tb"]) >= 1 for r in seed_rows]))
    adopted_a = [r for r in seed_rows if nconf(r["ta"]) >= 1]
    adopted_b = [r for r in seed_rows if nconf(r["tb"]) >= 1]
    comp_adopted_a = float(np.mean([r["ta"]["finalize"]["compound_frac"]
                                    for r in adopted_a])) if adopted_a else 0.0
    comp_adopted_b = float(np.mean([r["tb"]["finalize"]["compound_frac"]
                                    for r in adopted_b])) if adopted_b else 0.0
    jr_subset_a = (float(np.mean([r["jr_ratio_a"] for r in adopted_a]))
                   if adopted_a else 0.0)
    jr_subset_b = (float(np.mean([r["jr_ratio_b"] for r in adopted_b]))
                   if adopted_b else 0.0)
    spurious_g2s_a = [r["seed"] for r in seed_rows if nconf(r["g2a"]) >= 1]
    spurious_g2s_b = [r["seed"] for r in seed_rows if nconf(r["g2b"]) >= 1]
    spurious_g2r_a = [r["seed"] for r in seed_rows if nconf(r["g2ra"]) >= 1]
    spurious_g2r_b = [r["seed"] for r in seed_rows if nconf(r["g2rb"]) >= 1]
    fid_a_m = float(np.mean(col("A", "ctx_fidelity")))
    fid_b_m = float(np.mean(col("B", "ctx_fidelity")))
    adopt_conf_a_w = float(np.mean([nconf(r["wa"]) >= 1 for r in seed_rows]))
    adopt_conf_b_w = float(np.mean([nconf(r["wb"]) >= 1 for r in seed_rows]))

    # ---- 符号化判据（docs/276 §1.2 冻结；真值只进评估） ----
    def truth_side(s, side):
        if side == "A":
            return [1 - lb["ctx"] for lb in worlds_bidi[s][2]]
        return [lb["ctx"] for lb in wl_b_bidi[s]]

    cons_a, cons_b = [], []
    retests = []
    pred_a, pred_b = [], []
    compr_a, compr_b = [], []
    eta_a, eta_b = [], []
    dc_ref_a, dc_ref_b = [], []
    dc_sc2_a, dc_sc2_b = [], []
    sym_act_a, sym_act_b = [], []
    for r in seed_rows:
        s = r["seed"]
        tA, tB = tables[s]["M"]
        # C1a 同情境一致率（采纳种子）
        cA, _ = consistency_rate(r["ta"]["sym"]["c2seq"], truth_side(s, "A"))
        cB, _ = consistency_rate(r["tb"]["sym"]["c2seq"], truth_side(s, "B"))
        if nconf(r["ta"]) >= 1:
            cons_a.append(cA)
        if nconf(r["tb"]) >= 1:
            cons_b.append(cB)
        # C1b 边界重测（双侧）
        ra_ = boundary_retest(r["ta"]["sym"]["dseq"])
        rb_ = boundary_retest(r["tb"]["sym"]["dseq"])
        if ra_ is not None:
            retests.append(ra_)
        if rb_ is not None:
            retests.append(rb_)
        # C2 熵比（采纳种子）
        ea_ = entropy_ratio(r["ta"]["sym"]["c2seq"], r["ta"]["sym"]["dseq"])
        eb_ = entropy_ratio(r["tb"]["sym"]["c2seq"], r["tb"]["sym"]["dseq"])
        if nconf(r["ta"]) >= 1 and ea_ is not None:
            pred_a.append(ea_)
        if nconf(r["tb"]) >= 1 and eb_ is not None:
            pred_b.append(eb_)
        # C3 压缩（激活种子）
        if tA[0]:
            sym_act_a.append(s)
            nd = len(set(round(v, 3) for v in r["ta"]["sym"]["dseq"]
                         if v is not None))
            compr_a.append(nd / 2.0)
            eta_a.append(eta2_sym(r["ta"]["sym"]["dseq"], pre_m[s][4], tA[1]))
        if tB[0]:
            sym_act_b.append(s)
            nd = len(set(round(v, 3) for v in r["tb"]["sym"]["dseq"]
                         if v is not None))
            compr_b.append(nd / 2.0)
            eta_b.append(eta2_sym(r["tb"]["sym"]["dseq"], pre_m[s][6], tB[1]))
        # C4 DC 引用率（M 臂采纳种子——DC 臂是同一学习产物在信号切断下的引用测试；
        # DC 臂自身的 L2 验证会因寄存器常量符号使另一子条目零命中而拦（诊断），
        # 引用率是直接测量，与确认计数无关）
        if nconf(r["ta"]) >= 1:
            dc_ref_a.append(r["dca"]["sym"].get("dc_ref", 0.0))
        if nconf(r["tb"]) >= 1:
            dc_ref_b.append(r["dcb"]["sym"].get("dc_ref", 0.0))
        dc_sc2_a.append(r["dca"]["finalize"]["sc2"])
        dc_sc2_b.append(r["dcb"]["finalize"]["sc2"])
    c1a_ok = int((np.mean(cons_a) if cons_a else 0.0) >= SYM_STABILITY_MIN
                 and (np.mean(cons_b) if cons_b else 0.0) >= SYM_STABILITY_MIN)
    c1b_ok = int((max(retests) if retests else 1.0) <= SYM_RETEST_MAX)
    c2_ok = int((np.mean(pred_a) if pred_a else 0.0) <= SYM_PRED_RATIO_MAX
                and (np.mean(pred_b) if pred_b else 0.0) <= SYM_PRED_RATIO_MAX)
    c3a_ok = int((np.mean(compr_a) if compr_a else 0.0) >= SYM_COMPRESS_MIN
                 and (np.mean(compr_b) if compr_b else 0.0) >= SYM_COMPRESS_MIN)
    c3b_ok = int((np.mean(eta_a) if eta_a else 0.0) >= SYM_ETA2_MIN
                 and (np.mean(eta_b) if eta_b else 0.0) >= SYM_ETA2_MIN)
    c4a_ok = int((np.mean(dc_ref_a) if dc_ref_a else 0.0) >= DISCONTEXT_REF_MIN
                 and (np.mean(dc_ref_b) if dc_ref_b else 0.0) >= DISCONTEXT_REF_MIN)
    c4b_ok = int(min(dc_sc2_a) >= 1 and min(dc_sc2_b) >= 1)
    c1 = int(c1a_ok == 1 and c1b_ok == 1)
    c2 = int(c2_ok == 1)
    c3 = int(c3a_ok == 1 and c3b_ok == 1)
    c4 = int(c4a_ok == 1 and c4b_ok == 1)
    # C5 保持（docs/275 六条）
    c5_1 = int(adopt_conf_a >= ADOPT_FRAC_MIN and adopt_conf_b >= ADOPT_FRAC_MIN)
    c5_2 = int(jr_subset_a <= JR_RATIO_MAX and jr_subset_b <= JR_RATIO_MAX)
    c5_3 = int(len(spurious_g2r_a) == 0 and len(spurious_g2r_b) == 0)
    c5_4 = construction_ok
    c5_5 = int(transfer_a_m >= TRANSFER_FLOOR
               and transfer_a_m >= TRANSFER_REL * calib_a_m
               and transfer_b_m >= TRANSFER_FLOOR
               and transfer_b_m >= TRANSFER_REL * calib_b_m)
    c4a_struct_a = int(agg_g0["A"]["sc2_mean"] >= 1
                       and agg_g0["A"]["churn_mean"] < 0.3
                       and agg_g0["A"]["ratio_mean"] <= 1.5
                       and agg["A"]["sc2_mean"] >= 1
                       and agg["A"]["churn_mean"] < 0.3
                       and agg["A"]["ratio_mean"] <= 1.5)
    c4a_struct_b = int(agg_g0["B"]["sc2_mean"] >= 1
                       and agg_g0["B"]["churn_mean"] < 0.3
                       and agg_g0["B"]["ratio_mean"] <= 1.5
                       and agg["B"]["sc2_mean"] >= 1
                       and agg["B"]["churn_mean"] < 0.3
                       and agg["B"]["ratio_mean"] <= 1.5)
    c5_6 = int(c4a_struct_a == 1 and c4a_struct_b == 1)
    c5 = int(c5_1 == 1 and c5_2 == 1 and c5_3 == 1 and c5_4 == 1
             and c5_5 == 1 and c5_6 == 1)

    # 数据可用性（F2_BLOCKED 预防）：G0A/M 双侧逐种子 JR 有窗口（合成世界/回路运行
    # 成功）；采纳种子留出 eligible 非空（判据测量良定义——非采纳种子 ctx=None 导致
    # 无 eligible 窗是符号化设计的诚实形态（无符号不采纳），非数据不可用）
    blocked = int(not (all(r["g0a"]["jr"][1] >= 1 for r in seed_rows)
                       and all(r["g0b"]["jr"][1] >= 1 for r in seed_rows)
                       and all(r["ta"]["jr"][1] >= 1 for r in seed_rows)
                       and all(r["tb"]["jr"][1] >= 1 for r in seed_rows)
                       and (len(adopted_a) == 0
                            or all(r["held_elig_a"] >= 1 for r in adopted_a))
                       and (len(adopted_b) == 0
                            or all(r["held_elig_b"] >= 1 for r in adopted_b))))

    guards_ok = (g232_ok == 1 and g235_ok == 1 and construction_ok == 1
                 and prefix_ok_a == 1 and prefix_ok_b == 1
                 and two_phase_ok_a == 1 and two_phase_ok_b == 1
                 and repro_ok_a == 1 and repro_ok_b == 1
                 and c2_ok == 1 and c5a_ok == 1 and c5b_ok == 1
                 and c6_ok == 1 and c7_ok == 1 and world_ok == 1 and pre_ok == 1)
    if not guards_ok:
        verdict = "GUARD_FAIL"
        vnote = ("guard(s) failed: D232=%d, D235=%d, CONSTRUCTION=%d, "
                 "PREFIX_EQ_A=%d, PREFIX_EQ_B=%d, TWO_PHASE_EQ_A=%d, "
                 "TWO_PHASE_EQ_B=%d, REPRO_MAE_A=%d, REPRO_MAE_B=%d, "
                 "CELL2_REPRO=%d, CELL5_REPRO_A=%d, CELL5_REPRO_B=%d, "
                 "CELL6_REPRO=%d, CELL7_REPRO=%d, WORLD_EQ=%d, PRECOMPUTE=%d -> "
                 "implementation drift; fix implementation, do not judge "
                 "mechanism" % (g232_ok, g235_ok, construction_ok,
                                prefix_ok_a, prefix_ok_b, two_phase_ok_a,
                                two_phase_ok_b, repro_ok_a, repro_ok_b,
                                c2_ok, c5a_ok, c5b_ok, c6_ok, c7_ok,
                                world_ok, pre_ok))
    elif blocked:
        verdict = "F2_BLOCKED"
        vnote = ("synthetic environment unavailable (per-seed eligible/JR "
                 "windows missing on A/B sides); see per-seed numbers")
    elif c1 and c2 and c3 and c4 and c5:
        verdict = "SYMBOL_EMERGES"
        vnote = ("criteria C1-C5 all pass and all guards pass: the loop's own "
                 "signal/energy statistics yield stable, predictable, "
                 "compressed, discontext-referenceable symbols (C1 stability "
                 "%.3f/%.3f, C2 pred ratio %.3f/%.3f, C3 compression %.1f/%.1f "
                 "+ eta2 %.3f/%.3f, C4 DC ref %.3f/%.3f) and the docs/275 "
                 "communication base is kept (adopt %.2f/%.2f, JR subset "
                 "%.4f/%.4f, spurious(G2R) 0/0, construction 0/0, holdout "
                 "%.3f/%.3f) -> symbolization emerged as a pure representation-"
                 "layer addition (CELL7_REPRO=1)" % (
                     np.mean(cons_a), np.mean(cons_b),
                     np.mean(pred_a), np.mean(pred_b),
                     np.mean(compr_a), np.mean(compr_b),
                     np.mean(eta_a), np.mean(eta_b),
                     np.mean(dc_ref_a), np.mean(dc_ref_b),
                     adopt_conf_a, adopt_conf_b, jr_subset_a, jr_subset_b,
                     transfer_a_m, transfer_b_m))
    elif not c5:
        why5 = []
        if not c5_1:
            why5.append("adopt A=%.2f/B=%.2f < 0.6" % (adopt_conf_a, adopt_conf_b))
        if not c5_2:
            why5.append("JR subset A=%.4f/B=%.4f > 0.85"
                        % (jr_subset_a, jr_subset_b))
        if not c5_3:
            why5.append("spurious(G2R) A=%d/B=%d > 0"
                        % (len(spurious_g2r_a), len(spurious_g2r_b)))
        if not c5_4:
            why5.append("construction G0A/G1n nonzero")
        if not c5_5:
            why5.append("holdout transfer A=%.3f/B=%.3f fails dual gate"
                        % (transfer_a_m, transfer_b_m))
        if not c5_6:
            why5.append("structure (SC2/churn/ratio) fails")
        verdict = "PARTIAL"
        vnote = ("symbolization criteria pass but C5 (base keep) fails: %s; "
                 "symbolization emerged but broke the communication base -> "
                 "honest report, no rollback" % "; ".join(why5))
    elif not (c1 and c2 and c3 and c4):
        why = []
        if not c1:
            why.append("C1 stability fails (consistency %.3f/%.3f, retest max %.3f)"
                       % (np.mean(cons_a), np.mean(cons_b), (max(retests) if retests else 0)))
        if not c2:
            why.append("C2 predictability fails (pred ratio %.3f/%.3f > %.2f)"
                       % (np.mean(pred_a), np.mean(pred_b), SYM_PRED_RATIO_MAX))
        if not c3:
            why.append("C3 compression fails (compr %.1f/%.1f, eta2 %.3f/%.3f)"
                       % (np.mean(compr_a), np.mean(compr_b),
                          np.mean(eta_a), np.mean(eta_b)))
        if not c4:
            why.append("C4 discontext fails (DC ref %.3f/%.3f, SC2 %d/%d)"
                       % (np.mean(dc_ref_a), np.mean(dc_ref_b),
                          min(dc_sc2_a), min(dc_sc2_b)))
        verdict = "PARTIAL"
        vnote = ("; ".join(why) + " (C5 kept but symbolization criteria "
                 "partially failed; see R_L2N_CRIT* numbers)")
    else:
        verdict = "SYMBOL_FLAT"
        vnote = ("no symbols emerged (activation conditions not met: "
                 "signal-degenerate or energy-not-splittable); activation "
                 "counts A=%d/10, B=%d/10" % (len(sym_act_a), len(sym_act_b)))

    # ---- 工件（自描述 JSON） ----
    out = {
        "artifact": "lang_comm_test8",
        "doc_ref": "docs/63, docs/228, docs/231, docs/232, docs/235, docs/240, "
                   "docs/247, docs/252, docs/255, docs/258, docs/264, docs/266, "
                   "docs/268, docs/269, docs/270, docs/271, docs/273, docs/274, "
                   "docs/275, docs/276",
        "config": cfg,
        "guards": {"d232": {"ok": g232_ok, "detail": g232},
                   "d235": {"ok": g235_ok, "detail": g235},
                   "construction": {"ok": construction_ok},
                   "prefix_eq_a": prefix_ok_a, "prefix_eq_b": prefix_ok_b,
                   "two_phase_eq_a": two_phase_ok_a,
                   "two_phase_eq_b": two_phase_ok_b,
                   "repro_mae_a": repro_ok_a, "repro_mae_b": repro_ok_b,
                   "cell2_repro": {"ok": c2_ok},
                   "cell5_repro_a": {"ok": c5a_ok},
                   "cell5_repro_b": {"ok": c5b_ok},
                   "cell6_repro": {"ok": c6_ok, "detail": c6_detail},
                   "cell7_repro": {"ok": c7_ok, "detail": c7_detail},
                   "world_eq": {"ok": world_ok, "per_seed": world_oks},
                   "precompute": pre_detail},
        "per_seed": seed_rows,
        "arms": {k: {"mean_sd": agg[k], "mae_ci95": agg[k]["mae_ci95"],
                     "comp_ci95": agg[k]["comp_ci95"]}
                 for k in ("A", "B")},
        "arms_g0a": agg_g0,
        "jr": {"ratio_a_full": jr_ratio_a_m, "ratio_a_full_sd": jr_ratio_a_sd,
               "ratio_b_full": jr_ratio_b_m, "ratio_b_full_sd": jr_ratio_b_sd,
               "ratio_a_subset": jr_subset_a, "ratio_b_subset": jr_subset_b,
               "adopted_seeds_a": [r["seed"] for r in adopted_a],
               "adopted_seeds_b": [r["seed"] for r in adopted_b]},
        "adoption": {"adopt_a_confirmed": adopt_conf_a,
                     "adopt_b_confirmed": adopt_conf_b,
                     "comp_adopted_a": comp_adopted_a,
                     "comp_adopted_b": comp_adopted_b,
                     "adopt_a_w_confirmed": adopt_conf_a_w,
                     "adopt_b_w_confirmed": adopt_conf_b_w},
        "governance": {"spurious_g2s_orig_a": spurious_g2s_a,
                       "spurious_g2s_orig_b": spurious_g2s_b,
                       "spurious_g2r_a": spurious_g2r_a,
                       "spurious_g2r_b": spurious_g2r_b},
        "holdout": {"transfer_a_mean": transfer_a_m,
                    "transfer_a_sd": transfer_a_sd,
                    "calib_a_mean": calib_a_m,
                    "transfer_b_mean": transfer_b_m,
                    "transfer_b_sd": transfer_b_sd,
                    "calib_b_mean": calib_b_m},
        "diag": {"fid_a_mean": fid_a_m, "fid_b_mean": fid_b_m,
                 "design_win_a_frac": float(np.mean([l2j.design_window_a(
                     l2j.bidi_diag([lb["ctx"] for lb in worlds_bidi[s][2]],
                                   per_unit["MGA_%d" % s]["E"])) for s in seeds])),
                 "design_win_b_frac": float(np.mean([l2j.design_window_b(
                     l2j.bidi_diag([lb["ctx"] for lb in wl_b_bidi[s]],
                                   per_unit["MGB_%d" % s]["E"])) for s in seeds]))},
        "symbolization": {
            "tables_m_a": [tables[s]["M"][0] for s in seeds],
            "tables_m_b": [tables[s]["M"][1] for s in seeds],
            "activated_a": sym_act_a, "activated_b": sym_act_b,
            "consistency_a": cons_a, "consistency_b": cons_b,
            "retests": retests,
            "pred_ratio_a": pred_a, "pred_ratio_b": pred_b,
            "compress_a": compr_a, "compress_b": compr_b,
            "eta2_a": eta_a, "eta2_b": eta_b,
            "dc_ref_a": dc_ref_a, "dc_ref_b": dc_ref_b,
            "dc_sc2_a": dc_sc2_a, "dc_sc2_b": dc_sc2_b},
        "criteria": {"c1": c1, "c1a": c1a_ok, "c1b": c1b_ok,
                     "c2": c2, "c3": c3, "c3a": c3a_ok, "c3b": c3b_ok,
                     "c4": c4, "c4a": c4a_ok, "c4b": c4b_ok,
                     "c5": c5, "c5_1": c5_1, "c5_2": c5_2, "c5_3": c5_3,
                     "c5_4": c5_4, "c5_5": c5_5, "c5_6": c5_6,
                     "c1a_consistency_a": (float(np.mean(cons_a)) if cons_a else 0.0),
                     "c1a_consistency_b": (float(np.mean(cons_b)) if cons_b else 0.0),
                     "c1b_retest_max": (float(max(retests)) if retests else 0.0),
                     "c2_pred_ratio_a": (float(np.mean(pred_a)) if pred_a else 0.0),
                     "c2_pred_ratio_b": (float(np.mean(pred_b)) if pred_b else 0.0),
                     "c3_compress_a": (float(np.mean(compr_a)) if compr_a else 0.0),
                     "c3_compress_b": (float(np.mean(compr_b)) if compr_b else 0.0),
                     "c3_eta2_a": (float(np.mean(eta_a)) if eta_a else 0.0),
                     "c3_eta2_b": (float(np.mean(eta_b)) if eta_b else 0.0),
                     "c4_ref_a": (float(np.mean(dc_ref_a)) if dc_ref_a else 0.0),
                     "c4_ref_b": (float(np.mean(dc_ref_b)) if dc_ref_b else 0.0),
                     "jr_ratio_a_subset": jr_subset_a,
                     "jr_ratio_b_subset": jr_subset_b,
                     "jr_ratio_a_full": jr_ratio_a_m,
                     "jr_ratio_b_full": jr_ratio_b_m,
                     "adopt_a_confirmed": adopt_conf_a,
                     "adopt_b_confirmed": adopt_conf_b,
                     "transfer_a": transfer_a_m, "calib_a": calib_a_m,
                     "transfer_b": transfer_b_m, "calib_b": calib_b_m,
                     "spurious_g2r_a_seeds": spurious_g2r_a,
                     "spurious_g2r_b_seeds": spurious_g2r_b},
        "verdict": {"verdict": verdict, "note": vnote},
        "timing": {"elapsed_sec": round(time.time() - t0, 2)},
    }
    res_path = os.path.join(args.out_dir, "l2n_%s.json" % args.tag)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 摘要块：ASCII 标签，每行一个数字（顺序固定；无中文/日志/JSON） ----
    print("R_L2N_TAG=%s" % args.tag)
    print("R_L2N_SEEDS=%d" % len(seeds))
    print("R_L2N_FRAMES=%d" % N_FRAMES)
    print("R_L2N_WINDOWS=%d" % (N_FRAMES // WINDOW))
    print("R_L2N_NC=%d" % N_C)
    print("R_L2N_FRAME_SYNC=%s" % FRAME_SYNC)
    print("R_L2N_STAGGER_D=%d" % STAGGER_D)
    print("R_L2N_M1=%.1f" % B_M1)
    print("R_L2N_M2=%.1f" % B_M2)
    print("R_L2N_A1=%.1f" % A_M1)
    print("R_L2N_A2=%.1f" % A_M2)
    print("R_L2N_GATE_PURITY_MIN=%.2f" % GATE_PURITY_MIN)
    print("R_L2N_SYM_GROUP_MIN=%d" % SYM_GROUP_MIN)
    print("R_L2N_SYM_PURITY_MIN=%.2f" % SYM_PURITY_MIN)
    print("R_L2N_SYM_DISTINCT=%d" % SYM_DISTINCT)
    print("R_L2N_GUARD_D232=%d" % g232_ok)
    print("R_L2N_GUARD_D235=%d" % g235_ok)
    print("R_L2N_CONSTRUCTION=%d" % construction_ok)
    print("R_L2N_CONSTRUCTION_G0A=%d" % int(sum(cons_g0a)))
    print("R_L2N_CONSTRUCTION_G0B=%d" % int(sum(cons_g0b)))
    print("R_L2N_CONSTRUCTION_G1NA=%d" % int(sum(cons_g1na)))
    print("R_L2N_CONSTRUCTION_G1NB=%d" % int(sum(cons_g1nb)))
    print("R_L2N_PREFIX_EQ_A=%d" % prefix_ok_a)
    print("R_L2N_PREFIX_EQ_B=%d" % prefix_ok_b)
    print("R_L2N_TWO_PHASE_EQ_A=%d" % two_phase_ok_a)
    print("R_L2N_TWO_PHASE_EQ_B=%d" % two_phase_ok_b)
    print("R_L2N_REPRO_MAE_A=%d" % repro_ok_a)
    print("R_L2N_REPRO_MAE_B=%d" % repro_ok_b)
    print("R_L2N_CELL2_REPRO=%d" % c2_ok)
    print("R_L2N_CELL5_REPRO_A=%d" % c5a_ok)
    print("R_L2N_CELL5_REPRO_B=%d" % c5b_ok)
    print("R_L2N_CELL6_REPRO=%d" % c6_ok)
    print("R_L2N_CELL7_REPRO=%d" % c7_ok)
    print("R_L2N_CELL7_ADOPT_A=%.4f" % c7_detail["adopt_a"])
    print("R_L2N_CELL7_ADOPT_B=%.4f" % c7_detail["adopt_b"])
    print("R_L2N_CELL7_JR_SUBSET_A=%.6f" % c7_detail["jr_sub_a"])
    print("R_L2N_CELL7_JR_SUBSET_B=%.6f" % c7_detail["jr_sub_b"])
    print("R_L2N_CELL7_JR_FULL_A=%.6f" % c7_detail["jr_full_a"])
    print("R_L2N_CELL7_JR_FULL_B=%.6f" % c7_detail["jr_full_b"])
    print("R_L2N_CELL7_FID_A=%.4f" % c7_detail["fid_a"])
    print("R_L2N_CELL7_FID_B=%.4f" % c7_detail["fid_b"])
    print("R_L2N_CELL7_TRANSFER_A=%.4f" % c7_detail["transfer_a"])
    print("R_L2N_CELL7_TRANSFER_B=%.4f" % c7_detail["transfer_b"])
    print("R_L2N_CELL7_SPURIOUS_G2R_A=%d" % len(c7_detail["spurious_a"]))
    print("R_L2N_CELL7_SPURIOUS_G2R_B=%d" % len(c7_detail["spurious_b"]))
    print("R_L2N_CELL7_W_ADOPT_A=%.4f" % c7_detail["w_a"])
    print("R_L2N_CELL7_W_ADOPT_B=%.4f" % c7_detail["w_b"])
    print("R_L2N_WORLD_EQ=%d" % world_ok)
    print("R_L2N_WORLD_EQ_SEEDS=%d" % int(sum(world_oks)))
    print("R_L2N_PRECOMPUTE=%d" % pre_ok)
    print("R_L2N_PRECOMPUTE_DW_A=%d" % pre_detail["dw_a"])
    print("R_L2N_PRECOMPUTE_DW_B=%d" % pre_detail["dw_b"])
    for s in seeds:
        print("R_L2N_SEED=%d" % s)
        for side, key in (("A", "ta"), ("B", "tb")):
            mf = seed_rows[s][key]["finalize"]
            print("R_L2N_S%d_%s_M_MAE=%.6f" % (s, side, mf["mae_mean"]))
            print("R_L2N_S%d_%s_M_SC2=%d" % (s, side, mf["sc2"]))
            print("R_L2N_S%d_%s_M_COMP=%.4f" % (s, side, mf["compound_frac"]))
            print("R_L2N_S%d_%s_M_CHURN=%.4f" % (s, side, mf["churn_frac"]))
            print("R_L2N_S%d_%s_M_PROMO=%d" % (s, side, mf["n_promo"]))
            print("R_L2N_S%d_%s_M_FID=%.4f" % (s, side, mf["ctx_fidelity"]))
            print("R_L2N_S%d_%s_M_NCONF=%d" % (s, side,
                                                seed_rows[s][key]["verify"]["n_confirmed"]))
        tA, tB = tables[s]["M"]
        print("R_L2N_S%d_SYM_A_ACT=%d_TAU=%s_PUR=%.4f" % (
            s, int(tA[0]),
            ("NA" if tA[1] is None else "%.3f" % tA[1]), tA[2]))
        print("R_L2N_S%d_SYM_B_ACT=%d_TAU=%s_PUR=%.4f" % (
            s, int(tB[0]),
            ("NA" if tB[1] is None else "%.3f" % tB[1]), tB[2]))
        print("R_L2N_S%d_JR_RATIO_A=%.6f" % (s, seed_rows[s]["jr_ratio_a"]))
        print("R_L2N_S%d_JR_RATIO_B=%.6f" % (s, seed_rows[s]["jr_ratio_b"]))
        print("R_L2N_S%d_TRANSFER_A=%.6f" % (s, seed_rows[s]["transfer_a"]))
        print("R_L2N_S%d_CALIB_A=%.6f" % (s, seed_rows[s]["calib_a"]))
        print("R_L2N_S%d_TRANSFER_B=%.6f" % (s, seed_rows[s]["transfer_b"]))
        print("R_L2N_S%d_CALIB_B=%.6f" % (s, seed_rows[s]["calib_b"]))
        print("R_L2N_S%d_FIRST_PROMO_A=%s" % (
            s, ("NA" if seed_rows[s]["first_promo_a"] is None
                else str(seed_rows[s]["first_promo_a"]))))
        print("R_L2N_S%d_FIRST_PROMO_B=%s" % (
            s, ("NA" if seed_rows[s]["first_promo_b"] is None
                else str(seed_rows[s]["first_promo_b"]))))
        if seed_rows[s]["gate_a"] is None:
            print("R_L2N_S%d_GATE_PURITY_A=NA" % s)
        else:
            print("R_L2N_S%d_GATE_PURITY_A=%.4f" % (s, seed_rows[s]["gate_a"]["purity"]))
        if seed_rows[s]["gate_b"] is None:
            print("R_L2N_S%d_GATE_PURITY_B=NA" % s)
        else:
            print("R_L2N_S%d_GATE_PURITY_B=%.4f" % (s, seed_rows[s]["gate_b"]["purity"]))
    print("R_L2N_MAE_A=%.6f" % agg["A"]["mae_mean"])
    print("R_L2N_MAE_B=%.6f" % agg["B"]["mae_mean"])
    print("R_L2N_SC2_A=%.4f" % agg["A"]["sc2_mean"])
    print("R_L2N_SC2_B=%.4f" % agg["B"]["sc2_mean"])
    print("R_L2N_COMP_A=%.4f" % agg["A"]["comp_mean"])
    print("R_L2N_COMP_B=%.4f" % agg["B"]["comp_mean"])
    print("R_L2N_CHURN_A=%.4f" % agg["A"]["churn_mean"])
    print("R_L2N_CHURN_B=%.4f" % agg["B"]["churn_mean"])
    print("R_L2N_FID_A=%.4f" % fid_a_m)
    print("R_L2N_FID_B=%.4f" % fid_b_m)
    print("R_L2N_JR_RATIO_A_FULL=%.6f" % jr_ratio_a_m)
    print("R_L2N_JR_RATIO_B_FULL=%.6f" % jr_ratio_b_m)
    print("R_L2N_JR_RATIO_A_SUBSET=%.6f" % jr_subset_a)
    print("R_L2N_JR_RATIO_B_SUBSET=%.6f" % jr_subset_b)
    print("R_L2N_ADOPT_A_CONF=%.4f" % adopt_conf_a)
    print("R_L2N_ADOPT_B_CONF=%.4f" % adopt_conf_b)
    print("R_L2N_COMP_ADOPTED_A=%.4f" % comp_adopted_a)
    print("R_L2N_COMP_ADOPTED_B=%.4f" % comp_adopted_b)
    print("R_L2N_ADOPT_A_W_CONF=%.4f" % adopt_conf_a_w)
    print("R_L2N_ADOPT_B_W_CONF=%.4f" % adopt_conf_b_w)
    print("R_L2N_TRANSFER_A_MEAN=%.6f" % transfer_a_m)
    print("R_L2N_CALIB_A_MEAN=%.6f" % calib_a_m)
    print("R_L2N_TRANSFER_B_MEAN=%.6f" % transfer_b_m)
    print("R_L2N_CALIB_B_MEAN=%.6f" % calib_b_m)
    print("R_L2N_CRIT_C1=%d" % c1)
    print("R_L2N_CRIT_C1A_CONS_A=%.4f" % (np.mean(cons_a) if cons_a else 0.0))
    print("R_L2N_CRIT_C1A_CONS_B=%.4f" % (np.mean(cons_b) if cons_b else 0.0))
    print("R_L2N_CRIT_C1B_RETEST_MAX=%.4f" % (max(retests) if retests else 0.0))
    print("R_L2N_CRIT_C2=%d" % c2)
    print("R_L2N_CRIT_C2_PRED_RATIO_A=%.4f" % (np.mean(pred_a) if pred_a else 0.0))
    print("R_L2N_CRIT_C2_PRED_RATIO_B=%.4f" % (np.mean(pred_b) if pred_b else 0.0))
    print("R_L2N_CRIT_C3=%d" % c3)
    print("R_L2N_CRIT_C3_COMPR_A=%.2f" % (np.mean(compr_a) if compr_a else 0.0))
    print("R_L2N_CRIT_C3_COMPR_B=%.2f" % (np.mean(compr_b) if compr_b else 0.0))
    print("R_L2N_CRIT_C3_ETA2_A=%.4f" % (np.mean(eta_a) if eta_a else 0.0))
    print("R_L2N_CRIT_C3_ETA2_B=%.4f" % (np.mean(eta_b) if eta_b else 0.0))
    print("R_L2N_CRIT_C4=%d" % c4)
    print("R_L2N_CRIT_C4_REF_A=%.4f" % (np.mean(dc_ref_a) if dc_ref_a else 0.0))
    print("R_L2N_CRIT_C4_REF_B=%.4f" % (np.mean(dc_ref_b) if dc_ref_b else 0.0))
    print("R_L2N_CRIT_C4_SC2_A=%d" % min(dc_sc2_a))
    print("R_L2N_CRIT_C4_SC2_B=%d" % min(dc_sc2_b))
    print("R_L2N_CRIT_C5=%d" % c5)
    print("R_L2N_CRIT_C5_1_ADOPT=%.4f/%.4f" % (adopt_conf_a, adopt_conf_b))
    print("R_L2N_CRIT_C5_2_JR_SUBSET=%.6f/%.6f" % (jr_subset_a, jr_subset_b))
    print("R_L2N_CRIT_C5_3_SPURIOUS_G2R_A=%d" % len(spurious_g2r_a))
    print("R_L2N_CRIT_C5_3_SPURIOUS_G2R_B=%d" % len(spurious_g2r_b))
    print("R_L2N_CRIT_C5_3_SPURIOUS_ORIG_A=%d" % len(spurious_g2s_a))
    print("R_L2N_CRIT_C5_3_SPURIOUS_ORIG_B=%d" % len(spurious_g2s_b))
    print("R_L2N_CRIT_C5_4_CONSTRUCTION=%d" % construction_ok)
    print("R_L2N_CRIT_C5_5_TRANSFER=%.6f/%.6f" % (transfer_a_m, transfer_b_m))
    print("R_L2N_CRIT_C5_5_CALIB=%.6f/%.6f" % (calib_a_m, calib_b_m))
    print("R_L2N_CRIT_C5_6_STRUCTURE=%d" % c5_6)
    print("R_L2N_ACTIVATED_A=%d" % len(sym_act_a))
    print("R_L2N_ACTIVATED_B=%d" % len(sym_act_b))
    print("R_L2N_VERDICT=%s" % verdict)
    print("R_L2N_VERDICT_NOTE=%s" % vnote)
    print("R_L2N_ELAPSED=%.2f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
