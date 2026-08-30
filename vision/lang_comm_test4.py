"""vision/lang_comm_test4.py — 语言线第四格：双向信道与互采纳（G2 第二格：B 输出回流 A、
交换闭环；交付 docs/271）。

docs/271 §一 预注册冻结，运行后不改。机制基座 = docs/270 逐字继承（import lang_comm_test3/
lang_comm_test2/lang_comm_test 复用，零改写）：LangCommGateLoop 门条件④（GATE_PURITY_MIN=
0.80）、环境 m1=2.6/m2=4.2、判据阈值、信道递送语义、留出划分（N_C=14）全沿用（四格同尺
可比）。**本格唯一机制加法 = 双向信道 + 对称世界**：

1. 双向信道：CH1（A->B，docs/268 逐字：s_A=mean_x(A 事件, y<68)，ctx_B=sign(s_A-x_B)，
   同窗递送）；CH2（B->A，本格对称定义：s_B=mean_x(B 事件, y>=68)，ctx_A=sign(s_B-x_A)，
   **一窗滞后**——B 的 w 窗输出在 w 窗末才可得，帧同步管线（A 先闭窗）无法双窗同读，
   docs/271 §1.2 冻结）。
2. 对称世界：A 的乘子也依 ctx 切换（a(t)=a1=1.6 if ctx_prev==0 else a2=2.6，docs/271
   §1.1 环境预计算冻结——A 两态中位能量 514-560.5/704-781 同落 c0 档 2 + c1 档 1 恒定、
   双侧设计窗口 8/10、窗口级能量双侧零跌破 450）。

BiLangLoop = LangCommGateLoop 逐字继承，双侧参数化：
  - self_side：A='upper'（x_A=mean_x(y<68)）/ B='lower'（x_B=mean_x(y>=68)，与 docs/268
    逐字同代码）；
  - publish_side：A='upper'（= docs/268 s_A 定义逐字）/ B='lower'（s_B）；
  - 门（条件④）逐字继承，在双回路同时生效（C3 双侧零假阳性是本格核心主张）。

流（docs/271 §1.8）：双向世界（CH2 on）M-T（两阶段）/M-G（单阶段）/C（前缀）/G0A（双
off）/G1n（双 null）/G2s（双 scrambled，(seed+5)%10）；单向世界（CH2 off = docs/270 逐位）
CELL3_REPRO 六臂 + W（信道在但 A 无信息）。同一世界种子的双向四臂共享同一世界帧。

度量（§1.4 双侧化）：M1 预测 MAE（双侧）；M2 结构（双侧）；M3 联合残差 JR（双侧）；
M4 信号质量诊断 + 门诊断（双侧）；M5 信号留出归因（双侧，N_C=14）。判据（§1.5）：
C1 MUTUAL_VALUE（双侧 JR 配对比值 <=0.85 + 双侧 MAE==G0A abs<1e-9）、C2 MUTUAL_ADOPTION
（双侧 adopt>=0.6 且采纳 compound>=0.5 且 G0A/G1n 双侧零采纳）、C3 CLEAN_KEEP（双侧
spurious(G2s)==0）、C4 STRUCTURE_KEEP/SIG_HOLDOUT（双侧结构 + 双侧 transfer>=0.10 且
>=0.5*calib）。判定映射：MUTUAL_EMERGES/ONE_WAY/MUTUAL_FLAT（含 MUTUAL_NO_GAIN）/
PARTIAL/GUARD_FAIL/LANG_BLOCKED。

守卫（§1.6）：R_L2J_GUARD_D232、R_L2J_GUARD_D235、R_L2J_CELL2_REPRO（import docs/270
逐字）、R_L2J_CONSTRUCTION、R_L2J_PREFIX_EQ（双侧）、R_L2J_TWO_PHASE_EQ（双侧）、
R_L2J_REPRO_MAE（双侧）、R_L2J_DETERM（timing/main 逐位一致，外部核对）、R_L2J_SMOKE
（含双侧门语义 + CH2 一窗滞后）、R_L2J_CELL3_REPRO（单向关闭 ≡ docs/270 逐位：adopt
0.80/comp 1.0/JR 0.587463/transfer 0.69/spurious 0）、R_L2J_WORLD_EQ（a_ctx_dep=False
≡ make_world 逐位）、R_L2J_PRECOMPUTE（环境预计算核对）。

安全纪律（§1.11）：新文件仅本文件；stdout 只输出 ASCII 标签 + 每行一个数字的 R_L2J_*
摘要块；运行经 powershell 包装重定向到 logs/；数字用 vision/extract_r.py 抽取；禁止读
日志/JSON 原文；本格不读 DAVIS。

用法：
  python vision/lang_comm_test4.py --smoke
  python vision/lang_comm_test4.py --precompute
  python vision/lang_comm_test4.py --tag timing
  python vision/lang_comm_test4.py --tag main
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
from critical_point import mean_sd, bootstrap_ci, JITTER, N_BOOT, BOOT_SEED
from stream_test import LOOP_CFG

warnings.filterwarnings("ignore")

DEFAULT_OUT = os.path.join("vision", "out", "results")

# ---------------- 冻结常量（docs/271 §1.1/§1.7；运行后不改） ----------------
LVCODES = l2h.LVCODES
LV_WORLD = l2h.LV_WORLD
B_M1 = l2h.B_M1                 # 2.6（B 环境常量，docs/269 冻结）
B_M2 = l2h.B_M2                 # 4.2
# 本格唯一新冻结量（docs/271 §1.1 预计算冻结）：A 乘子（环境常量，非机制旋钮）
A_M1 = 1.6                      # a(t)=a1 if ctx_prev==0 else a2（mirror=False）
A_M2 = 2.6
A_MIRROR = False
N_C = l2h.N_C                   # 14
TRANSFER_FLOOR = l2h.TRANSFER_FLOOR
TRANSFER_REL = l2h.TRANSFER_REL
JR_RATIO_MAX = l2h.JR_RATIO_MAX
ADOPT_FRAC_MIN = l2h.ADOPT_FRAC_MIN
COMPOUND_MIN = l2h.COMPOUND_MIN
N_FRAMES = l2h.N_FRAMES
WINDOW = l2h.WINDOW
ENERGY_BINS = l2h.ENERGY_BINS
DESIGN_MIN_E = 450.0
DESIGN_RATIO = 1.30
K_CONSIST = 3
GATE_PURITY_MIN = l2i.GATE_PURITY_MIN      # 0.80（docs/270 冻结，双侧生效）
BAND2_LO = 450.0                           # c0 档 2 下界（窗口级零跌破核对）


# ---------------- 双向世界（docs/271 §1.1 冻结；a_ctx_dep=False ≡ docs/270 逐位） ----------------
def make_bidi_world(seed, m1=B_M1, m2=B_M2, a1=A_M1, a2=A_M2, mirror=A_MIRROR,
                    a_ctx_dep=True, n_frames=N_FRAMES):
    """双向世界（A 两态 if a_ctx_dep else A 匀速 = docs/270 make_world 逐位）。
    rng 流与 make_world 同（seed*7919 + 41*104729 + 13；每帧一次 rng.normal，draw 序列
    与 docs/268/269/270 完全一致——a_ctx_dep 只改变 A 的相位步进算术，不改变任何随机
    抽取）-> a_ctx_dep=False 时 A/B 帧与 docs/270 逐位相同（R_L2J_WORLD_EQ 证明）。
    A 乘子规则（冻结）：a(t) = a1 if ctx_prev==(1 if mirror else 0) else a2。"""
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
    frames_a, frames_b = [], []
    per_frame = []
    for t in range(n_frames):
        if a_ctx_dep:
            a = a1 if ctx_prev == (1 if mirror else 0) else a2
            th_a += 2 * np.pi * l2g.A_FREQ * a / 30.0
        else:
            th_a += 2 * np.pi * l2g.A_FREQ / 30.0
        ax = l2g.A_CENTER[0] + l2g.A_ORBIT * np.cos(th_a)
        ay = l2g.A_CENTER[1] + l2g.A_ORBIT * np.sin(th_a)
        m = m1 if ctx_prev == 0 else m2
        th_b += 2 * np.pi * l2g.B_FREQ * m / 30.0
        bx = l2g.B_CENTER[0] + l2g.B_ORBIT * np.cos(th_b)
        by = l2g.B_CENTER[1] + l2g.B_ORBIT * np.sin(th_b)
        noise = rng.normal(0, sigma, (120, 160)).astype(np.float32)
        img_a = bg.copy()
        l2g.cv2_circle(img_a, int(ax), int(ay), l2g.DISK_R, l2g.OBJ_GRAY)
        img_b = bg.copy()
        l2g.cv2_circle(img_b, int(bx), int(by), l2g.DISK_R, l2g.OBJ_GRAY)
        frames_a.append(np.clip(img_a + noise, 0, 255).astype(np.uint8))
        frames_b.append(np.clip(img_b + noise, 0, 255).astype(np.uint8))
        ctx = 0 if (ax - bx) < 0 else 1
        per_frame.append((ctx, m))
        ctx_prev = ctx
    win_labels = []
    w = n_frames // WINDOW
    for i in range(w):
        seg = per_frame[i * WINDOW:(i + 1) * WINDOW]
        ctxs = [s[0] for s in seg]
        n0 = sum(1 for c in ctxs if c == 0)
        ctx_w = 0 if n0 > len(ctxs) - n0 else 1
        ms = [s[1] for s in seg]
        win_labels.append(dict(ctx=ctx_w, b_mult=float(np.mean(ms)),
                               a_regime=None))
    return frames_a, frames_b, win_labels


# ---------------- BiLangLoop（docs/271 §1.3：LangCommGateLoop 逐字 + 双侧 c2/发布） ----------------
class BiLangLoop(l2i.LangCommGateLoop):
    """LangCommGateLoop 逐字继承；唯一覆写 _ctx_from_signal（self_side 参数化：B='lower'
    逐字同 docs/268、A='upper' 对称镜像）与 _on_window（publish_side 参数化发布）。
    c0/c1/账本/匹配/提升/回收/churn 会计/预测路径与门（条件④）逐字继承。"""

    def __init__(self, self_side="lower", publish_side="upper",
                 gate_purity_min=None, mode="pixel", **kw):
        super().__init__(gate_purity_min=gate_purity_min, mode=mode, **kw)
        self.self_side = self_side
        self.publish_side = publish_side

    def _ctx_from_signal(self, ev_win):
        """c2 = sign(s - x_self)。self_side='lower'（B）：与 docs/268 LangCommLoop
        _ctx_from_signal 逐字同代码（CELL3_REPRO 前提）；self_side='upper'（A）：对称
        镜像（x_A = mean_x(y<68)）。"""
        if self.self_side == "lower":
            lo = ev_win[int(l2g.CTX_SPLIT_Y):, :]
            lo_n = int(lo.sum())
            if lo_n >= l2g.SIG_SPARSE_PX:
                xb = float(np.mean(np.nonzero(lo)[1]))
            else:
                xb = None
            self._last_xB = xb
            if self.mode == "off":
                return None
            if self.mode == "null":
                s = l2g.NULL_SIGNAL
            else:                                   # comm / scrambled
                s = self.signal
                if s is None:
                    return None
            if xb is None:
                return None
            return 0 if (s - xb) < 0 else 1
        # upper（A 侧对称）
        up = ev_win[:int(l2g.CTX_SPLIT_Y), :]
        up_n = int(up.sum())
        if up_n >= l2g.SIG_SPARSE_PX:
            xa = float(np.mean(np.nonzero(up)[1]))
        else:
            xa = None
        self._last_xB = xa
        if self.mode == "off":
            return None
        if self.mode == "null":
            s = l2g.NULL_SIGNAL
        else:
            s = self.signal
            if s is None:
                return None
        if xa is None:
            return None
        return 0 if (s - xa) < 0 else 1

    def _on_window(self):
        """发布感知输出（publish_side：A='upper' = docs/268 s_A 定义逐字；B='lower' =
        s_B）；其余逐字同 LangCommLoop（组件/账本/匹配/提升/回收由 CompLoop._on_window
        承担——直接调用以跳过 LangCommLoop 的 up 区发布，避免 sA_trace 双计）。"""
        ev_win = self._ev_win if self._ev_win is not None else \
            np.zeros((120, 160), bool)
        if self.publish_side == "upper":
            up = ev_win[:int(l2g.CTX_SPLIT_Y), :]
            up_n = int(up.sum())
            s_pub = float(np.mean(np.nonzero(up)[1])) if up_n >= l2g.SIG_SPARSE_PX \
                else None
        else:
            lo = ev_win[int(l2g.CTX_SPLIT_Y):, :]
            lo_n = int(lo.sum())
            s_pub = float(np.mean(np.nonzero(lo)[1])) if lo_n >= l2g.SIG_SPARSE_PX \
                else None
        self.sA_trace.append(s_pub)
        self._last_xB = None
        super(l2g.LangCommLoop, self)._on_window()   # CompLoop._on_window 逐字
        self.xB_trace.append(self._last_xB)


# ---------------- 双向流运行（docs/271 §1.2/§1.8 冻结） ----------------
def run_bidi(fa, fb, wl, ch1="comm", ch2="comm", a_mode=None, two_phase=False,
             n_c=N_C, gate=GATE_PURITY_MIN, want_end_snap=False,
             sig1_fn=None, sig2_fn=None):
    """双向世界双回路（A/B 帧同步步进）：
    - CH1（A->B）同窗递送（docs/268 run_dual 步序逐字：A.step -> A 闭窗发布 s_A(w) ->
      B.set_signal -> B.step 同窗读取）；
    - CH2（B->A）一窗滞后（B 闭窗发布 s_B(w) -> A.set_signal -> A 的下一窗读取；
      docs/271 §1.2 冻结：A 的窗口 0 无信号）。
    ch1 = B 侧信道模式、ch2 = A 侧信道模式（off/null/scrambled 在窗起始预置信号；
    comm 由对方闭窗发布设置）；a_mode：A 回路模式（默认 = ch2；CELL3_REPRO 用
    a_mode='pixel' 复现 docs/270）。sig1_fn(w)/sig2_fn(w) = scrambled 信号源。
    返回 (out_a, loop_a, out_b, loop_b, snap_a, snap_b)。"""
    loop_a = BiLangLoop(mode=(a_mode if a_mode is not None else ch2),
                        self_side="upper", publish_side="upper",
                        gate_purity_min=gate, window=WINDOW, **LOOP_CFG)
    loop_b = BiLangLoop(mode=ch1, self_side="lower", publish_side="lower",
                        gate_purity_min=gate, window=WINDOW, **LOOP_CFG)
    n_frames = len(fb)
    n_w = n_frames // WINDOW
    phases = ([(0, n_c * WINDOW), (n_c * WINDOW, n_frames)] if two_phase
              else [(0, n_frames)])
    snap_a = snap_b = None
    a_closed = b_closed = 0
    for (f0, f1) in phases:
        for k in range(f0, f1):
            # A 侧信号（ch2）：新窗开始按模式预置（comm 例外：由 B 闭窗发布设置）
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
                if ch1 == "comm":                 # CH1 同窗递送（docs/268 逐字）
                    loop_b.set_signal(loop_a.sA_trace[a_closed])
                a_closed += 1
            # B 侧信号（ch1）：新窗开始按模式预置（comm 例外：由 A 闭窗发布设置）
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
                if ch2 == "comm":                 # CH2 发布 -> A 下一窗读取（一窗滞后）
                    loop_a.set_signal(loop_b.sA_trace[b_closed])
                b_closed += 1
        if two_phase and f0 == 0:
            snap_a = l2g.snapshot_b(loop_a)
            snap_b = l2g.snapshot_b(loop_b)
    if want_end_snap and snap_a is None:
        snap_a = l2g.snapshot_b(loop_a)
        snap_b = l2g.snapshot_b(loop_b)
    # 收尾：A 冲刷末窗 -> 发布 s_A 给 B 的 finalize 冲刷窗（docs/268 逐字）
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


def run_b_signal_bidi(frames_b, window=WINDOW):
    """B 回路单独跑完（收集 s_B 发布序列；G2s 错乱信号用另一种子的 B 帧）。"""
    loop_b = BiLangLoop(mode="off", self_side="lower", publish_side="lower",
                        window=window, **LOOP_CFG)
    n_w = len(frames_b) // window
    for k in range(len(frames_b)):
        loop_b.step(frames_b[k])
    if len(loop_b._frame_buf):
        loop_b.finalize(n_w, None)
    return loop_b


# ---------------- 环境预计算（docs/271 §1.1 协议；机制无关） ----------------
def bidi_diag(wl, E):
    """(真值 ctx 标签, 逐窗能量) 的两态中位能量/档位/比值/窗数诊断。"""
    by = {0: [], 1: []}
    for w, c in enumerate(wl):
        if c is not None and w < len(E):
            by[c].append(float(E[w]))
    meds = {c: (float(np.median(v)) if v else None) for c, v in by.items()}
    bands = {c: (None if meds[c] is None else l2g._band(meds[c], ENERGY_BINS))
             for c in (0, 1)}
    c1bands = {c: (None if meds[c] is None else l2g._band(meds[c], l2g.UPPER_BINS))
               for c in (0, 1)}
    ratio = None
    if meds[0] and meds[1]:
        ratio = max(meds[0], meds[1]) / max(1e-9, min(meds[0], meds[1]))
    n0 = sum(1 for c in wl if c == 0)
    n1 = sum(1 for c in wl if c == 1)
    return dict(med0=meds[0], med1=meds[1], band0=bands[0], band1=bands[1],
                c1b0=c1bands[0], c1b1=c1bands[1], ratio=ratio, n0=n0, n1=n1)


def design_window_a(d):
    """A 设计窗口（docs/271 §1.1）：两态 E 同落 c0 档 2 且 c1 两态同档（A 的 c1=band(E)，
    "自身不可分"签名条件）且比值>=1.30 且两态窗数>=3。"""
    return int(d["band0"] == 2 and d["band1"] == 2
               and d["c1b0"] == d["c1b1"]
               and d["ratio"] is not None and d["ratio"] >= DESIGN_RATIO
               and d["n0"] >= K_CONSIST and d["n1"] >= K_CONSIST)


def design_window_b(d):
    """B 设计窗口（docs/269 §1.1 逐字）：两态 E 同落 c0 档 2 且比值>=1.30 且两态窗数>=3。"""
    return int(d["band0"] == 2 and d["band1"] == 2
               and d["ratio"] is not None and d["ratio"] >= DESIGN_RATIO
               and d["n0"] >= K_CONSIST and d["n1"] >= K_CONSIST)


def precompute_ok_main():
    """主运行内的环境预计算核对（§1.6 R_L2J_PRECOMPUTE）：
    ① 锚点：a_ctx_dep=False 世界 B 侧中位能量 ≡ docs/269 预计算表（10/10 逐位）；
    ② 冻结候选 (1.6, 2.6, F)：双向世界 A/B 双侧设计窗口 8/10 + 设计种子窗口级能量双侧
    零跌破 450；③ 预计算 vs 回路能量逐位比对（双向 G0A 双 off 的 energy_trace，双侧）。"""
    seeds = list(range(10))
    exp_e0 = [481.0, 512.0, 513.5, 510.0, 367.0, 510.0, 520.5, 310.0, 517.0, 530.0]
    exp_e1 = [697.0, 724.0, 725.0, 712.0, 538.0, 710.0, 740.0, 452.0, 728.0, 746.0]
    anchor_ok = 1
    for s in seeds:
        _, fb, wl = make_bidi_world(s, a_ctx_dep=False)
        E = l2h.precompute_energies(fb)
        ctxs = [lb["ctx"] for lb in wl]
        d = bidi_diag(ctxs, E)
        ok = int(abs(d["med0"] - exp_e0[s]) < 0.15
                 and abs(d["med1"] - exp_e1[s]) < 0.15)
        anchor_ok &= ok
    dw_a = dw_b = 0
    under_a = under_b = 0
    eq_oks = []
    per_a = []
    per_b = []
    for s in seeds:
        fa, fb, wl = make_bidi_world(s, a1=A_M1, a2=A_M2, mirror=A_MIRROR)
        ctxs = [lb["ctx"] for lb in wl]
        Ea = l2h.precompute_energies(fa)
        Eb = l2h.precompute_energies(fb)
        da_ = bidi_diag(ctxs, Ea)
        db_ = bidi_diag(ctxs, Eb)
        wa = design_window_a(da_)
        wb = design_window_b(db_)
        dw_a += wa
        dw_b += wb
        per_a.append(wa)
        per_b.append(wb)
        if wa and wb:               # 设计种子（双侧）窗口级零跌破核对
            under_a += sum(1 for e in Ea if e < BAND2_LO)
            under_b += sum(1 for e in Eb if e < BAND2_LO)
        # 预计算 vs 回路能量（双向 G0A 双 off）
        out_a, loop_a, out_b, loop_b, _, _ = run_bidi(
            fa, fb, wl, ch1="off", ch2="off", a_mode="off")
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
    """--precompute：完整环境预计算核对（§1.1 协议复现 + 网格节选 + 锚点）。"""
    seeds = list(range(10))
    # ① 锚点：a_ctx_dep=False 世界 B 侧中位能量 vs docs/269 预计算表（逐位）
    exp_e0 = [481.0, 512.0, 513.5, 510.0, 367.0, 510.0, 520.5, 310.0, 517.0, 530.0]
    exp_e1 = [697.0, 724.0, 725.0, 712.0, 538.0, 710.0, 740.0, 452.0, 728.0, 746.0]
    anch = 1
    for s in seeds:
        _, fb, wl = make_bidi_world(s, a_ctx_dep=False)
        E = l2h.precompute_energies(fb)
        ctxs = [lb["ctx"] for lb in wl]
        d = bidi_diag(ctxs, E)
        ok = int(abs(d["med0"] - exp_e0[s]) < 0.15 and abs(d["med1"] - exp_e1[s]) < 0.15)
        anch &= ok
        print("R_L2J_PRECOMPUTE_ANCHOR_S%d=%d" % (s, ok))
        print("R_L2J_PRECOMPUTE_ANCHOR_S%d_E0=%.1f" % (s, d["med0"]))
        print("R_L2J_PRECOMPUTE_ANCHOR_S%d_E1=%.1f" % (s, d["med1"]))
    print("R_L2J_PRECOMPUTE_ANCHOR=%d" % anch)
    # ② 冻结候选 (1.6, 2.6, F)：双侧设计窗口 + 逐种子明细（窗口级零跌破仅统计双侧设计种子）
    dw_a = dw_b = 0
    under_a = under_b = 0
    for s in seeds:
        fa, fb, wl = make_bidi_world(s, a1=A_M1, a2=A_M2, mirror=A_MIRROR)
        ctxs = [lb["ctx"] for lb in wl]
        Ea = l2h.precompute_energies(fa)
        Eb = l2h.precompute_energies(fb)
        da_ = bidi_diag(ctxs, Ea)
        db_ = bidi_diag(ctxs, Eb)
        wa = design_window_a(da_)
        wb = design_window_b(db_)
        dw_a += wa
        dw_b += wb
        if wa and wb:
            under_a += sum(1 for e in Ea if e < BAND2_LO)
            under_b += sum(1 for e in Eb if e < BAND2_LO)
        print("R_L2J_PRECOMPUTE_S%d_A_E0=%.1f" % (s, da_["med0"]))
        print("R_L2J_PRECOMPUTE_S%d_A_E1=%.1f" % (s, da_["med1"]))
        print("R_L2J_PRECOMPUTE_S%d_A_RATIO=%.4f" % (s, da_["ratio"]))
        print("R_L2J_PRECOMPUTE_S%d_A_DW=%d" % (s, wa))
        print("R_L2J_PRECOMPUTE_S%d_B_E0=%.1f" % (s, db_["med0"]))
        print("R_L2J_PRECOMPUTE_S%d_B_E1=%.1f" % (s, db_["med1"]))
        print("R_L2J_PRECOMPUTE_S%d_B_RATIO=%.4f" % (s, db_["ratio"]))
        print("R_L2J_PRECOMPUTE_S%d_B_DW=%d" % (s, wb))
    print("R_L2J_PRECOMPUTE_DW_A=%d" % dw_a)
    print("R_L2J_PRECOMPUTE_DW_B=%d" % dw_b)
    print("R_L2J_PRECOMPUTE_UNDER450_A=%d" % under_a)
    print("R_L2J_PRECOMPUTE_UNDER450_B=%d" % under_b)
    # ③ 网格节选（a1 x a2 x mirror -> 双侧设计窗口计数；预计算快扫）
    grid_rows = []
    for (a1, a2) in ((1.4, 2.2), (1.4, 2.4), (1.4, 2.6), (1.6, 2.4), (1.6, 2.6)):
        row = []
        for mirror in (False, True):
            da = db = 0
            for s in seeds:
                fa, fb, wl = make_bidi_world(s, a1=a1, a2=a2, mirror=mirror)
                ctxs = [lb["ctx"] for lb in wl]
                da += design_window_a(bidi_diag(ctxs, l2h.precompute_energies(fa)))
                db += design_window_b(bidi_diag(ctxs, l2h.precompute_energies(fb)))
            row.append("m%d:%d/%d" % (int(mirror), da, db))
        grid_rows.append("a1=%.1f,a2=%.1f:%s" % (a1, a2, ";".join(row)))
    print("R_L2J_PRECOMPUTE_GRID=%s" % "|".join(grid_rows))
    return 0 if (anch == 1 and dw_a == 8 and dw_b == 8
                 and under_a == 0 and under_b == 0) else 1


# ---------------- 守卫 ----------------
def world_eq():
    """R_L2J_WORLD_EQ：make_bidi_world(a_ctx_dep=False) ≡ l2g.make_world(m1=2.6,m2=4.2)
    逐位（A/B 帧 + 逐窗真值 ctx，10/10）。"""
    oks = []
    for s in range(10):
        fa, fb, wl = make_bidi_world(s, a_ctx_dep=False)
        fa2, fb2, wl2 = l2g.make_world(s, m1=B_M1, m2=B_M2)
        wl2c = [lb["ctx"] for lb in wl2]
        same = (len(fa) == len(fa2) and len(fb) == len(fb2)
                and all(np.array_equal(x, y) for x, y in zip(fa, fa2))
                and all(np.array_equal(x, y) for x, y in zip(fb, fb2))
                and [lb["ctx"] for lb in wl] == wl2c)
        oks.append(int(same))
    return int(all(oks)), oks


# ---------------- R_L2J_CELL3_REPRO（docs/271 §1.6 复现锚：单向关闭 ≡ docs/270 逐位） ----------------
# 期望数字 = docs/270 §三/§四 冻结值（门开启、CH2 off = docs/270 世界；同代码路径 ->
# 期望位精确，容差取打印精度 + 余量）。来源行：docs/270 §3.1（逐种子 JR/首提升窗/门纯度/
# 保真度）、§3.2（逐种子 transfer/calib）、§3.3（聚合）、§3.6（fid）。
CELL3_EXP = {
    "jr_g0": [0.1699, 0.1440, 0.1615, 0.1542, 0.1224, 0.1689, 0.1490,
              0.1856, 0.1518, 0.1485],
    "jr_g1": [0.0774, 0.0647, 0.0500, 0.0981, 0.1224, 0.0599, 0.0935,
              0.1856, 0.0935, 0.0633],
    "jr_ratio": [0.4554, 0.4495, 0.3097, 0.6358, 1.0000, 0.3546, 0.6273,
                 1.0000, 0.6164, 0.4261],
    "transfer": [1.000, 0.900, 1.000, 1.000, 0.000, 1.000, 0.000,
                 0.000, 1.000, 1.000],
    "calib": [0.875, 0.857, 0.857, 0.667, 0.000, 0.875, 0.000,
              0.000, 0.500, 0.875],
    "first_promo": [6, 7, 7, 11, None, 6, 15, None, 12, 6],
    "gate_purity": [1.0, 1.0, 1.0, 0.8, None, 1.0, 0.8, None, 0.8, 1.0],
    "fid": [1.0000, 0.9583, 1.0000, 0.9167, 0.8333, 1.0000, 0.9167,
            0.7500, 0.9583, 0.9583],
    "g2s_promo_seeds": [],          # docs/270：spurious(G2s) = 0/10（门开启）
    "adopt_frac": 0.8000,
    "comp_adopted": 1.0000,
    "mae": 0.023023,
    "mae_sd": 0.002179,
    "jr_ratio_mean": 0.587463,
    "transfer_mean": 0.690,
    "calib_mean": 0.551,
    "fid_mean": 0.9292,
    "a_sc2": 1.0,
}


def repro_cell3():
    """单向关闭（CH2 off、A 匀速 = docs/270 世界逐位）：经本格 run_bidi 管线（BiLangLoop
    的 lower 路径 + A pixel）复现 docs/270 六臂 10 种子数字逐位。返回 (ok, detail)。"""
    seeds = list(range(10))
    jr_g0s, jr_g1s, jr_ratios = [], [], []
    trans, calibs, fids = [], [], []
    first_promos, g_purities, mae_t, mae_0 = [], [], [], []
    a_sc2s, compounds = [], []
    g2_promos = []
    two_phase_oks = []
    for s in seeds:
        fa, fb, wl = l2g.make_world(s, m1=B_M1, m2=B_M2)
        # G1T（两阶段，门开启，A pixel，CH2 off）
        out_ta, loop_a, out_tb, loop_t, _, snap_t = run_bidi(
            fa, fb, wl, ch1="comm", ch2="off", a_mode="pixel",
            two_phase=True, n_c=N_C)
        jr1 = l2g.jr_b(loop_t)[0]
        att = l2g.attribution(loop_t, N_C)
        # G1G（单阶段）-> TWO_PHASE_EQ（本格管线自洽）
        out_ga, _, out_gb, loop_g, _, _ = run_bidi(fa, fb, wl, ch1="comm",
                                                   ch2="off", a_mode="pixel",
                                                   two_phase=False)
        # C（前缀）-> PREFIX_EQ（本格管线自洽）
        out_ca, _, out_cb, _, _, snap_c = run_bidi(fa[:140], fb[:140], wl[:14],
                                                   ch1="comm", ch2="off",
                                                   a_mode="pixel",
                                                   two_phase=False,
                                                   want_end_snap=True)
        # G0
        out_0a, _, out_0b, loop0, _, _ = run_bidi(fa, fb, wl, ch1="off",
                                                  ch2="off", a_mode="pixel",
                                                  two_phase=False)
        jr0 = l2g.jr_b(loop0)[0]
        # G1N
        out_na, _, out_nb, loop_n, _, _ = run_bidi(fa, fb, wl, ch1="null",
                                                   ch2="off", a_mode="pixel",
                                                   two_phase=False)
        # G2S（错乱，门开启）
        other = (s + 5) % 10
        a_other = l2g.run_a_signal(l2g.make_world(other, m1=B_M1, m2=B_M2)[0])
        out_2a, _, out_2b, loop2, _, _ = run_bidi(
            fa, fb, wl, ch1="scrambled", ch2="off", a_mode="pixel",
            two_phase=False, sig1_fn=lambda w: a_other.sA_trace[w])
        jr_ratios.append(jr1 / max(jr0, 1e-12))
        jr_g0s.append(jr0)
        jr_g1s.append(jr1)
        trans.append(att["transfer_adopted_hit_rate"])
        calibs.append(att["calib_baseline"])
        fids.append(out_tb["ctx_fidelity"])
        mae_t.append(out_tb["mae_mean"])
        mae_0.append(out_0b["mae_mean"])
        first_promos.append(att["first_promo_win"])
        # 门纯度（首提升窗处）
        gp = None
        if att["first_promo_win"] is not None:
            for rec in loop_t.gate_attempts:
                if rec[0] == att["first_promo_win"]:
                    gp = rec[2]
                    break
        g_purities.append(gp)
        a_sc2s.append(loop_a.finalize(max(1, len(loop_a.energy_trace)),
                                      None)["sc2"])
        compounds.append(out_tb["compound_frac"])
        g2_promos.append(out_2b["n_promo"])
        two_phase_oks.append(l2g.two_phase_eq(
            l2h.unit_record2("G1G", s, out_gb, loop_g),
            l2h.unit_record2("G1T", s, out_tb, loop_t, snap=snap_t)))
    adopt_frac = float(np.mean([fp is not None for fp in first_promos]))
    adopted_comp = [c for s, c in enumerate(compounds)
                    if first_promos[s] is not None]
    comp_adopted = float(np.mean(adopted_comp)) if adopted_comp else 0.0
    mae_m, mae_sd = mean_sd(mae_0)
    jr_ratio_m, _ = mean_sd(jr_ratios)
    trans_m, _ = mean_sd(trans)
    calib_m, _ = mean_sd(calibs)
    fid_m, _ = mean_sd(fids)
    a_sc2_m = float(np.mean(a_sc2s))
    g2_spur = [s for s in seeds if g2_promos[s] >= 1]

    def chk(name, got, exp, tol=1e-4):
        return name, int(abs(got - exp) < tol), got

    checks = []
    for s in seeds:
        checks.append(chk("JR_RATIO_S%d" % s, jr_ratios[s],
                          CELL3_EXP["jr_ratio"][s], 1e-4))
        checks.append(chk("JR_G0_S%d" % s, jr_g0s[s], CELL3_EXP["jr_g0"][s], 1e-4))
        checks.append(chk("JR_G1_S%d" % s, jr_g1s[s], CELL3_EXP["jr_g1"][s], 1e-4))
        checks.append(chk("TRANSFER_S%d" % s, trans[s], CELL3_EXP["transfer"][s], 1e-3))
        checks.append(chk("CALIB_S%d" % s, calibs[s], CELL3_EXP["calib"][s], 1e-3))
        checks.append(chk("FID_S%d" % s, fids[s], CELL3_EXP["fid"][s], 1e-4))
        exp_fp = CELL3_EXP["first_promo"][s]
        checks.append(("FIRST_PROMO_S%d" % s,
                       int(first_promos[s] == exp_fp),
                       (first_promos[s] if first_promos[s] is not None else -1)))
        exp_gp = CELL3_EXP["gate_purity"][s]
        checks.append(("GATE_PURITY_S%d" % s,
                       int((g_purities[s] is None and exp_gp is None)
                           or (g_purities[s] is not None and exp_gp is not None
                               and abs(g_purities[s] - exp_gp) < 1e-4)),
                       (g_purities[s] if g_purities[s] is not None else -1)))
    checks.append(chk("ADOPT_FRAC", adopt_frac, CELL3_EXP["adopt_frac"], 1e-6))
    checks.append(chk("COMP_ADOPTED", comp_adopted, CELL3_EXP["comp_adopted"], 1e-4))
    checks.append(chk("MAE_G0", mae_m, CELL3_EXP["mae"], 1e-5))
    checks.append(chk("MAE_SD", mae_sd, CELL3_EXP["mae_sd"], 1e-5))
    checks.append(chk("JR_RATIO_MEAN", jr_ratio_m, CELL3_EXP["jr_ratio_mean"], 1e-4))
    checks.append(chk("TRANSFER_MEAN", trans_m, CELL3_EXP["transfer_mean"], 1e-3))
    checks.append(chk("CALIB_MEAN", calib_m, CELL3_EXP["calib_mean"], 1e-3))
    checks.append(chk("FID_MEAN", fid_m, CELL3_EXP["fid_mean"], 1e-4))
    checks.append(chk("A_SC2", a_sc2_m, CELL3_EXP["a_sc2"], 1e-4))
    checks.append(("G2S_SPURIOUS", int(g2_spur == CELL3_EXP["g2s_promo_seeds"]),
                   g2_spur))
    checks.append(("TWO_PHASE_EQ", int(all(two_phase_oks)),
                   int(sum(two_phase_oks))))
    ok = int(all(c[1] == 1 for c in checks))
    return ok, dict(checks=checks, adopt_frac=adopt_frac,
                    comp_adopted=comp_adopted, mae=mae_m, mae_sd=mae_sd,
                    jr_ratio_mean=jr_ratio_m, transfer=trans_m,
                    calib=calib_m, fid=fid_m, a_sc2=a_sc2_m,
                    g2_spurious=g2_spur, two_phase_ok=int(all(two_phase_oks)))


# ---------------- 构造冒烟（docs/271 §1.6-8；合成帧，非数据） ----------------
def smoke_main4():
    """构造冒烟：双侧四模式构造运行正常；off 与 null 逐窗一致；G0 无提升；退化不崩；
    归因不变量；baseline=0 不崩；**门语义双侧单元测试 + CH2 一窗滞后语义**。"""
    results = {}
    fb = l2g._synth_frames(30)
    fa = l2g._synth_frames(30, y0=26)
    labels = [dict(ctx=1, b_mult=1.0, a_regime=None)] * 3
    outs = {}
    loops = {}
    for ch1, ch2 in (("off", "off"), ("comm", "comm"), ("null", "null"),
                     ("scrambled", "scrambled")):
        out_a, loop_a, out_b, loop_b, _, _ = run_bidi(
            fa, fb, labels, ch1=ch1, ch2=ch2, two_phase=False, n_c=3)
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
                                  two_phase=False, n_c=5)
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
                                    two_phase=False, n_c=5)
    att_a3 = l2g.attribution(la3, n_c=5)
    att_b3 = l2g.attribution(lb3, n_c=5)
    results["baseline_zero_ok"] = int(
        att_a3["calib_baseline"] == 0.0 and att_a3["transfer_adopted_hits"] == 0
        and att_b3["calib_baseline"] == 0.0 and att_b3["transfer_adopted_hits"] == 0)
    # ---- 门语义双侧单元测试（docs/270 §1.1 门定义；合成账本组，非数据） ----
    loop_on_a = BiLangLoop(mode="comm", self_side="upper", publish_side="upper",
                           gate_purity_min=GATE_PURITY_MIN, window=10, **LOOP_CFG)
    loop_on_b = BiLangLoop(mode="comm", self_side="lower", publish_side="lower",
                           gate_purity_min=GATE_PURITY_MIN, window=10, **LOOP_CFG)
    loop_off = BiLangLoop(mode="comm", self_side="lower", publish_side="lower",
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
    # ---- CH2 一窗滞后语义（docs/271 §1.2 冻结）：A 的窗口 0 无信号（c2=None） ----
    # 构造：A 信号来自 B 发布；首个窗口 A 侧 c2 应为 None（尚无 s_B）
    _, la4, _, lb4, _, _ = run_bidi(fa2, fb2, lab2, ch1="comm", ch2="comm",
                                    two_phase=False, n_c=5)
    a_c2_first = la4.sig_trace[0][2]
    results["ch2_lag_first_win_none"] = int(a_c2_first is None)
    # null 模式双侧 c2 恒 1（单值 -> 无采纳条件②）
    _, la5, _, lb5, _, _ = run_bidi(fa2, fb2, lab2, ch1="null", ch2="null",
                                    two_phase=False, n_c=5)
    results["null_both_single_ctx"] = int(
        all(s[2] == 1 for s in la5.sig_trace if s[2] is not None)
        and all(s[2] == 1 for s in lb5.sig_trace if s[2] is not None))
    for k in sorted(results):
        print("R_L2J_SMOKE_%s=%d" % (k.upper(), results[k]))
    return 0 if all(results.values()) else 1


# ---------------- 单元记录（checkpoint/resume 用；自包含） ----------------
def unit_record4(arm, seed, out, loop, side, snap=None, jr=None, att=None):
    rec = l2h.unit_record2(arm, seed, out, loop, snap=snap, jr=jr, att=att)
    rec["side"] = side
    rec["gate"] = list(loop.gate_attempts)
    return rec


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="l2j")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--precompute", action="store_true",
                    help="环境预计算完整核对（§1.1 协议复现；机制无关）")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.smoke:
        return smoke_main4()
    if args.precompute:
        return precompute_main()
    t0 = time.time()
    seeds = list(range(10))

    cfg = {"tag": args.tag, "n_seeds": len(seeds), "frames": N_FRAMES,
           "window": WINDOW, "n_c": N_C, "jitter": JITTER,
           "b_m1": B_M1, "b_m2": B_M2, "a_m1": A_M1, "a_m2": A_M2,
           "a_mirror": A_MIRROR, "noise_sigma": l2g.NOISE_SIGMA,
           "world": {"a_center": list(l2g.A_CENTER), "a_orbit": l2g.A_ORBIT,
                     "a_freq": l2g.A_FREQ, "b_center": list(l2g.B_CENTER),
                     "b_orbit": l2g.B_ORBIT, "b_freq": l2g.B_FREQ,
                     "rng_lvcode": LV_WORLD},
           "channel": {"sparse_px": l2g.SIG_SPARSE_PX,
                       "null_signal": l2g.NULL_SIGNAL,
                       "ch2_lag_windows": 1},
           "gate": {"purity_min": GATE_PURITY_MIN},
           "criteria": {"jr_ratio_max": JR_RATIO_MAX,
                        "adopt_frac_min": ADOPT_FRAC_MIN,
                        "compound_min": COMPOUND_MIN,
                        "transfer_floor": TRANSFER_FLOOR,
                        "transfer_rel": TRANSFER_REL},
           "loop": LOOP_CFG}
    ck_tag = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    ckpt_path = os.path.join(args.out_dir, "ckpt_l2j_%s.json" % ck_tag)

    done = {}
    if args.resume and not args.no_resume and os.path.exists(ckpt_path):
        with open(ckpt_path, encoding="utf-8") as f:
            done = json.load(f).get("per_unit", {})

    per_unit = dict(done)
    worlds_bidi = {s: make_bidi_world(s, a1=A_M1, a2=A_M2, mirror=A_MIRROR)
                   for s in seeds}
    worlds_uniform = {s: l2g.make_world(s, m1=B_M1, m2=B_M2) for s in seeds}

    # G2s 错乱信号源（双向世界，每种子一次）：s_A 序列（A 回路单独跑）+ s_B 序列
    bidi_sig = {}
    for s in seeds:
        fa, fb, _ = worlds_bidi[s]
        la = l2g.run_a_signal(fa)
        lb = run_b_signal_bidi(fb)
        bidi_sig[s] = (list(la.sA_trace), list(lb.sA_trace))

    def need(arm, s):
        return "%s_%d" % (arm, s) not in per_unit

    for s in seeds:
        fa, fb, wl = worlds_bidi[s]
        if need("MTA", s) or need("MTB", s):
            out_a, loop_a, out_b, loop_b, snap_a, snap_b = run_bidi(
                fa, fb, wl, ch1="comm", ch2="comm", two_phase=True, n_c=N_C)
            per_unit["MTA_%d" % s] = unit_record4(
                "MTA", s, out_a, loop_a, "A", snap=snap_a,
                jr=l2g.jr_b(loop_a), att=l2g.attribution(loop_a, N_C))
            per_unit["MTB_%d" % s] = unit_record4(
                "MTB", s, out_b, loop_b, "B", snap=snap_b,
                jr=l2g.jr_b(loop_b), att=l2g.attribution(loop_b, N_C))
            print("PROGRESS", flush=True)
        if need("MGA", s) or need("MGB", s):
            out_a, loop_a, out_b, loop_b, _, _ = run_bidi(
                fa, fb, wl, ch1="comm", ch2="comm", two_phase=False)
            per_unit["MGA_%d" % s] = unit_record4(
                "MGA", s, out_a, loop_a, "A")
            per_unit["MGB_%d" % s] = unit_record4(
                "MGB", s, out_b, loop_b, "B")
            print("PROGRESS", flush=True)
        if need("CA", s) or need("CB", s):
            out_a, loop_a, out_b, loop_b, snap_a, snap_b = run_bidi(
                fa[:140], fb[:140], wl[:14], ch1="comm", ch2="comm",
                two_phase=False, want_end_snap=True)
            per_unit["CA_%d" % s] = unit_record4(
                "CA", s, out_a, loop_a, "A", snap=snap_a)
            per_unit["CB_%d" % s] = unit_record4(
                "CB", s, out_b, loop_b, "B", snap=snap_b)
            print("PROGRESS", flush=True)
        if need("G0AA", s) or need("G0AB", s):
            out_a, loop_a, out_b, loop_b, _, _ = run_bidi(
                fa, fb, wl, ch1="off", ch2="off", a_mode="off")
            per_unit["G0AA_%d" % s] = unit_record4(
                "G0AA", s, out_a, loop_a, "A", jr=l2g.jr_b(loop_a))
            per_unit["G0AB_%d" % s] = unit_record4(
                "G0AB", s, out_b, loop_b, "B", jr=l2g.jr_b(loop_b))
            print("PROGRESS", flush=True)
        if need("G1NA", s) or need("G1NB", s):
            out_a, loop_a, out_b, loop_b, _, _ = run_bidi(
                fa, fb, wl, ch1="null", ch2="null", a_mode="null")
            per_unit["G1NA_%d" % s] = unit_record4(
                "G1NA", s, out_a, loop_a, "A")
            per_unit["G1NB_%d" % s] = unit_record4(
                "G1NB", s, out_b, loop_b, "B")
            print("PROGRESS", flush=True)
        if need("G2SA", s) or need("G2SB", s):
            other = (s + 5) % 10
            sigA = bidi_sig[other][0]
            # A 侧错乱源（docs/271 §1.2 修正，§二 C3 记录）：冻结的"另一种子 s_B"源因
            # 几何嵌套（A.x∈[67,93] ⊃ B.x∈[70,90]）与真值 ctx 高度相关（corr 至 +1.0），
            # 违反 C3 判据"与真值 ctx 不相关"前提（门在错乱臂不可测）——改为确定性每种子
            # 随机二元 ctx 注入（docs/270 §二 C1 蒙特卡洛随机分裂同款模型；sign(v-x_A)
            # 用远阈值 131/31 直接实现随机位：v=131 → ctx=1、v=31 → ctx=0）。
            rng_bits = np.random.default_rng(s * 99991 + 12345)
            rand_bits = [int(rng_bits.random() < 0.5) for _ in range(
                len(wl))]
            sig2 = [131.0 if b else 31.0 for b in rand_bits]
            out_a, loop_a, out_b, loop_b, _, _ = run_bidi(
                fa, fb, wl, ch1="scrambled", ch2="scrambled",
                a_mode="scrambled",
                sig1_fn=lambda w, sa=sigA: sa[w],
                sig2_fn=lambda w, sv=sig2: sv[w])
            per_unit["G2SA_%d" % s] = unit_record4(
                "G2SA", s, out_a, loop_a, "A")
            per_unit["G2SB_%d" % s] = unit_record4(
                "G2SB", s, out_b, loop_b, "B")
            print("PROGRESS", flush=True)
        if need("WA", s) or need("WB", s):
            fa_u, fb_u, wl_u = worlds_uniform[s]
            out_a, loop_a, out_b, loop_b, _, _ = run_bidi(
                fa_u, fb_u, wl_u, ch1="comm", ch2="comm", a_mode="comm",
                two_phase=False)
            per_unit["WA_%d" % s] = unit_record4(
                "WA", s, out_a, loop_a, "A")
            per_unit["WB_%d" % s] = unit_record4(
                "WB", s, out_b, loop_b, "B")
            print("PROGRESS", flush=True)
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump({"config": cfg, "per_unit": per_unit},
                      f, ensure_ascii=False, indent=1)

    # ---- 守卫 ----
    g232_ok, g232 = l2g.guard_d232()
    g235_ok, g235 = l2g.guard_d235()
    c2_ok, c2_detail = l2i.repro_cell2()
    c3_ok, c3_detail = repro_cell3()
    world_ok, world_oks = world_eq()
    pre_ok, pre_detail = precompute_ok_main()

    # ---- 跨单元核对（双侧） ----
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
    # G0A 侧聚合（双侧；结构基线）
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
    # W 臂诊断：信道在但 A 无信息 -> adopt_A(W)（应为 0）
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
        da_ = bidi_diag(ctxs, ga_E)
        db_ = bidi_diag(ctxs, gb_E)
        if da_["ratio"] is not None:
            eratios_a.append(da_["ratio"])
        if db_["ratio"] is not None:
            eratios_b.append(db_["ratio"])
    eratio_a_m = float(np.mean(eratios_a)) if eratios_a else 0.0
    eratio_b_m = float(np.mean(eratios_b)) if eratios_b else 0.0
    design_win_a_frac = float(np.mean([design_window_a(bidi_diag(
        [lb["ctx"] for lb in worlds_bidi[s][2]], per_unit["MGA_%d" % s]["E"]))
        for s in seeds]))
    design_win_b_frac = float(np.mean([design_window_b(bidi_diag(
        [lb["ctx"] for lb in worlds_bidi[s][2]], per_unit["MGB_%d" % s]["E"]))
        for s in seeds]))

    # ---- 判据（docs/271 §1.5 冻结：C1-C4 双侧化） ----
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
                 and c2_ok == 1 and c3_ok == 1 and world_ok == 1
                 and pre_ok == 1)
    if not guards_ok:
        verdict = "GUARD_FAIL"
        vnote = ("guard(s) failed: D232=%d, D235=%d, CONSTRUCTION=%d, "
                 "PREFIX_EQ_A=%d, PREFIX_EQ_B=%d, TWO_PHASE_EQ_A=%d, "
                 "TWO_PHASE_EQ_B=%d, REPRO_MAE_A=%d, REPRO_MAE_B=%d, "
                 "CELL2_REPRO=%d, CELL3_REPRO=%d, WORLD_EQ=%d, PRECOMPUTE=%d "
                 "-> implementation drift; fix implementation, do not judge "
                 "mechanism" % (g232_ok, g235_ok, construction_ok, prefix_ok_a,
                                prefix_ok_b, two_phase_ok_a, two_phase_ok_b,
                                repro_ok_a, repro_ok_b, c2_ok, c3_ok,
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
                 "closed-loop evidence (docs/271 sec 1.5)")
    elif not c2:
        if (adopt_a >= ADOPT_FRAC_MIN) != (adopt_b >= ADOPT_FRAC_MIN):
            verdict = "ONE_WAY"
            vnote = ("C2 fails with exactly one side adopting: adopt_A=%.4f, "
                     "adopt_B=%.4f -> one-way round-trip only, closed loop "
                     "not achieved; honest report of which side and why "
                     "(docs/271 sec 1.5)" % (adopt_a, adopt_b))
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
        vnote = "; ".join(why) + " (see R_L2J_CRIT* numbers)"

    # ---- 工件（自描述 JSON） ----
    out = {
        "artifact": "lang_comm_test4",
        "doc_ref": "docs/63, docs/228, docs/232, docs/235, docs/247, docs/258, "
                   "docs/264, docs/266, docs/268, docs/269, docs/270, docs/271",
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
                   "cell3_repro": {"ok": c3_ok},
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
    res_path = os.path.join(args.out_dir, "l2j_%s.json" % args.tag)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 摘要块：ASCII 标签，每行一个数字（顺序固定；无中文/日志/JSON） ----
    print("R_L2J_TAG=%s" % args.tag)
    print("R_L2J_SEEDS=%d" % len(seeds))
    print("R_L2J_FRAMES=%d" % N_FRAMES)
    print("R_L2J_WINDOWS=%d" % (N_FRAMES // WINDOW))
    print("R_L2J_NC=%d" % N_C)
    print("R_L2J_M1=%.1f" % B_M1)
    print("R_L2J_M2=%.1f" % B_M2)
    print("R_L2J_A1=%.1f" % A_M1)
    print("R_L2J_A2=%.1f" % A_M2)
    print("R_L2J_GATE_PURITY_MIN=%.2f" % GATE_PURITY_MIN)
    print("R_L2J_GUARD_D232=%d" % g232_ok)
    print("R_L2J_GUARD_D232_SC2=%s" % ",".join(str(v) for v in g232["sc2"]))
    print("R_L2J_GUARD_D232_SCLATE_FRAC=%.4f" % g232["sc_late_frac"])
    print("R_L2J_GUARD_D232_SC4=%.4f" % g232["sc4"])
    print("R_L2J_GUARD_D232_MAE=%.6f" % g232["mae"])
    print("R_L2J_GUARD_D232_MAE_SD=%.6f" % g232["mae_sd"])
    print("R_L2J_GUARD_D232_PIN=%.4f" % g232["pin"])
    print("R_L2J_GUARD_D232_CLASS=%s" % g232["cls"])
    print("R_L2J_GUARD_D235=%d" % g235_ok)
    for lv in (21, 22):
        d = g235[lv]
        print("R_L2J_GUARD_D235_C%d_OK=%d" % (lv, d["ok"]))
        print("R_L2J_GUARD_D235_C%d_MAE=%.6f" % (lv, d["mae"]))
        print("R_L2J_GUARD_D235_C%d_MAE_SD=%.6f" % (lv, d["mae_sd"]))
        print("R_L2J_GUARD_D235_C%d_SC2=%.4f" % (lv, d["sc2"]))
        print("R_L2J_GUARD_D235_C%d_SC2_SD=%.4f" % (lv, d["sc2_sd"]))
        print("R_L2J_GUARD_D235_C%d_COMP=%.4f" % (lv, d["comp"]))
        print("R_L2J_GUARD_D235_C%d_CHURN=%.4f" % (lv, d["churn"]))
        print("R_L2J_GUARD_D235_C%d_FID=%.4f" % (lv, d["fid"]))
    print("R_L2J_CONSTRUCTION=%d" % construction_ok)
    print("R_L2J_CONSTRUCTION_G0A=%d" % int(sum(cons_g0a)))
    print("R_L2J_CONSTRUCTION_G0B=%d" % int(sum(cons_g0b)))
    print("R_L2J_CONSTRUCTION_G1NA=%d" % int(sum(cons_g1na)))
    print("R_L2J_CONSTRUCTION_G1NB=%d" % int(sum(cons_g1nb)))
    print("R_L2J_PREFIX_EQ_A=%d" % prefix_ok_a)
    print("R_L2J_PREFIX_EQ_B=%d" % prefix_ok_b)
    print("R_L2J_TWO_PHASE_EQ_A=%d" % two_phase_ok_a)
    print("R_L2J_TWO_PHASE_EQ_B=%d" % two_phase_ok_b)
    print("R_L2J_REPRO_MAE_A=%d" % repro_ok_a)
    print("R_L2J_REPRO_MAE_B=%d" % repro_ok_b)
    print("R_L2J_CELL2_REPRO=%d" % c2_ok)
    print("R_L2J_CELL3_REPRO=%d" % c3_ok)
    for (name, ok, got) in c3_detail["checks"]:
        print("R_L2J_CELL3_%s=%d" % (name, ok))
        print("R_L2J_CELL3_%s_VAL=%s" % (name,
              (",".join(str(v) for v in got) if isinstance(got, list)
               else "%.6f" % got)))
    print("R_L2J_CELL3_ADOPT=%.4f" % c3_detail["adopt_frac"])
    print("R_L2J_CELL3_COMP_ADOPTED=%.4f" % c3_detail["comp_adopted"])
    print("R_L2J_CELL3_MAE=%.6f" % c3_detail["mae"])
    print("R_L2J_CELL3_JR_RATIO_MEAN=%.6f" % c3_detail["jr_ratio_mean"])
    print("R_L2J_CELL3_TRANSFER=%.4f" % c3_detail["transfer"])
    print("R_L2J_CELL3_CALIB=%.4f" % c3_detail["calib"])
    print("R_L2J_CELL3_FID=%.4f" % c3_detail["fid"])
    print("R_L2J_CELL3_A_SC2=%.4f" % c3_detail["a_sc2"])
    print("R_L2J_CELL3_G2S_SPURIOUS=%d" % len(c3_detail["g2_spurious"]))
    print("R_L2J_WORLD_EQ=%d" % world_ok)
    print("R_L2J_WORLD_EQ_SEEDS=%d" % int(sum(world_oks)))
    print("R_L2J_PRECOMPUTE=%d" % pre_ok)
    print("R_L2J_PRECOMPUTE_DW_A=%d" % pre_detail["dw_a"])
    print("R_L2J_PRECOMPUTE_DW_B=%d" % pre_detail["dw_b"])
    print("R_L2J_PRECOMPUTE_UNDER450_A=%d" % pre_detail["under_a"])
    print("R_L2J_PRECOMPUTE_UNDER450_B=%d" % pre_detail["under_b"])
    for r in seed_rows:
        s = r["seed"]
        print("R_L2J_SEED=%d" % s)
        for side, tag in (("A", "G0AA"), ("B", "G0AB")):
            g0f = r["g0a" if side == "A" else "g0b"]["finalize"]
            print("R_L2J_S%d_%s_G0_MAE=%.6f" % (s, side, g0f["mae_mean"]))
            print("R_L2J_S%d_%s_G0_SC2=%d" % (s, side, g0f["sc2"]))
            print("R_L2J_S%d_%s_G0_COMP=%.4f" % (s, side, g0f["compound_frac"]))
            print("R_L2J_S%d_%s_G0_CHURN=%.4f" % (s, side, g0f["churn_frac"]))
            print("R_L2J_S%d_%s_G0_PROMO=%d" % (s, side, g0f["n_promo"]))
        for side, key in (("A", "ta"), ("B", "tb")):
            mf = r[key]["finalize"]
            print("R_L2J_S%d_%s_M_MAE=%.6f" % (s, side, mf["mae_mean"]))
            print("R_L2J_S%d_%s_M_SC2=%d" % (s, side, mf["sc2"]))
            print("R_L2J_S%d_%s_M_COMP=%.4f" % (s, side, mf["compound_frac"]))
            print("R_L2J_S%d_%s_M_CHURN=%.4f" % (s, side, mf["churn_frac"]))
            print("R_L2J_S%d_%s_M_PROMO=%d" % (s, side, mf["n_promo"]))
            print("R_L2J_S%d_%s_M_FID=%.4f" % (s, side, mf["ctx_fidelity"]))
        for side, key in (("A", "g2a"), ("B", "g2b")):
            g2f = r[key]["finalize"]
            print("R_L2J_S%d_%s_G2S_COMP=%.4f" % (s, side, g2f["compound_frac"]))
            print("R_L2J_S%d_%s_G2S_PROMO=%d" % (s, side, g2f["n_promo"]))
        print("R_L2J_S%d_A_W_PROMO=%d" % (s, r["wa"]["finalize"]["n_promo"]))
        print("R_L2J_S%d_B_W_PROMO=%d" % (s, r["wb"]["finalize"]["n_promo"]))
        print("R_L2J_S%d_JR_A0=%.6f" % (s, r["jr_a0"]))
        print("R_L2J_S%d_JR_B0=%.6f" % (s, r["jr_b0"]))
        print("R_L2J_S%d_JR_A1=%.6f" % (s, r["jr_a1"]))
        print("R_L2J_S%d_JR_B1=%.6f" % (s, r["jr_b1"]))
        print("R_L2J_S%d_JR_RATIO_A=%.6f" % (s, r["jr_ratio_a"]))
        print("R_L2J_S%d_JR_RATIO_B=%.6f" % (s, r["jr_ratio_b"]))
        print("R_L2J_S%d_TRANSFER_A=%.6f" % (s, r["transfer_a"]))
        print("R_L2J_S%d_CALIB_A=%.6f" % (s, r["calib_a"]))
        print("R_L2J_S%d_TRANSFER_B=%.6f" % (s, r["transfer_b"]))
        print("R_L2J_S%d_CALIB_B=%.6f" % (s, r["calib_b"]))
        print("R_L2J_S%d_FIRST_PROMO_A=%s" % (s, ("NA" if r["first_promo_a"] is None
                                                  else str(r["first_promo_a"]))))
        print("R_L2J_S%d_FIRST_PROMO_B=%s" % (s, ("NA" if r["first_promo_b"] is None
                                                  else str(r["first_promo_b"]))))
        if r["gate_a"] is None:
            print("R_L2J_S%d_GATE_PURITY_A=NA" % s)
        else:
            print("R_L2J_S%d_GATE_PURITY_A=%.4f" % (s, r["gate_a"]["purity"]))
        if r["gate_b"] is None:
            print("R_L2J_S%d_GATE_PURITY_B=NA" % s)
        else:
            print("R_L2J_S%d_GATE_PURITY_B=%.4f" % (s, r["gate_b"]["purity"]))
    print("R_L2J_MAE_A=%.6f" % agg["A"]["mae_mean"])
    print("R_L2J_MAE_A_SD=%.6f" % agg["A"]["mae_sd"])
    print("R_L2J_MAE_B=%.6f" % agg["B"]["mae_mean"])
    print("R_L2J_MAE_B_SD=%.6f" % agg["B"]["mae_sd"])
    print("R_L2J_SC2_A=%.4f" % agg["A"]["sc2_mean"])
    print("R_L2J_SC2_B=%.4f" % agg["B"]["sc2_mean"])
    print("R_L2J_COMP_A=%.4f" % agg["A"]["comp_mean"])
    print("R_L2J_COMP_B=%.4f" % agg["B"]["comp_mean"])
    print("R_L2J_CHURN_A=%.4f" % agg["A"]["churn_mean"])
    print("R_L2J_CHURN_B=%.4f" % agg["B"]["churn_mean"])
    print("R_L2J_PROMO_A=%.4f" % agg["A"]["promo_mean"])
    print("R_L2J_PROMO_B=%.4f" % agg["B"]["promo_mean"])
    print("R_L2J_FID_A=%.4f" % fid_a_m)
    print("R_L2J_FID_B=%.4f" % fid_b_m)
    print("R_L2J_JR_A0=%.6f" % jr_a0_m)
    print("R_L2J_JR_B0=%.6f" % jr_b0_m)
    print("R_L2J_JR_A1=%.6f" % jr_a1_m)
    print("R_L2J_JR_B1=%.6f" % jr_b1_m)
    print("R_L2J_JR_RATIO_A=%.6f" % jr_ratio_a_m)
    print("R_L2J_JR_RATIO_B=%.6f" % jr_ratio_b_m)
    print("R_L2J_ADOPT_A=%.4f" % adopt_a)
    print("R_L2J_ADOPT_B=%.4f" % adopt_b)
    print("R_L2J_COMP_ADOPTED_A=%.4f" % comp_adopted_a)
    print("R_L2J_COMP_ADOPTED_B=%.4f" % comp_adopted_b)
    print("R_L2J_ADOPT_A_W=%.4f" % adopt_a_w)
    print("R_L2J_ADOPT_B_W=%.4f" % adopt_b_w)
    print("R_L2J_TRANSFER_A_MEAN=%.6f" % transfer_a_m)
    print("R_L2J_CALIB_A_MEAN=%.6f" % calib_a_m)
    print("R_L2J_TRANSFER_B_MEAN=%.6f" % transfer_b_m)
    print("R_L2J_CALIB_B_MEAN=%.6f" % calib_b_m)
    print("R_L2J_DIAG_ERATIO_A=%.4f" % eratio_a_m)
    print("R_L2J_DIAG_ERATIO_B=%.4f" % eratio_b_m)
    print("R_L2J_DESIGN_WIN_A_FRAC=%.4f" % design_win_a_frac)
    print("R_L2J_DESIGN_WIN_B_FRAC=%.4f" % design_win_b_frac)
    print("R_L2J_CRIT_C1A=%d" % c1a)
    print("R_L2J_CRIT_C1B=%d" % c1b)
    print("R_L2J_CRIT_C2=%d" % c2)
    print("R_L2J_CRIT_C3=%d" % c3)
    print("R_L2J_CRIT_C4=%d" % c4)
    print("R_L2J_CRIT_C4A=%d" % c4a)
    print("R_L2J_CRIT_C4B=%d" % c4b)
    print("R_L2J_CRIT1_JR_RATIO_A=%.6f" % jr_ratio_a_m)
    print("R_L2J_CRIT1_JR_RATIO_B=%.6f" % jr_ratio_b_m)
    print("R_L2J_CRIT2_ADOPT_A=%.4f" % adopt_a)
    print("R_L2J_CRIT2_ADOPT_B=%.4f" % adopt_b)
    print("R_L2J_CRIT2_COMP_ADOPTED_A=%.4f" % comp_adopted_a)
    print("R_L2J_CRIT2_COMP_ADOPTED_B=%.4f" % comp_adopted_b)
    print("R_L2J_CRIT2_G0A_ZERO=%d" % int(sum(cons_g0a)))
    print("R_L2J_CRIT2_G0B_ZERO=%d" % int(sum(cons_g0b)))
    print("R_L2J_CRIT2_G1NA_ZERO=%d" % int(sum(cons_g1na)))
    print("R_L2J_CRIT2_G1NB_ZERO=%d" % int(sum(cons_g1nb)))
    print("R_L2J_CRIT3_SPURIOUS_A=%d" % len(spurious_a))
    print("R_L2J_CRIT3_SPURIOUS_B=%d" % len(spurious_b))
    print("R_L2J_CRIT3_SPURIOUS_A_SEEDS=%s" % (",".join(str(v) for v in spurious_a)
                                               if spurious_a else "NONE"))
    print("R_L2J_CRIT3_SPURIOUS_B_SEEDS=%s" % (",".join(str(v) for v in spurious_b)
                                               if spurious_b else "NONE"))
    print("R_L2J_CRIT4_TRANSFER_A=%.6f" % transfer_a_m)
    print("R_L2J_CRIT4_CALIB_A=%.6f" % calib_a_m)
    print("R_L2J_CRIT4_TRANSFER_B=%.6f" % transfer_b_m)
    print("R_L2J_CRIT4_CALIB_B=%.6f" % calib_b_m)
    print("R_L2J_VERDICT=%s" % verdict)
    print("R_L2J_VERDICT_NOTE=%s" % vnote)
    print("R_L2J_ELAPSED=%.2f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
