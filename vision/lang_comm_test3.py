"""vision/lang_comm_test3.py — 语言线第三格：采纳门收紧（测"信息量"从能量分离升级为
语义对应；交付 docs/270）。

docs/270 §一 预注册冻结，运行后不改。机制基座 = docs/269 逐字（import lang_comm_test2/
lang_comm_test 复用，零改写）：环境 m1=2.6/m2=4.2、LangCommLoop 信道版、判据 C1-C4、
守卫、流、旋钮全沿用（同尺可比）。**唯一机制加法 = 提升条件③后加条件④：组内能量
纯度门（GATE_PURITY_MIN=0.80）**——提升触发时，账本两个合格 ctx 组的窗口能量各自 >=
80% 落在两中位中点 mid 的自身一侧（纯行为可计算，不读真值 ctx；docs/270 §1.1 冻结）。

LangCommGateLoop = LangCommLoop 逐字继承，唯一覆写 _maybe_promote：
  - 门开启（gate_purity_min=0.80）：门不过 -> 不提升（父条目继续累积，无回退）；
  - 门关闭（gate_purity_min=None）：逐字透传 super()._maybe_promote -> 与 docs/269
    机制逐位一致（R_L2I_CELL2_REPRO 复现锚：门关闭 (2.6,4.2) 四臂复现 docs/269 数字）。
预测路径/匹配/创建/退休/churn 会计/信道定义零改动（REPRO_MAE/PREFIX_EQ/TWO_PHASE_EQ
不破）。

流（docs/270 §1.7 逐字沿用 docs/269）：G1-T（双回路两阶段）、G1-G（单阶段）、C
（校准前缀 [0,140)）、G0（off）、G1n（null）、G2s（scrambled，(seed+5)%10，**判据臂
C5**）。同一世界种子的四臂共享同一 B 帧。

度量（§1.3 沿用）：M1 预测 MAE；M2 结构；M3 联合残差 JR；M4 信号质量诊断 + 门诊断
（每组纯度/门中位比/门通过首窗）；M5 信号留出归因（N_C=14）。判据（§1.4）：C1a JR
配对比值 <=0.85、C1b MAE(G1)==MAE(G0) abs<1e-9、C2 adopt_frac>=0.6 且采纳种子
compound>=0.5 且 G0/G1n 零采纳、C3 结构保持、C4 transfer>=0.10 且 >=0.5*calib_baseline
+ **C5 GATE_CLEANLINESS：spurious(G2s)==0（判据级）**。判定映射：COMM_EMERGES_CLEAN
（C1-C5 全过）/ COMM_EMERGES（C1-C4 过但 C5 残留）/ COMM_FLAT（含 COMM_NO_GAIN）/
PARTIAL / GUARD_FAIL / LANG_BLOCKED。

守卫（§1.5 沿用 + 新增）：R_L2I_GUARD_D232、R_L2I_GUARD_D235、R_L2I_CONSTRUCTION、
R_L2I_PREFIX_EQ、R_L2I_TWO_PHASE_EQ、R_L2I_REPRO_MAE、R_L2I_DETERM（timing/main
逐位一致，外部核对）、R_L2I_SMOKE（含门语义冒烟）、R_L2I_CELL2_REPRO（门关闭四臂
复现 docs/269：adopt 0.80/comp 1.0/JR 0.557933/transfer 0.79/calib 0.672/fid 0.9292/
逐种子 JR/首提升窗/G2s spurious 6/10）。

安全纪律（§1.10）：新文件仅本文件；stdout 只输出 ASCII 标签 + 每行一个数字的
R_L2I_* 摘要块；运行经 powershell 包装重定向到 logs/；数字用 vision/extract_r.py
抽取；禁止读日志/JSON 原文；本格不读 DAVIS。

用法：
  python vision/lang_comm_test3.py --smoke
  python vision/lang_comm_test3.py --tag timing
  python vision/lang_comm_test3.py --tag main
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
from critical_point import mean_sd, bootstrap_ci, JITTER, N_BOOT, BOOT_SEED
from stream_test import LOOP_CFG

warnings.filterwarnings("ignore")

DEFAULT_OUT = os.path.join("vision", "out", "results")

# ---------------- 冻结常量（docs/270 §1.1/§1.6；运行后不改） ----------------
# 环境/流/判据常量逐字沿用 docs/269（import l2h 引用）：
LVCODES = l2h.LVCODES
LV_WORLD = l2h.LV_WORLD
B_M1 = l2h.B_M1                 # 2.6（环境常量，docs/269 冻结）
B_M2 = l2h.B_M2                 # 4.2
N_C = l2h.N_C                   # 14
TRANSFER_FLOOR = l2h.TRANSFER_FLOOR
TRANSFER_REL = l2h.TRANSFER_REL
JR_RATIO_MAX = l2h.JR_RATIO_MAX
ADOPT_FRAC_MIN = l2h.ADOPT_FRAC_MIN
COMPOUND_MIN = l2h.COMPOUND_MIN
N_FRAMES = l2h.N_FRAMES
WINDOW = l2h.WINDOW
ENERGY_BINS = l2h.ENERGY_BINS
DESIGN_MIN_E = l2h.DESIGN_MIN_E
DESIGN_RATIO = l2h.DESIGN_RATIO
K_CONSIST = l2h.K_CONSIST

# 本格唯一新冻结量（docs/270 §1.1 冻结）：组内能量纯度门阈值（机制门参数）
GATE_PURITY_MIN = 0.80          # "4/5 语义一致"：每组窗口能量 >= 80% 落两中位中点自身一侧


# ---------------- 门机制（docs/270 §1.2：LangCommLoop 继承 + 条件④） ----------------
class LangCommGateLoop(l2g.LangCommLoop):
    """LangCommLoop 逐字继承；唯一覆写 _maybe_promote：门开启时条件④（组内能量纯度
    门）不过 -> 不提升；门关闭（gate_purity_min=None）-> 逐字透传 super()（与 docs/269
    机制逐位一致，R_L2I_CELL2_REPRO 证明）。c0/c1/账本/匹配/提升/回收/churn 会计与
    预测路径全部逐字继承。"""

    def __init__(self, gate_purity_min=None, **kw):
        super().__init__(**kw)
        self.gate_purity_min = gate_purity_min
        self.gate_attempts = []      # 门评估记录：[win, ratio, min_purity, ok]（门开启时）

    def _maybe_promote(self, s2, nd):
        """条件①-③（super 逐字）+ 条件④（本格新增）：
        门开启时先评估组内能量纯度；不过 -> return（不提升，父条目继续累积账本证据，
        后续窗口仍可触发——无回退、无降级，docs/235 退休语义逐字不变）。"""
        if self.gate_purity_min is not None:
            ok, ratio, purity = self._gate_eval(nd)
            self.gate_attempts.append([int(self._win), ratio, purity, int(ok)])
            if not ok:
                return
        super()._maybe_promote(s2, nd)

    def _gate_eval(self, nd):
        """条件④评估（机制可计算，只读账本 c2_ene = (ctx_B, 窗口事件能量) 记录；
        不读真值 ctx——真值只进评估统计/守卫）。返回 (门过, 中位比, 组内纯度 min)。
        门关闭（gate_purity_min=None）时返回透传（True）——与 docs/269 机制逐位一致。"""
        if self.gate_purity_min is None:
            return True, 0.0, 1.0
        c2ene = nd["c2_ene"]
        quals = {k: v for k, v in c2ene.items()
                 if k is not None and len(v) >= self.k_consist}
        if len(quals) < 2:
            return True, 0.0, 1.0          # 组数不足：提升本不会发生，门不干预
        meds = {k: float(np.median(v)) for k, v in quals.items()}
        ratio = max(meds.values()) / max(1e-9, min(meds.values()))
        mid = (min(meds.values()) + max(meds.values())) / 2.0
        lo_k = min(meds, key=meds.get)
        pur = []
        for k, v in quals.items():
            if k == lo_k:
                frac = sum(1.0 for e in v if e < mid) / len(v)
            else:
                frac = sum(1.0 for e in v if e >= mid) / len(v)
            pur.append(frac)
        pmin = float(min(pur))
        return (pmin >= self.gate_purity_min), ratio, pmin


# ---------------- 门版流运行（docs/270 §1.2；docs/269 run_dual/run_b_only 逐字 +
# loop_b 类换为 LangCommGateLoop，仅此一处差异；门关闭时逐位一致） ----------------
def run_dual_g(seed, frames_a, frames_b, win_labels, mode="comm",
               two_phase=False, n_c=N_C, window=WINDOW, want_end_snap=False,
               gate_purity_min=GATE_PURITY_MIN):
    """双回路（同 docs/269 run_dual 逐字；loop_b = LangCommGateLoop）。A/B 两回路帧
    同步连续步进；A 每闭一窗即发布 s_A(w)，B 于同一步闭窗时读取该信号。"""
    loop_a = l2g.LangCommLoop(mode="pixel", window=window, **LOOP_CFG)
    loop_b = LangCommGateLoop(mode=mode, window=window,
                              gate_purity_min=gate_purity_min, **LOOP_CFG)
    n_frames = len(frames_b)
    n_w = n_frames // window
    phases = ([(0, n_c * window), (n_c * window, n_frames)] if two_phase
              else [(0, n_frames)])
    snap = None
    a_closed = 0
    for (f0, f1) in phases:
        for k in range(f0, f1):
            prev_a = len(loop_a.sA_trace)
            loop_a.step(frames_a[k])
            if len(loop_a.sA_trace) > prev_a:
                loop_b.set_signal(loop_a.sA_trace[a_closed])
                a_closed += 1
            loop_b.step(frames_b[k])
        if two_phase and f0 == 0:
            snap = l2g.snapshot_b(loop_b)
    if want_end_snap and snap is None:
        snap = l2g.snapshot_b(loop_b)
    if len(loop_a._frame_buf):
        loop_a.finalize(n_w, None)
        if len(loop_a.sA_trace) > a_closed:
            loop_b.set_signal(loop_a.sA_trace[-1])
    out = loop_b.finalize(n_w, win_labels)
    return out, loop_b, loop_a, snap


def run_b_only_g(frames_b, win_labels, mode, signal_fn=None, window=WINDOW,
                 gate_purity_min=GATE_PURITY_MIN):
    """B 单回路（同 docs/269 run_b_only 逐字；loop = LangCommGateLoop）。"""
    loop_b = LangCommGateLoop(mode=mode, window=window,
                              gate_purity_min=gate_purity_min, **LOOP_CFG)
    n_frames = len(frames_b)
    n_w = n_frames // window
    closed = 0
    for k in range(n_frames):
        if k > 0 and len(loop_b._frame_buf) == 0:
            if mode == "off":
                loop_b.set_signal(None)
            elif mode == "null":
                loop_b.set_signal(l2g.NULL_SIGNAL)
            else:
                loop_b.set_signal(signal_fn(closed) if signal_fn is not None
                                  else None)
        prev = len(loop_b.sA_trace)
        loop_b.step(frames_b[k])
        if len(loop_b.sA_trace) > prev:
            closed += 1
    out = loop_b.finalize(n_w, win_labels)
    return out, loop_b


# ---------------- 单元记录（docs/270；= docs/269 unit_record2 + 门评估记录） ----------------
def unit_record3(arm, seed, out, loop_b, loop_a=None, snap=None, jr=None, att=None):
    rec = l2h.unit_record2(arm, seed, out, loop_b, loop_a=loop_a, snap=snap,
                           jr=jr, att=att)
    rec["gate"] = list(loop_b.gate_attempts)
    return rec


# ---------------- 构造冒烟（docs/270 §1.5-8；= docs/269 同款 + 门语义冒烟） ----------------
def smoke_main3():
    """构造冒烟：各信道模式构造运行正常；off 与 null 逐窗一致；G0 无提升；
    DEGENERATE_OK；归因不变量；baseline=0 不崩；**门语义单元测试**。"""
    results = {}
    fb = l2g._synth_frames(30)
    fa = l2g._synth_frames(30, y0=26)
    labels = [dict(ctx=1, b_mult=1.0, a_regime=None)] * 3
    outs = {}
    loops = {}
    for mode in ("off", "comm", "null", "scrambled"):
        if mode in ("comm", "scrambled"):
            out, loop_b, _, _ = run_dual_g(0, fa, fb, labels, mode=mode,
                                           two_phase=False, n_c=N_C)
        else:
            sig_fn = (lambda w: l2g.NULL_SIGNAL) if mode == "null" else None
            out, loop_b = run_b_only_g(fb, labels, mode, signal_fn=sig_fn)
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
    _, loop2, _, _ = run_dual_g(0, fa2, fb2, lab2, mode="comm",
                                two_phase=False, n_c=5)
    att = l2g.attribution(loop2, n_c=5)
    rates = [att["calib_baseline"], att["transfer_adopted_hit_rate"]]
    results["attr_invariants"] = int(
        all(0.0 <= r <= 1.0 for r in rates)
        and att["transfer_adopted_hits"] >= 0
        and att["n_heldout_eligible"] >= 0
        and att["transfer_adopted_hit_rate"] <= 1.0)
    _, loop3 = run_b_only_g(fb2, lab2, "off", signal_fn=None)
    att3 = l2g.attribution(loop3, n_c=5)
    results["baseline_zero_ok"] = int(att3["calib_baseline"] == 0.0
                                      and att3["transfer_adopted_hits"] == 0
                                      and att3["transfer_adopted_hit_rate"] == 0.0)
    # ---- 门语义单元测试（docs/270 §1.1 门定义；合成账本组，非数据） ----
    loop_on = LangCommGateLoop(mode="comm", gate_purity_min=GATE_PURITY_MIN,
                               window=10, **LOOP_CFG)
    loop_off = LangCommGateLoop(mode="comm", gate_purity_min=None,
                                window=10, **LOOP_CFG)
    pure_nd = {"c2_ene": {0: [500.0, 510.0, 520.0, 515.0],
                          1: [700.0, 710.0, 720.0, 705.0]}}
    mixed_nd = {"c2_ene": {0: [500.0, 510.0, 700.0, 520.0],
                           1: [720.0, 705.0, 710.0, 690.0]}}
    # 纯组：中位 512.5/707.5、比 1.38>=1.30、纯度 1.0/1.0 -> 门过
    ok_pure, r_pure, p_pure = loop_on._gate_eval(pure_nd)
    results["gate_pure_pass"] = int(ok_pure and p_pure >= GATE_PURITY_MIN)
    # 混合组：中位 515/707.5、比 1.37>=1.30（条件③会过）但组内纯度 0.75<0.80 -> 门拦
    ok_mix, r_mix, p_mix = loop_on._gate_eval(mixed_nd)
    results["gate_mixed_block"] = int((not ok_mix) and p_mix < GATE_PURITY_MIN)
    # 门关闭：透传（不评估、不记录）
    ok_off, r_off, p_off = loop_off._gate_eval(pure_nd)
    results["gate_off_passthrough"] = int(ok_off and len(loop_off.gate_attempts) == 0)
    for k in sorted(results):
        print("R_L2I_SMOKE_%s=%d" % (k.upper(), results[k]))
    return 0 if all(results.values()) else 1


# ---------------- R_L2I_CELL2_REPRO（docs/270 §1.5 新增复现锚） ----------------
# 期望数字 = docs/269 §三/§四 冻结值（本格环境 m1=2.6/m2=4.2，门关闭运行；同代码路径
# -> 期望位精确，容差取打印精度 + 余量）。来源行：docs/269 §3.1（逐种子 JR/首提升窗/
# 保真度）、§3.2（逐种子 transfer/calib）、§3.3（聚合 MAE/JR/adopt/comp）、§3.6（fid）。
CELL2_EXP = {
    "jr_g0": [0.1699, 0.1440, 0.1615, 0.1542, 0.1224, 0.1689, 0.1490,
              0.1856, 0.1518, 0.1485],
    "jr_g1": [0.0774, 0.0647, 0.0500, 0.0780, 0.1224, 0.0599, 0.0766,
              0.1856, 0.0856, 0.0633],
    "jr_ratio": [0.4554, 0.4495, 0.3097, 0.5059, 1.0000, 0.3546, 0.5139,
                 1.0000, 0.5644, 0.4261],
    "transfer": [1.000, 0.900, 1.000, 1.000, 0.000, 1.000, 1.000,
                 0.000, 1.000, 1.000],
    "first_promo": [6, 7, 7, 9, None, 6, 8, None, 10, 6],
    "fid": [1.0000, 0.9583, 1.0000, 0.9167, 0.8333, 1.0000, 0.9167,
            0.7500, 0.9583, 0.9583],
    "g2s_promo_seeds": [0, 1, 3, 5, 6, 8],          # docs/269 §3.1：G2s spurious 6/10
    "adopt_frac": 0.8000,
    "comp_adopted": 1.0000,
    "mae": 0.023023,
    "mae_sd": 0.002179,
    "jr_ratio_mean": 0.557933,
    "transfer_mean": 0.790,
    "calib_mean": 0.672,
    "fid_mean": 0.9292,
}


def repro_cell2():
    """门关闭（gate_purity_min=None）在本格环境 (2.6, 4.2)：G0/G1G/G1T/G2S 四臂
    10 种子，与 docs/269 §三/§四 冻结数字逐位一致（浮点容差 1e-4/1e-3 按打印精度；
    离散精确）。返回 (ok, detail)。"""
    seeds = list(range(10))
    jr_ratios, jr_g0s, jr_g1s = [], [], []
    mae_g0s, mae_g1s, fids = [], [], []
    trans, calibs = [], []
    first_promos, g2_promos, compounds = [], [], []
    for s in seeds:
        fa, fb, wl = l2g.make_world(s, m1=B_M1, m2=B_M2)
        out_t, loop_t, _, _ = run_dual_g(s, fa, fb, wl, mode="comm",
                                         two_phase=True, n_c=N_C,
                                         gate_purity_min=None)
        jr1 = l2g.jr_b(loop_t)[0]
        att = l2g.attribution(loop_t, N_C)
        out_g, loop_g, _, _ = run_dual_g(s, fa, fb, wl, mode="comm",
                                         two_phase=False, gate_purity_min=None)
        out0, loop0 = run_b_only_g(fb, wl, "off", signal_fn=None,
                                   gate_purity_min=None)
        jr0 = l2g.jr_b(loop0)[0]
        other = (s + 5) % 10
        a_other = l2g.run_a_signal(l2g.make_world(other, m1=B_M1, m2=B_M2)[0])
        out2, loop2 = run_b_only_g(fb, wl, "scrambled",
                                   signal_fn=lambda w: a_other.sA_trace[w],
                                   gate_purity_min=None)
        jr_ratios.append(jr1 / max(jr0, 1e-12))
        jr_g0s.append(jr0)
        jr_g1s.append(jr1)
        mae_g0s.append(out0["mae_mean"])
        mae_g1s.append(out_t["mae_mean"])
        fids.append(out_t["ctx_fidelity"])
        trans.append(att["transfer_adopted_hit_rate"])
        calibs.append(att["calib_baseline"])
        first_promos.append(att["first_promo_win"])
        g2_promos.append(out2["n_promo"])
        compounds.append(out_t["compound_frac"])
    adopt_frac = float(np.mean([fp is not None for fp in first_promos]))
    adopted_comp = [c for s, c in enumerate(compounds)
                    if first_promos[s] is not None]
    comp_adopted = float(np.mean(adopted_comp)) if adopted_comp else 0.0
    mae_m, mae_sd = mean_sd(mae_g0s)
    jr_ratio_m, _ = mean_sd(jr_ratios)
    trans_m, _ = mean_sd(trans)
    calib_m, _ = mean_sd(calibs)
    fid_m, _ = mean_sd(fids)
    g2_spur = [s for s in seeds if g2_promos[s] >= 1]

    def chk(name, got, exp, tol=1e-4):
        return name, int(abs(got - exp) < tol), got

    checks = []
    for s in seeds:
        checks.append(chk("JR_RATIO_S%d" % s, jr_ratios[s],
                          CELL2_EXP["jr_ratio"][s], 1e-4))
        checks.append(chk("JR_G0_S%d" % s, jr_g0s[s], CELL2_EXP["jr_g0"][s], 1e-4))
        checks.append(chk("JR_G1_S%d" % s, jr_g1s[s], CELL2_EXP["jr_g1"][s], 1e-4))
        checks.append(chk("TRANSFER_S%d" % s, trans[s], CELL2_EXP["transfer"][s], 1e-3))
        checks.append(chk("FID_S%d" % s, fids[s], CELL2_EXP["fid"][s], 1e-4))
        exp_fp = CELL2_EXP["first_promo"][s]
        checks.append(("FIRST_PROMO_S%d" % s,
                       int(first_promos[s] == exp_fp),
                       (first_promos[s] if first_promos[s] is not None else -1)))
    checks.append(chk("ADOPT_FRAC", adopt_frac, CELL2_EXP["adopt_frac"], 1e-6))
    checks.append(chk("COMP_ADOPTED", comp_adopted, CELL2_EXP["comp_adopted"], 1e-4))
    checks.append(chk("MAE_G0", mae_m, CELL2_EXP["mae"], 1e-5))
    checks.append(chk("MAE_SD", mae_sd, CELL2_EXP["mae_sd"], 1e-5))
    checks.append(chk("JR_RATIO_MEAN", jr_ratio_m, CELL2_EXP["jr_ratio_mean"], 1e-4))
    checks.append(chk("TRANSFER_MEAN", trans_m, CELL2_EXP["transfer_mean"], 1e-3))
    checks.append(chk("CALIB_MEAN", calib_m, CELL2_EXP["calib_mean"], 1e-3))
    checks.append(chk("FID_MEAN", fid_m, CELL2_EXP["fid_mean"], 1e-4))
    checks.append(("G2S_SPURIOUS", int(g2_spur == CELL2_EXP["g2s_promo_seeds"]),
                   g2_spur))
    ok = int(all(c[1] == 1 for c in checks))
    return ok, dict(checks=checks, adopt_frac=adopt_frac,
                    comp_adopted=comp_adopted, mae=mae_m, mae_sd=mae_sd,
                    jr_ratio_mean=jr_ratio_m, transfer=trans_m,
                    calib=calib_m, fid=fid_m, g2_spurious=g2_spur)


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="l2i")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.smoke:
        return smoke_main3()
    t0 = time.time()
    seeds = list(range(10))

    cfg = {"tag": args.tag, "n_seeds": len(seeds), "frames": N_FRAMES,
           "window": WINDOW, "n_c": N_C, "jitter": JITTER,
           "b_m1": B_M1, "b_m2": B_M2,
           "noise_sigma": l2g.NOISE_SIGMA,
           "world": {"a_center": list(l2g.A_CENTER), "a_orbit": l2g.A_ORBIT,
                     "a_freq": l2g.A_FREQ, "b_center": list(l2g.B_CENTER),
                     "b_orbit": l2g.B_ORBIT, "b_freq": l2g.B_FREQ,
                     "rng_lvcode": LV_WORLD},
           "channel": {"sparse_px": l2g.SIG_SPARSE_PX,
                       "null_signal": l2g.NULL_SIGNAL},
           "gate": {"purity_min": GATE_PURITY_MIN},       # 本格唯一机制加法参数
           "criteria": {"jr_ratio_max": JR_RATIO_MAX,
                        "adopt_frac_min": ADOPT_FRAC_MIN,
                        "compound_min": COMPOUND_MIN,
                        "transfer_floor": TRANSFER_FLOOR,
                        "transfer_rel": TRANSFER_REL},
           "loop": LOOP_CFG}
    ck_tag = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    ckpt_path = os.path.join(args.out_dir, "ckpt_l2i_%s.json" % ck_tag)

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
            out, loop_b, loop_a, snap = run_dual_g(s, fa, fb, wl, mode="comm",
                                                   two_phase=True, n_c=N_C)
            per_unit["G1T_%d" % s] = unit_record3(
                "G1T", s, out, loop_b, loop_a=loop_a, snap=snap,
                jr=l2g.jr_b(loop_b), att=l2g.attribution(loop_b, N_C))
            print("PROGRESS", flush=True)
        if need("G1G", s):
            out, loop_b, loop_a, _ = run_dual_g(s, fa, fb, wl, mode="comm",
                                                two_phase=False)
            per_unit["G1G_%d" % s] = unit_record3(
                "G1G", s, out, loop_b, loop_a=loop_a)
            print("PROGRESS", flush=True)
        if need("C", s):
            out, loop_b, _, snap = run_dual_g(s, fa[:140], fb[:140], wl[:14],
                                              mode="comm", two_phase=False,
                                              want_end_snap=True)
            per_unit["C_%d" % s] = unit_record3("C", s, out, loop_b, snap=snap)
            print("PROGRESS", flush=True)
        if need("G0", s):
            out, loop_b = run_b_only_g(fb, wl, "off", signal_fn=None)
            per_unit["G0_%d" % s] = unit_record3("G0", s, out, loop_b,
                                                 jr=l2g.jr_b(loop_b))
            print("PROGRESS", flush=True)
        if need("G1N", s):
            out, loop_b = run_b_only_g(fb, wl, "null", signal_fn=None)
            per_unit["G1N_%d" % s] = unit_record3("G1N", s, out, loop_b)
            print("PROGRESS", flush=True)
        if need("G2S", s):
            other = (s + 5) % 10
            a_other = l2g.run_a_signal(worlds[other][0])
            out, loop_b = run_b_only_g(fb, wl, "scrambled",
                                       signal_fn=lambda w: a_other.sA_trace[w])
            per_unit["G2S_%d" % s] = unit_record3("G2S", s, out, loop_b)
            print("PROGRESS", flush=True)
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump({"config": cfg, "per_unit": per_unit},
                      f, ensure_ascii=False, indent=1)

    # ---- 守卫 ----
    g232_ok, g232 = l2g.guard_d232()
    g235_ok, g235 = l2g.guard_d235()
    c2_ok, c2_detail = repro_cell2()

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
        # 门诊断：首提升窗处的门评估记录（[win, ratio, min_purity, ok]）
        gate_info = None
        if att["first_promo_win"] is not None:
            for rec in t["gate"]:
                if rec[0] == att["first_promo_win"]:
                    gate_info = dict(win=rec[0], ratio=rec[1], purity=rec[2],
                                     ok=rec[3])
                    break
        seed_rows.append(dict(
            seed=s, g0=g0, g1t=t, g1g=g, c=c, g1n=gn, g2s=g2,
            jr_g0=jr_g0, jr_g1=jr_g1, jr_ratio=jr_ratio,
            transfer_rate=att["transfer_adopted_hit_rate"],
            calib_baseline=att["calib_baseline"],
            transfer_hits=att["transfer_adopted_hits"],
            held_elig=att["n_heldout_eligible"],
            first_promo=att["first_promo_win"],
            promo_wins=att["promo_wins"],
            flip=fd, energy=ed, gate=gate_info,
            design_win=l2h.design_window(dict(
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

    # ---- 判据（docs/270 §1.4 冻结：C1-C4 docs/269 逐字 + C5 新增） ----
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
    g2_spurious = [r["seed"] for r in seed_rows
                   if r["g2s"]["finalize"]["n_promo"] >= 1]
    c5 = int(len(g2_spurious) == 0)

    # 数据可用性（LANG_BLOCKED 预防）：逐种子 JR 有窗口、留出 eligible 非空
    blocked = int(not (all(r["g0"]["jr"][1] >= 1 for r in seed_rows)
                       and all(r["g1t"]["jr"][1] >= 1 for r in seed_rows)
                       and all(r["held_elig"] >= 1 for r in seed_rows)
                       and all(r["transfer_rate"] == r["transfer_rate"]
                               for r in seed_rows)))

    guards_ok = (g232_ok == 1 and g235_ok == 1 and construction_ok == 1
                 and prefix_ok == 1 and two_phase_ok == 1 and repro_ok == 1
                 and c2_ok == 1)
    if not guards_ok:
        verdict = "GUARD_FAIL"
        vnote = ("guard(s) failed: D232=%d, D235=%d, CONSTRUCTION=%d, "
                 "PREFIX_EQ=%d, TWO_PHASE_EQ=%d, REPRO_MAE=%d, CELL2_REPRO=%d "
                 "-> implementation drift; fix implementation, do not judge "
                 "mechanism" % (g232_ok, g235_ok, construction_ok, prefix_ok,
                                two_phase_ok, repro_ok, c2_ok))
    elif blocked:
        verdict = "LANG_BLOCKED"
        vnote = ("synthetic environment unavailable (per-seed eligible/JR "
                 "windows missing); see per-seed numbers")
    elif c1a and c1b and c2 and c3 and c4 and c5:
        verdict = "COMM_EMERGES_CLEAN"
        vnote = ("criteria C1-C5 all pass and all guards pass: other-agent "
                 "signal spontaneously adopted (ledger-driven, no switch) and "
                 "adoption lowers joint residual; structural behavior kept; "
                 "holdout transfer holds; G2s (scrambled) zero spurious "
                 "-> information criterion upgraded from energy-separation "
                 "proxy to semantic correspondence (docs/270 sec 1.4)")
    elif c1a and c1b and c2 and c3 and c4:
        verdict = "COMM_EMERGES"
        vnote = ("criteria C1-C4 pass but C5 fails: spurious(G2s)=%d > 0 -> "
                 "positive evidence with residual false positives; honest "
                 "report, gate threshold not retuned" % len(g2_spurious))
    elif not c2:
        verdict = "COMM_FLAT"
        vnote = ("C2 fails (signal not adopted): adopt_frac=%.4f < 0.6 or "
                 "G0/G1n non-zero adoption -> honest negative: no information "
                 "gain or channel ineffective or gate over-blocking; see "
                 "decomposition" % adopt_frac)
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
        vnote = "; ".join(why) + " (see R_L2I_CRIT* numbers)"

    # ---- 工件（自描述 JSON） ----
    out = {
        "artifact": "lang_comm_test3",
        "doc_ref": "docs/63, docs/228, docs/232, docs/235, docs/247, docs/258, "
                   "docs/264, docs/266, docs/268, docs/269, docs/270",
        "config": cfg,
        "guards": {"d232": {"ok": g232_ok, "detail": g232},
                   "d235": {"ok": g235_ok, "detail": g235},
                   "construction": {"ok": construction_ok,
                                    "g0_zero": int(sum(cons_g0)),
                                    "g1n_zero": int(sum(cons_g1n))},
                   "prefix_eq": prefix_ok, "two_phase_eq": two_phase_ok,
                   "repro_mae": repro_ok, "cell2_repro": {"ok": c2_ok},
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
                 "design_win_frac": design_win_frac,
                 "gate_purity_min": GATE_PURITY_MIN,
                 "gate_purity_at_promo": [r["gate"]["purity"]
                                          if r["gate"] else None
                                          for r in seed_rows],
                 "gate_ratio_at_promo": [r["gate"]["ratio"]
                                         if r["gate"] else None
                                         for r in seed_rows]},
        "criteria": {"c1a_comm_value": c1a, "c1b_mae_eq": c1b,
                     "c2_adoption_emerges": c2, "c3_structure_keep": c3,
                     "c4_sig_holdout": c4,
                     "c5_gate_cleanliness": c5,
                     "jr_ratio": jr_ratio_m, "adopt_frac": adopt_frac,
                     "comp_adopted": comp_adopted, "transfer_rate": transfer_m,
                     "calib_baseline": calib_m,
                     "c3_g0": c3_g0, "c3_g1": c3_g1, "c3_a": c3_a,
                     "g2_spurious_seeds": g2_spurious,
                     "g2_spurious_count": len(g2_spurious)},
        "verdict": {"verdict": verdict, "note": vnote},
        "timing": {"elapsed_sec": round(time.time() - t0, 2)},
    }
    res_path = os.path.join(args.out_dir, "l2i_%s.json" % args.tag)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 摘要块：ASCII 标签，每行一个数字（顺序固定；无中文/日志/JSON） ----
    print("R_L2I_TAG=%s" % args.tag)
    print("R_L2I_SEEDS=%d" % len(seeds))
    print("R_L2I_FRAMES=%d" % N_FRAMES)
    print("R_L2I_WINDOWS=%d" % (N_FRAMES // WINDOW))
    print("R_L2I_NC=%d" % N_C)
    print("R_L2I_M1=%.1f" % B_M1)
    print("R_L2I_M2=%.1f" % B_M2)
    print("R_L2I_GATE_PURITY_MIN=%.2f" % GATE_PURITY_MIN)
    print("R_L2I_GUARD_D232=%d" % g232_ok)
    print("R_L2I_GUARD_D232_SC2=%s" % ",".join(str(v) for v in g232["sc2"]))
    print("R_L2I_GUARD_D232_SCLATE_FRAC=%.4f" % g232["sc_late_frac"])
    print("R_L2I_GUARD_D232_SC4=%.4f" % g232["sc4"])
    print("R_L2I_GUARD_D232_MAE=%.6f" % g232["mae"])
    print("R_L2I_GUARD_D232_MAE_SD=%.6f" % g232["mae_sd"])
    print("R_L2I_GUARD_D232_PIN=%.4f" % g232["pin"])
    print("R_L2I_GUARD_D232_CLASS=%s" % g232["cls"])
    print("R_L2I_GUARD_D235=%d" % g235_ok)
    for lv in (21, 22):
        d = g235[lv]
        print("R_L2I_GUARD_D235_C%d_OK=%d" % (lv, d["ok"]))
        print("R_L2I_GUARD_D235_C%d_MAE=%.6f" % (lv, d["mae"]))
        print("R_L2I_GUARD_D235_C%d_MAE_SD=%.6f" % (lv, d["mae_sd"]))
        print("R_L2I_GUARD_D235_C%d_SC2=%.4f" % (lv, d["sc2"]))
        print("R_L2I_GUARD_D235_C%d_SC2_SD=%.4f" % (lv, d["sc2_sd"]))
        print("R_L2I_GUARD_D235_C%d_COMP=%.4f" % (lv, d["comp"]))
        print("R_L2I_GUARD_D235_C%d_CHURN=%.4f" % (lv, d["churn"]))
        print("R_L2I_GUARD_D235_C%d_FID=%.4f" % (lv, d["fid"]))
    print("R_L2I_CONSTRUCTION=%d" % construction_ok)
    print("R_L2I_CONSTRUCTION_G0=%d" % int(sum(cons_g0)))
    print("R_L2I_CONSTRUCTION_G1N=%d" % int(sum(cons_g1n)))
    print("R_L2I_PREFIX_EQ=%d" % prefix_ok)
    print("R_L2I_TWO_PHASE_EQ=%d" % two_phase_ok)
    print("R_L2I_REPRO_MAE=%d" % repro_ok)
    print("R_L2I_REPRO_MAE_SEEDS=%d" % int(sum(repro_oks)))
    print("R_L2I_CELL2_REPRO=%d" % c2_ok)
    for (name, ok, got) in c2_detail["checks"]:
        print("R_L2I_CELL2_%s=%d" % (name, ok))
        print("R_L2I_CELL2_%s_VAL=%s" % (name,
              (",".join(str(v) for v in got) if isinstance(got, list)
               else "%.6f" % got)))
    print("R_L2I_CELL2_ADOPT=%.4f" % c2_detail["adopt_frac"])
    print("R_L2I_CELL2_COMP_ADOPTED=%.4f" % c2_detail["comp_adopted"])
    print("R_L2I_CELL2_MAE=%.6f" % c2_detail["mae"])
    print("R_L2I_CELL2_TRANSFER=%.4f" % c2_detail["transfer"])
    print("R_L2I_CELL2_CALIB=%.4f" % c2_detail["calib"])
    print("R_L2I_CELL2_FID=%.4f" % c2_detail["fid"])
    for r in seed_rows:
        s = r["seed"]
        print("R_L2I_SEED=%d" % s)
        g0f = r["g0"]["finalize"]
        g1f = r["g1t"]["finalize"]
        gnf = r["g1n"]["finalize"]
        g2f = r["g2s"]["finalize"]
        print("R_L2I_S%d_G0_MAE=%.6f" % (s, g0f["mae_mean"]))
        print("R_L2I_S%d_G0_SC2=%d" % (s, g0f["sc2"]))
        print("R_L2I_S%d_G0_COMP=%.4f" % (s, g0f["compound_frac"]))
        print("R_L2I_S%d_G0_CHURN=%.4f" % (s, g0f["churn_frac"]))
        print("R_L2I_S%d_G0_PROMO=%d" % (s, g0f["n_promo"]))
        print("R_L2I_S%d_G0_RATIO=%.6f" % (s, r["g0"]["ratio"]))
        print("R_L2I_S%d_G1_MAE=%.6f" % (s, g1f["mae_mean"]))
        print("R_L2I_S%d_G1_SC2=%d" % (s, g1f["sc2"]))
        print("R_L2I_S%d_G1_COMP=%.4f" % (s, g1f["compound_frac"]))
        print("R_L2I_S%d_G1_CHURN=%.4f" % (s, g1f["churn_frac"]))
        print("R_L2I_S%d_G1_PROMO=%d" % (s, g1f["n_promo"]))
        print("R_L2I_S%d_G1_RATIO=%.6f" % (s, r["g1g"]["ratio"]))
        print("R_L2I_S%d_G1_FID=%.4f" % (s, g1f["ctx_fidelity"]))
        print("R_L2I_S%d_G1N_COMP=%.4f" % (s, gnf["compound_frac"]))
        print("R_L2I_S%d_G1N_PROMO=%d" % (s, gnf["n_promo"]))
        print("R_L2I_S%d_G2S_COMP=%.4f" % (s, g2f["compound_frac"]))
        print("R_L2I_S%d_G2S_PROMO=%d" % (s, g2f["n_promo"]))
        print("R_L2I_S%d_A_SC2=%s" % (s, ("NA" if r["g1t"]["a_sc2"] is None
                                          else str(r["g1t"]["a_sc2"]))))
        print("R_L2I_S%d_JR_G0=%.6f" % (s, r["jr_g0"]))
        print("R_L2I_S%d_JR_G1=%.6f" % (s, r["jr_g1"]))
        print("R_L2I_S%d_JR_RATIO=%.6f" % (s, r["jr_ratio"]))
        print("R_L2I_S%d_TRANSFER_RATE=%.6f" % (s, r["transfer_rate"]))
        print("R_L2I_S%d_CALIB_BASE=%.6f" % (s, r["calib_baseline"]))
        print("R_L2I_S%d_TRANSFER_HITS=%d" % (s, r["transfer_hits"]))
        print("R_L2I_S%d_HELD_ELIG=%d" % (s, r["held_elig"]))
        print("R_L2I_S%d_FIRST_PROMO=%s" % (s, ("NA" if r["first_promo"] is None
                                                else str(r["first_promo"]))))
        print("R_L2I_S%d_DESIGN_WIN=%d" % (s, r["design_win"]))
        ginfo = r["gate"]
        if ginfo is None:
            print("R_L2I_S%d_GATE_PURITY=NA" % s)
            print("R_L2I_S%d_GATE_RATIO=NA" % s)
        else:
            print("R_L2I_S%d_GATE_PURITY=%.4f" % (s, ginfo["purity"]))
            print("R_L2I_S%d_GATE_RATIO=%.4f" % (s, ginfo["ratio"]))
        ed = r["energy"]
        print("R_L2I_S%d_DIAG_E0_MED=%s" % (s, ("NA" if ed["med0"] is None
                                                else "%.1f" % ed["med0"])))
        print("R_L2I_S%d_DIAG_E1_MED=%s" % (s, ("NA" if ed["med1"] is None
                                                else "%.1f" % ed["med1"])))
        print("R_L2I_S%d_DIAG_EBAND0=%s" % (s, ("NA" if ed["band0"] is None
                                                else str(ed["band0"]))))
        print("R_L2I_S%d_DIAG_EBAND1=%s" % (s, ("NA" if ed["band1"] is None
                                                else str(ed["band1"]))))
        print("R_L2I_S%d_DIAG_ERATIO=%s" % (s, ("NA" if ed["ratio"] is None
                                                else "%.4f" % ed["ratio"])))
        print("R_L2I_S%d_DIAG_FLIPS=%d" % (s, r["flip"]["flips"]))
        print("R_L2I_S%d_DIAG_SEGOK=%d" % (s, r["flip"]["seg_ok"]))
    print("R_L2I_MAE_G0=%.6f" % agg["G0"]["mae_mean"])
    print("R_L2I_MAE_G0_SD=%.6f" % agg["G0"]["mae_sd"])
    print("R_L2I_MAE_G1=%.6f" % agg["G1"]["mae_mean"])
    print("R_L2I_MAE_G1_SD=%.6f" % agg["G1"]["mae_sd"])
    print("R_L2I_MAE_G1N=%.6f" % agg["G1N"]["mae_mean"])
    print("R_L2I_MAE_G1N_SD=%.6f" % agg["G1N"]["mae_sd"])
    print("R_L2I_MAE_G2S=%.6f" % agg["G2S"]["mae_mean"])
    print("R_L2I_MAE_G2S_SD=%.6f" % agg["G2S"]["mae_sd"])
    print("R_L2I_SC2_G0=%.4f" % agg["G0"]["sc2_mean"])
    print("R_L2I_SC2_G1=%.4f" % agg["G1"]["sc2_mean"])
    print("R_L2I_SC2_G1N=%.4f" % agg["G1N"]["sc2_mean"])
    print("R_L2I_SC2_G2S=%.4f" % agg["G2S"]["sc2_mean"])
    print("R_L2I_COMP_G0=%.4f" % agg["G0"]["comp_mean"])
    print("R_L2I_COMP_G1=%.4f" % agg["G1"]["comp_mean"])
    print("R_L2I_COMP_G1_SD=%.4f" % agg["G1"]["comp_sd"])
    print("R_L2I_COMP_G1N=%.4f" % agg["G1N"]["comp_mean"])
    print("R_L2I_COMP_G2S=%.4f" % agg["G2S"]["comp_mean"])
    print("R_L2I_CHURN_G0=%.4f" % agg["G0"]["churn_mean"])
    print("R_L2I_CHURN_G1=%.4f" % agg["G1"]["churn_mean"])
    print("R_L2I_PROMO_G1=%.4f" % agg["G1"]["promo_mean"])
    print("R_L2I_FID_G1=%.4f" % agg["G1"]["fid_mean"])
    print("R_L2I_MAE_CI95_G1_LO=%.6f" % agg["G1"]["mae_ci95"][0])
    print("R_L2I_MAE_CI95_G1_HI=%.6f" % agg["G1"]["mae_ci95"][1])
    print("R_L2I_COMP_CI95_G1_LO=%.4f" % agg["G1"]["comp_ci95"][0])
    print("R_L2I_COMP_CI95_G1_HI=%.4f" % agg["G1"]["comp_ci95"][1])
    print("R_L2I_JR_G0=%.6f" % jr_g0_m)
    print("R_L2I_JR_G0_SD=%.6f" % jr_g0_sd)
    print("R_L2I_JR_G1=%.6f" % jr_g1_m)
    print("R_L2I_JR_G1_SD=%.6f" % jr_g1_sd)
    print("R_L2I_JR_RATIO=%.6f" % jr_ratio_m)
    print("R_L2I_JR_RATIO_SD=%.6f" % jr_ratio_sd)
    print("R_L2I_ADOPT_FRAC=%.4f" % adopt_frac)
    print("R_L2I_COMP_ADOPTED=%.4f" % comp_adopted)
    print("R_L2I_A_SC2_MEAN=%.4f" % a_sc2_m)
    print("R_L2I_TRANSFER_RATE_MEAN=%.6f" % transfer_m)
    print("R_L2I_TRANSFER_RATE_SD=%.6f" % transfer_sd)
    print("R_L2I_CALIB_BASE_MEAN=%.6f" % calib_m)
    print("R_L2I_DIAG_CTXFID=%.4f" % fid_m)
    print("R_L2I_DIAG_ERATIO_MEAN=%.4f" % eratio_m)
    print("R_L2I_DIAG_BAND0_FRAC=%.4f" % band0_ok)
    print("R_L2I_DIAG_BAND1_FRAC=%.4f" % band1_ok)
    print("R_L2I_DIAG_SEGOK_FRAC=%.4f" % seg_ok_frac)
    print("R_L2I_DIAG_FLIPS_MEAN=%.2f" % flips_m)
    print("R_L2I_DESIGN_WIN_FRAC=%.4f" % design_win_frac)
    print("R_L2I_CRIT_C1A=%d" % c1a)
    print("R_L2I_CRIT_C1B=%d" % c1b)
    print("R_L2I_CRIT_C2=%d" % c2)
    print("R_L2I_CRIT_C3=%d" % c3)
    print("R_L2I_CRIT_C4=%d" % c4)
    print("R_L2I_CRIT_C5=%d" % c5)
    print("R_L2I_CRIT_C3_G0=%d" % c3_g0)
    print("R_L2I_CRIT_C3_G1=%d" % c3_g1)
    print("R_L2I_CRIT_C3_A=%d" % c3_a)
    print("R_L2I_CRIT1_JR_RATIO=%.6f" % jr_ratio_m)
    print("R_L2I_CRIT2_ADOPT_FRAC=%.4f" % adopt_frac)
    print("R_L2I_CRIT2_COMP_ADOPTED=%.4f" % comp_adopted)
    print("R_L2I_CRIT2_G0_ZERO=%d" % int(sum(cons_g0)))
    print("R_L2I_CRIT2_G1N_ZERO=%d" % int(sum(cons_g1n)))
    print("R_L2I_CRIT4_TRANSFER_RATE=%.6f" % transfer_m)
    print("R_L2I_CRIT4_CALIB_BASE=%.6f" % calib_m)
    print("R_L2I_CRIT4_FLOOR_OK=%d" % int(transfer_m >= TRANSFER_FLOOR))
    print("R_L2I_CRIT4_REL_OK=%d" % int(transfer_m >= TRANSFER_REL * calib_m))
    print("R_L2I_SPURIOUS_G2S=%d" % len(g2_spurious))
    print("R_L2I_SPURIOUS_G2S_SEEDS=%s" % (",".join(str(v) for v in g2_spurious)
                                           if g2_spurious else "NONE"))
    print("R_L2I_VERDICT=%s" % verdict)
    print("R_L2I_VERDICT_NOTE=%s" % vnote)
    print("R_L2I_ELAPSED=%.2f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
