"""vision/lang_comm_test7.py — 语言线第七格：JR 采纳子集口径 + spurious 治理联合
（攻 docs/274 两个缺口：JR_B 贴线 0.004 的均值稀释 + C3 spurious 残留；交付 docs/275）。

docs/275 §一 预注册冻结，运行后不改。机制基座 = docs/274 逐字继承（import
lang_comm_test6/lang_comm_test5/lang_comm_test4/lang_comm_test3/lang_comm_test2/
lang_comm_test 复用，零改写）：run_bidi6（frame_sync="stagger5" 半窗相位错位并行双
回路）、BiLangLoop（双向信道 CH1+CH2、对称世界 a1=1.6/a2=2.6、门条件④
GATE_PURITY_MIN=0.80 双回路）、环境 m1=2.6/m2=4.2、判据阈值、留出划分（N_C=14）全沿用
（七格同尺可比）。**本格唯一机制改动 = (a) JR 测量口径 + (b) spurious 治理**：

1. **JR 采纳子集口径（§1.1 冻结）**：JR 配对比值 = mean over 采纳种子（该侧该种子
   n_promo_confirmed >= 1）of JR_X(M)/JR_X(G0A)——分母 = 采纳种子数；未采纳种子（无
   机制事件、比值恒 1.0 的缺省值）不参与；全种子口径同尺双报；阈值 0.85 不变。
2. **spurious 治理（§1.2 冻结，两层联合）**：
   - L1 错乱源升级为真随机族：G2s'（G2R）判据臂的 B 侧错乱源 = 确定性每种子随机二元
     ctx 注入（default_rng((s+50)*99991+12345)，远阈值信号 v=131/31）——修复 C3 前提
     （原源 (seed+5) s_A 与真值弱相关 corr 0.58-0.66）；A 侧 rng(s*99991+12345) 逐字。
   - L2 门后二次验证（分裂后子条目持续命中检查）：提升（条件①-④全过）后验证窗
     [w+1, 末窗]，须① 每子条目命中 >=1 ② 合计命中 >= K_VERIFY=3 ③ 子条目命中能量-tag
     对应纯度 >= VERIFY_PURITY_MIN=0.65（tag 侧相对提升时父条目组中位 mid——与门同源
     同构的"门后复验"，不读真值）；通过才计入 n_promo_confirmed（确认制），不通过计入
     n_promo_unconfirmed（诊断）。
3. **守卫锚定**：CELL6_REPRO（stagger5 主测量原源 ≡ docs/274 逐位：adopt 0.70/0.60、
   JR 全种子 A 0.783315/B 0.854447、fid 0.8522/0.8000、spurious 原源 A{S7}/B{S1,S3,S8}、
   transfer 0.600/0.540、逐种子双侧 JR/fid/transfer/calib/首提升窗/门纯度——复用主测量
   原源记录，本格改动纯加法/纯测量口径）+ CELL5_REPRO 双锚（a_first ≡ docs/271、
   b_first ≡ docs/273，经 lang_comm_test5.repro_cell4 + lang_comm_test6.repro_cell5_b
   逐字节调用）。

流（§1.8）：双向世界（stagger5）M-T（两阶段）/M-G（单阶段）/C（前缀）/G0A（双 off）/
G1n（双 null）/G2s（原源 scrambled，诊断臂 + CELL6_REPRO 目标）/G2R（G2s' 真随机族
scrambled，判据臂 C3）/W（均匀世界双 ON）；CELL5_REPRO-A（a_first 12 臂）+
CELL5_REPRO-B（b_first 12 臂）。同一世界种子的双向四臂共享同一世界帧。

度量（§1.4 双侧化，docs/274 逐字 + M6 治理度量）：M1 预测 MAE；M2 结构；M3 联合残差
JR；M4 信号质量诊断 + 门诊断（双侧；B 侧真值 = 错位分段窗标签）；M5 信号留出归因
（双侧，N_C=14）；M6 治理度量（每提升验证明细/确认计数/JR 子集口径/spurious 对照）。
判据（§1.5）：C1 MUTUAL_VALUE（双侧 JR 采纳子集口径 <=0.85 + 全种子同尺双报 + 双侧
MAE==G0A abs<1e-9）、C2 MUTUAL_ADOPTION（确认制双侧 adopt >=0.6 且采纳 compound>=0.5
且 G0A/G1n 双侧零采纳）、C3 CLEAN_KEEP（**双侧 spurious(G2R)==0——治理后判据臂**；
原源 G2s 为诊断）、C4 STRUCTURE_KEEP/SIG_HOLDOUT（双侧结构 + 双侧 transfer>=0.10 且
>=0.5*calib）。判定映射：MUTUAL_EMERGES/ONE_WAY/MUTUAL_FLAT（含 MUTUAL_NO_GAIN）/
PARTIAL/GUARD_FAIL/LANG_BLOCKED。**预期 verdict（设计期实测，如实预注册）=
MUTUAL_EMERGES 最可能（C1 子集 A 0.6904/B 0.7574 <=0.85、C2 确认制 0.70/0.60、C3 治理
后 0/10 双侧、C4 沿用 + 守卫全过）**。

守卫（§1.6）：R_L2M_GUARD_D232、R_L2M_GUARD_D235、R_L2M_CELL2_REPRO（import docs/270
逐字）、R_L2M_CONSTRUCTION、R_L2M_PREFIX_EQ（双侧）、R_L2M_TWO_PHASE_EQ（双侧）、
R_L2M_REPRO_MAE（双侧）、R_L2M_DETERM（timing/main 逐位一致，外部核对）、R_L2M_SMOKE
（docs/274 全部 + 治理语义：验证单元测试/新源构造/确认制计数）、R_L2M_CELL5_REPRO 双锚、
R_L2M_CELL6_REPRO（stagger5 主测量原源 ≡ docs/274 逐位）、R_L2M_WORLD_EQ、
R_L2M_PRECOMPUTE。

安全纪律（§1.11）：新文件仅本文件；stdout 只输出 ASCII 标签 + 每行一个数字的 R_L2M_*
摘要块；运行经 powershell 包装重定向到 logs/；数字用 vision/extract_r.py 抽取；禁止读
日志/JSON 原文；本格不读 DAVIS。

用法：
  python vision/lang_comm_test7.py --smoke
  python vision/lang_comm_test7.py --tag timing
  python vision/lang_comm_test7.py --tag main
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
from critical_point import mean_sd, bootstrap_ci, JITTER, N_BOOT, BOOT_SEED
from stream_test import LOOP_CFG

warnings.filterwarnings("ignore")

DEFAULT_OUT = os.path.join("vision", "out", "results")

# ---------------- 冻结常量（docs/275 §1.1/§1.2/§1.7；运行后不改；docs/274 逐字沿用） ----------------
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
GATE_PURITY_MIN = l2i.GATE_PURITY_MIN      # 0.80（docs/270 冻结，双侧生效；零改动）
FRAME_SYNC = "stagger5"                    # docs/274 冻结（逐字沿用）
STAGGER_D = 5
# 本格新冻结量（docs/275 §1.2/§1.7）：
VERIFY_PURITY_MIN = 0.65       # L2 门后二次验证：子条目命中能量-tag 对应纯度阈值
K_VERIFY = 3                   # L2 验证窗内合计命中下界（与 k_consist 同量级）
G2R_B_RNG_OFFSET = 50          # L1：G2s'（G2R）B 侧随机二元注入的 rng 种子偏移


# ---------------- 门后二次验证（docs/275 §1.2 L2 冻结；测量层，不改机制代码路径） ----------------
def verify_promotions(loop):
    """门后二次验证（分裂后子条目持续命中检查）：对 loop 记录（_entry_log 含父条目
    c2_ene、match_trace、energy_trace）事后计算每个提升的验证结果。
    验证窗 = [w+1, 末窗]；条件：① 每子条目命中 >=1（无"零命中子条目"）② 合计命中
    >= K_VERIFY ③ 子条目命中能量-tag 对应纯度 >= VERIFY_PURITY_MIN（tag 侧相对提升时
    父条目组中位 mid，与门同源同构——不读真值 ctx）。
    返回 (details, n_confirmed, n_unconfirmed)；details = [(w, children, ok, purity,
    total, per_hits)]。"""
    log = loop._entry_log
    E = loop.energy_trace
    nw = len(E)
    details = []
    n_conf = 0
    n_unconf = 0
    for e in log:
        if not e.get("retired"):
            continue
        w = e.get("retired_at")
        pkey = tuple(e["key"])
        children = [c["key"] for c in log
                    if c["arity"] == 3 and c["created"] == w
                    and tuple(c["key"][:2]) == pkey]
        meds = {}
        for c2, vs in e.get("c2_ene", {}).items():
            if c2 is not None and len(vs) >= l2h.K_CONSIST:
                meds[c2] = float(np.median(vs))
        if len(meds) < 2 or not children:
            # 提升本不该发生（条件②/门已保证 >=2 组）——防御性记未确认
            details.append((w, [tuple(k) for k in children], 0, 0.0, 0, []))
            n_unconf += 1
            continue
        lo_k = min(meds, key=meds.get)
        hi_k = max(meds, key=meds.get)
        mid = (meds[lo_k] + meds[hi_k]) / 2.0
        per_hits = []
        correct = 0
        total = 0
        for k in children:
            tag = tuple(k)[2]
            hits = [w2 for w2, kk in loop.match_trace
                    if w + 1 <= w2 <= nw - 1
                    and kk is not None and tuple(kk) == tuple(k)]
            per_hits.append(len(hits))
            for w2 in hits:
                total += 1
                e2 = E[w2]
                if (tag == lo_k and e2 < mid) or (tag == hi_k and e2 >= mid):
                    correct += 1
        purity = (correct / total) if total else 0.0
        ok = int(all(p >= 1 for p in per_hits)
                 and total >= K_VERIFY
                 and purity >= VERIFY_PURITY_MIN)
        details.append((w, [tuple(k) for k in children], ok, purity, total,
                        per_hits))
        if ok:
            n_conf += 1
        else:
            n_unconf += 1
    return details, n_conf, n_unconf


def unit_record7(arm, seed, out, loop, side, snap=None, jr=None, att=None):
    """单元记录 = unit_record4 逐字 + 治理字段（verify 明细/确认计数）。"""
    rec = l2j.unit_record4(arm, seed, out, loop, side, snap=snap, jr=jr, att=att)
    vd, n_conf, n_unconf = verify_promotions(loop)
    rec["verify"] = {"details": [[w, [list(k) for k in kids], ok, round(p, 6),
                                  tot, per] for (w, kids, ok, p, tot, per) in vd],
                     "n_confirmed": n_conf, "n_unconfirmed": n_unconf}
    return rec


# ---------------- R_L2M_CELL6_REPRO（docs/275 §1.6 复现锚：stagger5 主测量原源 ≡ docs/274 逐位） ----------------
# 期望数字 = docs/274 §三/§四 冻结值（主测量原源 = docs/274 run_bidi6 逐字节；同代码
# 路径 -> 期望位精确，容差取打印精度 + 余量）。来源行：docs/274 §3.1（逐种子 JR/首提升窗/
# 门纯度/保真度）、§3.2（逐种子 transfer/calib）、§3.3（聚合）、§3.4（判据）、§3.5-3.6。
CELL6_EXP = {
    "adopt_a": 0.7000,
    "adopt_b": 0.6000,
    "comp_a": 1.0000,
    "comp_b": 1.0000,
    "jr_ratio_a": 0.783315,          # 全种子口径（docs/274 C1a 口径；同尺双报）
    "jr_ratio_b": 0.854447,
    "fid_a": 0.8522,
    "fid_b": 0.8000,
    "transfer_a": 0.600,
    "transfer_b": 0.540,
    "calib_a": 0.503,
    "calib_b": 0.386,
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
    "calib_a_per": [0.0, 0.800, 0.857, 0.0, 0.0, 0.857, 0.857, 0.0, 0.857,
                    0.800],
    "calib_b_per": [0.0, 0.000, 0.667, 0.857, 0.0, 0.0, 0.750, 0.0, 0.750,
                    0.833],
    "spurious_a": [7],
    "spurious_b": [1, 3, 8],
    "w_a": 0.0,
    "w_b": 0.7,
}


def repro_cell6(per_unit, seeds):
    """CELL6_REPRO：stagger5 主测量原源记录（M-T/M-G/G0A/G1n/G2s(原源)/W 双侧 12 臂 ×
    10 种子——同代码路径 run_bidi6 逐字节）≡ docs/274 §三/§四 冻结数字逐位一致。
    复用主测量记录（零额外运行）。返回 (ok, detail)。"""
    jr_a0s, jr_a1s, jr_b0s, jr_b1s = [], [], [], []
    ratio_as, ratio_bs = [], []
    fids_a, fids_b = [], []
    trans_a, trans_b, calibs_a, calibs_b = [], [], [], []
    fp_as, fp_bs, gate_as, gate_bs = [], [], [], []
    comp_a, comp_b = [], []
    g2a_promos, g2b_promos = [], []
    w_a, w_b = [], []
    for s in seeds:
        ta = per_unit["MTA_%d" % s]
        tb = per_unit["MTB_%d" % s]
        g0a = per_unit["G0AA_%d" % s]
        g0b = per_unit["G0AB_%d" % s]
        g2a = per_unit["G2SA_%d" % s]
        g2b = per_unit["G2SB_%d" % s]
        wa = per_unit["WA_%d" % s]
        wb = per_unit["WB_%d" % s]
        jr_a0s.append(g0a["jr"][0])
        jr_b0s.append(g0b["jr"][0])
        jr_a1s.append(ta["jr"][0])
        jr_b1s.append(tb["jr"][0])
        ratio_as.append(ta["jr"][0] / max(g0a["jr"][0], 1e-12))
        ratio_bs.append(tb["jr"][0] / max(g0b["jr"][0], 1e-12))
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
        g2a_promos.append(g2a["finalize"]["n_promo"])
        g2b_promos.append(g2b["finalize"]["n_promo"])
        w_a.append(wa["finalize"]["n_promo"])
        w_b.append(wb["finalize"]["n_promo"])
        ga_ = None
        if fp_as[-1] is not None:
            for rec in ta["gate"]:
                if rec[0] == fp_as[-1]:
                    ga_ = rec[2]
                    break
        gb_ = None
        if fp_bs[-1] is not None:
            for rec in tb["gate"]:
                if rec[0] == fp_bs[-1]:
                    gb_ = rec[2]
                    break
        gate_as.append(ga_)
        gate_bs.append(gb_)
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
                          CELL6_EXP["ratio_a"][s], 1e-4))
        checks.append(chk("JR_RATIO_B_S%d" % s, ratio_bs[s],
                          CELL6_EXP["ratio_b"][s], 1e-4))
        checks.append(chk("JR_A0_S%d" % s, jr_a0s[s], CELL6_EXP["jr_a0"][s], 1e-4))
        checks.append(chk("JR_A1_S%d" % s, jr_a1s[s], CELL6_EXP["jr_a1"][s], 1e-4))
        checks.append(chk("JR_B0_S%d" % s, jr_b0s[s], CELL6_EXP["jr_b0"][s], 1e-4))
        checks.append(chk("JR_B1_S%d" % s, jr_b1s[s], CELL6_EXP["jr_b1"][s], 1e-4))
        checks.append(chk("FID_A_S%d" % s, fids_a[s], CELL6_EXP["fid_a_per"][s], 1e-4))
        checks.append(chk("FID_B_S%d" % s, fids_b[s], CELL6_EXP["fid_b_per"][s], 1e-4))
        checks.append(chk("TRANSFER_A_S%d" % s, trans_a[s],
                          CELL6_EXP["transfer_a_per"][s], 1e-3))
        checks.append(chk("TRANSFER_B_S%d" % s, trans_b[s],
                          CELL6_EXP["transfer_b_per"][s], 1e-3))
        checks.append(chk("CALIB_A_S%d" % s, calibs_a[s],
                          CELL6_EXP["calib_a_per"][s], 1e-3))
        checks.append(chk("CALIB_B_S%d" % s, calibs_b[s],
                          CELL6_EXP["calib_b_per"][s], 1e-3))
        exp_fpa = CELL6_EXP["fp_a"][s]
        checks.append(("FP_A_S%d" % s, int(fp_as[s] == exp_fpa),
                       (fp_as[s] if fp_as[s] is not None else -1)))
        exp_fpb = CELL6_EXP["fp_b"][s]
        checks.append(("FP_B_S%d" % s, int(fp_bs[s] == exp_fpb),
                       (fp_bs[s] if fp_bs[s] is not None else -1)))
        exp_ga = CELL6_EXP["gate_a"][s]
        checks.append(("GATE_A_S%d" % s,
                       int((gate_as[s] is None and exp_ga is None)
                           or (gate_as[s] is not None and exp_ga is not None
                               and abs(gate_as[s] - exp_ga) < 1e-4)),
                       (gate_as[s] if gate_as[s] is not None else -1)))
        exp_gb = CELL6_EXP["gate_b"][s]
        checks.append(("GATE_B_S%d" % s,
                       int((gate_bs[s] is None and exp_gb is None)
                           or (gate_bs[s] is not None and exp_gb is not None
                               and abs(gate_bs[s] - exp_gb) < 1e-4)),
                       (gate_bs[s] if gate_bs[s] is not None else -1)))
    checks.append(chk("ADOPT_A", adopt_a, CELL6_EXP["adopt_a"], 1e-6))
    checks.append(chk("ADOPT_B", adopt_b, CELL6_EXP["adopt_b"], 1e-6))
    checks.append(chk("COMP_ADOPTED_A", comp_adopted_a, CELL6_EXP["comp_a"], 1e-4))
    checks.append(chk("COMP_ADOPTED_B", comp_adopted_b, CELL6_EXP["comp_b"], 1e-4))
    checks.append(chk("JR_RATIO_A_MEAN", jr_ratio_a_m, CELL6_EXP["jr_ratio_a"], 1e-3))
    checks.append(chk("JR_RATIO_B_MEAN", jr_ratio_b_m, CELL6_EXP["jr_ratio_b"], 1e-3))
    checks.append(chk("FID_A_MEAN", fid_a_m, CELL6_EXP["fid_a"], 1e-4))
    checks.append(chk("FID_B_MEAN", fid_b_m, CELL6_EXP["fid_b"], 1e-4))
    checks.append(chk("TRANSFER_A_MEAN", trans_a_m, CELL6_EXP["transfer_a"], 1e-3))
    checks.append(chk("TRANSFER_B_MEAN", trans_b_m, CELL6_EXP["transfer_b"], 1e-3))
    checks.append(chk("CALIB_A_MEAN", calib_a_m, CELL6_EXP["calib_a"], 1e-3))
    checks.append(chk("CALIB_B_MEAN", calib_b_m, CELL6_EXP["calib_b"], 1e-3))
    checks.append(chk("W_ADOPT_A", w_a_m, CELL6_EXP["w_a"], 1e-6))
    checks.append(chk("W_ADOPT_B", w_b_m, CELL6_EXP["w_b"], 1e-3))
    checks.append(("SPURIOUS_ORIG_A", int(g2a_spur == CELL6_EXP["spurious_a"]),
                   g2a_spur))
    checks.append(("SPURIOUS_ORIG_B", int(g2b_spur == CELL6_EXP["spurious_b"]),
                   g2b_spur))
    ok = int(all(c[1] == 1 for c in checks))
    return ok, dict(checks=checks, adopt_a=adopt_a, adopt_b=adopt_b,
                    comp_a=comp_adopted_a, comp_b=comp_adopted_b,
                    jr_ratio_a=jr_ratio_a_m, jr_ratio_b=jr_ratio_b_m,
                    fid_a=fid_a_m, fid_b=fid_b_m, transfer_a=trans_a_m,
                    transfer_b=trans_b_m, calib_a=calib_a_m, calib_b=calib_b_m,
                    w_a=w_a_m, w_b=w_b_m, spurious_a=g2a_spur,
                    spurious_b=g2b_spur, fp_a=fp_as, fp_b=fp_bs)


# ---------------- 构造冒烟（docs/275 §1.6；合成帧 + 治理语义单元测试，非数据） ----------------
def smoke_main7():
    """构造冒烟：docs/274 smoke_main6 全部 + 本格治理语义（验证单元测试：真实型过/
    随机型拦/零命中否决/合计不足否决；确认制计数 = confirmed+unconfirmed == 原始
    提升数；新源 G2R 构造运行正常）。"""
    results = {}
    # ---- docs/274 既有冒烟（逐字复制关键项） ----
    fb = l2g._synth_frames(30)
    fa = l2g._synth_frames(30, y0=26)
    labels = [dict(ctx=1, b_mult=1.0, a_regime=None)] * 3
    labels_b = [dict(ctx=0, b_mult=1.0, a_regime=None)] * 3
    for ch1, ch2 in (("off", "off"), ("comm", "comm"), ("null", "null"),
                     ("scrambled", "scrambled")):
        out_a, loop_a, out_b, loop_b, _, _ = l2l.run_bidi6(
            fa, fb, labels, labels_b, ch1=ch1, ch2=ch2, two_phase=False, n_c=3)
        results["construct_%s_%s" % (ch1, ch2)] = int(
            isinstance(out_a, dict) and isinstance(out_b, dict)
            and len(out_a.get("mae_trace", [])) >= 1
            and len(out_b.get("mae_trace", [])) >= 1)
    fb2 = l2g._synth_frames(100)
    fa2 = l2g._synth_frames(100, y0=26)
    lab2 = [dict(ctx=1, b_mult=1.0, a_regime=None)] * 10
    lab2b = [dict(ctx=0, b_mult=1.0, a_regime=None)] * 10
    _, la4, _, lb4, _, _ = l2l.run_bidi6(fa2, fb2, lab2, lab2b, ch1="comm",
                                         ch2="comm", two_phase=False, n_c=5)
    results["stagger_a_first_win_none"] = int(la4.sig_trace[0][2] is None)
    results["stagger_b_first_win_non_none"] = int(lb4.sig_trace[0][2] is not None)
    _, la5, _, _, _, _ = l2k.run_bidi(fa2, fb2, lab2, ch1="comm", ch2="comm",
                                      two_phase=False, n_c=5, timing="a_first")
    results["a_first_ch2_lag_first_win_none"] = int(la5.sig_trace[0][2] is None)
    _, _, _, lb6, _, _ = l2k.run_bidi(fa2, fb2, lab2, ch1="comm", ch2="comm",
                                      two_phase=False, n_c=5, timing="b_first")
    results["b_first_ch1_lag_first_win_none"] = int(lb6.sig_trace[0][2] is None)
    # ---- 门语义（docs/270 同款，双侧） ----
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
    ok_p, _, p_p = loop_on_a._gate_eval(pure_nd)
    results["gate_pure_pass_a"] = int(ok_p and p_p >= GATE_PURITY_MIN)
    ok_pb, _, p_pb = loop_on_b._gate_eval(pure_nd)
    results["gate_pure_pass_b"] = int(ok_pb and p_pb >= GATE_PURITY_MIN)
    ok_m, _, p_m = loop_on_a._gate_eval(mixed_nd)
    results["gate_mixed_block_a"] = int((not ok_m) and p_m < GATE_PURITY_MIN)
    ok_mb, _, p_mb = loop_on_b._gate_eval(mixed_nd)
    results["gate_mixed_block_b"] = int((not ok_mb) and p_mb < GATE_PURITY_MIN)
    ok_o, _, p_o = loop_off._gate_eval(pure_nd)
    results["gate_off_passthrough"] = int(ok_o and len(loop_off.gate_attempts) == 0)
    # ---- 本格治理语义：验证单元测试（合成 loop 记录；docs/275 §1.2 L2） ----
    class FakeLoop(object):
        pass

    def fake_loop(E, match_trace, parent):
        lp = FakeLoop()
        lp.energy_trace = E
        lp.match_trace = match_trace
        kids = [dict(key=[2, 0, 0], arity=3, created=parent["retired_at"],
                     c2_ene={}),
                dict(key=[2, 0, 1], arity=3, created=parent["retired_at"],
                     c2_ene={})]
        lp._entry_log = [parent] + kids
        return lp

    # 真实型：两子条目各 3 命中（不同窗）、能量全在 tag 侧 -> 过
    parent_real = dict(key=[2, 0], retired=True, retired_at=3, arity=2,
                       c2_ene={0: [510.0, 520.0, 500.0], 1: [700.0, 710.0, 690.0]})
    E_real = [0.0] * 20
    mt_real = [(4, [2, 0, 0]), (5, [2, 0, 0]), (6, [2, 0, 0]),
               (7, [2, 0, 1]), (8, [2, 0, 1]), (9, [2, 0, 1])]
    for w, k in mt_real:
        E_real[w] = 510.0 if k[2] == 0 else 710.0
    d, nc, nu = verify_promotions(fake_loop(E_real, mt_real, parent_real))
    results["verify_real_pass"] = int(nc == 1 and nu == 0 and d[0][2] == 1
                                      and d[0][3] == 1.0)
    # 随机型：子命中能量与 tag 交错（纯度 ~0.5 < 0.65）-> 拦
    parent_rnd = dict(key=[2, 0], retired=True, retired_at=3, arity=2,
                      c2_ene={0: [510.0, 520.0, 500.0], 1: [700.0, 710.0, 690.0]})
    E_rnd = [0.0] * 20
    mt_rnd = [(4, [2, 0, 0]), (5, [2, 0, 0]), (4, [2, 0, 1]), (5, [2, 0, 1])]
    for w, k in mt_rnd:
        E_rnd[w] = 510.0 if k[2] == 1 else 710.0     # 能量与 tag 反相
    d, nc, nu = verify_promotions(fake_loop(E_rnd, mt_rnd, parent_rnd))
    results["verify_random_block"] = int(nc == 0 and nu == 1 and d[0][2] == 0)
    # 零命中子条目否决：某子条目验证窗内无命中 -> 拦
    parent_zero = dict(key=[2, 0], retired=True, retired_at=3, arity=2,
                       c2_ene={0: [510.0, 520.0, 500.0], 1: [700.0, 710.0, 690.0]})
    E_zero = [0.0] * 20
    mt_zero = [(4, [2, 0, 0]), (5, [2, 0, 0]), (6, [2, 0, 0]),
               (7, [2, 0, 0]), (8, [2, 0, 0])]
    for w, k in mt_zero:
        E_zero[w] = 510.0
    d, nc, nu = verify_promotions(fake_loop(E_zero, mt_zero, parent_zero))
    results["verify_zero_hit_child_block"] = int(
        nc == 0 and nu == 1 and d[0][2] == 0)
    # 合计不足否决：总命中 < K_VERIFY -> 拦
    parent_low = dict(key=[2, 0], retired=True, retired_at=3, arity=2,
                      c2_ene={0: [510.0, 520.0, 500.0], 1: [700.0, 710.0, 690.0]})
    E_low = [0.0] * 20
    mt_low = [(4, [2, 0, 0]), (5, [2, 0, 1])]
    for w, k in mt_low:
        E_low[w] = 510.0 if k[2] == 0 else 710.0
    d, nc, nu = verify_promotions(fake_loop(E_low, mt_low, parent_low))
    results["verify_total_low_block"] = int(nc == 0 and nu == 1 and d[0][2] == 0)
    # ---- 确认制计数 + 新源 G2R 构造 ----
    out_g2r, loop_g2r_a, _, loop_g2r_b, _, _ = l2l.run_bidi6(
        fa2, fb2, lab2, lab2b, ch1="scrambled", ch2="scrambled",
        a_mode="scrambled", n_c=5,
        sig1_fn=lambda w: 131.0 if w % 2 == 0 else 31.0,
        sig2_fn=lambda w: 31.0 if w % 2 == 0 else 131.0)
    results["g2r_construct"] = int(isinstance(out_g2r, dict))
    vd_a, nc_a, nu_a = verify_promotions(loop_g2r_a)
    vd_b, nc_b, nu_b = verify_promotions(loop_g2r_b)
    results["verify_counting"] = int(
        nc_a + nu_a == len(vd_a) and nc_b + nu_b == len(vd_b))
    # 确认制与原始提升数一致（本合成流无提升 -> 0+0 == 0）
    results["verify_no_promo_consistent"] = int(
        nc_a + nu_a == out_g2r["n_promo"]
        and nc_b + nu_b == out_g2r["n_promo"])
    for k in sorted(results):
        print("R_L2M_SMOKE_%s=%d" % (k.upper(), results[k]))
    return 0 if all(results.values()) else 1


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="l2m")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.smoke:
        return smoke_main7()
    t0 = time.time()
    seeds = list(range(10))

    cfg = {"tag": args.tag, "n_seeds": len(seeds), "frames": N_FRAMES,
           "window": WINDOW, "n_c": N_C, "jitter": JITTER,
           "b_m1": B_M1, "b_m2": B_M2, "a_m1": A_M1, "a_m2": A_M2,
           "a_mirror": A_MIRROR, "noise_sigma": l2g.NOISE_SIGMA,
           "frame_sync": FRAME_SYNC, "stagger_d": STAGGER_D,
           "jr_measure": {"subset_denominator": "adopted_seeds_confirmed",
                          "full_report": 1, "threshold": JR_RATIO_MAX},
           "governance": {"verify_purity_min": VERIFY_PURITY_MIN,
                          "k_verify": K_VERIFY,
                          "g2r_b_rng_offset": G2R_B_RNG_OFFSET},
           "world": {"a_center": list(l2g.A_CENTER), "a_orbit": l2g.A_ORBIT,
                     "a_freq": l2g.A_FREQ, "b_center": list(l2g.B_CENTER),
                     "b_orbit": l2g.B_ORBIT, "b_freq": l2g.B_FREQ,
                     "rng_lvcode": LV_WORLD},
           "channel": {"sparse_px": l2g.SIG_SPARSE_PX,
                       "null_signal": l2g.NULL_SIGNAL,
                       "ch1_halfwin_overlap": 1, "ch2_halfwin_lag": 1},
           "gate": {"purity_min": GATE_PURITY_MIN},
           "criteria": {"jr_ratio_max": JR_RATIO_MAX,
                        "adopt_frac_min": ADOPT_FRAC_MIN,
                        "compound_min": COMPOUND_MIN,
                        "transfer_floor": TRANSFER_FLOOR,
                        "transfer_rel": TRANSFER_REL},
           "loop": LOOP_CFG}
    ck_tag = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    ckpt_path = os.path.join(args.out_dir, "ckpt_l2m_%s.json" % ck_tag)

    done = {}
    if args.resume and not args.no_resume and os.path.exists(ckpt_path):
        with open(ckpt_path, encoding="utf-8") as f:
            done = json.load(f).get("per_unit", {})

    per_unit = dict(done)
    worlds_bidi = {s: l2j.make_bidi_world(s, a1=A_M1, a2=A_M2, mirror=A_MIRROR)
                   for s in seeds}
    worlds_uniform = {s: l2g.make_world(s, m1=B_M1, m2=B_M2) for s in seeds}
    wl_b_bidi = {s: l2l.stagger_labels(l2l.per_frame_ctx(s)) for s in seeds}
    wl_b_uniform = {s: l2l.stagger_labels(l2l.per_frame_ctx(s, a_ctx_dep=False))
                    for s in seeds}

    # 错乱信号源：G2s 原源（docs/274 逐字）+ G2s' 真随机族（docs/275 §1.2 L1）
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

    g2r_sig1 = {s: rand_sig((s + G2R_B_RNG_OFFSET) * 99991 + 12345)
                for s in seeds}                       # B 侧新源（L1 冻结）
    g2r_sig2 = {s: rand_sig(s * 99991 + 12345)
                for s in seeds}                       # A 侧（docs/271 模型逐字）

    def need(arm, s):
        return "%s_%d" % (arm, s) not in per_unit

    for s in seeds:
        fa, fb, wl = worlds_bidi[s]
        wlb = wl_b_bidi[s]
        if need("MTA", s) or need("MTB", s):
            out_a, loop_a, out_b, loop_b, snap_a, snap_b = l2l.run_bidi6(
                fa, fb, wl, wlb, ch1="comm", ch2="comm", two_phase=True,
                n_c=N_C)
            per_unit["MTA_%d" % s] = unit_record7(
                "MTA", s, out_a, loop_a, "A", snap=snap_a,
                jr=l2g.jr_b(loop_a), att=l2g.attribution(loop_a, N_C))
            per_unit["MTB_%d" % s] = unit_record7(
                "MTB", s, out_b, loop_b, "B", snap=snap_b,
                jr=l2g.jr_b(loop_b), att=l2g.attribution(loop_b, N_C))
            print("PROGRESS", flush=True)
        if need("MGA", s) or need("MGB", s):
            out_a, loop_a, out_b, loop_b, _, _ = l2l.run_bidi6(
                fa, fb, wl, wlb, ch1="comm", ch2="comm", two_phase=False)
            per_unit["MGA_%d" % s] = unit_record7(
                "MGA", s, out_a, loop_a, "A")
            per_unit["MGB_%d" % s] = unit_record7(
                "MGB", s, out_b, loop_b, "B")
            print("PROGRESS", flush=True)
        if need("CA", s) or need("CB", s):
            out_a, loop_a, out_b, loop_b, snap_a, snap_b = l2l.run_bidi6(
                fa[:140], fb[:140], wl[:14], wlb[:14], ch1="comm", ch2="comm",
                two_phase=False, want_end_snap=True)
            per_unit["CA_%d" % s] = unit_record7(
                "CA", s, out_a, loop_a, "A", snap=snap_a)
            per_unit["CB_%d" % s] = unit_record7(
                "CB", s, out_b, loop_b, "B", snap=snap_b)
            print("PROGRESS", flush=True)
        if need("G0AA", s) or need("G0AB", s):
            out_a, loop_a, out_b, loop_b, _, _ = l2l.run_bidi6(
                fa, fb, wl, wlb, ch1="off", ch2="off", a_mode="off")
            per_unit["G0AA_%d" % s] = unit_record7(
                "G0AA", s, out_a, loop_a, "A", jr=l2g.jr_b(loop_a))
            per_unit["G0AB_%d" % s] = unit_record7(
                "G0AB", s, out_b, loop_b, "B", jr=l2g.jr_b(loop_b))
            print("PROGRESS", flush=True)
        if need("G1NA", s) or need("G1NB", s):
            out_a, loop_a, out_b, loop_b, _, _ = l2l.run_bidi6(
                fa, fb, wl, wlb, ch1="null", ch2="null", a_mode="null")
            per_unit["G1NA_%d" % s] = unit_record7(
                "G1NA", s, out_a, loop_a, "A")
            per_unit["G1NB_%d" % s] = unit_record7(
                "G1NB", s, out_b, loop_b, "B")
            print("PROGRESS", flush=True)
        if need("G2SA", s) or need("G2SB", s):
            # G2s 原源（docs/274 逐字：A 随机二元注入、B (seed+5)%10 的 s_A）——诊断臂
            other = (s + 5) % 10
            sigA = bidi_sig[other][0]
            rng_bits = np.random.default_rng(s * 99991 + 12345)
            rand_bits = [int(rng_bits.random() < 0.5) for _ in range(
                len(wl))]
            sig2 = [131.0 if b else 31.0 for b in rand_bits]
            out_a, loop_a, out_b, loop_b, _, _ = l2l.run_bidi6(
                fa, fb, wl, wlb, ch1="scrambled", ch2="scrambled",
                a_mode="scrambled",
                sig1_fn=lambda w, sa=sigA: sa[w],
                sig2_fn=lambda w, sv=sig2: sv[w])
            per_unit["G2SA_%d" % s] = unit_record7(
                "G2SA", s, out_a, loop_a, "A")
            per_unit["G2SB_%d" % s] = unit_record7(
                "G2SB", s, out_b, loop_b, "B")
            print("PROGRESS", flush=True)
        if need("G2RA", s) or need("G2RB", s):
            # G2s' 真随机族（docs/275 §1.2 L1：A rng(s*99991+12345) 逐字、B 新 rng）——
            # 判据臂 C3
            out_a, loop_a, out_b, loop_b, _, _ = l2l.run_bidi6(
                fa, fb, wl, wlb, ch1="scrambled", ch2="scrambled",
                a_mode="scrambled",
                sig1_fn=lambda w, sv=g2r_sig1[s]: sv[w],
                sig2_fn=lambda w, sv=g2r_sig2[s]: sv[w])
            per_unit["G2RA_%d" % s] = unit_record7(
                "G2RA", s, out_a, loop_a, "A")
            per_unit["G2RB_%d" % s] = unit_record7(
                "G2RB", s, out_b, loop_b, "B")
            print("PROGRESS", flush=True)
        if need("WA", s) or need("WB", s):
            fa_u, fb_u, wl_u = worlds_uniform[s]
            wl_bu = wl_b_uniform[s]
            out_a, loop_a, out_b, loop_b, _, _ = l2l.run_bidi6(
                fa_u, fb_u, wl_u, wl_bu, ch1="comm", ch2="comm", a_mode="comm")
            per_unit["WA_%d" % s] = unit_record7(
                "WA", s, out_a, loop_a, "A")
            per_unit["WB_%d" % s] = unit_record7(
                "WB", s, out_b, loop_b, "B")
            print("PROGRESS", flush=True)
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump({"config": cfg, "per_unit": per_unit},
                      f, ensure_ascii=False, indent=1)

    # ---- 守卫 ----
    g232_ok, g232 = l2g.guard_d232()
    g235_ok, g235 = l2g.guard_d235()
    c2_ok, c2_detail = l2i.repro_cell2()
    c5a_ok, c5a_detail = l2k.repro_cell4()      # CELL5_REPRO-A：a_first ≡ docs/271 逐位
    c5b_ok, c5b_detail = l2l.repro_cell5_b()    # CELL5_REPRO-B：b_first ≡ docs/273 逐位
    c6_ok, c6_detail = repro_cell6(per_unit, seeds)   # CELL6_REPRO：原源 ≡ docs/274 逐位
    world_ok, world_oks = l2j.world_eq()
    pre_ok, pre_detail = l2l.precompute_ok_main()

    # ---- 跨单元核对（双侧，stagger5） ----
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
            g2ra=g2ra, g2rb=g2rb, wa=wa, wb=wb,
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

    # ---- 治理读数（docs/275 §1.2 确认制 + §1.1 采纳子集口径） ----
    def nconf(rec):
        return rec["verify"]["n_confirmed"]

    adopt_raw_a = float(np.mean([r["ta"]["finalize"]["n_promo"] >= 1
                                 for r in seed_rows]))
    adopt_raw_b = float(np.mean([r["tb"]["finalize"]["n_promo"] >= 1
                                 for r in seed_rows]))
    adopt_conf_a = float(np.mean([nconf(r["ta"]) >= 1 for r in seed_rows]))
    adopt_conf_b = float(np.mean([nconf(r["tb"]) >= 1 for r in seed_rows]))
    adopt_conf_a_w = float(np.mean([nconf(r["wa"]) >= 1 for r in seed_rows]))
    adopt_conf_b_w = float(np.mean([nconf(r["wb"]) >= 1 for r in seed_rows]))
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
    spurious_g2s_a = [r["seed"] for r in seed_rows
                      if nconf(r["g2a"]) >= 1]
    spurious_g2s_b = [r["seed"] for r in seed_rows
                      if nconf(r["g2b"]) >= 1]
    spurious_g2r_a = [r["seed"] for r in seed_rows
                      if nconf(r["g2ra"]) >= 1]
    spurious_g2r_b = [r["seed"] for r in seed_rows
                      if nconf(r["g2rb"]) >= 1]
    fid_a_m = float(np.mean(col("A", "ctx_fidelity")))
    fid_b_m = float(np.mean(col("B", "ctx_fidelity")))
    # 双向世界诊断（M-G A/B 能量带）
    eratios_a = []
    eratios_b = []
    for s in seeds:
        fa, fb, wl = worlds_bidi[s]
        ctxs = [lb["ctx"] for lb in wl]
        ga_E = per_unit["MGA_%d" % s]["E"]
        gb_E = per_unit["MGB_%d" % s]["E"]
        da_ = l2j.bidi_diag(ctxs, ga_E)
        db_ = l2j.bidi_diag([lb["ctx"] for lb in wl_b_bidi[s]], gb_E)
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
        [lb["ctx"] for lb in wl_b_bidi[s]], per_unit["MGB_%d" % s]["E"]))
        for s in seeds]))

    # ---- 判据（docs/275 §1.5 冻结：C1a 采纳子集口径 + C3 治理后判据臂） ----
    c1a = int(jr_subset_a <= JR_RATIO_MAX and jr_subset_b <= JR_RATIO_MAX)
    c1b = repro_ok_a == 1 and repro_ok_b == 1
    c2 = int(adopt_conf_a >= ADOPT_FRAC_MIN and adopt_conf_b >= ADOPT_FRAC_MIN
             and comp_adopted_a >= COMPOUND_MIN
             and comp_adopted_b >= COMPOUND_MIN
             and all(cons_g0a) and all(cons_g0b)
             and all(cons_g1na) and all(cons_g1nb))
    c3 = int(len(spurious_g2r_a) == 0 and len(spurious_g2r_b) == 0)
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
                 and c2_ok == 1 and c5a_ok == 1 and c5b_ok == 1
                 and c6_ok == 1 and world_ok == 1 and pre_ok == 1)
    if not guards_ok:
        verdict = "GUARD_FAIL"
        vnote = ("guard(s) failed: D232=%d, D235=%d, CONSTRUCTION=%d, "
                 "PREFIX_EQ_A=%d, PREFIX_EQ_B=%d, TWO_PHASE_EQ_A=%d, "
                 "TWO_PHASE_EQ_B=%d, REPRO_MAE_A=%d, REPRO_MAE_B=%d, "
                 "CELL2_REPRO=%d, CELL5_REPRO_A=%d, CELL5_REPRO_B=%d, "
                 "CELL6_REPRO=%d, WORLD_EQ=%d, PRECOMPUTE=%d -> "
                 "implementation drift; fix implementation, do not judge "
                 "mechanism" % (g232_ok, g235_ok, construction_ok,
                                prefix_ok_a, prefix_ok_b, two_phase_ok_a,
                                two_phase_ok_b, repro_ok_a, repro_ok_b,
                                c2_ok, c5a_ok, c5b_ok, c6_ok,
                                world_ok, pre_ok))
    elif blocked:
        verdict = "LANG_BLOCKED"
        vnote = ("synthetic environment unavailable (per-seed eligible/JR "
                 "windows missing on A/B sides); see per-seed numbers")
    elif c1a and c1b and c2 and c3 and c4:
        verdict = "MUTUAL_EMERGES"
        vnote = ("criteria C1-C4 all pass and all guards pass: A and B both "
                 "spontaneously adopt the other's signal (ledger-driven, no "
                 "switch, gate on both loops), joint residual drops on both "
                 "sides under the adoption-subset JR measure, gate + "
                 "post-gate verification keep both loops clean "
                 "(spurious(G2R)=0 both), holdout transfer holds both sides "
                 "-> minimal measurable closed-loop evidence (docs/275 sec "
                 "1.5)")
    elif not c2:
        if (adopt_conf_a >= ADOPT_FRAC_MIN) != (adopt_conf_b >= ADOPT_FRAC_MIN):
            verdict = "ONE_WAY"
            vnote = ("C2 fails with exactly one side adopting (confirmed): "
                     "adopt_A=%.4f, adopt_B=%.4f -> one-way round-trip only, "
                     "closed loop not achieved; honest report of which side "
                     "and why (docs/275 sec 1.5)" % (adopt_conf_a,
                                                     adopt_conf_b))
        else:
            verdict = "MUTUAL_FLAT"
            vnote = ("C2 fails (signal not mutually adopted): adopt_A=%.4f, "
                     "adopt_B=%.4f (confirmed; <0.6 or G0A/G1n non-zero "
                     "adoption) -> honest negative; see bilateral "
                     "decomposition" % (adopt_conf_a, adopt_conf_b))
    elif not (c1a and c1b):
        verdict = "MUTUAL_FLAT"
        vnote = ("C2 passes but C1 fails (adopted but joint residual not "
                 "lowered on at least one side under adoption-subset JR "
                 "measure) -> MUTUAL_NO_GAIN sub-form: JR_subset_A=%.4f, "
                 "JR_subset_B=%.4f > 0.85; mechanism has no rollback "
                 "(docs/268 sec 5.6)" % (jr_subset_a, jr_subset_b))
    else:
        why = []
        if not c3:
            why.append("C3 CLEAN_KEEP fails: spurious(G2R)_A=%d, "
                       "spurious(G2R)_B=%d (gate + post-gate verification "
                       "must hold on both loops)" % (len(spurious_g2r_a),
                                                     len(spurious_g2r_b)))
        if not c4a:
            why.append("C4a STRUCTURE_KEEP fails (see SC2/churn/ratio per side)")
        if not c4b:
            why.append("C4b SIG_HOLDOUT fails (transfer_A=%.4f/calib_A=%.4f, "
                       "transfer_B=%.4f/calib_B=%.4f)" % (transfer_a_m,
                                                           calib_a_m,
                                                           transfer_b_m,
                                                           calib_b_m))
        verdict = "PARTIAL"
        vnote = "; ".join(why) + " (see R_L2M_CRIT* numbers)"

    # ---- 工件（自描述 JSON） ----
    out = {
        "artifact": "lang_comm_test7",
        "doc_ref": "docs/63, docs/228, docs/232, docs/235, docs/247, docs/255, "
                   "docs/258, docs/264, docs/266, docs/268, docs/269, docs/270, "
                   "docs/271, docs/273, docs/274, docs/275",
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
                   "cell5_repro_a": {"ok": c5a_ok},
                   "cell5_repro_b": {"ok": c5b_ok},
                   "cell6_repro": {"ok": c6_ok, "detail": c6_detail},
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
               "ratio_a_full": jr_ratio_a_m, "ratio_a_full_sd": jr_ratio_a_sd,
               "ratio_b_full": jr_ratio_b_m, "ratio_b_full_sd": jr_ratio_b_sd,
               "ratio_a_subset": jr_subset_a, "ratio_b_subset": jr_subset_b,
               "adopted_seeds_a": [r["seed"] for r in adopted_a],
               "adopted_seeds_b": [r["seed"] for r in adopted_b],
               "per_seed_ratio_a": jr_ratio_as,
               "per_seed_ratio_b": jr_ratio_bs},
        "adoption": {"adopt_a_raw": adopt_raw_a, "adopt_b_raw": adopt_raw_b,
                     "adopt_a_confirmed": adopt_conf_a,
                     "adopt_b_confirmed": adopt_conf_b,
                     "comp_adopted_a": comp_adopted_a,
                     "comp_adopted_b": comp_adopted_b,
                     "adopt_a_w_confirmed": adopt_conf_a_w,
                     "adopt_b_w_confirmed": adopt_conf_b_w},
        "governance": {"verify_purity_min": VERIFY_PURITY_MIN,
                       "k_verify": K_VERIFY,
                       "spurious_g2s_orig_a": spurious_g2s_a,
                       "spurious_g2s_orig_b": spurious_g2s_b,
                       "spurious_g2r_a": spurious_g2r_a,
                       "spurious_g2r_b": spurious_g2r_b},
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
        "criteria": {"c1a_mutual_value_subset": c1a, "c1b_mae_eq": c1b,
                     "c2_mutual_adoption": c2, "c3_clean_keep": c3,
                     "c4_structure_holdout": c4, "c4a": c4a, "c4b": c4b,
                     "jr_ratio_a_subset": jr_subset_a,
                     "jr_ratio_b_subset": jr_subset_b,
                     "jr_ratio_a_full": jr_ratio_a_m,
                     "jr_ratio_b_full": jr_ratio_b_m,
                     "adopt_a_confirmed": adopt_conf_a,
                     "adopt_b_confirmed": adopt_conf_b,
                     "comp_adopted_a": comp_adopted_a,
                     "comp_adopted_b": comp_adopted_b,
                     "transfer_a": transfer_a_m, "calib_a": calib_a_m,
                     "transfer_b": transfer_b_m, "calib_b": calib_b_m,
                     "spurious_g2r_a_seeds": spurious_g2r_a,
                     "spurious_g2r_b_seeds": spurious_g2r_b},
        "verdict": {"verdict": verdict, "note": vnote},
        "timing": {"elapsed_sec": round(time.time() - t0, 2)},
    }
    res_path = os.path.join(args.out_dir, "l2m_%s.json" % args.tag)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 摘要块：ASCII 标签，每行一个数字（顺序固定；无中文/日志/JSON） ----
    print("R_L2M_TAG=%s" % args.tag)
    print("R_L2M_SEEDS=%d" % len(seeds))
    print("R_L2M_FRAMES=%d" % N_FRAMES)
    print("R_L2M_WINDOWS=%d" % (N_FRAMES // WINDOW))
    print("R_L2M_NC=%d" % N_C)
    print("R_L2M_FRAME_SYNC=%s" % FRAME_SYNC)
    print("R_L2M_STAGGER_D=%d" % STAGGER_D)
    print("R_L2M_M1=%.1f" % B_M1)
    print("R_L2M_M2=%.1f" % B_M2)
    print("R_L2M_A1=%.1f" % A_M1)
    print("R_L2M_A2=%.1f" % A_M2)
    print("R_L2M_GATE_PURITY_MIN=%.2f" % GATE_PURITY_MIN)
    print("R_L2M_VERIFY_PURITY_MIN=%.2f" % VERIFY_PURITY_MIN)
    print("R_L2M_K_VERIFY=%d" % K_VERIFY)
    print("R_L2M_GUARD_D232=%d" % g232_ok)
    print("R_L2M_GUARD_D232_SC2=%s" % ",".join(str(v) for v in g232["sc2"]))
    print("R_L2M_GUARD_D232_SCLATE_FRAC=%.4f" % g232["sc_late_frac"])
    print("R_L2M_GUARD_D232_SC4=%.4f" % g232["sc4"])
    print("R_L2M_GUARD_D232_MAE=%.6f" % g232["mae"])
    print("R_L2M_GUARD_D232_MAE_SD=%.6f" % g232["mae_sd"])
    print("R_L2M_GUARD_D232_PIN=%.4f" % g232["pin"])
    print("R_L2M_GUARD_D232_CLASS=%s" % g232["cls"])
    print("R_L2M_GUARD_D235=%d" % g235_ok)
    for lv in (21, 22):
        d = g235[lv]
        print("R_L2M_GUARD_D235_C%d_OK=%d" % (lv, d["ok"]))
        print("R_L2M_GUARD_D235_C%d_MAE=%.6f" % (lv, d["mae"]))
        print("R_L2M_GUARD_D235_C%d_MAE_SD=%.6f" % (lv, d["mae_sd"]))
        print("R_L2M_GUARD_D235_C%d_SC2=%.4f" % (lv, d["sc2"]))
        print("R_L2M_GUARD_D235_C%d_SC2_SD=%.4f" % (lv, d["sc2_sd"]))
        print("R_L2M_GUARD_D235_C%d_COMP=%.4f" % (lv, d["comp"]))
        print("R_L2M_GUARD_D235_C%d_CHURN=%.4f" % (lv, d["churn"]))
        print("R_L2M_GUARD_D235_C%d_FID=%.4f" % (lv, d["fid"]))
    print("R_L2M_CONSTRUCTION=%d" % construction_ok)
    print("R_L2M_CONSTRUCTION_G0A=%d" % int(sum(cons_g0a)))
    print("R_L2M_CONSTRUCTION_G0B=%d" % int(sum(cons_g0b)))
    print("R_L2M_CONSTRUCTION_G1NA=%d" % int(sum(cons_g1na)))
    print("R_L2M_CONSTRUCTION_G1NB=%d" % int(sum(cons_g1nb)))
    print("R_L2M_PREFIX_EQ_A=%d" % prefix_ok_a)
    print("R_L2M_PREFIX_EQ_B=%d" % prefix_ok_b)
    print("R_L2M_TWO_PHASE_EQ_A=%d" % two_phase_ok_a)
    print("R_L2M_TWO_PHASE_EQ_B=%d" % two_phase_ok_b)
    print("R_L2M_REPRO_MAE_A=%d" % repro_ok_a)
    print("R_L2M_REPRO_MAE_B=%d" % repro_ok_b)
    print("R_L2M_CELL2_REPRO=%d" % c2_ok)
    print("R_L2M_CELL5_REPRO_A=%d" % c5a_ok)
    for (name, ok, got) in c5a_detail["checks"]:
        print("R_L2M_CELL5A_%s=%d" % (name, ok))
        print("R_L2M_CELL5A_%s_VAL=%s" % (name,
              (",".join(str(v) for v in got) if isinstance(got, list)
               else "%.6f" % got)))
    print("R_L2M_CELL5A_ADOPT_A=%.4f" % c5a_detail["adopt_a"])
    print("R_L2M_CELL5A_ADOPT_B=%.4f" % c5a_detail["adopt_b"])
    print("R_L2M_CELL5A_JR_RATIO_A=%.6f" % c5a_detail["jr_ratio_a"])
    print("R_L2M_CELL5A_JR_RATIO_B=%.6f" % c5a_detail["jr_ratio_b"])
    print("R_L2M_CELL5A_FID_A=%.4f" % c5a_detail["fid_a"])
    print("R_L2M_CELL5A_FID_B=%.4f" % c5a_detail["fid_b"])
    print("R_L2M_CELL5A_TRANSFER_A=%.4f" % c5a_detail["transfer_a"])
    print("R_L2M_CELL5A_TRANSFER_B=%.4f" % c5a_detail["transfer_b"])
    print("R_L2M_CELL5A_SPURIOUS_A=%d" % len(c5a_detail["spurious_a"]))
    print("R_L2M_CELL5A_SPURIOUS_B=%d" % len(c5a_detail["spurious_b"]))
    print("R_L2M_CELL5_REPRO_B=%d" % c5b_ok)
    for (name, ok, got) in c5b_detail["checks"]:
        print("R_L2M_CELL5B_%s=%d" % (name, ok))
        print("R_L2M_CELL5B_%s_VAL=%s" % (name,
              (",".join(str(v) for v in got) if isinstance(got, list)
               else "%.6f" % got)))
    print("R_L2M_CELL5B_ADOPT_A=%.4f" % c5b_detail["adopt_a"])
    print("R_L2M_CELL5B_ADOPT_B=%.4f" % c5b_detail["adopt_b"])
    print("R_L2M_CELL5B_JR_RATIO_A=%.6f" % c5b_detail["jr_ratio_a"])
    print("R_L2M_CELL5B_JR_RATIO_B=%.6f" % c5b_detail["jr_ratio_b"])
    print("R_L2M_CELL5B_FID_A=%.4f" % c5b_detail["fid_a"])
    print("R_L2M_CELL5B_FID_B=%.4f" % c5b_detail["fid_b"])
    print("R_L2M_CELL5B_TRANSFER_A=%.4f" % c5b_detail["transfer_a"])
    print("R_L2M_CELL5B_TRANSFER_B=%.4f" % c5b_detail["transfer_b"])
    print("R_L2M_CELL5B_SPURIOUS_A=%d" % len(c5b_detail["spurious_a"]))
    print("R_L2M_CELL5B_SPURIOUS_B=%d" % len(c5b_detail["spurious_b"]))
    print("R_L2M_CELL6_REPRO=%d" % c6_ok)
    for (name, ok, got) in c6_detail["checks"]:
        print("R_L2M_CELL6_%s=%d" % (name, ok))
        print("R_L2M_CELL6_%s_VAL=%s" % (name,
              (",".join(str(v) for v in got) if isinstance(got, list)
               else "%.6f" % got)))
    print("R_L2M_CELL6_ADOPT_A=%.4f" % c6_detail["adopt_a"])
    print("R_L2M_CELL6_ADOPT_B=%.4f" % c6_detail["adopt_b"])
    print("R_L2M_CELL6_JR_RATIO_A=%.6f" % c6_detail["jr_ratio_a"])
    print("R_L2M_CELL6_JR_RATIO_B=%.6f" % c6_detail["jr_ratio_b"])
    print("R_L2M_CELL6_FID_A=%.4f" % c6_detail["fid_a"])
    print("R_L2M_CELL6_FID_B=%.4f" % c6_detail["fid_b"])
    print("R_L2M_CELL6_TRANSFER_A=%.4f" % c6_detail["transfer_a"])
    print("R_L2M_CELL6_TRANSFER_B=%.4f" % c6_detail["transfer_b"])
    print("R_L2M_CELL6_SPURIOUS_ORIG_A=%d" % len(c6_detail["spurious_a"]))
    print("R_L2M_CELL6_SPURIOUS_ORIG_B=%d" % len(c6_detail["spurious_b"]))
    print("R_L2M_WORLD_EQ=%d" % world_ok)
    print("R_L2M_WORLD_EQ_SEEDS=%d" % int(sum(world_oks)))
    print("R_L2M_PRECOMPUTE=%d" % pre_ok)
    print("R_L2M_PRECOMPUTE_DW_A=%d" % pre_detail["dw_a"])
    print("R_L2M_PRECOMPUTE_DW_B=%d" % pre_detail["dw_b"])
    print("R_L2M_PRECOMPUTE_UNDER450_A=%d" % pre_detail["under_a"])
    print("R_L2M_PRECOMPUTE_UNDER450_B_FULL=%d" % pre_detail["under_b_full"])
    print("R_L2M_PRECOMPUTE_UNDER450_B_PARTIAL=%d" % pre_detail["under_b_partial"])
    for r in seed_rows:
        s = r["seed"]
        print("R_L2M_SEED=%d" % s)
        for side, tag in (("A", "G0AA"), ("B", "G0AB")):
            g0f = r["g0a" if side == "A" else "g0b"]["finalize"]
            print("R_L2M_S%d_%s_G0_MAE=%.6f" % (s, side, g0f["mae_mean"]))
            print("R_L2M_S%d_%s_G0_SC2=%d" % (s, side, g0f["sc2"]))
            print("R_L2M_S%d_%s_G0_COMP=%.4f" % (s, side, g0f["compound_frac"]))
            print("R_L2M_S%d_%s_G0_CHURN=%.4f" % (s, side, g0f["churn_frac"]))
            print("R_L2M_S%d_%s_G0_PROMO=%d" % (s, side, g0f["n_promo"]))
        for side, key in (("A", "ta"), ("B", "tb")):
            mf = r[key]["finalize"]
            print("R_L2M_S%d_%s_M_MAE=%.6f" % (s, side, mf["mae_mean"]))
            print("R_L2M_S%d_%s_M_SC2=%d" % (s, side, mf["sc2"]))
            print("R_L2M_S%d_%s_M_COMP=%.4f" % (s, side, mf["compound_frac"]))
            print("R_L2M_S%d_%s_M_CHURN=%.4f" % (s, side, mf["churn_frac"]))
            print("R_L2M_S%d_%s_M_PROMO=%d" % (s, side, mf["n_promo"]))
            print("R_L2M_S%d_%s_M_FID=%.4f" % (s, side, mf["ctx_fidelity"]))
            print("R_L2M_S%d_%s_M_NCONF=%d" % (s, side,
                                                r[key]["verify"]["n_confirmed"]))
            print("R_L2M_S%d_%s_M_NUNCONF=%d" % (s, side,
                                                  r[key]["verify"]["n_unconfirmed"]))
            for i, (w, kids, okv, p, tot, per) in enumerate(
                    r[key]["verify"]["details"]):
                print("R_L2M_S%d_%s_VERIFY_P%d_W=%d" % (s, side, i, w))
                print("R_L2M_S%d_%s_VERIFY_P%d_OK=%d" % (s, side, i, okv))
                print("R_L2M_S%d_%s_VERIFY_P%d_PURITY=%.6f" % (s, side, i, p))
                print("R_L2M_S%d_%s_VERIFY_P%d_TOTAL=%d" % (s, side, i, tot))
                print("R_L2M_S%d_%s_VERIFY_P%d_PER=%s" % (
                    s, side, i, ",".join(str(v) for v in per)))
        for side, key in (("A", "g2a"), ("B", "g2b")):
            g2f = r[key]["finalize"]
            print("R_L2M_S%d_%s_G2S_COMP=%.4f" % (s, side, g2f["compound_frac"]))
            print("R_L2M_S%d_%s_G2S_PROMO=%d" % (s, side, g2f["n_promo"]))
            print("R_L2M_S%d_%s_G2S_NCONF=%d" % (s, side,
                                                  r[key]["verify"]["n_confirmed"]))
        for side, key in (("A", "g2ra"), ("B", "g2rb")):
            g2f = r[key]["finalize"]
            print("R_L2M_S%d_%s_G2R_COMP=%.4f" % (s, side, g2f["compound_frac"]))
            print("R_L2M_S%d_%s_G2R_PROMO=%d" % (s, side, g2f["n_promo"]))
            print("R_L2M_S%d_%s_G2R_NCONF=%d" % (s, side,
                                                  r[key]["verify"]["n_confirmed"]))
            for i, (w, kids, okv, p, tot, per) in enumerate(
                    r[key]["verify"]["details"]):
                print("R_L2M_S%d_%s_G2R_VERIFY_P%d_W=%d" % (s, side, i, w))
                print("R_L2M_S%d_%s_G2R_VERIFY_P%d_OK=%d" % (s, side, i, okv))
                print("R_L2M_S%d_%s_G2R_VERIFY_P%d_PURITY=%.6f" % (s, side, i, p))
        print("R_L2M_S%d_A_W_PROMO=%d" % (s, r["wa"]["finalize"]["n_promo"]))
        print("R_L2M_S%d_B_W_PROMO=%d" % (s, r["wb"]["finalize"]["n_promo"]))
        print("R_L2M_S%d_A_W_NCONF=%d" % (s, r["wa"]["verify"]["n_confirmed"]))
        print("R_L2M_S%d_B_W_NCONF=%d" % (s, r["wb"]["verify"]["n_confirmed"]))
        print("R_L2M_S%d_JR_A0=%.6f" % (s, r["jr_a0"]))
        print("R_L2M_S%d_JR_B0=%.6f" % (s, r["jr_b0"]))
        print("R_L2M_S%d_JR_A1=%.6f" % (s, r["jr_a1"]))
        print("R_L2M_S%d_JR_B1=%.6f" % (s, r["jr_b1"]))
        print("R_L2M_S%d_JR_RATIO_A=%.6f" % (s, r["jr_ratio_a"]))
        print("R_L2M_S%d_JR_RATIO_B=%.6f" % (s, r["jr_ratio_b"]))
        print("R_L2M_S%d_TRANSFER_A=%.6f" % (s, r["transfer_a"]))
        print("R_L2M_S%d_CALIB_A=%.6f" % (s, r["calib_a"]))
        print("R_L2M_S%d_TRANSFER_B=%.6f" % (s, r["transfer_b"]))
        print("R_L2M_S%d_CALIB_B=%.6f" % (s, r["calib_b"]))
        print("R_L2M_S%d_FIRST_PROMO_A=%s" % (s, ("NA" if r["first_promo_a"] is None
                                                   else str(r["first_promo_a"]))))
        print("R_L2M_S%d_FIRST_PROMO_B=%s" % (s, ("NA" if r["first_promo_b"] is None
                                                   else str(r["first_promo_b"]))))
        if r["gate_a"] is None:
            print("R_L2M_S%d_GATE_PURITY_A=NA" % s)
        else:
            print("R_L2M_S%d_GATE_PURITY_A=%.4f" % (s, r["gate_a"]["purity"]))
        if r["gate_b"] is None:
            print("R_L2M_S%d_GATE_PURITY_B=NA" % s)
        else:
            print("R_L2M_S%d_GATE_PURITY_B=%.4f" % (s, r["gate_b"]["purity"]))
    print("R_L2M_MAE_A=%.6f" % agg["A"]["mae_mean"])
    print("R_L2M_MAE_A_SD=%.6f" % agg["A"]["mae_sd"])
    print("R_L2M_MAE_B=%.6f" % agg["B"]["mae_mean"])
    print("R_L2M_MAE_B_SD=%.6f" % agg["B"]["mae_sd"])
    print("R_L2M_SC2_A=%.4f" % agg["A"]["sc2_mean"])
    print("R_L2M_SC2_B=%.4f" % agg["B"]["sc2_mean"])
    print("R_L2M_COMP_A=%.4f" % agg["A"]["comp_mean"])
    print("R_L2M_COMP_B=%.4f" % agg["B"]["comp_mean"])
    print("R_L2M_CHURN_A=%.4f" % agg["A"]["churn_mean"])
    print("R_L2M_CHURN_B=%.4f" % agg["B"]["churn_mean"])
    print("R_L2M_PROMO_A=%.4f" % agg["A"]["promo_mean"])
    print("R_L2M_PROMO_B=%.4f" % agg["B"]["promo_mean"])
    print("R_L2M_FID_A=%.4f" % fid_a_m)
    print("R_L2M_FID_B=%.4f" % fid_b_m)
    print("R_L2M_JR_A0=%.6f" % jr_a0_m)
    print("R_L2M_JR_B0=%.6f" % jr_b0_m)
    print("R_L2M_JR_A1=%.6f" % jr_a1_m)
    print("R_L2M_JR_B1=%.6f" % jr_b1_m)
    print("R_L2M_JR_RATIO_A_FULL=%.6f" % jr_ratio_a_m)
    print("R_L2M_JR_RATIO_B_FULL=%.6f" % jr_ratio_b_m)
    print("R_L2M_JR_RATIO_A_SUBSET=%.6f" % jr_subset_a)
    print("R_L2M_JR_RATIO_B_SUBSET=%.6f" % jr_subset_b)
    print("R_L2M_ADOPT_A_RAW=%.4f" % adopt_raw_a)
    print("R_L2M_ADOPT_B_RAW=%.4f" % adopt_raw_b)
    print("R_L2M_ADOPT_A_CONF=%.4f" % adopt_conf_a)
    print("R_L2M_ADOPT_B_CONF=%.4f" % adopt_conf_b)
    print("R_L2M_COMP_ADOPTED_A=%.4f" % comp_adopted_a)
    print("R_L2M_COMP_ADOPTED_B=%.4f" % comp_adopted_b)
    print("R_L2M_ADOPT_A_W_CONF=%.4f" % adopt_conf_a_w)
    print("R_L2M_ADOPT_B_W_CONF=%.4f" % adopt_conf_b_w)
    print("R_L2M_TRANSFER_A_MEAN=%.6f" % transfer_a_m)
    print("R_L2M_CALIB_A_MEAN=%.6f" % calib_a_m)
    print("R_L2M_TRANSFER_B_MEAN=%.6f" % transfer_b_m)
    print("R_L2M_CALIB_B_MEAN=%.6f" % calib_b_m)
    print("R_L2M_DIAG_ERATIO_A=%.4f" % eratio_a_m)
    print("R_L2M_DIAG_ERATIO_B=%.4f" % eratio_b_m)
    print("R_L2M_DESIGN_WIN_A_FRAC=%.4f" % design_win_a_frac)
    print("R_L2M_DESIGN_WIN_B_FRAC=%.4f" % design_win_b_frac)
    print("R_L2M_CRIT_C1A=%d" % c1a)
    print("R_L2M_CRIT_C1B=%d" % c1b)
    print("R_L2M_CRIT_C2=%d" % c2)
    print("R_L2M_CRIT_C3=%d" % c3)
    print("R_L2M_CRIT_C4=%d" % c4)
    print("R_L2M_CRIT_C4A=%d" % c4a)
    print("R_L2M_CRIT_C4B=%d" % c4b)
    print("R_L2M_CRIT1_JR_RATIO_A_SUBSET=%.6f" % jr_subset_a)
    print("R_L2M_CRIT1_JR_RATIO_B_SUBSET=%.6f" % jr_subset_b)
    print("R_L2M_CRIT1_JR_RATIO_A_FULL=%.6f" % jr_ratio_a_m)
    print("R_L2M_CRIT1_JR_RATIO_B_FULL=%.6f" % jr_ratio_b_m)
    print("R_L2M_CRIT2_ADOPT_A=%.4f" % adopt_conf_a)
    print("R_L2M_CRIT2_ADOPT_B=%.4f" % adopt_conf_b)
    print("R_L2M_CRIT2_COMP_ADOPTED_A=%.4f" % comp_adopted_a)
    print("R_L2M_CRIT2_COMP_ADOPTED_B=%.4f" % comp_adopted_b)
    print("R_L2M_CRIT2_G0A_ZERO=%d" % int(sum(cons_g0a)))
    print("R_L2M_CRIT2_G0B_ZERO=%d" % int(sum(cons_g0b)))
    print("R_L2M_CRIT2_G1NA_ZERO=%d" % int(sum(cons_g1na)))
    print("R_L2M_CRIT2_G1NB_ZERO=%d" % int(sum(cons_g1nb)))
    print("R_L2M_CRIT3_SPURIOUS_G2R_A=%d" % len(spurious_g2r_a))
    print("R_L2M_CRIT3_SPURIOUS_G2R_B=%d" % len(spurious_g2r_b))
    print("R_L2M_CRIT3_SPURIOUS_G2R_A_SEEDS=%s" % (
        ",".join(str(v) for v in spurious_g2r_a) if spurious_g2r_a else "NONE"))
    print("R_L2M_CRIT3_SPURIOUS_G2R_B_SEEDS=%s" % (
        ",".join(str(v) for v in spurious_g2r_b) if spurious_g2r_b else "NONE"))
    print("R_L2M_CRIT3_SPURIOUS_ORIG_A=%d" % len(spurious_g2s_a))
    print("R_L2M_CRIT3_SPURIOUS_ORIG_B=%d" % len(spurious_g2s_b))
    print("R_L2M_CRIT3_SPURIOUS_ORIG_A_SEEDS=%s" % (
        ",".join(str(v) for v in spurious_g2s_a) if spurious_g2s_a else "NONE"))
    print("R_L2M_CRIT3_SPURIOUS_ORIG_B_SEEDS=%s" % (
        ",".join(str(v) for v in spurious_g2s_b) if spurious_g2s_b else "NONE"))
    print("R_L2M_CRIT4_TRANSFER_A=%.6f" % transfer_a_m)
    print("R_L2M_CRIT4_CALIB_A=%.6f" % calib_a_m)
    print("R_L2M_CRIT4_TRANSFER_B=%.6f" % transfer_b_m)
    print("R_L2M_CRIT4_CALIB_B=%.6f" % calib_b_m)
    print("R_L2M_VERDICT=%s" % verdict)
    print("R_L2M_VERDICT_NOTE=%s" % vnote)
    print("R_L2M_ELAPSED=%.2f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
