"""lineB-motion-coupling/scripts/motion_coupling_test.py — B 路第一格：运动耦合判别
（合成受控三臂，docs: lineB-motion-coupling/docs/B1-运动耦合判别-预注册设计.md §一 冻结）。

核心：把 docs/260-261 的阴影判别从几何路（V1 闭合轮廓 / V3 主轴）换到行为路（运动耦合）。
三臂（同一合成场景族变体，docs/260 风格 + 独立运动学）：
  A 真实阴影   ：遮挡物（亮圆 255，docs/260 逐字）+ 有限长投影阴影带（沿 −L 长 40px、
                 宽 20px、×0.5、只投地面）——带质心 = 遮挡物质心 + 常数偏移 → 共享运动场；
  B 运动物体   ：遮挡物（无阴影）+ 独立轨道暗物体（gray 40，轨道 (44,22) r=11 f=0.13 Hz）；
  C 暗色移动物体：遮挡物（无阴影）+ 独立轨道暗圆盘（gray 32 = 阴影最暗值，docs/261 对照臂
                 移动版、adversarial 亮度匹配）；B/C 为配对臂（唯一差异 = 灰度 40 vs 32）。

行为量（零几何规则；全从两候选质心轨迹 + 预测回路自身状态算，§1.3 冻结）：
  1. 轨迹耦合度 COUPLING：coup = Pearson([vx_d,vy_d],[vx_o,vy_o])，v(t)=c(t+2)−c(t−2)；
  2. 相位滞后 PHASE_LAG：去均值轨迹互相关 ρ(τ), τ∈[−10,10]；lag_peak=argmax、peak_val=max；
  3. 事件-预测比 EVENT_PRED：pred_ratio = 1 − S_dark/S_occ（|L−bg_slow| 于暗候选 vs 遮挡物
     掩码的均值比；docs/187 慢背景吸收载体）。
分类规则（冻结）：shadow = (coup≥0.70) ∧ (peak_val≥0.60) ∧ (|lag_peak|≤2) ∧ (pred≥0.30)；
coverage（两候选同时有效帧占比）< 0.6 → coup/peak = NaN → 判为物体。

判据（§1.4 冻结，docs/247 标签 [L1][机制][合成受控]）：
  C1 COUPLING_SEP : frac_A(coup≥0.70) ≥ 0.70 且 frac_B ≤ 0.30 且 frac_C ≤ 0.30
  C2 SHADOW_CLS   : 三臂 pooled 分类正确率 ≥ 0.75
  C3 DARK_OBJ_REJECT : C 臂被判为"物体"比例 ≥ 0.70
  C4 KEEP 守卫    : R_B1_GUARD_CELL260 + R_B1_GUARD_CP5 + R_B1_GUARD_MAD + R_B1_REPRO 全 = 1
判定（§1.5）：全过 = COUPLING_PASS；COUPLING_SEP 过但 C2/C3 不过 = PARTIAL；
COUPLING_SEP 不过 = COUPLING_FAIL；守卫不过 = GUARD_FAIL。

守卫（§1.6 冻结）：
  R_B1_GUARD_CELL260：import 第一格 run_unit 重跑全部 docs/260 原场景 (级,种子)，聚合
    det/fp/ld_med/sfs 与 docs/260 §3.2 公布值（1.0000/0.0000/0.4821/4.6258）逐位一致（容差 1e-3）
  R_B1_GUARD_CP5 / R_B1_GUARD_MAD：import 第一格 guard_cp5 / guard_mad
  R_B1_REPRO：--repro 时 A/B/C 120 运行整体重跑第二遍（不读 checkpoint），关键数字位级一致

安全纪律（docs/228/234/235）：stdout 只输出 ASCII 标签 + 每行一个数字的 R_B1_* 摘要块
（顺序固定）；JSON 归档 lineB-motion-coupling/out/mc_<tag>.json + checkpoint
ckpt_mc_<hash>.json（--resume 断点续跑）；数字用 vision/extract_r.py 纯正则抽取；
禁止读取 lineB-motion-coupling/out/*.log 与 lineB-motion-coupling/out/*.json 原文。
**未修改任何主线既有脚本**（vision/ 下全部不动；只 import 复用）。

用法：
  python lineB-motion-coupling/scripts/motion_coupling_test.py --levels 30,31,32,33 --n-seeds 10 --tag main --repro
  python lineB-motion-coupling/scripts/motion_coupling_test.py --levels 30 --n-seeds 1 --tag timing
"""
import argparse
import hashlib
import json
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))          # 项目根（lineB-motion-coupling/ 的父目录）
VISION = os.path.join(PROJ, "vision")
for _p in (VISION, PROJ):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from critical_point import CPLoop, mean_sd, bootstrap_ci, JITTER
from light_shadow_test import (
    make_shadow_scene, run_unit, largest_component, iou, circ_err_deg,
    guard_cp5, guard_mad,
    W, H, BG_CELL, BG_DARK, BG_BRIGHT,
    SPHERE_C, SPHERE_R, OCC_R, OCC_GRAY, ORBIT_C, ORBIT_R, OCC_FREQ,
    SHADOW_MULT, NOISE_SIGMA,
    DELTA_SHADOW, A_MIN, K_MOVE, MOVE_IOU, WARMUP, SFS_LO, SFS_HI,
    OCC_LUM_THRESH, LIGHT_AZIMUTH,
    IOU_DET, IOU_FP, ANG_TOL,
)

warnings.filterwarnings("ignore")

DEFAULT_OUT = os.path.join("lineB-motion-coupling", "out")
N_BOOT = 2000
BOOT_SEED = 20260828

# ---- 三臂场景旋钮（docs/B1 §1.2 冻结）----
LEN_SHADOW = 40.0        # 有限长阴影带长度（沿 −L），px
# 诊断轮 D0 修复（§二 记录）：§1.2 初值轨道 (42,28,r=15) 与球面上缘相交（40/40 单位
# 各 1-5 帧，违反冻结语义"不与球面相交"）；修复值 (44,22,r=11)——圆盘 y 上界 43 < 44
# （球面顶），相交不可能（构造保证）；与遮挡物轨道角区复核 0 相交；独立运动场语义不变
OBJ_ORBIT_C = (44.0, 22.0)
OBJ_ORBIT_R = 11.0
OBJ_FREQ = 0.13          # Hz（独立轨道，与遮挡物 0.18 Hz 不同 → 独立运动场）
OBJ_R = 10.0
OBJ_GRAY_B = 40.0        # B 臂暗物体灰度（可检出：log(65)−log(41)≈0.46 > δ）
OBJ_GRAY_C = 32.0        # C 臂暗圆盘灰度 = 阴影最暗值 64×0.5（docs/261 CTRL_OCC_GRAY 同款）
OBJ_RNG_OFFSET = 99991   # 对象相位 rng 偏移常数（与遮挡物相位独立的确定性派生，B/C 配对）

# ---- 行为量旋钮（docs/B1 §1.3 冻结）----
COUP_HIGH = 0.70
PEAK_MIN = 0.60
LAG_TOL = 2
PRED_MIN = 0.30
COV_MIN = 0.60
LAG_MAX = 10
VEL_DT = 2               # 速度 = 5 帧中心差分（v(t) = c(t+2) − c(t−2)）

# ---- 判据阈值（docs/B1 §1.4 冻结）----
SEP_A_MIN = 0.70
SEP_BC_MAX = 0.30
CLS_MIN = 0.75
REJ_MIN = 0.70

ARMS = ["A", "B", "C"]
REPRO_KEYS = ["coup", "lag_peak", "peak_val", "pred_ratio", "coverage",
              "shadow", "sfs_err"]
REPRO_KEYS_A = REPRO_KEYS + ["det_rate", "fp_rate", "ld_med", "ld_acc"]


# ---------------- 三臂场景（docs/B1 §1.2 冻结） ----------------
def make_arm_scene(lvcode, seed, arm, n_frames=240, jitter=JITTER):
    """生成 (臂,级,种子) 的灰度帧序列 + GT 掩码（只用于评估，绝不进入机制）。

    A：make_shadow_scene（docs/260 逐字：遮挡物 + 半无限阴影带）→ 阴影带截为有限长 40px
       （new_band ⊆ old_shadow：超出部分恢复 bg×0.5，噪声场不变）；GT shadow = new_band。
    B/C：恢复全部阴影（img[sh] += bg×0.5）→ GT shadow 置空 → 叠加独立轨道暗物体
       （B gray 40 / C gray 32，相位由 rng_obj = default_rng(seed×7919+lvcode×104729+13+
       99991) 独立抽取——B/C 同 (级,种子) 同相位 → 配对臂）。确定性：全部 rng 调用由
       (seed, lvcode) 派生，无全局状态。
    """
    frames, gts = make_shadow_scene(lvcode, seed, n_frames=n_frames, jitter=jitter)
    bg = np.zeros((H, W), np.float32)
    for y in range(0, H, BG_CELL):
        for x in range(0, W, BG_CELL):
            bg[y:y + BG_CELL, x:x + BG_CELL] = \
                BG_DARK if ((x // BG_CELL) + (y // BG_CELL)) % 2 == 0 else BG_BRIGHT
    theta = LIGHT_AZIMUTH[lvcode]
    Lv = np.array([np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))])
    sphere = gts["sphere_mask"]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    out_frames = []
    if arm == "A":
        for t, g in enumerate(frames):
            img = g.astype(np.float32).copy()
            gt = gts["per_frame"][t]
            ox, oy = gt["occ_pos"]
            vx = xx - ox
            vy = yy - oy
            vdot = vx * Lv[0] + vy * Lv[1]
            vp_x = vx - vdot * Lv[0]
            vp_y = vy - vdot * Lv[1]
            dist_perp = np.sqrt(vp_x * vp_x + vp_y * vp_y)
            old_sh = gt["shadow"]
            new_band = (dist_perp <= OCC_R) & (vdot < 0) & (vdot >= -LEN_SHADOW) \
                & (~sphere) & (~gt["occ"])
            undo = old_sh & (~new_band)
            if undo.any():
                img[undo] += bg[undo] * SHADOW_MULT      # 恢复 ×0.5（噪声场不变）
            out_frames.append(np.clip(img, 0, 255).astype(np.uint8))
            gt["shadow"] = new_band
        return out_frames, gts

    # B / C：无阴影 + 独立轨道暗物体
    gray = OBJ_GRAY_B if arm == "B" else OBJ_GRAY_C
    rng_obj = np.random.default_rng(seed * 7919 + lvcode * 104729 + 13 + OBJ_RNG_OFFSET)
    phase = rng_obj.uniform(0, 2 * np.pi)
    for t, g in enumerate(frames):
        img = g.astype(np.float32).copy()
        gt = gts["per_frame"][t]
        sh = gt["shadow"]
        if sh.any():
            img[sh] += bg[sh] * SHADOW_MULT
        ang = 2 * np.pi * OBJ_FREQ * t / 30.0 + phase
        ocx = OBJ_ORBIT_C[0] + OBJ_ORBIT_R * np.cos(ang)
        ocy = OBJ_ORBIT_C[1] + OBJ_ORBIT_R * np.sin(ang)
        omask = (xx - ocx) ** 2 + (yy - ocy) ** 2 <= OBJ_R * OBJ_R
        img[omask] = gray
        out_frames.append(np.clip(img, 0, 255).astype(np.uint8))
        gt["shadow"] = np.zeros((H, W), bool)
        gt["obj"] = omask
    return out_frames, gts


def _interp_traj(centroids, lo, hi, gap=5):
    """质心序列（eval 窗内）：缺口 ≤ gap 帧线性插值；返回 (xs, ys) 长度 = hi−lo。"""
    n = hi - lo
    xs = np.full(n, np.nan)
    ys = np.full(n, np.nan)
    for i in range(n):
        c = centroids[lo + i]
        if c is not None:
            xs[i], ys[i] = c[0], c[1]
    i = 0
    while i < n:
        if not np.isnan(xs[i]):
            i += 1
            continue
        j = i
        while j < n and np.isnan(xs[j]):
            j += 1
        if j - i <= gap and i > 0 and j < n:
            x0, y0 = xs[i - 1], ys[i - 1]
            x1, y1 = xs[j], ys[j]
            for k in range(i, j):
                w = (k - (i - 1)) / (j - (i - 1))
                xs[k] = x0 + w * (x1 - x0)
                ys[k] = y0 + w * (y1 - y0)
        i = j
    return xs, ys


def pearson(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 2 or len(a) != len(b):
        return float("nan")
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _z(a):
    """逐轴标准化（诊断轮 D2 修复，§二 记录）：相位互相关的拼接 Pearson 在轴间常数
    偏移不同（如 L33 带偏移 (−16.9,+16.9)）时被轴偏移差污染（逐轴相关 0.9999 而拼接
    0.18）；逐轴标准化后拼接 ≡ 逐轴 Pearson 平均（机制语义不变：去均值互相关的峰值
    滞后/峰值）。"""
    a = np.asarray(a, float)
    s = a.std()
    if s > 0:
        return (a - a.mean()) / s
    return a - a.mean()


def run_arm(lvcode, seed, arm, n_frames=240, jitter=JITTER):
    """跑 (臂,级,种子) 一次完整运行 → 行为量 + 分类 + KEEP 对齐度量。"""
    theta = LIGHT_AZIMUTH[lvcode]
    frames, gts = make_arm_scene(lvcode, seed, arm, n_frames=n_frames, jitter=jitter)
    loop = CPLoop(window=10)
    eval_lo = WARMUP
    eval_hi = n_frames
    n_eval = max(1, eval_hi - eval_lo)
    occ_thresh_log = np.log(OCC_LUM_THRESH + 1.0)

    # ---- Pass A：适应（docs/260 诊断轮 D2 冻结口径，逐字同实现）----
    buf = np.empty((n_frames, H, W), np.float32)
    for t, g in enumerate(frames):
        buf[t] = g.astype(np.float32)
    buf_masked = buf.copy()
    for t, g in enumerate(frames):
        Lf = np.log(np.maximum(g.astype(np.float32), 1.0))
        buf_masked[t][Lf > occ_thresh_log] = np.nan
    ref_dark = np.nanpercentile(buf_masked, 95.0, axis=0)
    ref_dark = np.nan_to_num(ref_dark, nan=0.0)
    ref_dark_log = np.log(np.maximum(ref_dark, 1.0))

    c_d = [None] * n_frames
    c_o = [None] * n_frames
    shadow_masks = [None] * n_frames
    S_dark = 0.0
    S_occ = 0.0
    n_dark = 0
    n_occ = 0
    det_flags, fp_flags, ld_errs = [], [], []
    gray_mean = np.zeros((H, W), np.float64)

    # ---- Pass B：判断（预测回路 + 检测 + 行为量逐帧累积）----
    for t, g in enumerate(frames):
        L = np.log(np.maximum(g.astype(np.float32), 1.0))
        loop.step(g)
        if t >= eval_lo:
            gray_mean += g.astype(np.float64) / n_eval
        dark = (ref_dark_log - L) > DELTA_SHADOW
        bright = L > occ_thresh_log
        d_mask, d_area, d_c = largest_component(dark)
        b_mask, b_area, b_c = largest_component(bright)
        shadow_masks[t] = d_mask
        c_d[t] = d_c if d_area >= A_MIN else None
        c_o[t] = b_c if b_area >= A_MIN else None

        # 事件-预测比累积（docs/187 慢背景吸收载体；bg_slow = 回路自身状态）
        res = np.abs(L - loop.bg_slow)
        if d_area >= A_MIN:
            S_dark += float(res[d_mask].mean())
            n_dark += 1
        if b_area >= A_MIN:
            S_occ += float(res[b_mask].mean())
            n_occ += 1

        # KEEP 对齐度量（A 臂 docs/260 口径；B/C 臂 docs/260 口径盲区 fp 暴露）
        gt_sh = gts["per_frame"][t]["shadow"]
        if t >= eval_lo:
            active = (d_area >= A_MIN)
            if t >= eval_lo + K_MOVE:
                m0 = shadow_masks[t - K_MOVE]
                move = 1.0 - iou(d_mask, m0)
                active = active and (move >= MOVE_IOU)
            iou_v = iou(d_mask, gt_sh)
            if arm == "A":
                det_flags.append(1.0 if (active and iou_v >= IOU_DET) else 0.0)
                fp_flags.append(1.0 if (active and iou_v < IOU_FP) else 0.0)
                if active and d_c is not None and b_c is not None:
                    ex, ey = b_c[0] - d_c[0], b_c[1] - d_c[1]
                    est = float(np.rad2deg(np.arctan2(ey, ex))) % 360.0
                    ld_errs.append(circ_err_deg(est, theta))
            else:
                fp_flags.append(1.0 if active else 0.0)   # GT 阴影空 → 任何 active 都是 FP

    # ---- 轨迹耦合度 + 相位滞后（零几何规则，§1.3 冻结）----
    both = [1 if (c_d[t] is not None and c_o[t] is not None)
            else 0 for t in range(eval_lo, eval_hi)]
    coverage = float(np.mean(both))
    xd, yd = _interp_traj(c_d, eval_lo, eval_hi, gap=5)
    xo, yo = _interp_traj(c_o, eval_lo, eval_hi, gap=5)
    coup = float("nan")
    lag_peak = float("nan")
    peak_val = float("nan")
    if coverage >= COV_MIN:
        vdx = np.array([xd[i + VEL_DT] - xd[i - VEL_DT] for i in range(VEL_DT, n_eval - VEL_DT)])
        vdy = np.array([yd[i + VEL_DT] - yd[i - VEL_DT] for i in range(VEL_DT, n_eval - VEL_DT)])
        vox = np.array([xo[i + VEL_DT] - xo[i - VEL_DT] for i in range(VEL_DT, n_eval - VEL_DT)])
        voy = np.array([yo[i + VEL_DT] - yo[i - VEL_DT] for i in range(VEL_DT, n_eval - VEL_DT)])
        coup = pearson(np.concatenate([vdx, vdy]), np.concatenate([vox, voy]))
        best_tau, best_val = 0, -1.0
        for tau in range(-LAG_MAX, LAG_MAX + 1):
            if tau >= 0:
                ia = list(range(tau, n_eval))
                ib = [i - tau for i in ia]
            else:
                ib = list(range(-tau, n_eval))
                ia = [i + tau for i in ib]
            # 诊断轮 D2 修复（§二 记录）：逐轴标准化后拼接（≡ 逐轴 Pearson 平均），
            # 消除轴间常数偏移差对拼接 Pearson 的伪相关（L33 带偏移 (−16.9,+16.9)）
            a = np.concatenate([_z(np.array([xd[i] for i in ia])),
                                _z(np.array([yd[i] for i in ia]))])
            b = np.concatenate([_z(np.array([xo[i] for i in ib])),
                                _z(np.array([yo[i] for i in ib]))])
            v = pearson(a, b)
            if v == v and v > best_val:
                best_val = v
                best_tau = tau
        peak_val = float(best_val)
        lag_peak = float(best_tau)

    # ---- 事件-预测比 + 分类（冻结规则 §1.3）----
    if n_occ > 0 and n_dark > 0:
        pred_ratio = 1.0 - (S_dark / max(1, n_dark)) / (S_occ / max(1, n_occ))
    else:
        pred_ratio = 0.0
    pred_ratio = float(np.clip(pred_ratio, -1.0, 1.0))
    shadow = bool(coup >= COUP_HIGH and peak_val >= PEAK_MIN
                  and abs(lag_peak) <= LAG_TOL and pred_ratio >= PRED_MIN)

    # ---- SfS（docs/260 §1.3 同实现，KEEP 报告；球面不变 → 三臂应一致）----
    dark_cap, cap_area, cap_c = largest_component(gray_mean < SFS_LO, min_area=50)
    if cap_area > 0 and cap_c is not None:
        pidx = int(np.argmax(gray_mean))
        py, px = divmod(pidx, W)
        sfs_err = circ_err_deg(
            float(np.rad2deg(np.arctan2(py - cap_c[1], px - cap_c[0]))) % 360.0, theta)
    else:
        sfs_err = float("nan")

    det_rate = float(np.mean(det_flags)) if det_flags else 0.0
    fp_rate = float(np.mean(fp_flags)) if fp_flags else 0.0
    ld_med = float(np.median(ld_errs)) if ld_errs else float("nan")
    ld_acc = float(np.mean([1.0 if e <= ANG_TOL else 0.0 for e in ld_errs])) \
        if ld_errs else 0.0

    return dict(kind=arm, seed=seed, lvcode=lvcode, theta=theta,
                coup=coup, lag_peak=lag_peak, peak_val=peak_val,
                pred_ratio=pred_ratio, coverage=coverage, shadow=shadow,
                det_rate=det_rate, fp_rate=fp_rate,
                ld_med=ld_med, ld_acc=ld_acc, sfs_err=sfs_err)


# ---------------- 守卫（docs/B1 §1.6 冻结） ----------------
def guard_cell260(levels, seeds, n_frames=240):
    """同代码路径复现 docs/260 三连数字：import run_unit 重跑全部原场景 (级,种子)，
    聚合 det_rate/fp_rate/ld_med/sfs_mean vs docs/260 §3.2 公布值（1.0000/0.0000/
    0.4821/4.6258，容差 1e-3）。返回 (ok, det, fp, ld, sfs)。"""
    dets, fps, lds, sfss = [], [], [], []
    for lv in levels:
        for seed in seeds:
            r = run_unit(lv, seed, n_frames=n_frames)
            dets.append(r["det_rate"])
            fps.append(r["fp_rate"])
            lds.append(r["ld_med"])
            sfss.append(r["sfs_err"])
    det_m = float(np.mean(dets))
    fp_m = float(np.mean(fps))
    ld_m = float(np.nanmean(lds))
    sfs_m = float(np.nanmean(sfss))
    ok = (abs(det_m - 1.0000) <= 1e-3 and abs(fp_m - 0.0000) <= 1e-3
          and abs(ld_m - 0.4821) <= 1e-3 and abs(sfs_m - 4.6258) <= 1e-3)
    return (1 if ok else 0), det_m, fp_m, ld_m, sfs_m


# ---------------- 统计外壳（critical_point 同款） ----------------
def agg_col(vals):
    vals = [v for v in vals if v == v]     # 滤 NaN
    if not vals:
        return float("nan"), 0.0, [float("nan"), float("nan")]
    m, s = mean_sd(vals)
    lo, hi = bootstrap_ci(vals)
    return m, s, [float(lo), float(hi)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="30,31,32,33")
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--jitter", type=float, default=JITTER)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--tag", default="mc")
    ap.add_argument("--repro", action="store_true")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    levels = [int(x) for x in args.levels.split(",") if x.strip() != ""]
    seeds = list(range(args.first_seed, args.first_seed + args.n_seeds))
    t0 = time.time()

    cfg = {"levels": levels, "n_seeds": args.n_seeds, "first_seed": args.first_seed,
           "frames": args.frames, "jitter": args.jitter, "tag": args.tag,
           "arms": ARMS,
           "scene": {"light_azimuth": LIGHT_AZIMUTH, "sphere_c": list(SPHERE_C),
                     "sphere_r": SPHERE_R, "orbit_c": list(ORBIT_C),
                     "orbit_r": ORBIT_R, "occ_r": OCC_R, "occ_freq": OCC_FREQ,
                     "occ_gray": OCC_GRAY, "noise": NOISE_SIGMA,
                     "len_shadow": LEN_SHADOW,
                     "obj_orbit_c": list(OBJ_ORBIT_C), "obj_orbit_r": OBJ_ORBIT_R,
                     "obj_freq": OBJ_FREQ, "obj_r": OBJ_R,
                     "obj_gray_b": OBJ_GRAY_B, "obj_gray_c": OBJ_GRAY_C},
           "behavior": {"coup_high": COUP_HIGH, "peak_min": PEAK_MIN,
                        "lag_tol": LAG_TOL, "pred_min": PRED_MIN,
                        "cov_min": COV_MIN, "lag_max": LAG_MAX, "vel_dt": VEL_DT},
           "criteria": {"sep_a_min": SEP_A_MIN, "sep_bc_max": SEP_BC_MAX,
                        "cls_min": CLS_MIN, "rej_min": REJ_MIN},
           "detect": {"delta_shadow": DELTA_SHADOW, "a_min": A_MIN,
                      "k_move": K_MOVE, "move_iou": MOVE_IOU, "warmup": WARMUP,
                      "sfs_lo": SFS_LO, "occ_lum_thresh": OCC_LUM_THRESH}}
    ck_tag = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    ckpt_path = os.path.join(args.out_dir, "ckpt_mc_%s.json" % ck_tag)

    def run_all(use_resume=True):
        done = {}
        if use_resume and args.resume and not args.no_resume and os.path.exists(ckpt_path):
            with open(ckpt_path, encoding="utf-8") as f:
                done = json.load(f).get("per_unit", {})
        per_unit = dict(done)
        for arm in ARMS:
            for lv in levels:
                for seed in seeds:
                    key = "%s_%d_%d" % (arm, lv, seed)
                    if key in per_unit:
                        continue
                    per_unit[key] = run_arm(lv, seed, arm,
                                            n_frames=args.frames, jitter=args.jitter)
                    with open(ckpt_path, "w", encoding="utf-8") as f:
                        json.dump({"config": cfg, "per_unit": per_unit},
                                  f, ensure_ascii=False, indent=1)
                    print("PROGRESS", flush=True)
        return per_unit

    per_unit = run_all()
    units = {arm: [per_unit["%s_%d_%d" % (arm, lv, s)]
                   for lv in levels for s in seeds] for arm in ARMS}
    all_units = units["A"] + units["B"] + units["C"]

    def frac_high(arm_units):
        return float(np.mean([1.0 if (u["coup"] == u["coup"] and u["coup"] >= COUP_HIGH)
                              else 0.0 for u in arm_units]))

    frac_a = frac_high(units["A"])
    frac_b = frac_high(units["B"])
    frac_c = frac_high(units["C"])
    cls_acc = float(np.mean([1.0 if u["shadow"] == (u["kind"] == "A") else 0.0
                             for u in all_units]))
    c_rej = float(np.mean([1.0 if not u["shadow"] else 0.0 for u in units["C"]]))

    c1 = (frac_a >= SEP_A_MIN) and (frac_b <= SEP_BC_MAX) and (frac_c <= SEP_BC_MAX)
    c2 = cls_acc >= CLS_MIN
    c3 = c_rej >= REJ_MIN

    agg = {}
    for arm in ARMS:
        au = units[arm]
        coup_m, coup_s, coup_ci = agg_col([u["coup"] for u in au])
        lag_m, _, _ = agg_col([u["lag_peak"] for u in au])
        peak_m, _, _ = agg_col([u["peak_val"] for u in au])
        pred_m, pred_s, _ = agg_col([u["pred_ratio"] for u in au])
        cov_m, _, _ = agg_col([u["coverage"] for u in au])
        shd_m = float(np.mean([1.0 if u["shadow"] else 0.0 for u in au]))
        agg[arm] = {"coup": coup_m, "coup_sd": coup_s, "coup_ci": coup_ci,
                    "lag_peak": lag_m, "peak_val": peak_m,
                    "pred_ratio": pred_m, "pred_ratio_sd": pred_s,
                    "coverage": cov_m, "shadow_frac": shd_m,
                    "high_frac": {"A": frac_a, "B": frac_b, "C": frac_c}[arm]}
    # A 臂 KEEP 对齐（docs/260 口径，同代码路径）
    au = units["A"]
    agg["A"]["det_rate"] = float(np.mean([u["det_rate"] for u in au]))
    agg["A"]["fp_rate"] = float(np.mean([u["fp_rate"] for u in au]))
    agg["A"]["ld_med"] = float(np.nanmean([u["ld_med"] for u in au]))
    agg["A"]["ld_acc"] = float(np.mean([u["ld_acc"] for u in au]))
    agg["A"]["sfs"] = float(np.nanmean([u["sfs_err"] for u in au]))
    for arm in ("B", "C"):
        agg[arm]["fp_260"] = float(np.mean([u["fp_rate"] for u in units[arm]]))

    # ---- 守卫 ----
    g_cell, g_det, g_fp, g_ld, g_sfs = guard_cell260(levels, seeds, n_frames=args.frames)
    g_cp5, g_cp5_sc2, g_cp5_scl = guard_cp5(n_seeds=args.n_seeds, n_frames=args.frames)
    g_mad, g_mad_sig, g_mad_exp = guard_mad()
    guards_ok = (g_cell == 1) and (g_cp5 == 1) and (g_mad == 1)

    # ---- 判定（docs/B1 §1.5 冻结）----
    if not guards_ok:
        verdict = "GUARD_FAIL"
    elif c1 and c2 and c3:
        verdict = "COUPLING_PASS"
    elif not c1:
        verdict = "COUPLING_FAIL"
    else:
        verdict = "PARTIAL"

    # ---- 内部确定性复现（docs/B1 §1.6-4；第二遍强制重算，不读 checkpoint）----
    repro = 1
    if args.repro:
        per_unit2 = run_all(use_resume=False)
        for arm in ARMS:
            keys = REPRO_KEYS_A if arm == "A" else REPRO_KEYS
            for lv in levels:
                for s in seeds:
                    k = "%s_%d_%d" % (arm, lv, s)
                    for kk in keys:
                        if per_unit[k][kk] != per_unit2[k][kk]:
                            repro = 0

    out = {
        "artifact": "motion_coupling_test",
        "doc_ref": "lineB-motion-coupling/docs/B1",
        "config": cfg,
        "per_unit": per_unit,
        "aggregate": agg,
        "criteria": {"c1_coupling_sep": bool(c1), "c2_shadow_cls": bool(c2),
                     "c3_dark_obj_reject": bool(c3),
                     "frac_a_high": frac_a, "frac_b_high": frac_b,
                     "frac_c_high": frac_c, "cls_acc": cls_acc, "c_rej": c_rej},
        "guards": {"cell260": g_cell, "cell260_det": g_det, "cell260_fp": g_fp,
                   "cell260_ld": g_ld, "cell260_sfs": g_sfs,
                   "cp5": g_cp5, "cp5_sc2": g_cp5_sc2, "cp5_sc_late": g_cp5_scl,
                   "mad": g_mad, "mad_sig": g_mad_sig, "mad_expected": g_mad_exp,
                   "repro": repro},
        "verdict": verdict,
        "timing": {"elapsed_sec": round(time.time() - t0, 2)},
    }
    res_path = os.path.join(args.out_dir, "mc_%s.json" % args.tag)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 摘要块：ASCII 标签 + 每行一个数字（顺序固定）----
    print("R_B1_TAG=%s" % args.tag)
    print("R_B1_LEVELS=%d" % len(levels))
    print("R_B1_SEEDS=%d" % len(seeds))
    for lv in levels:
        for arm in ARMS:
            lu = [u for u in units[arm] if u["lvcode"] == lv]
            if not lu:
                continue
            print("R_B1_L%d_%s_COUP=%.4f" % (lv, arm, float(np.nanmean([u["coup"] for u in lu]))))
            print("R_B1_L%d_%s_LAG=%.4f" % (lv, arm, float(np.nanmean([u["lag_peak"] for u in lu]))))
            print("R_B1_L%d_%s_PEAK=%.4f" % (lv, arm, float(np.nanmean([u["peak_val"] for u in lu]))))
            print("R_B1_L%d_%s_PRED=%.4f" % (lv, arm, float(np.nanmean([u["pred_ratio"] for u in lu]))))
            print("R_B1_L%d_%s_SHADOW=%.4f" % (lv, arm, float(np.mean([1.0 if u["shadow"] else 0.0 for u in lu]))))
    for arm in ARMS:
        a = agg[arm]
        print("R_B1_%s_COUP=%.4f" % (arm, a["coup"]))
        print("R_B1_%s_COUP_SD=%.4f" % (arm, a["coup_sd"]))
        print("R_B1_%s_COUP_CI_LO=%.4f" % (arm, a["coup_ci"][0]))
        print("R_B1_%s_COUP_CI_HI=%.4f" % (arm, a["coup_ci"][1]))
        print("R_B1_%s_LAG=%.4f" % (arm, a["lag_peak"]))
        print("R_B1_%s_PEAK=%.4f" % (arm, a["peak_val"]))
        print("R_B1_%s_PRED=%.4f" % (arm, a["pred_ratio"]))
        print("R_B1_%s_COV=%.4f" % (arm, a["coverage"]))
        print("R_B1_%s_SHADOW=%.4f" % (arm, a["shadow_frac"]))
    for lv in levels:
        lu = [u for u in units["A"] if u["lvcode"] == lv]
        if lu:
            print("R_B1_L%d_A_DET=%.4f" % (lv, float(np.mean([u["det_rate"] for u in lu]))))
            print("R_B1_L%d_A_FP=%.4f" % (lv, float(np.mean([u["fp_rate"] for u in lu]))))
            print("R_B1_L%d_A_LD=%.4f" % (lv, float(np.nanmean([u["ld_med"] for u in lu]))))
            print("R_B1_L%d_A_SFS=%.4f" % (lv, float(np.nanmean([u["sfs_err"] for u in lu]))))
    print("R_B1_A_DET=%.4f" % agg["A"]["det_rate"])
    print("R_B1_A_FP=%.4f" % agg["A"]["fp_rate"])
    print("R_B1_A_LD_MED=%.4f" % agg["A"]["ld_med"])
    print("R_B1_A_SFS=%.4f" % agg["A"]["sfs"])
    print("R_B1_B_FP_260=%.4f" % agg["B"]["fp_260"])
    print("R_B1_C_FP_260=%.4f" % agg["C"]["fp_260"])
    print("R_B1_FRAC_A_HIGH=%.4f" % frac_a)
    print("R_B1_FRAC_B_HIGH=%.4f" % frac_b)
    print("R_B1_FRAC_C_HIGH=%.4f" % frac_c)
    print("R_B1_CLS_ACC=%.4f" % cls_acc)
    print("R_B1_C_REJ=%.4f" % c_rej)
    print("R_B1_C1=%s" % ("PASS" if c1 else "FAIL"))
    print("R_B1_C2=%s" % ("PASS" if c2 else "FAIL"))
    print("R_B1_C3=%s" % ("PASS" if c3 else "FAIL"))
    print("R_B1_GUARD_CELL260=%d" % g_cell)
    print("R_B1_GUARD_CELL260_DET=%.4f" % g_det)
    print("R_B1_GUARD_CELL260_FP=%.4f" % g_fp)
    print("R_B1_GUARD_CELL260_LD=%.4f" % g_ld)
    print("R_B1_GUARD_CELL260_SFS=%.4f" % g_sfs)
    print("R_B1_GUARD_CP5=%d" % g_cp5)
    print("R_B1_GUARD_CP5_SC2=%.4f" % g_cp5_sc2)
    print("R_B1_GUARD_MAD=%d" % g_mad)
    print("R_B1_GUARD_MAD_SIG=%.4f" % g_mad_sig)
    print("R_B1_REPRO=%d" % repro)
    print("R_B1_VERDICT=%s" % verdict)
    print("R_B1_ELAPSED=%.2f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
