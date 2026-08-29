"""vision/light_shadow_test.py — 纹理光影解耦：阴影检测 + 光照方向估计 + 表面朝向恢复雏形
（docs/260 预注册设计，判据/旋钮/守卫冻结；本脚本为唯一新增文件，import 复用既有脚本）。

场景（已知光源合成受控，docs/260 §1.2）：
  160×120 灰度，240 帧 @30fps；24px 棋盘格背景 64/96；静态朗伯球面（圆心 (18,60)
  半径 16，亮度 = 25 + 170·max(0, n·L)，平面内光 L_z=0）；移动亮圆遮挡物（轨道
  (80,60) 半径 14、频率 0.18 Hz、圆半径 10、灰度 255）；投影阴影 = dist_perp(p,o,L)
  ≤ 10 且 (p−o)·L < 0（半无限暗带，×0.5 亮度，只投背景地面）；噪声 σ=3.0 × jitter。
  光源方位 θ_L（场景参数，真值已知）：lvcode 30-33 → 45/135/225/315 度。

机制（docs/260 §1.3，预测路径零改动）：
  CPLoop（LOOP_CFG 冻结值）逐帧 step；检测读回路自己的 bg_slow（慢背景）与 L。
  dark_mask = (bg_slow − L) > δ_shadow（δ=0.35 log 域）；bright_mask = (L − bg_slow)
  > δ_bright；8-连通域（cv2.connectedComponents），面积 ≥ A_min=25；
  最大暗域 = 阴影候选、最大亮域 = 遮挡物候选；
  时间门（docs/199b 两条件 AND 的明度版）：shadow_active[t] = 面积 ≥ A_min AND
  质心位移 |c_t − c_{t−10}| ≥ move_min=3px。
  SfS：M = mean(灰度[60:])；region = 最大连通域 of {(M<45) OR (M>115)}；包围盒中心
  = 球心；p_max = region 内 argmax M；θ_sfs = angle(p_max − 球心)。

判据（docs/260 §1.4，冻结）：
  C1 SHADOW_DETECT  [新][机制][合成受控]：det_rate（IoU(阴影候选,GT)≥0.30 帧占比）≥ 0.80
                                         且 fp_rate（active 且 IoU<0.20）≤ 0.10
  C2 LIGHT_DIR      [新][机制][合成受控]：θ_est = angle(遮挡物质心 − 阴影质心)；
                                         每运行中位误差跨运行均值 ≤ 15° 且准确率(≤15°) ≥ 0.70
  C3 SFS            [新][机制][合成受控]：跨运行平均误差 ≤ 15° 且正确率(≤15°) ≥ 0.75
  判定：三判据全过 + 守卫全过 = SHADOW_DECOUPLE_PASS；C1/C2/C3 不过按名报
  （SHADOW_FAIL/LIGHT_DIR_FAIL/SFS_FAIL）；守卫不过 = GUARD_FAIL。

守卫（docs/260 §1.6，冻结）：
  R_LT_GUARD_CP5：critical_point.make_scene(5, seed) + CPLoop(window=10)，10 种子 →
                 复现 docs/232 L5：SC2 均值 ∈ [2.0,3.6]（2.8±0.42）且 SC_late 均值 ≥ 0.5
  R_LT_GUARD_MAD：恒定灰度 64 + 高斯噪声 σ=3.0 → CPLoop σ̂（log 域 MAD of ΔL）应收敛
                 到 √2·3/64 ≈ 0.0663（±25%）
  R_LT_REPRO：--repro 时整体重跑第二遍，逐项比对 R_LT_* 关键数字位级一致

安全纪律（docs/228/234/235）：stdout 只输出 ASCII 标签 + 每行一个数字的 R_LT_* 摘要块
（顺序固定）；JSON 归档 vision/out/results/ls_<tag>.json + checkpoint
ckpt_ls_<hash>.json（--resume 断点续跑）；数字用 vision/extract_r.py 纯正则抽取；
禁止读取 logs/*.log 与 vision/out/results/*.json 原文。

用法：
  python vision/light_shadow_test.py --levels 30,31,32,33 --n-seeds 10 --tag main --repro
  python vision/light_shadow_test.py --levels 30 --n-seeds 1 --frames 240 --tag timing
"""
import argparse
import hashlib
import json
import os
import sys
import time
import warnings

import numpy as np
import cv2

from critical_point import CPLoop, make_scene, mean_sd, bootstrap_ci, JITTER

warnings.filterwarnings("ignore")

DEFAULT_OUT = os.path.join("vision", "out", "results")
N_BOOT = 2000
BOOT_SEED = 20260828

# ---- 场景常量（docs/260 §1.2 冻结）----
W, H = 160, 120
FPS = 30
BG_CELL = 24
BG_DARK, BG_BRIGHT = 64.0, 96.0
SPHERE_C = (18.0, 60.0)
SPHERE_R = 16.0
SPHERE_BASE = 25.0
SPHERE_ALBEDO = 170.0
OCC_R = 10.0
OCC_GRAY = 255.0
ORBIT_C = (80.0, 60.0)
ORBIT_R = 14.0
OCC_FREQ = 0.18            # Hz（compose_test C0 a_freq 同款）
SHADOW_MULT = 0.5          # critical_point dips light=0.5 同款
NOISE_SIGMA = 3.0          # compose_test C0-C3 noise 同款

# ---- 机制旋钮（docs/260 §1.3 冻结）----
DELTA_SHADOW = 0.35        # log 域 = log(2)/2（0.5 倍阴影深度半量）
DELTA_BRIGHT = 0.35
A_MIN = 25                 # px
K_MOVE = 10                # 帧（= window）
MOVE_MIN = 3.0             # px
WARMUP = 60                # 帧（评估窗口 [WARMUP, N_FRAMES)）
SFS_LO, SFS_HI = 45.0, 115.0
# 判亮场景常量阈值（诊断轮 D2 冻结，§二 记录）：遮挡物灰度 255 与球面峰值 195 的
# 几何中点 sqrt(255·195)≈223，取整 220 → 亮检测 = L > log(221)（亮度显著性，人眼式，
# 不依赖背景参考；遮挡物 5.545 > 5.398，球面峰值 5.278 < 5.398，余量 ~0.12 log ≈ 2.7σ）
OCC_LUM_THRESH = 220.0
# 时间移动门（诊断轮 D2 冻结，§二 记录）：掩码帧间 IoU 变化下限——半无限阴影带质心
# 位移被带长变化抵消（中位 2.42px/10帧），移动签名改用掩码随时间的变化（docs/199b
# 帧间 |Δ| 在掩码空间的等价物）；静态内容已被 ref_dark 吸收，暗掩码只剩阴影带，
# IoU(t,t−K) ≤ 0.95 即显著帧间变化（实测 100% 评估帧覆盖，静态球面重叠 0.0）
MOVE_IOU = 0.05

# ---- 判据阈值（docs/260 §1.4 冻结）----
IOU_DET = 0.30
IOU_FP = 0.20
ANG_TOL = 15.0             # 度
C1_DET_MIN = 0.80
C1_FP_MAX = 0.10
C2_MED_MAX = 15.0
C2_ACC_MIN = 0.70
C3_MEAN_MAX = 15.0
C3_ACC_MIN = 0.75

# ---- 光源方位（lvcode -> θ_L 度，docs/260 §1.2 冻结）----
LIGHT_AZIMUTH = {30: 45.0, 31: 135.0, 32: 225.0, 33: 315.0}


def light_vec(theta_deg):
    t = np.deg2rad(theta_deg)
    return np.array([np.cos(t), np.sin(t)])


def make_shadow_scene(lvcode, seed, n_frames=240, width=W, height=H, fps=FPS,
                      jitter=JITTER):
    """生成 (级,种子) 的灰度帧序列 + 逐帧 GT 掩码（阴影/遮挡物/球面，只用于评估）。

    确定性：rng 由 (seed, lvcode) 派生（critical_point 同款公式；lvcode 30-33 与
    0-6 / 20-23 错开）。返回 (frames, gts)：
      frames: list[uint8 (H,W)]
      gts:    dict(light_azimuth=float, sphere_mask=bool(H,W),
                   per_frame=[dict(shadow=bool, occ=bool, occ_pos=(cx,cy))])
    """
    theta = LIGHT_AZIMUTH[lvcode]
    Lv = light_vec(theta)
    rng = np.random.default_rng(seed * 7919 + lvcode * 104729 + 13)
    noise_mult = rng.uniform(1 - jitter, 1 + jitter) if jitter > 0 else 1.0
    sigma = NOISE_SIGMA * noise_mult
    phase = rng.uniform(0, 2 * np.pi)

    # 背景棋盘格（critical_point 同款）
    bg = np.zeros((height, width), np.float32)
    for y in range(0, height, BG_CELL):
        for x in range(0, width, BG_CELL):
            bg[y:y + BG_CELL, x:x + BG_CELL] = \
                BG_DARK if ((x // BG_CELL) + (y // BG_CELL)) % 2 == 0 else BG_BRIGHT

    # 静态球面（朗伯，平面内光 L_z=0）
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    dx = xx - SPHERE_C[0]
    dy = yy - SPHERE_C[1]
    d2 = dx * dx + dy * dy
    sphere_mask = d2 <= SPHERE_R * SPHERE_R
    nx = np.zeros((height, width), np.float32)
    ny = np.zeros((height, width), np.float32)
    nx[sphere_mask] = dx[sphere_mask] / SPHERE_R
    ny[sphere_mask] = dy[sphere_mask] / SPHERE_R
    ndot = nx * Lv[0] + ny * Lv[1]
    sphere_lum = SPHERE_BASE + SPHERE_ALBEDO * np.maximum(ndot, 0.0)

    frames = []
    per_frame = []
    for t in range(n_frames):
        ang = 2 * np.pi * OCC_FREQ * t / fps + phase
        ocx = ORBIT_C[0] + ORBIT_R * np.cos(ang)
        ocy = ORBIT_C[1] + ORBIT_R * np.sin(ang)
        # 遮挡物圆盘掩码
        od2 = (xx - ocx) ** 2 + (yy - ocy) ** 2
        occ_mask = od2 <= OCC_R * OCC_R

        # 阴影：背景地面像素 p，dist_perp(p,o,L) ≤ R_occ 且 (p−o)·L < 0
        # 只投背景（非球面、非遮挡物）——docs/260 §五6 诚实边界
        vx = xx - ocx
        vy = yy - ocy
        vdot = vx * Lv[0] + vy * Lv[1]
        vperp_x = vx - vdot * Lv[0]
        vperp_y = vy - vdot * Lv[1]
        dist_perp = np.sqrt(vperp_x * vperp_x + vperp_y * vperp_y)
        shadow_geom = (dist_perp <= OCC_R) & (vdot < 0)
        ground = ~(sphere_mask | occ_mask)
        shadow_mask = shadow_geom & ground

        img = bg.copy()
        img[sphere_mask] = sphere_lum[sphere_mask]
        img[shadow_mask] *= SHADOW_MULT
        img[occ_mask] = OCC_GRAY
        img = img + rng.normal(0, sigma, img.shape).astype(np.float32)
        frames.append(np.clip(img, 0, 255).astype(np.uint8))
        per_frame.append(dict(shadow=shadow_mask, occ=occ_mask,
                              occ_pos=(float(ocx), float(ocy))))

    gts = dict(light_azimuth=theta, sphere_mask=sphere_mask, per_frame=per_frame)
    return frames, gts


def largest_component(mask_bool, min_area=A_MIN):
    """8-连通域：返回 (最大域掩码, 面积, 质心 or None)。mask_bool: bool(H,W)。"""
    if mask_bool.sum() == 0:
        return np.zeros_like(mask_bool), 0, None
    n, lab = cv2.connectedComponents(mask_bool.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask_bool), 0, None
    areas = np.bincount(lab.ravel())
    areas[0] = 0
    best = int(np.argmax(areas))
    if areas[best] < min_area:
        return np.zeros_like(mask_bool), 0, None
    comp = lab == best
    ys, xs = np.nonzero(comp)
    return comp, int(areas[best]), (float(xs.mean()), float(ys.mean()))


def circ_err_deg(a_deg, b_deg):
    d = abs((a_deg - b_deg) % 360.0)
    return min(d, 360.0 - d)


def iou(a, b):
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def run_unit(lvcode, seed, n_frames=240):
    """跑 (级,种子) 一次完整运行，返回指标 dict（判据 C1/C2/C3 的每运行标量）。

    人眼式两遍法（诊断轮 D2 修复，§二 记录；机制语义不变）：
      Pass A（适应）：全程 240 帧 → 每像素 0.85 分位亮度 = "该像素常见的无阴影亮度"
        （阴影占比 ≤0.65 < 0.85 → 免疫阴影低端污染；遮挡物 255 是高端污染，判暗时
        被抬高只会更保守，无害）。
      Pass B（判断）：判暗 = ref_dark − L > δ（阴影带）；判亮 = L > 场景常量阈值
        log(220+1)（遮挡物 255 与球面峰值 195 的几何中点，场景标定）——遮挡物是全局
        最亮移动物，用亮度显著性而非背景参考（人眼不靠背景参考判亮）。
    """
    theta = LIGHT_AZIMUTH[lvcode]
    frames, gts = make_shadow_scene(lvcode, seed, n_frames=n_frames)
    loop = CPLoop(window=10)
    eval_lo = WARMUP
    eval_hi = n_frames

    # ---- Pass A：适应（排除遮挡物帧后的亮度分位参考）----
    # 诊断轮 D2 修复（§二 记录，docs/199b 排除带同构）：0.85 分位对阴影（低端污染
    # ≤0.65）免疫，但对遮挡物 255（高端污染 ≤0.35）不免疫 → 轨道内缘像素参考被抬
    # 高到 255，遮挡物不在时的背景时刻被误判为暗（FP ~700px，污染阴影质心）。
    # 修复：统计参考时排除遮挡物帧（L > log(221)，场景常量阈值，观测驱动），对
    # 剩余帧取 0.95 分位——轨道像素非遮挡物帧中阴影占 ~92%，0.95 分位落在仅存的
    # 背景区；阴影主干（f_shadow 0.65）0.95 分位 = 背景；球面静态 = 自身亮度。
    occ_thresh_log = np.log(OCC_LUM_THRESH + 1.0)
    buf = np.empty((n_frames, H, W), np.float32)
    for t, g in enumerate(frames):
        buf[t] = g.astype(np.float32)
    buf_masked = buf.copy()
    for t, g in enumerate(frames):
        L = np.log(np.maximum(g.astype(np.float32), 1.0))
        buf_masked[t][L > occ_thresh_log] = np.nan
    ref_dark = np.nanpercentile(buf_masked, 95.0, axis=0)
    ref_dark = np.nan_to_num(ref_dark, nan=0.0)
    ref_dark_log = np.log(np.maximum(ref_dark, 1.0))

    shadow_masks = []      # 每帧阴影候选掩码（时间移动 IoU 门用）
    occ_cent = []          # 每帧遮挡物候选质心 or None
    det_flags = []         # 每评估帧 IoU ≥ IOU_DET
    fp_flags = []          # 每评估帧 active 且 IoU < IOU_FP
    ld_errs = []           # 每评估帧光源方位误差（度）
    gray_mean = np.zeros((H, W), np.float64)

    # ---- Pass B：判断（预测回路 + 检测）----
    for t, g in enumerate(frames):
        L = np.log(np.maximum(g.astype(np.float32), 1.0))
        loop.step(g)
        if t >= eval_lo:
            gray_mean += g.astype(np.float64) / max(1, eval_hi - eval_lo)
        dark = (ref_dark_log - L) > DELTA_SHADOW
        bright = L > occ_thresh_log
        d_mask, d_area, d_c = largest_component(dark)
        b_mask, b_area, b_c = largest_component(bright)
        shadow_masks.append(d_mask)
        occ_cent.append(b_c)

        if t >= eval_lo:
            gt = gts["per_frame"][t]
            gt_sh = gt["shadow"]
            active = (d_area >= A_MIN)
            if t >= eval_lo + K_MOVE:
                m0 = shadow_masks[t - K_MOVE]
                # 时间移动门（诊断轮 D2 修复，§二 记录）：掩码帧间 IoU 变化——
                # 半无限阴影带质心位移被带长变化抵消（中位 2.42px），移动签名改用
                # 掩码随时间的变化（docs/199b 帧间 |Δ| 在掩码空间的等价物）；静态
                # 内容已被 ref_dark 吸收，暗掩码只剩阴影带，IoU 下降即移动
                move = 1.0 - iou(d_mask, m0)
                active = active and (move >= MOVE_IOU)
            iou_v = iou(d_mask, gt_sh)
            det_flags.append(1.0 if (active and iou_v >= IOU_DET) else 0.0)
            fp_flags.append(1.0 if (active and iou_v < IOU_FP) else 0.0)
            if active and d_c is not None and b_c is not None:
                ex, ey = b_c[0] - d_c[0], b_c[1] - d_c[1]
                est = float(np.rad2deg(np.arctan2(ey, ex))) % 360.0
                ld_errs.append(circ_err_deg(est, theta))

    n_eval = max(1, eval_hi - eval_lo)
    det_rate = float(np.mean(det_flags)) if det_flags else 0.0
    fp_rate = float(np.mean(fp_flags)) if fp_flags else 0.0
    ld_med = float(np.median(ld_errs)) if ld_errs else float("nan")
    ld_acc = float(np.mean([1.0 if e <= ANG_TOL else 0.0 for e in ld_errs])) \
        if ld_errs else 0.0

    # ---- SfS（docs/260 §1.3，静态亮度场；暗帽质心 + 全局最亮像素连线 = L 方向）----
    dark_cap, cap_area, cap_c = largest_component(gray_mean < SFS_LO, min_area=50)
    if cap_area > 0 and cap_c is not None:
        pidx = int(np.argmax(gray_mean))
        py, px = divmod(pidx, W)
        sfs_err = circ_err_deg(
            float(np.rad2deg(np.arctan2(py - cap_c[1], px - cap_c[0]))) % 360.0, theta)
        sfs_px, sfs_py = int(px), int(py)
    else:
        sfs_err = float("nan")
        sfs_px = sfs_py = -1

    return dict(seed=seed, lvcode=lvcode, theta=theta,
                det_rate=det_rate, fp_rate=fp_rate,
                ld_med=ld_med, ld_acc=ld_acc,
                sfs_err=sfs_err, sfs_px=sfs_px, sfs_py=sfs_py,
                n_ld=len(ld_errs))


# ---------------- 守卫（docs/260 §1.6 冻结） ----------------
def guard_cp5(n_seeds=10, n_frames=240):
    """同代码路径复现 docs/232 L5：SC2 ∈ [2.0,3.6]、SC_late ≥ 0.5。"""
    sc2s, scls = [], []
    for seed in range(n_seeds):
        frames = make_scene(5, seed, n_frames=n_frames)
        loop = CPLoop(window=10)
        for g in frames:
            loop.step(g)
        out = loop.finalize(max(1, n_frames // 10))
        sc2s.append(out["sc2"])
        scls.append(out["sc_late"])
    sc2_mean = float(np.mean(sc2s))
    scl_mean = float(np.mean(scls))
    ok = (2.0 <= sc2_mean <= 3.6) and (scl_mean >= 0.5)
    return 1 if ok else 0, sc2_mean, scl_mean


def guard_mad(n_frames=200, sigma=3.0, gray=64.0):
    """MAD 噪声估计复现：恒定灰度 + 高斯噪声 → σ̂ ≈ √2·σ/gray（±25%）。"""
    expected = np.sqrt(2.0) * sigma / gray
    loop = CPLoop(window=10)
    rng = np.random.default_rng(20260828)
    g = np.full((H, W), gray, np.float32)
    for _ in range(n_frames):
        f = np.clip(g + rng.normal(0, sigma, g.shape), 0, 255).astype(np.uint8)
        loop.step(f)
    sig = loop.sigma_hat if loop.sigma_hat is not None else 0.0
    ok = abs(sig - expected) / expected <= 0.25
    return 1 if ok else 0, sig, expected


# ---------------- 统计外壳（critical_point 同款） ----------------
def aggregate(units):
    """units: list of per-unit dict → 跨运行聚合标量 + mean±SD + bootstrap CI。"""
    def col(k):
        return [u[k] for u in units if u[k] == u[k]]      # 滤 NaN
    def ms(k):
        m, s = mean_sd(col(k))
        return m, s
    det_m, det_s = ms("det_rate")
    fp_m, fp_s = ms("fp_rate")
    ld_m, ld_s = ms("ld_med")
    acc_m, acc_s = ms("ld_acc")
    sfs_m, sfs_s = ms("sfs_err")
    sfs_acc = float(np.mean([1.0 if u["sfs_err"] <= ANG_TOL else 0.0
                             for u in units if u["sfs_err"] == u["sfs_err"]]))
    det_ci = bootstrap_ci(col("det_rate"))
    fp_ci = bootstrap_ci(col("fp_rate"))
    ld_ci = bootstrap_ci(col("ld_med"))
    acc_ci = bootstrap_ci(col("ld_acc"))
    sfs_ci = bootstrap_ci(col("sfs_err"))
    return dict(det_rate=det_m, det_sd=det_s, det_ci=list(det_ci),
                fp_rate=fp_m, fp_sd=fp_s, fp_ci=list(fp_ci),
                ld_med=ld_m, ld_sd=ld_s, ld_ci=list(ld_ci),
                ld_acc=acc_m, ld_acc_sd=acc_s, ld_acc_ci=list(acc_ci),
                sfs_err=sfs_m, sfs_sd=sfs_s, sfs_ci=list(sfs_ci),
                sfs_acc=sfs_acc)


def verdict_of(agg):
    c1 = (agg["det_rate"] >= C1_DET_MIN) and (agg["fp_rate"] <= C1_FP_MAX)
    c2 = (agg["ld_med"] <= C2_MED_MAX) and (agg["ld_acc"] >= C2_ACC_MIN)
    c3 = (agg["sfs_err"] <= C3_MEAN_MAX) and (agg["sfs_acc"] >= C3_ACC_MIN)
    fails = []
    if not c1:
        fails.append("SHADOW_FAIL")
    if not c2:
        fails.append("LIGHT_DIR_FAIL")
    if not c3:
        fails.append("SFS_FAIL")
    return c1, c2, c3, (fails if fails else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="30,31,32,33")
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--jitter", type=float, default=JITTER)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--tag", default="ls")
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
           "mechanism": {"delta_shadow": DELTA_SHADOW, "delta_bright": DELTA_BRIGHT,
                         "a_min": A_MIN, "k_move": K_MOVE, "move_min": MOVE_MIN,
                         "warmup": WARMUP, "sfs_lo": SFS_LO, "sfs_hi": SFS_HI},
           "criteria": {"iou_det": IOU_DET, "iou_fp": IOU_FP, "ang_tol": ANG_TOL,
                        "c1_det_min": C1_DET_MIN, "c1_fp_max": C1_FP_MAX,
                        "c2_med_max": C2_MED_MAX, "c2_acc_min": C2_ACC_MIN,
                        "c3_mean_max": C3_MEAN_MAX, "c3_acc_min": C3_ACC_MIN},
           "scene": {"light_azimuth": LIGHT_AZIMUTH, "sphere_c": list(SPHERE_C),
                     "sphere_r": SPHERE_R, "orbit_c": list(ORBIT_C),
                     "orbit_r": ORBIT_R, "occ_r": OCC_R, "occ_freq": OCC_FREQ,
                     "shadow_mult": SHADOW_MULT, "noise": NOISE_SIGMA}}
    ck_tag = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    ckpt_path = os.path.join(args.out_dir, "ckpt_ls_%s.json" % ck_tag)

    def run_all(use_resume=True):
        done = {}
        if use_resume and args.resume and not args.no_resume and os.path.exists(ckpt_path):
            with open(ckpt_path, encoding="utf-8") as f:
                done = json.load(f).get("per_unit", {})
        per_unit = dict(done)
        for lv in levels:
            for seed in seeds:
                key = "%d_%d" % (lv, seed)
                if key in per_unit:
                    continue
                per_unit[key] = run_unit(lv, seed, n_frames=args.frames)
                with open(ckpt_path, "w", encoding="utf-8") as f:
                    json.dump({"config": cfg, "per_unit": per_unit},
                              f, ensure_ascii=False, indent=1)
                print("PROGRESS", flush=True)
        return per_unit

    per_unit = run_all()
    units = [per_unit["%d_%d" % (lv, s)] for lv in levels for s in seeds]
    agg = aggregate(units)
    c1, c2, c3, fails = verdict_of(agg)

    # ---- 守卫 ----
    g_cp5, g_cp5_sc2, g_cp5_scl = guard_cp5(n_seeds=args.n_seeds, n_frames=args.frames)
    g_mad, g_mad_sig, g_mad_exp = guard_mad()
    guards_ok = (g_cp5 == 1) and (g_mad == 1)

    # ---- 判定（docs/260 §1.5 冻结）----
    if not guards_ok:
        verdict = "GUARD_FAIL"
    elif fails is None:
        verdict = "SHADOW_DECOUPLE_PASS"
    else:
        verdict = "_".join(fails)

    # ---- 内部确定性复现（docs/260 §1.6-3；第二遍强制重算，不读 checkpoint）----
    repro = 1
    if args.repro:
        per_unit2 = run_all(use_resume=False)
        keys = ["det_rate", "fp_rate", "ld_med", "ld_acc", "sfs_err"]
        for lv in levels:
            for s in seeds:
                k = "%d_%d" % (lv, s)
                for kk in keys:
                    if per_unit[k][kk] != per_unit2[k][kk]:
                        repro = 0

    out = {
        "artifact": "light_shadow_test",
        "doc_ref": "docs/260",
        "config": cfg,
        "per_unit": per_unit,
        "aggregate": agg,
        "criteria": {"c1": bool(c1), "c2": bool(c2), "c3": bool(c3), "fails": fails},
        "guards": {"cp5": g_cp5, "cp5_sc2": g_cp5_sc2, "cp5_sc_late": g_cp5_scl,
                   "mad": g_mad, "mad_sig": g_mad_sig, "mad_expected": g_mad_exp,
                   "repro": repro},
        "verdict": verdict,
        "timing": {"elapsed_sec": round(time.time() - t0, 2)},
    }
    res_path = os.path.join(args.out_dir, "ls_%s.json" % args.tag)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 摘要块：ASCII 标签 + 每行一个数字（顺序固定）----
    # 每级 3 行：R_LT_L<lv>_DET / R_LT_L<lv>_LD_MED / R_LT_L<lv>_SFS_ERR
    print("R_LT_LEVELS=%d" % len(levels))
    print("R_LT_SEEDS=%d" % len(seeds))
    for lv in levels:
        lu = [u for u in units if u["lvcode"] == lv]
        if lu:
            print("R_LT_L%d_DET=%.4f" % (lv, float(np.mean([u["det_rate"] for u in lu]))))
            print("R_LT_L%d_LD_MED=%.4f" % (lv, float(np.nanmean([u["ld_med"] for u in lu]))))
            print("R_LT_L%d_SFS_ERR=%.4f" % (lv, float(np.nanmean([u["sfs_err"] for u in lu]))))
    print("R_LT_GUARD_CP5=%d" % g_cp5)
    print("R_LT_GUARD_CP5_SC2=%.4f" % g_cp5_sc2)
    print("R_LT_GUARD_MAD=%d" % g_mad)
    print("R_LT_DET_RATE=%.4f" % agg["det_rate"])
    print("R_LT_DET_RATE_SD=%.4f" % agg["det_sd"])
    print("R_LT_DET_CI_LO=%.4f" % agg["det_ci"][0])
    print("R_LT_DET_CI_HI=%.4f" % agg["det_ci"][1])
    print("R_LT_FP_RATE=%.4f" % agg["fp_rate"])
    print("R_LT_FP_RATE_SD=%.4f" % agg["fp_sd"])
    print("R_LT_LD_MED_ERR=%.4f" % agg["ld_med"])
    print("R_LT_LD_MED_SD=%.4f" % agg["ld_sd"])
    print("R_LT_LD_ACC=%.4f" % agg["ld_acc"])
    print("R_LT_LD_ACC_SD=%.4f" % agg["ld_acc_sd"])
    print("R_LT_SFS_MEAN_ERR=%.4f" % agg["sfs_err"])
    print("R_LT_SFS_MEAN_SD=%.4f" % agg["sfs_sd"])
    print("R_LT_SFS_ACC=%.4f" % agg["sfs_acc"])
    print("R_LT_C1=%s" % ("PASS" if c1 else "FAIL"))
    print("R_LT_C2=%s" % ("PASS" if c2 else "FAIL"))
    print("R_LT_C3=%s" % ("PASS" if c3 else "FAIL"))
    print("R_LT_VERDICT=%s" % verdict)
    print("R_LT_REPRO=%d" % repro)
    print("R_LT_ELAPSED=%.2f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
