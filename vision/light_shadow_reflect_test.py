"""vision/light_shadow_reflect_test.py — 光影判别第三格：颜色反射率连续性
（docs/263 预注册设计，判据/旋钮/守卫冻结；本脚本为唯一新增文件，import 复用第一格
vision/light_shadow_test.py、第二格 vision/light_shadow_gate_test.py、
vision/critical_point.py、vision/davis_suspicious.py，未修改任何既有脚本）。

目标（docs/263 §1.1）：把反射率连续性（intrinsic image，Barrow & Tenenbaum 1978；
图像 = 反射率 × 光照）做成显式判别——候选暗区边界两侧的色相/饱和度连续性：
  阴影边界两侧 = 同一反射率（H/S 不变仅 V 变，docs/196 亮度稳健）→ 判连续（E3 证据）
  物体边界两侧 = 不同反射率（H/S 跳变）→ 判跳变 → V4 否决（判为物体/非阴影）
V4 追加为 docs/261 否决门链第 3 门：gated = legacy AND NOT(v1 OR v3 OR v4)。

彩色合成场景（docs/263 §1.2）：docs/260/261 灰度帧逐字复用，彩色化 RGB = tint × Y
（tint = 表面反射率向量，归一化使灰度投影系数 = 1 → 灰度投影逐位 = 前两格）：
  背景棋盘 tint_BG（H=30° S=0.6）、球面 tint_SPH（H=60° S=0.5）、遮挡物白 255
  （无反射率）、对照圆盘/开放暗带 tint_DISK（H=140° S=0.6）；阴影 = 同表面反射率
  × 亮度 0.5（H/S 不变）。三臂：
  main  = docs/260 逐字 + 彩色化（阴影同背景反射率）→ 真阴影正样本（测连续 + 保持）
  ctrl  = docs/261 对照臂逐字 + 彩色化（暗圆盘紫，闭合不投影）→ 暗物体（测跳变）
  band  = main 灰度 + 阴影区域反射率替换为紫 + GT 阴影清空 → 开放暗带（触边界、
          沿光照 → V1/V3 不触发，V4 独立承担分离）

判据（docs/263 §1.4，冻结）：
  C1 REFLECT_CONT     [新][机制][合成受控]：main 臂 legacy_active 帧中边界判连续占比 ≥ 0.90
  C2 REFLECT_DISCONT  [新][机制][合成受控]：ctrl+band 臂 legacy_active 帧中边界判跳变占比 ≥ 0.90
  C3 KEEP             [新][机制][合成受控]：main det_gated ≥ 0.80 且 Δdet ≤ 0.05 且
                      ld_med_gated ≤ 15°；ctrl fp_legacy−fp_gated ≥ 0.50 且 fp_gated ≤ 0.10；
                      band fp_gated ≤ 0.10
  C4 REFLECT_VETO     [新][机制][合成受控]：band 臂 gated 否决率 ≥ 0.90（V4 独立承担）
                      且 main 臂真阴影帧 V4 假否决率 ≤ 0.10
  判定：四判据全过 + 守卫全过 = REFLECT_CONT_PASS；不过按名报
  （REFLECT_CONT_FAIL / REFLECT_DISCONT_FAIL / KEEP_FAIL / REFLECT_VETO_FAIL）；
  守卫不过 = GUARD_FAIL。

守卫（docs/263 §1.6，冻结）：
  R_RC_GUARD_CELL1：import 第一格 run_unit 重跑 40 main (级,种子)，本脚本 main-legacy
    的 det_rate/fp_rate/ld_med/sfs_err 逐位一致（明度通道逐位 = docs/260）
  R_RC_GUARD_CELL2：import 第二格 run_unit_gated("ctrl") 重跑 40 ctrl (级,种子)，本脚本
    ctrl-legacy 的 fp_legacy/fp_gated/veto_rate/sfs_err 逐位一致（明度通道逐位 = docs/261）
  R_RC_GUARD_CP5：import 第一格 guard_cp5（复现 docs/232 L5：SC2 ∈ [2.0,3.6]、SC_late ≥ 0.5）
  R_RC_GUARD_MAD：import 第一格 guard_mad（σ̂ 与理论 √2·3/64 ≈ 0.0663 偏差 ≤ 25%）
  R_RC_REPRO：--repro 时 120 单位整体重跑第二遍，逐项位级一致

安全纪律（docs/228/234/235）：stdout 只输出 ASCII 标签 + 每行一个数字的 R_RC_* 摘要块
（顺序固定，见 SUMMARY_LINES 注释）；JSON 归档 vision/out/results/rc_<tag>.json +
checkpoint ckpt_rc_<hash>.json（--resume 断点续跑）；数字用 vision/extract_r.py 纯正则
抽取；禁止读取 logs/*.log 与 vision/out/results/*.json 原文。

用法：
  python vision/light_shadow_reflect_test.py --levels 30,31,32,33 --n-seeds 10 --tag main --repro
  python vision/light_shadow_reflect_test.py --levels 30 --n-seeds 1 --frames 240 --tag timing
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

from critical_point import CPLoop, mean_sd, bootstrap_ci, JITTER
from light_shadow_test import (
    make_shadow_scene, run_unit, largest_component, iou, circ_err_deg,
    guard_cp5, guard_mad,
    W, H, BG_CELL, BG_DARK, BG_BRIGHT,
    SPHERE_C, SPHERE_R, OCC_R, OCC_GRAY, ORBIT_C, ORBIT_R, OCC_FREQ,
    SHADOW_MULT, NOISE_SIGMA,
    DELTA_SHADOW, A_MIN, K_MOVE, MOVE_IOU, WARMUP, SFS_LO, SFS_HI,
    OCC_LUM_THRESH, LIGHT_AZIMUTH,
    IOU_DET, IOU_FP, ANG_TOL, C2_MED_MAX,
)
from light_shadow_gate_test import (
    make_control_scene, run_unit_gated,
    touches_boundary, pca_axis, axis_err_deg,
    TOL_AXIS, RATIO_MIN, CTRL_OCC_GRAY,
)
from davis_suspicious import circ, SAT_MIN_TARGET

warnings.filterwarnings("ignore")

DEFAULT_OUT = os.path.join("vision", "out", "results")
N_BOOT = 2000
BOOT_SEED = 20260828

# ---- 反射率 tint 表（docs/263 §1.2 冻结；场景参数，已知反射率）----
# OpenCV H 单位（docs/219 惯例：0-179 = 0-358°）；S ∈ [0,1]。
TINT_BG_H, TINT_BG_S = 30.0, 0.6     # 背景棋盘（橙/黄），64/96 两格共用
TINT_SPH_H, TINT_SPH_S = 60.0, 0.5   # 静态朗伯球面（绿；S=0.5 为 D0 复核冻结）
TINT_DISK_H, TINT_DISK_S = 140.0, 0.6  # 对照圆盘 / 开放暗带（紫）
# 遮挡物 = 白 255（无反射率，docs/260/261 逐字，不乘 tint）

# ---- 反射率判别旋钮（docs/263 §1.3 冻结）----
TOL_H = 15.0          # 色相连续性容差（OpenCV 单位；docs/219 确认带宽 30° 的一半）
TOL_S = 80.0          # 饱和度连续性容差（HSV S 0-255；本格无独立 S 跳变样本，报告性）
BAND_EDGE = 2         # 边界环带半宽（px；5×5 椭圆核）
SAT_MIN = int(SAT_MIN_TARGET)   # 彩色像素过滤（docs/219 同款 = 60）

# ---- 判据阈值（docs/263 §1.4 冻结）----
CONT_MIN = 0.90           # C1 连续判别正确率下限
DISCONT_MIN = 0.90        # C2 跳变判别正确率下限
POS_DET_MIN = 0.80        # C3a docs/260 C1 同阈值
POS_DET_DROP_MAX = 0.05   # C3a docs/261 C2 同阈值
CTRL_DROP_MIN = 0.50      # C3b docs/261 C1 同阈值
CTRL_FP_MAX = 0.10        # C3b/c docs/260/261 同阈值
VETO_BAND_MIN = 0.90      # C4a band 臂 gated 否决率下限（V4 独立承担）
V4_FALSE_MAX = 0.10       # C4b main 臂真阴影帧 V4 假否决率上限

# ---- 内部确定性复现键（docs/263 §1.6-5）----
REPRO_KEYS_MAIN = ["det_legacy", "fp_legacy", "det_gated", "fp_gated",
                   "ld_med_legacy", "ld_med_gated", "sfs_err",
                   "cont_rate", "discont_rate", "v4_false_rate",
                   "veto_rate", "v1_rate", "v3_rate", "v4_rate", "e3_rate"]
REPRO_KEYS_CTRL = ["fp_legacy", "fp_gated", "discont_rate", "veto_rate", "sfs_err"]
REPRO_KEYS_BAND = ["fp_legacy", "fp_gated", "discont_rate", "veto_rate",
                   "v1_rate", "v3_rate", "v4_rate", "sfs_err"]


def hsv_tint(h_deg, s):
    """OpenCV H 单位（0-180 = 真实 0-360°）→ 归一化反射率 RGB tint。

    返回 tint 使 0.299·t_R + 0.587·t_G + 0.114·t_B = 1（灰度投影逐位 = 前两格）。
    色相/饱和度在归一化下不变（乘性缩放）。"""
    hh = (h_deg * 2.0) % 360.0          # 真实色相度数
    c = float(s)
    x = c * (1.0 - abs((hh / 60.0) % 2.0 - 1.0))
    m = 1.0 - c
    if hh < 60:
        r, g, b = c, x, 0.0
    elif hh < 120:
        r, g, b = x, c, 0.0
    elif hh < 180:
        r, g, b = 0.0, c, x
    elif hh < 240:
        r, g, b = 0.0, x, c
    elif hh < 300:
        r, g, b = x, 0.0, c
    else:
        r, g, b = c, 0.0, x
    rgb = np.array([r + m, g + m, b + m], np.float64)
    y = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return rgb / y


TINT_BG = hsv_tint(TINT_BG_H, TINT_BG_S)
TINT_SPH = hsv_tint(TINT_SPH_H, TINT_SPH_S)
TINT_DISK = hsv_tint(TINT_DISK_H, TINT_DISK_S)


def colorize(gray, sphere, occ, shadow, band_tint=None, occ_tint=None):
    """灰度帧 → 彩色帧（docs/263 §1.2）：RGB = tint × gray（乘性，Y 投影逐位）。

    gray: uint8 (H,W)（docs/260/261 帧，含明度噪声）；sphere/occ/shadow: bool 掩码。
    occ 默认 = 遮挡物白 255（无反射率，不乘 tint）；occ_tint 非 None 时 occ 处 =
    对照圆盘反射率（ctrl 臂，32×tint_DISK）；阴影 = 背景 tint（band_tint=None）
    或 band_tint（band 臂）。"""
    g = gray.astype(np.float32)
    rgb = g[..., None] * TINT_BG.astype(np.float32)      # 背景（默认，含阴影同反射率）
    rgb[sphere] = g[sphere][..., None] * TINT_SPH.astype(np.float32)
    if occ_tint is not None:
        rgb[occ] = g[occ][..., None] * occ_tint.astype(np.float32)   # 对照圆盘（ctrl）
    else:
        rgb[occ] = g[occ][..., None]                     # 遮挡物白 255（= gray 已 clip 255）
    if band_tint is not None and shadow.any():
        rgb[shadow] = g[shadow][..., None] * band_tint.astype(np.float32)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def make_rgb_frames(lvcode, seed, arm, n_frames=240, jitter=JITTER):
    """生成 (臂,级,种子) 的 (灰度帧序列, 彩色帧序列, GT)。

    arm: "main" | "ctrl" | "band"。灰度逐位 = docs/260（main/band）/ docs/261
    （ctrl）；彩色 = 灰度 × 反射率 tint。band 的 GT 阴影清空。"""
    if arm == "ctrl":
        frames, gts = make_control_scene(lvcode, seed, n_frames=n_frames, jitter=jitter)
    else:
        frames, gts = make_shadow_scene(lvcode, seed, n_frames=n_frames, jitter=jitter)
    sphere = gts["sphere_mask"]
    rgb_frames = []
    for t, g in enumerate(frames):
        occ = gts["per_frame"][t]["occ"]
        if arm == "ctrl":
            sh = np.zeros((H, W), bool)
            rgb_frames.append(colorize(g, sphere, occ, sh, band_tint=None,
                                       occ_tint=TINT_DISK))
        else:
            sh = gts["per_frame"][t]["shadow"]
            if arm == "band":
                rgb_frames.append(colorize(g, sphere, occ, sh, band_tint=TINT_DISK))
                gts["per_frame"][t]["shadow"] = np.zeros((H, W), bool)
            else:
                rgb_frames.append(colorize(g, sphere, occ, sh, band_tint=None))
    return frames, rgb_frames, gts


def boundary_bands(mask, k=BAND_EDGE):
    """候选掩码边界环带：内侧（掩码内 k px）与外侧（掩码外 k px）。"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    er = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
    di = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    inner = mask & ~er
    outer = di & ~mask
    return inner, outer


def reflect_stats(rgb, mask, k=BAND_EDGE, sat_min=SAT_MIN):
    """候选边界两侧反射率观测：ΔH（色距，docs/219 circ 口径）+ ΔS + 两侧中位数。

    返回 dict(dh, ds, hue_in, hue_out, n_in, n_out) or None（彩色像素不足）。"""
    inner, outer = boundary_bands(mask, k)
    if inner.sum() < 5 or outer.sum() < 5:
        return None
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s = hsv[:, :, 0], hsv[:, :, 1]
    si = s[inner] > sat_min
    so = s[outer] > sat_min
    if int(si.sum()) < 5 or int(so.sum()) < 5:
        return None
    hue_in = float(np.median(h[inner][si]))
    hue_out = float(np.median(h[outer][so]))
    sat_in = float(np.median(s[inner][si]))
    sat_out = float(np.median(s[outer][so]))
    return dict(dh=float(circ(hue_in, hue_out)), ds=abs(sat_in - sat_out),
                hue_in=hue_in, hue_out=hue_out,
                n_in=int(si.sum()), n_out=int(so.sum()))


def run_unit_reflect(lvcode, seed, arm, n_frames=240, jitter=JITTER):
    """跑 (臂,级,种子) 一次完整运行，返回反射率判别 + legacy/gated 双口径指标。

    明度通道（ref_dark/dark/bright/面积/IoU 移动门/SFS）在灰度帧上逐字复用
    docs/260/261（守卫 CELL1/CELL2 逐位基础）；反射率判别（V4/E3）在彩色帧上。
    """
    theta = LIGHT_AZIMUTH[lvcode]
    frames, rgb_frames, gts = make_rgb_frames(lvcode, seed, arm,
                                              n_frames=n_frames, jitter=jitter)
    loop = CPLoop(window=10)
    eval_lo = WARMUP
    eval_hi = n_frames
    n_eval = max(1, eval_hi - eval_lo)

    # ---- Pass A：适应（docs/260 诊断轮 D2 冻结口径，逐字同实现，灰度通道）----
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

    shadow_masks = []          # 每帧阴影候选掩码（时间移动 IoU 门用）
    det_leg, fp_leg = [], []   # legacy 口径 det/fp（docs/260 C1）
    det_gat, fp_gat = [], []   # gated 口径 det/fp（含 V1/V3/V4）
    ld_leg_errs, ld_gat_errs = [], []
    cont_flags, discont_flags = [], []     # C1/C2：边界判连续/跳变（legacy_active 帧）
    v1_flags, v3_flags, v4_flags = [], [], []   # 各否决门触发率（d_area ≥ A_MIN 帧）
    veto_flags = []                        # 整体否决率（legacy_active 帧）
    v4_false_flags = []                    # main：真阴影帧被 V4 否决（C4b）
    e3_flags = []                          # main：真阴影帧 E3 证据（C1 佐证）
    dh_vals, ds_vals = [], []              # 报告性：ΔH/ΔS 逐帧
    n_insuff = 0
    gray_mean = np.zeros((H, W), np.float64)

    # ---- Pass B：判断（预测回路 + 检测 + 否决门 + 反射率判别）----
    for t, g in enumerate(frames):
        L = np.log(np.maximum(g.astype(np.float32), 1.0))
        loop.step(g)
        if t >= eval_lo:
            gray_mean += g.astype(np.float64) / n_eval
        dark = (ref_dark_log - L) > DELTA_SHADOW
        bright = L > occ_thresh_log
        d_mask, d_area, d_c = largest_component(dark)
        b_mask, b_area, b_c = largest_component(bright)
        shadow_masks.append(d_mask)

        # legacy active（docs/260 C1 口径：空间相干 AND 帧间移动）
        active = (d_area >= A_MIN)
        if t >= eval_lo + K_MOVE:
            m0 = shadow_masks[t - K_MOVE]
            move = 1.0 - iou(d_mask, m0)
            active = active and (move >= MOVE_IOU)

        # ---- 否决门（docs/261 V1/V3 逐字 import + 本格 V4）----
        v1 = False
        v3 = False
        v4 = False
        if d_area >= A_MIN:
            v1 = not touches_boundary(d_mask)          # V1 闭合轮廓否决
            pr = pca_axis(d_mask)
            if pr is not None:
                alpha, ratio = pr
                if ratio >= RATIO_MIN:                 # 各向同性 → 跳过 V3
                    v3 = axis_err_deg(alpha, theta) > TOL_AXIS   # V3 主轴偏离光照否决
            rs = reflect_stats(rgb_frames[t], d_mask)  # V4 反射率跳变否决
            if rs is not None:
                dh_vals.append(rs["dh"])
                ds_vals.append(rs["ds"])
                e3 = (rs["dh"] <= TOL_H) and (rs["ds"] <= TOL_S)
                v4 = not e3
            else:
                n_insuff += 1
        veto = v1 or v3 or v4
        gated_active = active and (not veto)

        if t >= eval_lo:
            gt_sh = gts["per_frame"][t]["shadow"]
            iou_v = iou(d_mask, gt_sh)
            det_leg.append(1.0 if (active and iou_v >= IOU_DET) else 0.0)
            fp_leg.append(1.0 if (active and iou_v < IOU_FP) else 0.0)
            det_gat.append(1.0 if (gated_active and iou_v >= IOU_DET) else 0.0)
            fp_gat.append(1.0 if (gated_active and iou_v < IOU_FP) else 0.0)
            if active:
                veto_flags.append(1.0 if veto else 0.0)
                # 反射率判别（C1/C2 载体）：legacy_active 帧的边界判连续/跳变
                if d_area >= A_MIN and rs is not None:
                    cont_flags.append(1.0 if e3 else 0.0)
                    discont_flags.append(0.0 if e3 else 1.0)
                # 否决门分率（候选帧）
                if d_area >= A_MIN:
                    v1_flags.append(1.0 if v1 else 0.0)
                    v3_flags.append(1.0 if v3 else 0.0)
                    v4_flags.append(1.0 if v4 else 0.0)
                    if arm == "main" and iou_v >= IOU_DET:   # 真阴影帧
                        v4_false_flags.append(1.0 if v4 else 0.0)
                        e3_flags.append(1.0 if e3 else 0.0)
            if active and d_c is not None and b_c is not None:
                ex, ey = b_c[0] - d_c[0], b_c[1] - d_c[1]
                est = float(np.rad2deg(np.arctan2(ey, ex))) % 360.0
                ld_leg_errs.append(circ_err_deg(est, theta))
            if gated_active and d_c is not None and b_c is not None:
                ex, ey = b_c[0] - d_c[0], b_c[1] - d_c[1]
                est = float(np.rad2deg(np.arctan2(ey, ex))) % 360.0
                ld_gat_errs.append(circ_err_deg(est, theta))

    det_leg_r = float(np.mean(det_leg)) if det_leg else 0.0
    fp_leg_r = float(np.mean(fp_leg)) if fp_leg else 0.0
    det_gat_r = float(np.mean(det_gat)) if det_gat else 0.0
    fp_gat_r = float(np.mean(fp_gat)) if fp_gat else 0.0
    ld_med_leg = float(np.median(ld_leg_errs)) if ld_leg_errs else float("nan")
    ld_med_gat = float(np.median(ld_gat_errs)) if ld_gat_errs else float("nan")
    cont_rate = float(np.mean(cont_flags)) if cont_flags else float("nan")
    discont_rate = float(np.mean(discont_flags)) if discont_flags else float("nan")
    veto_rate = float(np.mean(veto_flags)) if veto_flags else float("nan")
    v1_rate = float(np.mean(v1_flags)) if v1_flags else float("nan")
    v3_rate = float(np.mean(v3_flags)) if v3_flags else float("nan")
    v4_rate = float(np.mean(v4_flags)) if v4_flags else float("nan")
    v4_false_rate = float(np.mean(v4_false_flags)) if v4_false_flags else float("nan")
    e3_rate = float(np.mean(e3_flags)) if e3_flags else float("nan")

    # ---- SfS（docs/260 §1.3 同实现，报告性；灰度通道）----
    dark_cap, cap_area, cap_c = largest_component(gray_mean < SFS_LO, min_area=50)
    if cap_area > 0 and cap_c is not None:
        pidx = int(np.argmax(gray_mean))
        py, px = divmod(pidx, W)
        sfs_err = circ_err_deg(
            float(np.rad2deg(np.arctan2(py - cap_c[1], px - cap_c[0]))) % 360.0, theta)
    else:
        sfs_err = float("nan")

    return dict(arm=arm, seed=seed, lvcode=lvcode, theta=theta,
                det_legacy=det_leg_r, fp_legacy=fp_leg_r,
                det_gated=det_gat_r, fp_gated=fp_gat_r,
                ld_med_legacy=ld_med_leg, ld_med_gated=ld_med_gat,
                sfs_err=sfs_err,
                cont_rate=cont_rate, discont_rate=discont_rate,
                veto_rate=veto_rate, v1_rate=v1_rate, v3_rate=v3_rate,
                v4_rate=v4_rate, v4_false_rate=v4_false_rate, e3_rate=e3_rate,
                dh_mean=float(np.mean(dh_vals)) if dh_vals else float("nan"),
                ds_mean=float(np.mean(ds_vals)) if ds_vals else float("nan"),
                n_insuff=n_insuff, n_active=len(veto_flags))


# ---------------- 守卫（docs/263 §1.6 冻结） ----------------
def guard_cell1(units_main, levels, seeds, n_frames=240):
    """同代码路径复现第一格：import run_unit 重跑全部 main (级,种子)，本脚本
    main-legacy 的 det_rate/fp_rate/ld_med/sfs_err 与 run_unit 逐位一致。"""
    n_bad = 0
    n = 0
    for lv in levels:
        for seed in seeds:
            ru = run_unit(lv, seed, n_frames=n_frames)
            mine = [u for u in units_main if u["lvcode"] == lv and u["seed"] == seed]
            if not mine:
                n_bad += 1
                continue
            u = mine[0]
            n += 1
            if not (u["det_legacy"] == ru["det_rate"]
                    and u["fp_legacy"] == ru["fp_rate"]
                    and u["ld_med_legacy"] == ru["ld_med"]
                    and u["sfs_err"] == ru["sfs_err"]):
                n_bad += 1
    return 1 if (n_bad == 0 and n > 0) else 0, n, n_bad


def guard_cell2(units_ctrl, levels, seeds, n_frames=240):
    """同代码路径复现第二格：import run_unit_gated("ctrl") 重跑全部 ctrl (级,种子)，
    本脚本 ctrl-legacy 的 fp_legacy/fp_gated/veto_rate/sfs_err 逐位一致
    （ctrl 上 V1 全否决 → gated 链含 V4 后指标不变，逐位保持）。"""
    n_bad = 0
    n = 0
    for lv in levels:
        for seed in seeds:
            rg = run_unit_gated(lv, seed, "ctrl", n_frames=n_frames)
            mine = [u for u in units_ctrl if u["lvcode"] == lv and u["seed"] == seed]
            if not mine:
                n_bad += 1
                continue
            u = mine[0]
            n += 1
            if not (u["fp_legacy"] == rg["fp_legacy"]
                    and u["fp_gated"] == rg["fp_gated"]
                    and u["veto_rate"] == rg["veto_rate"]
                    and u["sfs_err"] == rg["sfs_err"]):
                n_bad += 1
    return 1 if (n_bad == 0 and n > 0) else 0, n, n_bad


# ---------------- 统计外壳（critical_point 同款） ----------------
def agg_col(vals):
    vals = [v for v in vals if v == v]     # 滤 NaN
    if not vals:
        return float("nan"), 0.0, [float("nan"), float("nan")]
    m, s = mean_sd(vals)
    lo, hi = bootstrap_ci(vals)
    return m, s, [float(lo), float(hi)]


def verdict_of(agg):
    c1 = agg["cont_rate_main"] >= CONT_MIN
    c2 = agg["discont_rate_ctrl_band"] >= DISCONT_MIN
    c3 = (agg["det_gated_main"] >= POS_DET_MIN
          and agg["det_drop_main"] <= POS_DET_DROP_MAX
          and agg["ld_med_gated_main"] <= C2_MED_MAX
          and agg["fp_drop_ctrl"] >= CTRL_DROP_MIN
          and agg["fp_gated_ctrl"] <= CTRL_FP_MAX
          and agg["fp_gated_band"] <= CTRL_FP_MAX)
    c4 = (agg["veto_rate_band"] >= VETO_BAND_MIN
          and agg["v4_false_main"] <= V4_FALSE_MAX)
    fails = []
    if not c1:
        fails.append("REFLECT_CONT_FAIL")
    if not c2:
        fails.append("REFLECT_DISCONT_FAIL")
    if not c3:
        fails.append("KEEP_FAIL")
    if not c4:
        fails.append("REFLECT_VETO_FAIL")
    return c1, c2, c3, c4, (fails if fails else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="30,31,32,33")
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--jitter", type=float, default=JITTER)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--tag", default="rc")
    ap.add_argument("--repro", action="store_true")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    levels = [int(x) for x in args.levels.split(",") if x.strip() != ""]
    seeds = list(range(args.first_seed, args.first_seed + args.n_seeds))
    arms = ["main", "ctrl", "band"]
    t0 = time.time()

    cfg = {"levels": levels, "n_seeds": args.n_seeds, "first_seed": args.first_seed,
           "frames": args.frames, "jitter": args.jitter, "tag": args.tag,
           "arms": arms,
           "mechanism": {"delta_shadow": DELTA_SHADOW, "a_min": A_MIN,
                         "k_move": K_MOVE, "move_iou": MOVE_IOU,
                         "warmup": WARMUP, "sfs_lo": SFS_LO,
                         "veto": {"tol_axis": TOL_AXIS, "ratio_min": RATIO_MIN,
                                  "tol_h": TOL_H, "tol_s": TOL_S,
                                  "band_edge": BAND_EDGE, "sat_min": SAT_MIN},
                         "reflectance": {"tint_bg_h": TINT_BG_H, "tint_bg_s": TINT_BG_S,
                                         "tint_sph_h": TINT_SPH_H, "tint_sph_s": TINT_SPH_S,
                                         "tint_disk_h": TINT_DISK_H, "tint_disk_s": TINT_DISK_S}},
           "criteria": {"iou_det": IOU_DET, "iou_fp": IOU_FP, "ang_tol": ANG_TOL,
                        "cont_min": CONT_MIN, "discont_min": DISCONT_MIN,
                        "pos_det_min": POS_DET_MIN, "pos_det_drop_max": POS_DET_DROP_MAX,
                        "ctrl_drop_min": CTRL_DROP_MIN, "ctrl_fp_max": CTRL_FP_MAX,
                        "veto_band_min": VETO_BAND_MIN, "v4_false_max": V4_FALSE_MAX,
                        "c2_med_max": C2_MED_MAX},
           "scene": {"light_azimuth": LIGHT_AZIMUTH, "sphere_c": list(SPHERE_C),
                     "sphere_r": SPHERE_R, "orbit_c": list(ORBIT_C),
                     "orbit_r": ORBIT_R, "occ_r": OCC_R, "occ_freq": OCC_FREQ,
                     "occ_gray": OCC_GRAY, "ctrl_occ_gray": CTRL_OCC_GRAY,
                     "shadow_mult": SHADOW_MULT, "noise": NOISE_SIGMA}}
    ck_tag = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    ckpt_path = os.path.join(args.out_dir, "ckpt_rc_%s.json" % ck_tag)

    def run_all(use_resume=True):
        done = {}
        if use_resume and args.resume and not args.no_resume and os.path.exists(ckpt_path):
            with open(ckpt_path, encoding="utf-8") as f:
                done = json.load(f).get("per_unit", {})
        per_unit = dict(done)
        for arm in arms:
            for lv in levels:
                for seed in seeds:
                    key = "%s_%d_%d" % (arm[0].upper(), lv, seed)
                    if key in per_unit:
                        continue
                    per_unit[key] = run_unit_reflect(lv, seed, arm,
                                                     n_frames=args.frames,
                                                     jitter=args.jitter)
                    with open(ckpt_path, "w", encoding="utf-8") as f:
                        json.dump({"config": cfg, "per_unit": per_unit},
                                  f, ensure_ascii=False, indent=1)
                    print("PROGRESS", flush=True)
        return per_unit

    per_unit = run_all()
    units_main = [per_unit["M_%d_%d" % (lv, s)] for lv in levels for s in seeds]
    units_ctrl = [per_unit["C_%d_%d" % (lv, s)] for lv in levels for s in seeds]
    units_band = [per_unit["B_%d_%d" % (lv, s)] for lv in levels for s in seeds]

    def col(units, k):
        return [u[k] for u in units]

    cont_main, cont_main_sd, cont_main_ci = agg_col(col(units_main, "cont_rate"))
    dis_c, _, _ = agg_col(col(units_ctrl, "discont_rate"))
    dis_b, _, _ = agg_col(col(units_band, "discont_rate"))
    dis_comb = [u["discont_rate"] for u in units_ctrl + units_band]
    dis_comb = [v for v in dis_comb if v == v]
    dis_cb_m, dis_cb_sd = mean_sd(dis_comb)
    dis_cb_lo, dis_cb_hi = bootstrap_ci(dis_comb)
    det_leg_main, det_leg_main_sd, _ = agg_col(col(units_main, "det_legacy"))
    det_gat_main, det_gat_main_sd, det_gat_main_ci = agg_col(col(units_main, "det_gated"))
    ld_gat_main, ld_gat_main_sd, ld_gat_main_ci = agg_col(col(units_main, "ld_med_gated"))
    fp_leg_ctrl, fp_leg_ctrl_sd, _ = agg_col(col(units_ctrl, "fp_legacy"))
    fp_gat_ctrl, fp_gat_ctrl_sd, fp_gat_ctrl_ci = agg_col(col(units_ctrl, "fp_gated"))
    fp_gat_band, fp_gat_band_sd, fp_gat_band_ci = agg_col(col(units_band, "fp_gated"))
    veto_band, _, _ = agg_col(col(units_band, "veto_rate"))
    v4_false_main, _, _ = agg_col(col(units_main, "v4_false_rate"))
    v1_band, _, _ = agg_col(col(units_band, "v1_rate"))
    v3_band, _, _ = agg_col(col(units_band, "v3_rate"))
    v4_band, _, _ = agg_col(col(units_band, "v4_rate"))
    v4_ctrl, _, _ = agg_col(col(units_ctrl, "v4_rate"))
    e3_main, _, _ = agg_col(col(units_main, "e3_rate"))
    dh_main, _, _ = agg_col(col(units_main, "dh_mean"))
    dh_ctrl, _, _ = agg_col(col(units_ctrl, "dh_mean"))
    dh_band, _, _ = agg_col(col(units_band, "dh_mean"))
    sfs_main, _, _ = agg_col(col(units_main, "sfs_err"))
    sfs_ctrl, _, _ = agg_col(col(units_ctrl, "sfs_err"))
    sfs_band, _, _ = agg_col(col(units_band, "sfs_err"))

    fp_drop_ctrl = fp_leg_ctrl - fp_gat_ctrl
    det_drop_main = det_leg_main - det_gat_main

    agg = {"cont_rate_main": cont_main, "cont_rate_main_sd": cont_main_sd,
           "cont_rate_main_ci": cont_main_ci,
           "discont_rate_ctrl": dis_c, "discont_rate_band": dis_b,
           "discont_rate_ctrl_band": dis_cb_m, "discont_rate_ctrl_band_sd": dis_cb_sd,
           "discont_rate_ctrl_band_ci": [dis_cb_lo, dis_cb_hi],
           "det_legacy_main": det_leg_main, "det_legacy_main_sd": det_leg_main_sd,
           "det_gated_main": det_gat_main, "det_gated_main_sd": det_gat_main_sd,
           "det_gated_main_ci": det_gat_main_ci,
           "det_drop_main": det_drop_main,
           "ld_med_gated_main": ld_gat_main, "ld_med_gated_main_sd": ld_gat_main_sd,
           "ld_med_gated_main_ci": ld_gat_main_ci,
           "fp_legacy_ctrl": fp_leg_ctrl, "fp_legacy_ctrl_sd": fp_leg_ctrl_sd,
           "fp_gated_ctrl": fp_gat_ctrl, "fp_gated_ctrl_sd": fp_gat_ctrl_sd,
           "fp_gated_ctrl_ci": fp_gat_ctrl_ci,
           "fp_drop_ctrl": fp_drop_ctrl,
           "fp_gated_band": fp_gat_band, "fp_gated_band_sd": fp_gat_band_sd,
           "fp_gated_band_ci": fp_gat_band_ci,
           "veto_rate_band": veto_band,
           "v4_false_main": v4_false_main,
           "v1_rate_band": v1_band, "v3_rate_band": v3_band, "v4_rate_band": v4_band,
           "v4_rate_ctrl": v4_ctrl,
           "e3_rate_main": e3_main,
           "dh_mean_main": dh_main, "dh_mean_ctrl": dh_ctrl, "dh_mean_band": dh_band,
           "sfs_main": sfs_main, "sfs_ctrl": sfs_ctrl, "sfs_band": sfs_band}

    c1, c2, c3, c4, fails = verdict_of(agg)

    # ---- 守卫 ----
    g_cell1, g_cell1_n, g_cell1_bad = guard_cell1(units_main, levels, seeds,
                                                  n_frames=args.frames)
    g_cell2, g_cell2_n, g_cell2_bad = guard_cell2(units_ctrl, levels, seeds,
                                                  n_frames=args.frames)
    g_cp5, g_cp5_sc2, g_cp5_scl = guard_cp5(n_seeds=args.n_seeds, n_frames=args.frames)
    g_mad, g_mad_sig, g_mad_exp = guard_mad()
    guards_ok = (g_cell1 == 1) and (g_cell2 == 1) and (g_cp5 == 1) and (g_mad == 1)

    # ---- 判定（docs/263 §1.5 冻结）----
    if not guards_ok:
        verdict = "GUARD_FAIL"
    elif fails is None:
        verdict = "REFLECT_CONT_PASS"
    else:
        verdict = "_".join(fails)

    # ---- 内部确定性复现（docs/263 §1.6-5；第二遍强制重算，不读 checkpoint）----
    repro = 1
    if args.repro:
        per_unit2 = run_all(use_resume=False)
        for arm in arms:
            keys = {"main": REPRO_KEYS_MAIN, "ctrl": REPRO_KEYS_CTRL,
                    "band": REPRO_KEYS_BAND}[arm]
            for lv in levels:
                for s in seeds:
                    k = "%s_%d_%d" % (arm[0].upper(), lv, s)
                    for kk in keys:
                        if per_unit[k][kk] != per_unit2[k][kk]:
                            repro = 0

    out = {
        "artifact": "light_shadow_reflect_test",
        "doc_ref": "docs/263",
        "config": cfg,
        "per_unit": per_unit,
        "aggregate": agg,
        "criteria": {"c1_reflect_cont": bool(c1), "c2_reflect_discont": bool(c2),
                     "c3_keep": bool(c3), "c4_reflect_veto": bool(c4), "fails": fails},
        "guards": {"cell1": g_cell1, "cell1_n": g_cell1_n, "cell1_bad": g_cell1_bad,
                   "cell2": g_cell2, "cell2_n": g_cell2_n, "cell2_bad": g_cell2_bad,
                   "cp5": g_cp5, "cp5_sc2": g_cp5_sc2, "cp5_sc_late": g_cp5_scl,
                   "mad": g_mad, "mad_sig": g_mad_sig, "mad_expected": g_mad_exp,
                   "repro": repro},
        "verdict": verdict,
        "timing": {"elapsed_sec": round(time.time() - t0, 2)},
    }
    res_path = os.path.join(args.out_dir, "rc_%s.json" % args.tag)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 摘要块：ASCII 标签，每行一个数字（顺序固定）----
    print("R_RC_LEVELS=%d" % len(levels))
    print("R_RC_SEEDS=%d" % len(seeds))
    for lv in levels:
        lm = [u for u in units_main if u["lvcode"] == lv]
        lc = [u for u in units_ctrl if u["lvcode"] == lv]
        lb = [u for u in units_band if u["lvcode"] == lv]
        if lm:
            print("R_RC_M%d_DET_GAT=%.4f" % (lv, float(np.mean([u["det_gated"] for u in lm]))))
            print("R_RC_M%d_CONT=%.4f" % (lv, float(np.nanmean([u["cont_rate"] for u in lm]))))
        if lc:
            print("R_RC_C%d_FP_GAT=%.4f" % (lv, float(np.mean([u["fp_gated"] for u in lc]))))
            print("R_RC_C%d_DISCONT=%.4f" % (lv, float(np.nanmean([u["discont_rate"] for u in lc]))))
        if lb:
            print("R_RC_B%d_FP_GAT=%.4f" % (lv, float(np.mean([u["fp_gated"] for u in lb]))))
            print("R_RC_B%d_DISCONT=%.4f" % (lv, float(np.nanmean([u["discont_rate"] for u in lb]))))
            print("R_RC_B%d_VETO=%.4f" % (lv, float(np.nanmean([u["veto_rate"] for u in lb]))))
    print("R_RC_CONT_RATE=%.4f" % cont_main)
    print("R_RC_CONT_RATE_SD=%.4f" % cont_main_sd)
    print("R_RC_CONT_CI_LO=%.4f" % cont_main_ci[0])
    print("R_RC_CONT_CI_HI=%.4f" % cont_main_ci[1])
    print("R_RC_DISCONT_RATE=%.4f" % dis_cb_m)
    print("R_RC_DISCONT_RATE_SD=%.4f" % dis_cb_sd)
    print("R_RC_DISCONT_CI_LO=%.4f" % dis_cb_lo)
    print("R_RC_DISCONT_CI_HI=%.4f" % dis_cb_hi)
    print("R_RC_DISCONT_CTRL=%.4f" % dis_c)
    print("R_RC_DISCONT_BAND=%.4f" % dis_b)
    print("R_RC_DET_LEG_MAIN=%.4f" % det_leg_main)
    print("R_RC_DET_GAT_MAIN=%.4f" % det_gat_main)
    print("R_RC_DET_GAT_MAIN_SD=%.4f" % det_gat_main_sd)
    print("R_RC_DET_GAT_MAIN_CI_LO=%.4f" % det_gat_main_ci[0])
    print("R_RC_DET_GAT_MAIN_CI_HI=%.4f" % det_gat_main_ci[1])
    print("R_RC_DET_DROP=%.4f" % det_drop_main)
    print("R_RC_LD_MED_GAT_MAIN=%.4f" % ld_gat_main)
    print("R_RC_LD_MED_GAT_MAIN_SD=%.4f" % ld_gat_main_sd)
    print("R_RC_FP_LEG_CTRL=%.4f" % fp_leg_ctrl)
    print("R_RC_FP_GAT_CTRL=%.4f" % fp_gat_ctrl)
    print("R_RC_FP_GAT_CTRL_SD=%.4f" % fp_gat_ctrl_sd)
    print("R_RC_FP_GAT_CTRL_CI_LO=%.4f" % fp_gat_ctrl_ci[0])
    print("R_RC_FP_GAT_CTRL_CI_HI=%.4f" % fp_gat_ctrl_ci[1])
    print("R_RC_FP_DROP_CTRL=%.4f" % fp_drop_ctrl)
    print("R_RC_FP_GAT_BAND=%.4f" % fp_gat_band)
    print("R_RC_FP_GAT_BAND_SD=%.4f" % fp_gat_band_sd)
    print("R_RC_FP_GAT_BAND_CI_LO=%.4f" % fp_gat_band_ci[0])
    print("R_RC_FP_GAT_BAND_CI_HI=%.4f" % fp_gat_band_ci[1])
    print("R_RC_VETO_BAND=%.4f" % veto_band)
    print("R_RC_V4_FALSE_MAIN=%.4f" % v4_false_main)
    print("R_RC_V1_BAND=%.4f" % v1_band)
    print("R_RC_V3_BAND=%.4f" % v3_band)
    print("R_RC_V4_BAND=%.4f" % v4_band)
    print("R_RC_V4_CTRL=%.4f" % v4_ctrl)
    print("R_RC_E3_MAIN=%.4f" % e3_main)
    print("R_RC_DH_MAIN=%.4f" % dh_main)
    print("R_RC_DH_CTRL=%.4f" % dh_ctrl)
    print("R_RC_DH_BAND=%.4f" % dh_band)
    print("R_RC_SFS_MAIN=%.4f" % sfs_main)
    print("R_RC_SFS_CTRL=%.4f" % sfs_ctrl)
    print("R_RC_SFS_BAND=%.4f" % sfs_band)
    print("R_RC_C1_CONT=%s" % ("PASS" if c1 else "FAIL"))
    print("R_RC_C2_DISCONT=%s" % ("PASS" if c2 else "FAIL"))
    print("R_RC_C3_KEEP=%s" % ("PASS" if c3 else "FAIL"))
    print("R_RC_C4_VETO=%s" % ("PASS" if c4 else "FAIL"))
    print("R_RC_GUARD_CELL1=%d" % g_cell1)
    print("R_RC_GUARD_CELL1_N=%d" % g_cell1_n)
    print("R_RC_GUARD_CELL1_BAD=%d" % g_cell1_bad)
    print("R_RC_GUARD_CELL2=%d" % g_cell2)
    print("R_RC_GUARD_CELL2_N=%d" % g_cell2_n)
    print("R_RC_GUARD_CELL2_BAD=%d" % g_cell2_bad)
    print("R_RC_GUARD_CP5=%d" % g_cp5)
    print("R_RC_GUARD_CP5_SC2=%.4f" % g_cp5_sc2)
    print("R_RC_GUARD_MAD=%d" % g_mad)
    print("R_RC_GUARD_MAD_SIG=%.4f" % g_mad_sig)
    print("R_RC_REPRO=%d" % repro)
    print("R_RC_VERDICT=%s" % verdict)
    print("R_RC_ELAPSED=%.2f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
